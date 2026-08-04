#!/usr/bin/env python3
"""Production-shape BF16 baseline-vs-accelerated training parity gate."""

from __future__ import annotations

import argparse
import gc
import json
from dataclasses import replace
from pathlib import Path

import torch
import torch.distributed as dist

from fast_wam.checkpoint import load_megatron_dcp
from fast_wam.config import FastWAMConfig
from fast_wam.distributed import initialize, transformer_config
from fast_wam.model import FastWAMModel


def _inputs(cfg: FastWAMConfig, device: torch.device) -> dict:
    generator = torch.Generator(device="cpu").manual_seed(20260804)

    def normal(*shape: int) -> torch.Tensor:
        return torch.randn(*shape, generator=generator).to(device=device, dtype=torch.bfloat16)

    return {
        "input_latents": normal(1, 48, 3, 14, 28),
        "action": normal(1, 32, 12),
        "context": normal(1, 128, 4096),
        "context_mask": torch.ones(1, 128, dtype=torch.bool, device=device),
        "context_is_dense": True,
        "proprio": normal(1, 16),
        "image_is_pad": torch.zeros(1, 9, dtype=torch.bool, device=device),
        "action_is_pad": torch.zeros(1, 32, dtype=torch.bool, device=device),
        "noise_video": normal(1, 48, 3, 14, 28),
        "noise_action": normal(1, 32, 12),
        "timestep_video": torch.tensor([425.0], dtype=torch.bfloat16, device=device),
        "timestep_action": torch.tensor([675.0], dtype=torch.bfloat16, device=device),
    }


def _signature(model: FastWAMModel) -> dict[str, dict[str, float | list[float]]]:
    output: dict[str, dict[str, float | list[float]]] = {}
    for name, parameter in model.named_parameters():
        gradient = parameter.grad
        if gradient is None:
            continue
        flat = gradient.detach().reshape(-1).float()
        indices = torch.tensor(
            sorted({0, flat.numel() // 2, flat.numel() - 1}),
            device=flat.device,
        )
        output[name] = {
            "norm": float(torch.linalg.vector_norm(flat).cpu()),
            "samples": flat[indices].cpu().tolist(),
        }
    return output


def _run(
    *,
    base: FastWAMConfig,
    checkpoint: Path,
    device: torch.device,
    backend: str,
    kernel: str,
) -> dict:
    cfg = replace(
        base,
        training_attention_backend=backend,
        training_kernel_mode=kernel,
    )
    model = FastWAMModel(
        cfg,
        transformer_config(cfg, 1, torch.bfloat16),
    ).to(device=device, dtype=torch.bfloat16)
    load_megatron_dcp(model, checkpoint)
    model.train()
    inputs = _inputs(cfg, device)
    torch.cuda.reset_peak_memory_stats(device)
    loss, metrics = model.training_loss_encoded(**inputs)
    loss.backward()
    torch.cuda.synchronize(device)
    result = {
        "backend": backend,
        "kernel": kernel,
        "loss": float(loss.detach().cpu()),
        "loss_video": float(metrics["loss_video"].detach().cpu()),
        "loss_action": float(metrics["loss_action"].detach().cpu()),
        "gradient_signature": _signature(model),
        "peak_memory_gib": torch.cuda.max_memory_allocated(device) / 2**30,
    }
    del loss, metrics, inputs, model
    gc.collect()
    torch.cuda.empty_cache()
    return result


def _compare(reference: dict, candidate: dict) -> dict:
    ref_sig = reference["gradient_signature"]
    cand_sig = candidate["gradient_signature"]
    if ref_sig.keys() != cand_sig.keys():
        raise RuntimeError("Gradient parameter sets differ")
    max_sample_error = 0.0
    max_norm_relative_error = 0.0
    worst_sample = ""
    worst_norm = ""
    for name in ref_sig:
        expected = torch.tensor(ref_sig[name]["samples"])
        actual = torch.tensor(cand_sig[name]["samples"])
        sample_error = float((actual - expected).abs().max())
        norm_error = abs(cand_sig[name]["norm"] - ref_sig[name]["norm"]) / max(
            abs(ref_sig[name]["norm"]), 1.0e-12
        )
        if sample_error > max_sample_error:
            max_sample_error, worst_sample = sample_error, name
        if norm_error > max_norm_relative_error:
            max_norm_relative_error, worst_norm = norm_error, name
    losses = {}
    for key in ("loss", "loss_video", "loss_action"):
        absolute = abs(candidate[key] - reference[key])
        losses[key] = {
            "absolute_error": absolute,
            "relative_error": absolute / max(abs(reference[key]), 1.0e-12),
        }
    return {
        "losses": losses,
        "max_gradient_sample_error": max_sample_error,
        "max_gradient_norm_relative_error": max_norm_relative_error,
        "worst_gradient_sample": worst_sample,
        "worst_gradient_norm": worst_norm,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--initial-dcp", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--candidate-backend",
        choices=("sdpa", "structured_sdpa"),
        default="structured_sdpa",
    )
    parser.add_argument(
        "--candidate-kernel",
        choices=("reference", "optimized"),
        default="optimized",
    )
    args = parser.parse_args()

    info = initialize(1)
    try:
        device = torch.device("cuda", info.global_rank)
        base = FastWAMConfig()
        base = replace(
            base,
            action=replace(base.action, action_dim=12),
            proprio_dim=16,
            joint_action_video_attention=True,
        )
        reference = _run(
            base=base,
            checkpoint=args.initial_dcp,
            device=device,
            backend="sdpa",
            kernel="reference",
        )
        candidate = _run(
            base=base,
            checkpoint=args.initial_dcp,
            device=device,
            backend=args.candidate_backend,
            kernel=args.candidate_kernel,
        )
        errors = _compare(reference, candidate)
        max_loss_relative = max(
            value["relative_error"] for value in errors["losses"].values()
        )
        passed = (
            max_loss_relative <= 5.0e-3
            and errors["max_gradient_sample_error"] <= 5.0e-2
            and errors["max_gradient_norm_relative_error"] <= 2.0e-2
        )
        payload = {
            "status": "PASS" if passed else "FAIL",
            "initial_dcp": str(args.initial_dcp.resolve()),
            "shape_contract": {
                "latents": [1, 48, 3, 14, 28],
                "action": [1, 32, 12],
                "context": [1, 128, 4096],
                "proprio": [1, 16],
                "joint_action_video_attention": True,
                "dtype": "bfloat16",
            },
            "reference": {
                key: value for key, value in reference.items() if key != "gradient_signature"
            },
            "candidate": {
                key: value for key, value in candidate.items() if key != "gradient_signature"
            },
            "errors": errors,
        }
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(payload, indent=2) + "\n")
        print(json.dumps(payload, indent=2), flush=True)
        if not passed:
            raise SystemExit("Production RoboCasa training parity gate failed")
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()


if __name__ == "__main__":
    main()
