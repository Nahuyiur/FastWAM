#!/usr/bin/env python3
"""BF16 semantic parity gate for FlexAttention and structured SDPA."""

from __future__ import annotations

import argparse
import gc
import json
import time
from dataclasses import replace

import torch
import torch.distributed as dist

from fast_wam.config import FastWAMConfig
from fast_wam.distributed import initialize, transformer_config
from fast_wam.model import FastWAMModel


def _inputs(
    cfg: FastWAMConfig,
    device: torch.device,
    scale: str,
    seed: int,
) -> dict:
    generator = torch.Generator(device="cpu").manual_seed(seed + 6)
    latent_shape = (3, 4, 8) if scale == "tiny" else (6, 14, 28)
    image_frames = 9 if scale == "tiny" else 21

    def normal(*shape):
        return torch.randn(*shape, generator=generator).to(
            device=device,
            dtype=torch.bfloat16,
        )

    return {
        "input_latents": normal(1, cfg.video.in_dim, *latent_shape),
        "action": normal(1, cfg.action_horizon, cfg.action.action_dim),
        "context": normal(1, 8, cfg.video.text_dim),
        "context_mask": torch.ones(1, 8, dtype=torch.bool, device=device),
        "proprio": normal(1, cfg.proprio_dim),
        "image_is_pad": torch.zeros(
            1,
            image_frames,
            dtype=torch.bool,
            device=device,
        ),
        "action_is_pad": torch.zeros(
            1,
            cfg.action_horizon,
            dtype=torch.bool,
            device=device,
        ),
        "noise_video": normal(1, cfg.video.in_dim, *latent_shape),
        "noise_action": normal(1, cfg.action_horizon, cfg.action.action_dim),
        "timestep_video": torch.tensor(
            [425.0],
            device=device,
            dtype=torch.bfloat16,
        ),
        "timestep_action": torch.tensor(
            [675.0],
            device=device,
            dtype=torch.bfloat16,
        ),
        "context_is_dense": True,
    }


def _tensor_samples(tensor: torch.Tensor, count: int = 8) -> list[float]:
    flat = tensor.detach().reshape(-1)
    if flat.numel() <= count:
        values = flat
    else:
        indices = (
            torch.arange(
                count,
                device=flat.device,
                dtype=torch.int64,
            )
            * (flat.numel() - 1)
            // (count - 1)
        )
        values = flat[indices]
    return values.float().cpu().tolist()


def _model_signature(model: FastWAMModel, *, gradients: bool) -> dict:
    signature = {}
    for name, parameter in model.named_parameters():
        tensor = parameter.grad if gradients else parameter
        if tensor is None:
            continue
        value = tensor.detach().float()
        signature[name] = {
            "norm": float(torch.linalg.vector_norm(value).cpu()),
            "samples": _tensor_samples(value),
        }
        del value
    return signature


def _run_backend(
    model: FastWAMModel,
    backend: str,
    inputs: dict,
    device: torch.device,
) -> dict:
    model.fast_wam_config = replace(
        model.fast_wam_config,
        training_attention_backend=backend,
        training_kernel_mode="optimized",
    )
    model.mot.training_attention_backend = backend
    model.zero_grad(set_to_none=True)
    model.train()
    weight_signature = _model_signature(model, gradients=False)
    torch.cuda.reset_peak_memory_stats(device)
    started = time.perf_counter()
    loss, metrics = model.training_loss_encoded(**inputs)
    loss.backward()
    torch.cuda.synchronize(device)
    elapsed = time.perf_counter() - started
    result = {
        "backend": backend,
        "loss": float(loss.detach().cpu()),
        "loss_video": float(metrics["loss_video"].cpu()),
        "loss_action": float(metrics["loss_action"].cpu()),
        "elapsed_seconds": elapsed,
        "peak_memory_gib": torch.cuda.max_memory_allocated(device) / 2**30,
        "weights": weight_signature,
        "gradients": _model_signature(model, gradients=True),
    }
    del loss, metrics
    return result


def _compare_signatures(reference: dict, candidate: dict) -> dict:
    if reference.keys() != candidate.keys():
        missing = sorted(reference.keys() - candidate.keys())
        extra = sorted(candidate.keys() - reference.keys())
        raise RuntimeError(f"signature keys differ: missing={missing}, extra={extra}")
    max_sample_error = 0.0
    max_norm_relative_error = 0.0
    worst_sample = ""
    worst_norm = ""
    for name in reference:
        expected = torch.tensor(reference[name]["samples"])
        actual = torch.tensor(candidate[name]["samples"])
        sample_error = float((actual - expected).abs().max())
        if sample_error > max_sample_error:
            max_sample_error = sample_error
            worst_sample = name
        expected_norm = reference[name]["norm"]
        actual_norm = candidate[name]["norm"]
        norm_relative_error = abs(actual_norm - expected_norm) / max(
            abs(expected_norm),
            1.0e-12,
        )
        if norm_relative_error > max_norm_relative_error:
            max_norm_relative_error = norm_relative_error
            worst_norm = name
    return {
        "max_sample_error": max_sample_error,
        "max_norm_relative_error": max_norm_relative_error,
        "worst_sample_parameter": worst_sample,
        "worst_norm_parameter": worst_norm,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--scale", choices=("tiny", "production"), default="tiny")
    parser.add_argument("--seed", type=int, default=17)
    args = parser.parse_args()

    info = initialize(1)
    try:
        device = torch.device("cuda", info.global_rank)
        if args.scale == "tiny":
            base_cfg = FastWAMConfig.tiny()
        else:
            base_cfg = FastWAMConfig()
            base_cfg = replace(
                base_cfg,
                action=replace(base_cfg.action, action_dim=12),
                proprio_dim=16,
            )
        base_cfg = replace(
            base_cfg,
            training_attention_backend="structured_sdpa",
            training_kernel_mode="optimized",
            joint_action_video_attention=True,
        )
        mcore_config = transformer_config(base_cfg, 1, torch.bfloat16)
        torch.manual_seed(args.seed)
        model = FastWAMModel(base_cfg, mcore_config).to(
            device=device,
            dtype=torch.bfloat16,
        )
        inputs = _inputs(base_cfg, device, args.scale, args.seed)
        reference = _run_backend(
            model,
            "structured_sdpa",
            inputs,
            device,
        )
        candidate = _run_backend(
            model,
            "flex",
            inputs,
            device,
        )
        weight_errors = _compare_signatures(
            reference["weights"],
            candidate["weights"],
        )
        gradient_errors = _compare_signatures(
            reference["gradients"],
            candidate["gradients"],
        )
        errors = {
            "loss_abs_error": abs(candidate["loss"] - reference["loss"]),
            "loss_relative_error": abs(candidate["loss"] - reference["loss"])
            / max(abs(reference["loss"]), 1.0e-12),
            "loss_video_abs_error": abs(
                candidate["loss_video"] - reference["loss_video"]
            ),
            "loss_video_relative_error": abs(
                candidate["loss_video"] - reference["loss_video"]
            )
            / max(abs(reference["loss_video"]), 1.0e-12),
            "loss_action_abs_error": abs(
                candidate["loss_action"] - reference["loss_action"]
            ),
            "loss_action_relative_error": abs(
                candidate["loss_action"] - reference["loss_action"]
            )
            / max(abs(reference["loss_action"]), 1.0e-12),
            "reference_losses": {
                "loss": reference["loss"],
                "loss_video": reference["loss_video"],
                "loss_action": reference["loss_action"],
            },
            "candidate_losses": {
                "loss": candidate["loss"],
                "loss_video": candidate["loss_video"],
                "loss_action": candidate["loss_action"],
            },
            "weight_signature": weight_errors,
            "gradient_signature": gradient_errors,
        }
        if weight_errors["max_sample_error"] != 0.0:
            raise RuntimeError(f"model weights changed between backends: {errors}")
        max_loss_relative_error = max(
            errors["loss_relative_error"],
            errors["loss_video_relative_error"],
            errors["loss_action_relative_error"],
        )
        if max_loss_relative_error > 2.5e-3:
            raise RuntimeError(f"loss parity failed: {errors}")
        if gradient_errors["max_sample_error"] > 5.0e-2:
            raise RuntimeError(f"sampled gradient parity failed: {errors}")
        if gradient_errors["max_norm_relative_error"] > 1.0e-2:
            raise RuntimeError(f"gradient norm parity failed: {errors}")
        summary = {
            "status": "PASS",
            "scale": args.scale,
            "seed": args.seed,
            "num_parameters": sum(parameter.numel() for parameter in model.parameters()),
            "reference": {
                key: value
                for key, value in reference.items()
                if key not in {"weights", "gradients"}
            },
            "candidate": {
                key: value
                for key, value in candidate.items()
                if key not in {"weights", "gradients"}
            },
            "errors": errors,
        }
        if info.global_rank == 0:
            print(json.dumps(summary, indent=2, sort_keys=True), flush=True)
        del model
        gc.collect()
        torch.cuda.empty_cache()
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()


if __name__ == "__main__":
    main()
