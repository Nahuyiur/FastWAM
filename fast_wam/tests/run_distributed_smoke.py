"""Real Megatron-Core TP forward/backward/optimizer/DCP smoke test."""

from __future__ import annotations

import argparse
import json
from dataclasses import replace
from pathlib import Path

import torch
import torch.distributed as dist

from fast_wam.checkpoint import load_megatron_dcp, save_megatron_dcp
from fast_wam.config import FastWAMConfig
from fast_wam.distributed import initialize, transformer_config
from fast_wam.model import FastWAMModel


def _inputs(cfg: FastWAMConfig, device: torch.device):
    generator = torch.Generator(device="cpu").manual_seed(1234)
    batch = 1
    value = {
        "input_latents": torch.randn(batch, cfg.video.in_dim, 3, 4, 8, generator=generator),
        "action": torch.randn(batch, cfg.action_horizon, cfg.action.action_dim, generator=generator),
        "context": torch.randn(batch, 5, cfg.video.text_dim, generator=generator),
        "context_mask": torch.ones(batch, 5, dtype=torch.bool),
        "proprio": torch.randn(batch, cfg.action_horizon, cfg.proprio_dim, generator=generator),
        "image_is_pad": torch.zeros(batch, 9, dtype=torch.bool),
        "action_is_pad": torch.zeros(batch, cfg.action_horizon, dtype=torch.bool),
        "stochastic_seed": 77,
    }
    return {key: item.to(device) if isinstance(item, torch.Tensor) else item for key, item in value.items()}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tp", type=int, default=2)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    info = initialize(args.tp)
    device = torch.device("cuda", int(torch.cuda.current_device()))
    dtype = torch.bfloat16
    cfg = replace(
        FastWAMConfig.tiny(),
        training_attention_backend="structured_sdpa",
        training_kernel_mode="optimized",
    )
    torch.manual_seed(42)
    torch.cuda.manual_seed(42)
    model = FastWAMModel(cfg, transformer_config(cfg, args.tp, dtype)).to(device, dtype)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1.0e-4)
    loss, metrics = model(**_inputs(cfg, device))
    loss.backward()
    grad_finite = all(
        parameter.grad is None or torch.isfinite(parameter.grad).all().item()
        for parameter in model.parameters()
    )
    optimizer.step()
    optimizer.zero_grad(set_to_none=True)

    losses = [None for _ in range(dist.get_world_size())]
    dist.all_gather_object(losses, float(loss.detach()))
    if max(losses) - min(losses) > 1.0e-5:
        raise RuntimeError(f"TP ranks disagree on loss: {losses}")
    if not grad_finite:
        raise RuntimeError("Non-finite gradient in distributed smoke test")

    output = Path(args.output).resolve()
    save_megatron_dcp(model, output)
    reloaded = FastWAMModel(cfg, transformer_config(cfg, args.tp, dtype)).to(device, dtype)
    load_megatron_dcp(reloaded, output)
    for expected, actual in zip(model.parameters(), reloaded.parameters(), strict=True):
        torch.testing.assert_close(actual, expected, atol=0, rtol=0)

    if info.global_rank == 0:
        print(
            json.dumps(
                {
                    "passed": True,
                    "world_size": dist.get_world_size(),
                    "tp_size": args.tp,
                    "loss": losses[0],
                    "loss_video": float(metrics["loss_video"]),
                    "loss_action": float(metrics["loss_action"]),
                    "grad_finite": grad_finite,
                    "dcp_roundtrip": True,
                },
                indent=2,
            ),
            flush=True,
        )
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
