#!/usr/bin/env python3
"""Real-MCore BF16 reference/optimized training-kernel parity smoke."""

from __future__ import annotations

from dataclasses import replace

import torch
import torch.distributed as dist

from fast_wam.config import FastWAMConfig
from fast_wam.distributed import initialize, transformer_config
from fast_wam.model import FastWAMModel


def _inputs(cfg: FastWAMConfig, device: torch.device) -> dict:
    generator = torch.Generator(device="cpu").manual_seed(23)
    batch = 2

    def normal(*shape):
        return torch.randn(*shape, generator=generator).to(
            device=device,
            dtype=torch.bfloat16,
        )

    return {
        "input_latents": normal(batch, cfg.video.in_dim, 3, 4, 8),
        "action": normal(
            batch,
            cfg.action_horizon,
            cfg.action.action_dim,
        ),
        "context": normal(batch, 5, cfg.video.text_dim),
        "context_mask": torch.ones(
            batch,
            5,
            dtype=torch.bool,
            device=device,
        ),
        "proprio": normal(batch, cfg.proprio_dim),
        "image_is_pad": torch.zeros(
            batch,
            9,
            dtype=torch.bool,
            device=device,
        ),
        "action_is_pad": torch.zeros(
            batch,
            cfg.action_horizon,
            dtype=torch.bool,
            device=device,
        ),
        "noise_video": normal(batch, cfg.video.in_dim, 3, 4, 8),
        "noise_action": normal(
            batch,
            cfg.action_horizon,
            cfg.action.action_dim,
        ),
        "timestep_video": torch.tensor(
            [125.0, 725.0],
            device=device,
            dtype=torch.bfloat16,
        ),
        "timestep_action": torch.tensor(
            [250.0, 850.0],
            device=device,
            dtype=torch.bfloat16,
        ),
        "context_is_dense": True,
    }


def main() -> None:
    info = initialize(1)
    device = torch.device("cuda", info.global_rank)
    cfg = FastWAMConfig.tiny()
    mcore_config = transformer_config(cfg, 1, torch.bfloat16)
    torch.manual_seed(17)
    reference = FastWAMModel(cfg, mcore_config).to(
        device=device,
        dtype=torch.bfloat16,
    )
    optimized_cfg = replace(
        cfg,
        training_attention_backend="structured_sdpa",
        training_kernel_mode="optimized",
    )
    optimized = FastWAMModel(optimized_cfg, mcore_config).to(
        device=device,
        dtype=torch.bfloat16,
    )
    optimized.load_state_dict(reference.state_dict(), strict=True)
    values = _inputs(cfg, device)

    expected, _ = reference.training_loss_encoded(**values)
    expected.backward()
    actual, _ = optimized.training_loss_encoded(**values)
    actual.backward()
    torch.cuda.synchronize()

    loss_error = float((actual - expected).abs().detach())
    reference_grad = reference.video_expert.patch_embedding.weight.grad
    optimized_grad = optimized.video_expert.patch_embedding.weight.grad
    grad_error = float((optimized_grad - reference_grad).abs().max())
    if loss_error > 5.0e-3 or grad_error > 5.0e-3:
        raise RuntimeError(
            f"kernel parity failed: loss_error={loss_error}, "
            f"grad_error={grad_error}"
        )
    if info.global_rank == 0:
        print(
            "PASS "
            f"loss_reference={float(expected):.9f} "
            f"loss_optimized={float(actual):.9f} "
            f"loss_error={loss_error:.9f} "
            f"grad_max_error={grad_error:.9f}",
            flush=True,
        )
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
