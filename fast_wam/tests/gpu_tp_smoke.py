#!/usr/bin/env python3
"""Small real-Megatron TP inference smoke; run with torchrun."""

from __future__ import annotations

import argparse
import os

import torch
import torch.distributed as dist

from fast_wam.checkpoint import load_lerobot_checkpoint
from fast_wam.config import FastWAMConfig
from fast_wam.distributed import initialize, transformer_config
from fast_wam.model import FastWAMModel


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--tp", type=int, required=True)
    parser.add_argument("--checkpoint")
    args = parser.parse_args()
    info = initialize(args.tp)
    device = torch.device("cuda", int(os.environ["LOCAL_RANK"]))
    cfg = FastWAMConfig.from_pretrained(args.checkpoint) if args.checkpoint else FastWAMConfig.tiny()
    model = FastWAMModel(cfg, transformer_config(cfg, args.tp, torch.float32)).to(device)
    if args.checkpoint:
        load_lerobot_checkpoint(model, args.checkpoint, strict=True)
        latent_shape = (1, cfg.video.in_dim, 1, 14, 28)
        context_len = cfg.tokenizer_max_len
    else:
        latent_shape = (1, cfg.video.in_dim, 1, 4, 8)
        context_len = 5
    generator = torch.Generator(device="cpu").manual_seed(123)
    latents = torch.randn(latent_shape, generator=generator)
    context = torch.randn((1, context_len, cfg.video.text_dim), generator=generator)
    context_mask = torch.ones((1, context_len), dtype=torch.bool)
    proprio = torch.randn((1, cfg.proprio_dim), generator=generator)
    output = model.infer_action(latents, context, context_mask, proprio).to(device)
    gathered = [torch.empty_like(output) for _ in range(dist.get_world_size())]
    dist.all_gather(gathered, output)
    max_rank_diff = max(float((item - gathered[0]).abs().max()) for item in gathered)
    if not torch.isfinite(output).all() or max_rank_diff != 0.0:
        raise RuntimeError(f"TP smoke failed: finite={torch.isfinite(output).all()} diff={max_rank_diff}")
    if info.global_rank == 0:
        print(
            f"PASS tp={args.tp} checkpoint={bool(args.checkpoint)} "
            f"shape={tuple(output.shape)} max_rank_diff={max_rank_diff}"
        )
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
