#!/usr/bin/env python3
"""CPU-only 1-sample overfit verifier for the Wan tiny preset.

This is not the production training path; it exists so a sandbox without CUDA
or Slurm can still exercise the exact model/loss implementation end to end.
Use `wan/scripts/overfit.sh` for Megatron DCP/GPU overfit.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
import torch.nn.functional as F

from wan.model.config import PRESETS
from wan.model.scheduler import WanFlowMatchScheduler
from wan.model.wan_dit import WanFlowTrainingModel


@torch.no_grad()
def sample(model, context, shape, steps, seed):
    scheduler = WanFlowMatchScheduler()
    scheduler.set_timesteps(steps, shift=5.0, training=False)
    gen = torch.Generator(device="cpu")
    gen.manual_seed(seed)
    latents = torch.randn(shape, generator=gen)
    model.eval()
    for timestep in scheduler.timesteps:
        timestep = timestep.unsqueeze(0)
        pred = model.dit(latents, timestep=timestep, context=context)
        latents = scheduler.step(pred, timestep, latents)
    return latents


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sample", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--iters", type=int, default=200)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--train-timesteps", type=int, default=32)
    parser.add_argument("--eval-steps", type=int, default=32)
    parser.add_argument("--seed", type=int, default=1234)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    item = torch.load(args.sample, map_location="cpu", weights_only=False)
    gt = item.get("input_latents", item.get("latents")).float().unsqueeze(0)
    context = item["context"].float().unsqueeze(0)

    cfg = PRESETS["tiny"]
    model = WanFlowTrainingModel(
        cfg,
        train_timesteps=args.train_timesteps,
        disable_timestep_weight=True,
    )
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.0)

    losses = []
    model.train()
    for step in range(1, args.iters + 1):
        opt.zero_grad(set_to_none=True)
        loss = model(gt, context)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        losses.append(float(loss.detach()))
        if step == 1 or step % max(1, args.iters // 10) == 0:
            print(f"iter={step} loss={losses[-1]:.8f}")

    pred = sample(model, context, gt.shape, args.eval_steps, args.seed)
    latent_mse = F.mse_loss(pred.float(), gt.float()).item()
    result = {
        "losses": losses,
        "initial_loss": losses[0],
        "final_loss": losses[-1],
        "latent_mse": latent_mse,
        "pred_latents": pred,
        "gt_latents": gt,
        "sample": args.sample,
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(result, output)
    print(f"initial_loss={losses[0]:.8f}")
    print(f"final_loss={losses[-1]:.8f}")
    print(f"latent_mse={latent_mse:.8f}")
    print(f"saved {output}")


if __name__ == "__main__":
    main()
