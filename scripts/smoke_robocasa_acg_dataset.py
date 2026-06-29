#!/usr/bin/env python
import argparse
import json
import sys
from pathlib import Path

import hydra
import torch
from hydra.utils import instantiate
from omegaconf import OmegaConf
from torch.utils.data import DataLoader

REPO_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_DIR / "src"))

from fastwam.utils.config_resolvers import register_default_resolvers


def tensor_summary(x: torch.Tensor) -> dict:
    x_float = x.detach().to(torch.float32)
    return {
        "shape": list(x.shape),
        "dtype": str(x.dtype),
        "min": float(x_float.min().item()),
        "max": float(x_float.max().item()),
        "mean": float(x_float.mean().item()),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", default="robocasa_acg_v1_fastwam_smoke_dataset")
    parser.add_argument("--num-batches", type=int, default=2)
    parser.add_argument("--require-text-cache", action="store_true")
    args, unknown = parser.parse_known_args()

    register_default_resolvers()
    with hydra.initialize_config_dir(config_dir=str(REPO_DIR / "configs"), version_base="1.3"):
        overrides = [f"task={args.task}", *unknown]
        if args.require_text_cache:
            overrides += [
                "data.train.allow_missing_text_embeds=false",
                "data.val.allow_missing_text_embeds=false",
            ]
        cfg = hydra.compose(config_name="train", overrides=overrides)

    train_ds = instantiate(cfg.data.train)
    val_ds = instantiate(cfg.data.val)
    loader = DataLoader(train_ds, batch_size=1, shuffle=False, num_workers=0)

    batches = []
    for batch_idx, batch in enumerate(loader):
        if batch_idx >= args.num_batches:
            break
        video = batch["video"]
        action = batch["action"]
        proprio = batch["proprio"]
        context = batch["context"]
        context_mask = batch["context_mask"]
        assert video.ndim == 5, video.shape
        assert action.shape[-2:] == (cfg.data.train.action_horizon, 12), action.shape
        assert proprio.shape[-2:] == (cfg.data.train.action_horizon, 16), proprio.shape
        assert context.shape[-2:] == (cfg.data.train.context_len, 4096), context.shape
        assert context_mask.shape[-1] == cfg.data.train.context_len, context_mask.shape
        batches.append(
            {
                "batch_idx": batch_idx,
                "episode_index": int(batch["episode_index"][0].item()),
                "window_start": int(batch["window_start"][0].item()),
                "video": tensor_summary(video),
                "action": tensor_summary(action),
                "proprio": tensor_summary(proprio),
                "context_mask_true": int(context_mask.sum().item()),
            }
        )

    result = {
        "task": args.task,
        "train_len": len(train_ds),
        "val_len": len(val_ds),
        "batches": batches,
    }
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
