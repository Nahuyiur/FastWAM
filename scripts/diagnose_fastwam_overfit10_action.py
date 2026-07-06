#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import re
import time
from pathlib import Path

import numpy as np
import torch
from hydra import compose, initialize_config_dir
from hydra.utils import instantiate
from omegaconf import OmegaConf

from fastwam.runtime import _mixed_precision_to_model_dtype, _normalize_mixed_precision
from fastwam.utils.config_resolvers import register_default_resolvers


def step_from_checkpoint(path: Path) -> int:
    match = re.search(r"step_(\d+)\.pt$", path.name)
    return int(match.group(1)) if match else -1


def latest_weight(run_dir: Path) -> Path:
    candidates = sorted((run_dir / "checkpoints" / "weights").glob("step_*.pt"))
    if not candidates:
        raise FileNotFoundError(f"No step_*.pt under {run_dir / 'checkpoints' / 'weights'}")
    return max(candidates, key=step_from_checkpoint)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, default=None)
    parser.add_argument("--task", default=None)
    parser.add_argument("--config-dir", type=Path, default=Path("configs"))
    parser.add_argument("--split", choices=["train", "val"], default="train")
    parser.add_argument("--num-samples", type=int, default=10)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--num-inference-steps", type=int, default=10)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--output-json", type=Path, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run_dir = args.run_dir.expanduser().resolve()
    ckpt = (args.checkpoint or latest_weight(run_dir)).expanduser().resolve()
    register_default_resolvers()
    if args.task:
        config_dir = args.config_dir.expanduser().resolve()
        with initialize_config_dir(version_base="1.3", config_dir=str(config_dir)):
            cfg = compose(config_name="train", overrides=[f"task={args.task}"])
    else:
        cfg = OmegaConf.load(run_dir / "config.yaml")
    precision = _normalize_mixed_precision(str(cfg.get("mixed_precision", "bf16")))
    dtype = _mixed_precision_to_model_dtype(precision)
    model = instantiate(cfg.model, model_dtype=dtype, device=str(args.device))
    model.load_checkpoint(str(ckpt))
    model.eval()
    ds = instantiate(cfg.data[args.split])
    n = min(int(args.num_samples), len(ds))
    rows = []
    for i in range(n):
        sample = ds[i]
        input_image = sample["video"][:, 0].unsqueeze(0)
        proprio = sample["proprio"][0]
        gt = sample["action"].float()
        t0 = time.time()
        with torch.no_grad():
            pred = model.infer_action(
                prompt=None,
                input_image=input_image,
                action_horizon=int(gt.shape[0]),
                num_video_frames=int(sample["video"].shape[1]),
                proprio=proprio,
                context=sample["context"],
                context_mask=sample["context_mask"],
                text_cfg_scale=1.0,
                num_inference_steps=int(args.num_inference_steps),
                seed=int(args.seed) + i,
                rand_device="cpu",
                tiled=False,
            )["action"].float()
        norm = ds.processor.normalizer.normalizers["action"]["default"]
        pred_raw = norm.backward(pred).float()
        gt_raw = norm.backward(gt).float()
        rows.append(
            {
                "idx": i,
                "episode_index": int(sample["episode_index"]),
                "window_start": int(sample["window_start"]),
                "prompt": str(sample["prompt"]),
                "mae_norm": float((pred - gt).abs().mean()),
                "rmse_norm": float(torch.sqrt(((pred - gt) ** 2).mean())),
                "mae_raw": float((pred_raw - gt_raw).abs().mean()),
                "dim_mae_raw": [round(float(x), 6) for x in (pred_raw - gt_raw).abs().mean(dim=0)],
                "pred_raw_first": [round(float(x), 6) for x in pred_raw[0]],
                "gt_raw_first": [round(float(x), 6) for x in gt_raw[0]],
                "seconds": round(time.time() - t0, 3),
            }
        )
    summary = {
        "run_dir": str(run_dir),
        "checkpoint": str(ckpt),
        "split": args.split,
        "num_samples": len(rows),
        "mean_mae_norm": float(np.mean([r["mae_norm"] for r in rows])) if rows else None,
        "mean_rmse_norm": float(np.mean([r["rmse_norm"] for r in rows])) if rows else None,
        "mean_mae_raw": float(np.mean([r["mae_raw"] for r in rows])) if rows else None,
        "rows": rows,
    }
    text = json.dumps(summary, indent=2, ensure_ascii=False)
    print(text)
    if args.output_json is not None:
        args.output_json.expanduser().resolve().parent.mkdir(parents=True, exist_ok=True)
        args.output_json.expanduser().resolve().write_text(text, encoding="utf-8")


if __name__ == "__main__":
    main()
