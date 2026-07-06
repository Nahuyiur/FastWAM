#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from hydra import compose, initialize_config_dir
from hydra.utils import instantiate

from fastwam.runtime import _mixed_precision_to_model_dtype, _normalize_mixed_precision
from fastwam.utils.config_resolvers import register_default_resolvers


def load_cfg(task: str, config_dir: Path):
    with initialize_config_dir(version_base="1.3", config_dir=str(config_dir.resolve())):
        return compose(config_name="train", overrides=[f"task={task}"])


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", default="robocasa_acg_v1_fastwam_overfit10")
    parser.add_argument("--config-dir", type=Path, default=Path("configs"))
    parser.add_argument("--baseline-checkpoint", type=Path, required=True)
    parser.add_argument("--overfit-checkpoint", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--num-samples", type=int, default=10)
    parser.add_argument("--num-inference-steps", type=int, default=10)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--device", default="cuda:0")
    return parser.parse_args()


def infer_checkpoint(model, dataset, checkpoint: Path, n: int, seed: int, steps: int) -> tuple[np.ndarray, dict]:
    model.load_checkpoint(str(checkpoint.expanduser().resolve()))
    model.eval()
    norm = dataset.processor.normalizer.normalizers["action"]["default"]
    preds = []
    gts = []
    rows = []
    for i in range(n):
        sample = dataset[i]
        gt_norm = sample["action"].float()
        with torch.no_grad():
            pred_norm = model.infer_action(
                prompt=None,
                input_image=sample["video"][:, 0].unsqueeze(0),
                action_horizon=int(gt_norm.shape[0]),
                num_video_frames=int(sample["video"].shape[1]),
                proprio=sample["proprio"][0],
                context=sample["context"],
                context_mask=sample["context_mask"],
                text_cfg_scale=1.0,
                num_inference_steps=int(steps),
                seed=int(seed) + i,
                rand_device="cpu",
                tiled=False,
            )["action"].float()
        pred = norm.backward(pred_norm).float().cpu().numpy()
        gt = norm.backward(gt_norm).float().cpu().numpy()
        diff = pred - gt
        preds.append(pred)
        gts.append(gt)
        rows.append(
            {
                "idx": i,
                "episode_index": int(sample["episode_index"]),
                "window_start": int(sample["window_start"]),
                "mae": float(np.abs(diff).mean()),
                "rmse": float(np.sqrt(np.square(diff).mean())),
                "max_abs": float(np.abs(diff).max()),
                "dim_mae": np.abs(diff).mean(axis=0).round(6).tolist(),
            }
        )
    pred_arr = np.stack(preds, axis=0)
    gt_arr = np.stack(gts, axis=0)
    diff_arr = pred_arr - gt_arr
    summary = {
        "checkpoint": str(checkpoint.expanduser().resolve()),
        "num_samples": int(n),
        "mean_mae": float(np.abs(diff_arr).mean()),
        "mean_rmse": float(np.sqrt(np.square(diff_arr).mean())),
        "mean_max_abs_per_sample": float(np.mean([r["max_abs"] for r in rows])),
        "dim_mae": np.abs(diff_arr).mean(axis=(0, 1)).round(6).tolist(),
        "rows": rows,
    }
    return pred_arr, {"gt": gt_arr, "summary": summary}


def main() -> None:
    args = parse_args()
    register_default_resolvers()
    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    cfg = load_cfg(args.task, args.config_dir)
    precision = _normalize_mixed_precision(str(cfg.get("mixed_precision", "bf16")))
    dtype = _mixed_precision_to_model_dtype(precision)
    model = instantiate(cfg.model, model_dtype=dtype, device=str(args.device))
    dataset = instantiate(cfg.data.train)
    n = min(int(args.num_samples), len(dataset))

    baseline_pred, baseline_payload = infer_checkpoint(
        model, dataset, args.baseline_checkpoint, n, args.seed, args.num_inference_steps
    )
    overfit_pred, overfit_payload = infer_checkpoint(
        model, dataset, args.overfit_checkpoint, n, args.seed, args.num_inference_steps
    )
    gt = baseline_payload["gt"]

    npz_path = output_dir / "policy_action_full_arrays.npz"
    np.savez_compressed(npz_path, gt=gt, baseline_pred=baseline_pred, overfit_pred=overfit_pred)

    summary = {
        "task": args.task,
        "num_samples": n,
        "action_shape": list(gt.shape),
        "num_inference_steps": int(args.num_inference_steps),
        "seed": int(args.seed),
        "baseline": baseline_payload["summary"],
        "overfit": overfit_payload["summary"],
        "npz_path": str(npz_path),
    }
    summary_path = output_dir / "policy_action_full_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
