#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import torch
from hydra.utils import instantiate
from omegaconf import OmegaConf

from fastwam.evaluation.open_loop_wam import run_autoregressive_open_loop_wam_eval
from fastwam.runtime import _mixed_precision_to_model_dtype, _normalize_mixed_precision
from fastwam.utils.config_resolvers import register_default_resolvers


def _latest_weights_checkpoint(run_dir: Path) -> Path:
    weights_dir = run_dir / "checkpoints" / "weights"
    candidates = list(weights_dir.glob("step_*.pt"))
    if not candidates:
        raise FileNotFoundError(f"No step_*.pt checkpoint found under {weights_dir}")

    def step_num(path: Path) -> int:
        match = re.search(r"step_(\d+)\.pt$", path.name)
        return int(match.group(1)) if match else -1

    return max(candidates, key=step_num)


def _step_from_checkpoint(path: Path) -> int:
    match = re.search(r"step_(\d+)\.pt$", path.name)
    return int(match.group(1)) if match else 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run FastWAM GEMBench autoregressive open-loop WAM eval.")
    parser.add_argument("--run-dir", type=Path, default=None, help="Training run dir containing config.yaml/checkpoints.")
    parser.add_argument("--config", type=Path, default=None, help="Config YAML. Defaults to <run-dir>/config.yaml.")
    parser.add_argument("--checkpoint", type=Path, default=None, help="Weights checkpoint. Defaults to latest step_*.pt.")
    parser.add_argument("--output-dir", type=Path, default=None, help="Output root. Defaults to <run-dir>/eval_open_loop.")
    parser.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--split", choices=("train", "val"), default="val")
    parser.add_argument("--num-samples", type=int, default=1)
    parser.add_argument("--rollout-chunks", type=int, default=4)
    parser.add_argument("--chunk-stride", type=int, default=32)
    parser.add_argument("--num-inference-steps", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--video-fps", type=int, default=8)
    parser.add_argument("--tiled", action="store_true")
    parser.add_argument("--no-save-video", action="store_true")
    return parser.parse_args()


def main() -> None:
    register_default_resolvers()
    args = parse_args()
    run_dir = args.run_dir.expanduser().resolve() if args.run_dir is not None else None
    cfg_path = args.config
    if cfg_path is None:
        if run_dir is None:
            raise ValueError("--config is required when --run-dir is not provided.")
        cfg_path = run_dir / "config.yaml"
    cfg_path = cfg_path.expanduser().resolve()
    if not cfg_path.exists():
        raise FileNotFoundError(f"Missing config: {cfg_path}")

    checkpoint = args.checkpoint
    if checkpoint is None:
        if run_dir is None:
            raise ValueError("--checkpoint is required when --run-dir is not provided.")
        checkpoint = _latest_weights_checkpoint(run_dir)
    checkpoint = checkpoint.expanduser().resolve()
    if not checkpoint.exists():
        raise FileNotFoundError(f"Missing checkpoint: {checkpoint}")

    output_dir = args.output_dir
    if output_dir is None:
        output_dir = (run_dir / "eval_open_loop") if run_dir is not None else Path("eval_open_loop")
    output_dir = output_dir.expanduser().resolve()

    cfg = OmegaConf.load(cfg_path)
    mixed_precision = _normalize_mixed_precision(str(cfg.get("mixed_precision", "bf16")))
    model_dtype = _mixed_precision_to_model_dtype(mixed_precision)
    model_cfg = OmegaConf.create(OmegaConf.to_container(cfg.model, resolve=True))
    model = instantiate(model_cfg, model_dtype=model_dtype, device=str(args.device))
    model.load_checkpoint(str(checkpoint))
    model = model.to(args.device).eval()

    data_cfg = OmegaConf.create(OmegaConf.to_container(cfg.data[args.split], resolve=True))
    if "vae_latent_cache_dir" in data_cfg:
        data_cfg.vae_latent_cache_dir = None
    dataset = instantiate(data_cfg)

    with torch.no_grad():
        result = run_autoregressive_open_loop_wam_eval(
            model=model,
            dataset=dataset,
            output_dir=output_dir,
            global_step=_step_from_checkpoint(checkpoint),
            num_samples=args.num_samples,
            rollout_chunks=args.rollout_chunks,
            chunk_stride=args.chunk_stride,
            num_inference_steps=args.num_inference_steps,
            seed=args.seed,
            save_video=not args.no_save_video,
            video_fps=args.video_fps,
            tiled=args.tiled,
        )

    print(json.dumps(result, ensure_ascii=True, indent=2))


if __name__ == "__main__":
    main()
