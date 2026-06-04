#!/usr/bin/env python
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import torch
from hydra import compose, initialize_config_dir
from hydra.core.global_hydra import GlobalHydra
from hydra.utils import instantiate
from omegaconf import OmegaConf

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT / "src"))
if str(PROJECT_ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

from check_gembench_loss_grad_parity import _autocast_context, _next_batch, _seed_everything
from fastwam.runtime import _mixed_precision_to_model_dtype, _normalize_mixed_precision
from fastwam.utils.config_resolvers import register_default_resolvers


def _compose_cfg(task: str, overrides: list[str]):
    cfg = compose(config_name="train", overrides=[f"task={task}", *overrides])
    OmegaConf.resolve(cfg)
    return cfg


def _set_trainable_like_trainer(model) -> None:
    model.eval()
    model.requires_grad_(False)
    model.dit.train()
    model.dit.requires_grad_(True)
    proprio_encoder = getattr(model, "proprio_encoder", None)
    if proprio_encoder is not None:
        proprio_encoder.train()
        proprio_encoder.requires_grad_(True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Single-process GEMBench FastWAM backward profiler.")
    parser.add_argument("--task", default="gembench_keysteps_bbox_3cam224_vaecache_b4a1_1e-4")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--skip-batches", type=int, default=0)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--mixed-precision", default=None)
    parser.add_argument("--sort-by", default="self_cuda_time_total")
    parser.add_argument("--row-limit", type=int, default=80)
    parser.add_argument("--table-output", default="runs/gembench_profiler/single_step_key_averages.txt")
    parser.add_argument("--json-output", default="runs/gembench_profiler/single_step_summary.json")
    parser.add_argument("overrides", nargs="*", help="Additional Hydra overrides.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    os.chdir(PROJECT_ROOT)
    register_default_resolvers()
    GlobalHydra.instance().clear()

    common_overrides = [
        "wandb.enabled=false",
        f"batch_size={args.batch_size}",
        "num_workers=0",
        *args.overrides,
    ]
    with initialize_config_dir(config_dir=str(PROJECT_ROOT / "configs"), version_base="1.3"):
        cfg = _compose_cfg(args.task, common_overrides)

    precision = args.mixed_precision or str(cfg.mixed_precision)
    dtype = _mixed_precision_to_model_dtype(_normalize_mixed_precision(precision))
    device = torch.device(args.device)

    _seed_everything(args.seed)
    dataset = instantiate(cfg.data.train)
    batch = _next_batch(dataset, args.batch_size, args.skip_batches)
    model = instantiate(cfg.model, model_dtype=dtype, device=str(device))
    _set_trainable_like_trainer(model)

    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(device)

    model.zero_grad(set_to_none=True)
    activities = [torch.profiler.ProfilerActivity.CPU]
    if device.type == "cuda":
        activities.append(torch.profiler.ProfilerActivity.CUDA)

    with torch.profiler.profile(
        activities=activities,
        record_shapes=False,
        profile_memory=False,
        with_stack=False,
    ) as prof:
        with _autocast_context(device, dtype):
            loss, loss_dict = model.training_loss(batch)
        loss.backward()
        if device.type == "cuda":
            torch.cuda.synchronize(device)

    table = prof.key_averages().table(sort_by=args.sort_by, row_limit=args.row_limit)
    table_path = Path(args.table_output)
    table_path.parent.mkdir(parents=True, exist_ok=True)
    table_path.write_text(table, encoding="utf-8")

    summary = {
        "task": args.task,
        "batch_size": args.batch_size,
        "skip_batches": args.skip_batches,
        "seed": args.seed,
        "device": str(device),
        "mixed_precision": precision,
        "loss": float(loss.detach().float().cpu().item()),
        "loss_metrics": {k: float(v) for k, v in loss_dict.items()},
        "peak_memory_gb": (
            float(torch.cuda.max_memory_allocated(device) / (1024**3)) if device.type == "cuda" else 0.0
        ),
        "table_output": str(table_path),
    }
    json_path = Path(args.json_output)
    json_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.write_text(__import__("json").dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(table)
    print(f"profile_table={table_path}")
    print(f"peak_memory_gb={summary['peak_memory_gb']:.3f}")


if __name__ == "__main__":
    main()
