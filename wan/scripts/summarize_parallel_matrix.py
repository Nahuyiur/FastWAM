#!/usr/bin/env python3
"""Summarize Wan parallel matrix speed and approximate MFU.

The matrix uses Megatron's GPT dummy args, so Megatron's built-in TFLOP/MFU
helpers are not meaningful for Wan. This parser uses Wan config + the actual
pre-encoded sample shape, then combines the estimate with logged stable step
times from each case.
"""

from __future__ import annotations

import argparse
import csv
import re
from pathlib import Path

import torch

from wan.model.config import PRESETS

_ITER_RE = re.compile(
    r"iteration\s+(\d+)/\s*(\d+).*?"
    r"elapsed time per iteration \(ms\):\s*([0-9.]+).*?"
    r"mse loss:\s*([0-9.E+-]+).*?"
    r"grad norm:\s*([0-9.]+).*?"
    r"number of skipped iterations:\s*(\d+).*?"
    r"number of nan iterations:\s*(\d+)"
)
_MEMORY_RE = re.compile(
    r"\[Rank\s+(\d+)\].*?memory \(MB\).*?"
    r"allocated:\s*([0-9.]+).*?"
    r"max allocated:\s*([0-9.]+).*?"
    r"reserved:\s*([0-9.]+).*?"
    r"max reserved:\s*([0-9.]+)"
)


def _load_shape(sample: Path, preset: str):
    obj = torch.load(sample, map_location="cpu", weights_only=False)
    latents = obj.get("input_latents", obj.get("latents"))
    if latents.ndim == 5:
        latents = latents[0]
    context = obj["context"]
    cfg = PRESETS[preset]
    pt, ph, pw = cfg.patch_size
    video_tokens = (latents.shape[1] // pt) * (latents.shape[2] // ph) * (latents.shape[3] // pw)
    context_tokens = context.shape[-2]
    return cfg, video_tokens, context_tokens


def _forward_flops(cfg, video_tokens: int, context_tokens: int):
    s = video_tokens
    c = context_tokens
    d = cfg.dim
    ffn = cfg.ffn_dim
    patch_volume = cfg.patch_size[0] * cfg.patch_size[1] * cfg.patch_size[2]
    patch_in = cfg.in_dim * patch_volume
    head_out = cfg.out_dim * patch_volume

    patch = 2 * s * patch_in * d
    text = 2 * c * cfg.text_dim * d + 2 * c * d * d
    time_tokens = s if cfg.seperated_timestep else 1
    time = 2 * time_tokens * cfg.freq_dim * d + 2 * time_tokens * d * d + 2 * time_tokens * d * (6 * d)
    head = 2 * s * d * head_out

    self_proj = 8 * s * d * d
    self_attn = 4 * s * s * d
    cross_proj = 4 * s * d * d + 4 * c * d * d
    cross_attn = 4 * s * c * d
    mlp = 4 * s * d * ffn
    block = self_proj + self_attn + cross_proj + cross_attn + mlp

    blocks = cfg.num_layers * block
    forward = patch + text + time + blocks + head
    return {
        "forward": float(forward),
        "block_forward": float(blocks),
        "train_useful": float(3 * forward),
        "train_with_recompute": float(3 * forward + blocks),
    }


def _parse_log(path: Path, warmup: int):
    if not path.exists():
        return [], None, {}
    iterations = []
    memory_by_rank = {}
    for line in path.read_text(errors="replace").splitlines():
        match = _ITER_RE.search(line)
        if match:
            iteration, total, ms, loss, grad_norm, skipped, nan = match.groups()
            iterations.append(
                {
                    "iter": int(iteration),
                    "total": int(total),
                    "ms": float(ms),
                    "loss": float(loss),
                    "grad_norm": float(grad_norm),
                    "skipped": int(skipped),
                    "nan": int(nan),
                }
            )
            continue
        match = _MEMORY_RE.search(line)
        if match:
            rank, allocated, max_allocated, reserved, max_reserved = match.groups()
            rank = int(rank)
            current = memory_by_rank.get(rank, {})
            memory_by_rank[rank] = {
                "allocated_mb": max(float(allocated), current.get("allocated_mb", 0.0)),
                "max_allocated_mb": max(float(max_allocated), current.get("max_allocated_mb", 0.0)),
                "reserved_mb": max(float(reserved), current.get("reserved_mb", 0.0)),
                "max_reserved_mb": max(float(max_reserved), current.get("max_reserved_mb", 0.0)),
            }
    stable = [item for item in iterations if item["iter"] > warmup]
    if not stable:
        stable = iterations
    avg_ms = sum(item["ms"] for item in stable) / len(stable) if stable else None
    return iterations, avg_ms, memory_by_rank


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", required=True, type=Path)
    parser.add_argument("--sample", required=True, type=Path)
    parser.add_argument("--preset", default="ti2v-5b")
    parser.add_argument("--peak-tflops-per-gpu", type=float, default=989.0)
    parser.add_argument("--warmup-iters", type=int, default=2)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    cfg, video_tokens, context_tokens = _load_shape(args.sample, args.preset)
    flops = _forward_flops(cfg, video_tokens, context_tokens)
    peak_per_gpu = args.peak_tflops_per_gpu * 1e12

    summary_path = args.root / "summary.tsv"
    rows = []
    if summary_path.exists():
        with summary_path.open() as handle:
            for row in csv.DictReader(handle, delimiter="\t"):
                rows.append(row)
    else:
        for case_dir in sorted(p for p in args.root.iterdir() if p.is_dir()):
            rows.append({"case": case_dir.name, "gpus": "", "status": "", "seconds": ""})

    output_rows = []
    for row in rows:
        case = row["case"]
        log_path = args.root / case / "overfit.log"
        iterations, avg_ms, memory_by_rank = _parse_log(log_path, args.warmup_iters)
        last = iterations[-1] if iterations else None
        max_allocated_mb = max((m["max_allocated_mb"] for m in memory_by_rank.values()), default=None)
        max_reserved_mb = max((m["max_reserved_mb"] for m in memory_by_rank.values()), default=None)
        gpus = int(row["gpus"]) if row.get("gpus") else 0
        status = int(row["status"]) if row.get("status") else (0 if iterations else 1)
        useful_mfu = None
        recompute_hfu = None
        if avg_ms and gpus:
            seconds = avg_ms / 1000.0
            useful_mfu = flops["train_useful"] / (seconds * gpus * peak_per_gpu)
            recompute_hfu = flops["train_with_recompute"] / (seconds * gpus * peak_per_gpu)
        output_rows.append(
            {
                "case": case,
                "gpus": gpus,
                "status": status,
                "iters": len(iterations),
                "avg_ms_excl_warmup": avg_ms,
                "last_loss": None if last is None else last["loss"],
                "last_grad_norm": None if last is None else last["grad_norm"],
                "skipped": None if last is None else last["skipped"],
                "nan": None if last is None else last["nan"],
                "max_allocated_gb": None if max_allocated_mb is None else max_allocated_mb / 1024.0,
                "max_reserved_gb": None if max_reserved_mb is None else max_reserved_mb / 1024.0,
                "mfu_useful": useful_mfu,
                "hfu_with_recompute": recompute_hfu,
            }
        )

    out = args.output or (args.root / "speed_mfu_summary.tsv")
    with out.open("w", newline="") as handle:
        fieldnames = [
            "case",
            "gpus",
            "status",
            "iters",
            "avg_ms_excl_warmup",
            "last_loss",
            "last_grad_norm",
            "skipped",
            "nan",
            "max_allocated_gb",
            "max_reserved_gb",
            "mfu_useful",
            "hfu_with_recompute",
        ]
        writer = csv.DictWriter(handle, fieldnames=fieldnames, delimiter="\t")
        writer.writeheader()
        for row in output_rows:
            writer.writerow(row)

    print(f"video_tokens={video_tokens} context_tokens={context_tokens} layers={cfg.num_layers} dim={cfg.dim}")
    print(f"forward_tflops={flops['forward'] / 1e12:.3f}")
    print(f"train_useful_tflops={flops['train_useful'] / 1e12:.3f}")
    print(f"train_with_recompute_tflops={flops['train_with_recompute'] / 1e12:.3f}")
    print(f"peak_tflops_per_gpu={args.peak_tflops_per_gpu:.1f}")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
