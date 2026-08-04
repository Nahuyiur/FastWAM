#!/usr/bin/env python3
"""Certify a full-model RoboCasa training smoke before a formal launch."""

from __future__ import annotations

import argparse
import json
import re
import statistics
from pathlib import Path


ITERATION_RE = re.compile(
    r"iteration\s+(?P<iteration>\d+)/\s*(?P<maximum>\d+)\s*\|(?P<body>.*)"
)

FATAL_PATTERN = re.compile(
    r"Traceback \(most recent call last\)|CUDA out of memory|FloatingPointError|"
    r"WORKER_FAILED|NCCL (?:WARN|ERROR)|"
    r"nccl(?:Unhandled|System|Internal|Invalid|Remote)[A-Za-z]*Error|"
    r"ProcessGroupNCCL.*(?:error|abort|watchdog)",
    re.IGNORECASE,
)


def _field(body: str, name: str) -> float | None:
    match = re.search(rf"(?:^|\|)\s*{re.escape(name)}:\s*([^|]+)", body)
    if match is None:
        return None
    try:
        return float(match.group(1).strip())
    except ValueError:
        return None


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--initial-dcp", type=Path, required=True)
    parser.add_argument("--latent-cache", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--min-iteration", type=int, default=40)
    parser.add_argument("--warmup-iterations", type=int, default=20)
    parser.add_argument("--baseline-step-seconds", type=float, default=4.476276)
    parser.add_argument(
        "--expected-attention-backend",
        choices=("sdpa", "structured_sdpa"),
        required=True,
    )
    parser.add_argument(
        "--expected-kernel-mode",
        choices=("reference",),
        default="reference",
    )
    args = parser.parse_args()

    root = args.run_root.resolve()
    log_path = root / "train.log"
    text = log_path.read_text(errors="ignore")
    rows = []
    for line in text.splitlines():
        match = ITERATION_RE.search(line)
        if match is None:
            continue
        body = match.group("body")
        milliseconds = _field(body, "elapsed time per iteration (ms)")
        if milliseconds is None:
            continue
        rows.append(
            {
                "iteration": int(match.group("iteration")),
                "seconds": milliseconds / 1000.0,
                "loss": _field(body, "loss"),
                "skipped": _field(body, "number of skipped iterations"),
                "nan": _field(body, "number of nan iterations"),
            }
        )
    latest = max((row["iteration"] for row in rows), default=-1)
    steady = [
        row["seconds"]
        for row in rows
        if row["iteration"] > args.warmup_iterations
    ]
    ordered = sorted(steady)
    p90 = ordered[min(len(ordered) - 1, int(0.9 * len(ordered)))] if ordered else None
    mean = statistics.fmean(steady) if steady else None
    checkpoint = args.checkpoint.resolve()
    distcp_files = list(checkpoint.glob("__*_0.distcp"))
    checks = {
        "log_exists": log_path.is_file(),
        "iteration_reached": latest >= args.min_iteration,
        "steady_rows": len(steady) >= 10,
        "finite_losses": bool(rows)
        and all(row["loss"] is not None and row["loss"] == row["loss"] for row in rows),
        "no_skipped": bool(rows) and all(row["skipped"] == 0 for row in rows),
        "no_nan": bool(rows) and all(row["nan"] == 0 for row in rows),
        "no_fatal": FATAL_PATTERN.search(text) is None,
        "cached_input": "input=ordinary latents=cached" in text,
        "initial_dcp": str(args.initial_dcp.resolve()) in text,
        "attention_kernel": (
            f"attention={args.expected_attention_backend}, "
            f"kernels={args.expected_kernel_mode}"
        )
        in text,
        "optimizer_contract": (
            "optimizer contract: AdamW weight_decay applies to every trainable parameter"
            in text
        ),
        "checkpoint_complete": (checkpoint / "common.pt").is_file()
        and (checkpoint / ".metadata").is_file()
        and len(distcp_files) == 4,
        "faster_than_baseline": mean is not None and mean < args.baseline_step_seconds,
    }
    passed = all(checks.values())
    payload = {
        "status": "PASS" if passed else "FAIL",
        "run_root": str(root),
        "checkpoint": str(checkpoint),
        "initial_dcp": str(args.initial_dcp.resolve()),
        "latent_cache": str(args.latent_cache.resolve()),
        "candidate": {
            "attention_backend": args.expected_attention_backend,
            "kernel_mode": args.expected_kernel_mode,
            "optimizer": "AdamW",
            "weight_decay_policy": "all_trainable",
        },
        "latest_iteration": latest,
        "warmup_iterations": args.warmup_iterations,
        "steady_step_seconds": {
            "count": len(steady),
            "mean": mean,
            "median": statistics.median(steady) if steady else None,
            "p90": p90,
        },
        "baseline_step_seconds": args.baseline_step_seconds,
        "speedup_vs_baseline": args.baseline_step_seconds / mean if mean else None,
        "checks": checks,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n")
    print(json.dumps(payload, indent=2))
    if not passed:
        raise SystemExit("RoboCasa full-model training smoke certification failed")


if __name__ == "__main__":
    main()
