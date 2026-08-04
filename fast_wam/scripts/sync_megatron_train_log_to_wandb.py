#!/usr/bin/env python3
"""Backfill and follow Megatron training metrics from its text log into W&B."""

from __future__ import annotations

import argparse
import json
import math
import os
import re
import time
from pathlib import Path


_ITERATION_RE = re.compile(
    r"\[(?P<timestamp>\d{4}-\d{2}-\d{2} [^\]]+)\]\s+"
    r"iteration\s+(?P<iteration>\d+)/\s*(?P<max_iterations>\d+)\s*\|"
    r"(?P<body>.*)"
)

_FIELDS = {
    "consumed samples": ("train/consumed_samples", int),
    "elapsed time per iteration (ms)": ("performance/iteration_time_ms", float),
    "learning rate": ("train/learning_rate", float),
    "global batch size": ("train/global_batch_size", int),
    "loss": ("train/loss", float),
    "video loss": ("train/video_loss", float),
    "action loss": ("train/action_loss", float),
    "loss scale": ("train/loss_scale", float),
    "grad norm": ("train/grad_norm", float),
    "number of skipped iterations": ("train/skipped_iterations", int),
    "number of nan iterations": ("train/nan_iterations", int),
}


def parse_iteration_line(line: str) -> tuple[int, dict] | None:
    match = _ITERATION_RE.search(line)
    if match is None:
        return None

    iteration = int(match.group("iteration"))
    max_iterations = int(match.group("max_iterations"))
    body = match.group("body")
    payload: dict[str, int | float | str] = {
        "train/iteration": iteration,
        "train/max_iterations": max_iterations,
        "train/progress": iteration / max(max_iterations, 1),
        "source/timestamp": match.group("timestamp"),
    }
    for field, (metric, value_type) in _FIELDS.items():
        value_match = re.search(rf"(?:^|\|)\s*{re.escape(field)}:\s*([^|]+)", body)
        if value_match is None:
            continue
        raw_value = value_match.group(1).strip()
        try:
            value = value_type(float(raw_value)) if value_type is int else value_type(raw_value)
        except ValueError:
            continue
        if isinstance(value, float) and not math.isfinite(value):
            continue
        payload[metric] = value
    if "train/loss" not in payload:
        return None
    return iteration, payload


def parse_log(path: Path) -> list[tuple[int, dict]]:
    rows: dict[int, dict] = {}
    with path.open(errors="ignore") as stream:
        for line in stream:
            parsed = parse_iteration_line(line)
            if parsed is not None:
                rows[parsed[0]] = parsed[1]
    return sorted(rows.items())


def _read_last_step(path: Path) -> int:
    try:
        return int(json.loads(path.read_text())["last_logged_step"])
    except (FileNotFoundError, KeyError, TypeError, ValueError, json.JSONDecodeError):
        return -1


def _write_last_step(path: Path, step: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps({"last_logged_step": step}, indent=2) + "\n")
    temporary.replace(path)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--log-path", required=True, type=Path)
    parser.add_argument("--state-path", required=True, type=Path)
    parser.add_argument("--entity", required=True)
    parser.add_argument("--project", required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--run-name", required=True)
    parser.add_argument("--wandb-dir", required=True, type=Path)
    parser.add_argument("--poll-seconds", type=float, default=30.0)
    parser.add_argument("--stop-file", type=Path, default=None)
    parser.add_argument("--follow", action="store_true")
    args = parser.parse_args()

    import wandb

    args.wandb_dir.mkdir(parents=True, exist_ok=True)
    run = wandb.init(
        entity=args.entity,
        project=args.project,
        id=args.run_id,
        name=args.run_name,
        resume="allow",
        dir=str(args.wandb_dir),
        job_type="training-log-sync",
        tags=["robocasa", "megatron", "log-sync"],
        config={"metric_source": str(args.log_path), "sync_type": "training_log_sidecar"},
    )
    run.define_metric("train/iteration")
    run.define_metric("*", step_metric="train/iteration")
    try:
        while True:
            last_step = _read_last_step(args.state_path)
            if args.log_path.exists():
                new_rows = [(step, payload) for step, payload in parse_log(args.log_path) if step > last_step]
                for step, payload in new_rows:
                    payload["source/log"] = str(args.log_path)
                    run.log(payload)
                    _write_last_step(args.state_path, step)
                if new_rows:
                    print(f"[wandb-sync] logged through iteration={new_rows[-1][0]}", flush=True)
            else:
                print(f"[wandb-sync] waiting for {args.log_path}", flush=True)
            if not args.follow or (args.stop_file is not None and args.stop_file.exists()):
                return 0
            time.sleep(args.poll_seconds)
    finally:
        run.finish()


if __name__ == "__main__":
    raise SystemExit(main())
