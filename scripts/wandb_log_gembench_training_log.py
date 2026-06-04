#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import os
import re
import time
from pathlib import Path


def parse_rows(text: str) -> list[dict]:
    pattern = re.compile(
        r"(?P<date>\d\d/\d\d) \[(?P<time>\d\d:\d\d:\d\d)\].*?"
        r"epoch=(?P<epoch>\d+) step=(?P<step>\d+)/(?P<max>\d+)"
        r"(?P<body>.{0,700}?)(?=\n\d\d/\d\d \[|\Z)",
        re.S,
    )
    rows: list[dict] = []
    for match in pattern.finditer(text):
        body = match.group("body")
        values = {}
        for key in ("loss", "loss_action", "loss_video", "lr", "speed"):
            found = re.search(rf"{key}=([0-9.eE+-]+)", body)
            if found:
                value = float(found.group(1))
                if math.isfinite(value):
                    values[key] = value
        if "loss" not in values:
            continue
        rows.append(
            {
                "date": match.group("date"),
                "time": match.group("time"),
                "epoch": int(match.group("epoch")),
                "step": int(match.group("step")),
                "max_steps": int(match.group("max")),
                **values,
            }
        )
    return rows


def build_payload(row: dict, source: str, subproject: str) -> dict:
    payload = {
        "train/loss": row["loss"],
        "progress/epoch": row["epoch"],
        "progress/max_steps": row["max_steps"],
        "progress/fraction": row["step"] / max(row["max_steps"], 1),
        "source/date": row["date"],
        "source/time": row["time"],
        "source/log": source,
        "source/subproject": subproject,
    }
    if "loss_action" in row:
        payload["train/loss_action"] = row["loss_action"]
    if "loss_video" in row:
        payload["train/loss_video"] = row["loss_video"]
    if "lr" in row:
        payload["train/lr"] = row["lr"]
    if "speed" in row:
        payload["performance/steps_per_sec"] = row["speed"]
    return payload


def main() -> int:
    parser = argparse.ArgumentParser(description="Sync wrapped GEMBench trainer logs to W&B.")
    parser.add_argument("--log-path", required=True)
    parser.add_argument("--project", default=os.environ.get("WANDB_PROJECT", "trace-gembench"))
    parser.add_argument("--entity", default=os.environ.get("WANDB_ENTITY") or os.environ.get("WANDB_WORKSPACE"))
    parser.add_argument("--group", required=True)
    parser.add_argument("--name", required=True)
    parser.add_argument("--run-id", default=None)
    parser.add_argument("--subproject", required=True)
    parser.add_argument("--run-dir", default=None)
    parser.add_argument("--poll-seconds", type=float, default=60.0)
    parser.add_argument("--follow", action="store_true")
    parser.add_argument("--state-path", default=None)
    args = parser.parse_args()

    import wandb

    log_path = Path(args.log_path)
    state_path = Path(args.state_path) if args.state_path else log_path.with_suffix(log_path.suffix + ".wandb_state.json")
    last_logged_step = 0
    if state_path.exists():
        try:
            last_logged_step = int(json.loads(state_path.read_text()).get("last_logged_step", 0))
        except Exception:
            last_logged_step = 0

    config = {
        "source_log": str(log_path),
        "source_run_dir": args.run_dir,
        "subproject": args.subproject,
        "sync_type": "log_sidecar",
    }
    run = wandb.init(
        entity=args.entity or None,
        project=args.project,
        name=args.name,
        id=args.run_id,
        resume="allow" if args.run_id else None,
        group=args.group,
        job_type="log-sync",
        tags=["gembench", args.subproject, "log-sync"],
        config=config,
    )
    try:
        while True:
            if log_path.exists():
                rows = parse_rows(log_path.read_text(errors="ignore"))
                new_rows = [row for row in rows if row["step"] > last_logged_step]
                for row in new_rows:
                    wandb.log(build_payload(row, str(log_path), args.subproject), step=row["step"])
                    last_logged_step = row["step"]
                if new_rows:
                    state_path.write_text(json.dumps({"last_logged_step": last_logged_step}, indent=2))
                    print(f"[wandb-log-sync] logged through step={last_logged_step}", flush=True)
            else:
                print(f"[wandb-log-sync] waiting for log: {log_path}", flush=True)
            if not args.follow:
                break
            time.sleep(args.poll_seconds)
    finally:
        run.finish()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
