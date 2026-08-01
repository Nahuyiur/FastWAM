#!/usr/bin/env python3
"""Summarize steady-state Megatron iteration timings from a training log."""

from __future__ import annotations

import argparse
import json
import re
import statistics
from pathlib import Path


PATTERN = re.compile(
    r"iteration\s+(?P<iteration>\d+)/\s*\d+.*?elapsed time per iteration \(ms\):\s*"
    r"(?P<milliseconds>[0-9.]+)"
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("log")
    parser.add_argument("--warmup-iters", type=int, default=40)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    rows = [
        (int(match.group("iteration")), float(match.group("milliseconds")) / 1000.0)
        for match in PATTERN.finditer(Path(args.log).read_text(errors="replace"))
    ]
    steady = [seconds for iteration, seconds in rows if iteration > args.warmup_iters]
    if len(steady) < 20:
        raise RuntimeError(
            f"Need at least 20 steady-state timings after warmup; found {len(steady)}"
        )
    ordered = sorted(steady)
    payload = {
        "log": str(Path(args.log).resolve()),
        "warmup_iters": args.warmup_iters,
        "num_logged_iters": len(rows),
        "num_steady_iters": len(steady),
        "step_seconds_mean": statistics.fmean(steady),
        "step_seconds_median": statistics.median(steady),
        "step_seconds_p90": ordered[int(0.9 * (len(ordered) - 1))],
        "step_seconds_min": min(steady),
        "step_seconds_max": max(steady),
        "steady_step_seconds": steady,
    }
    output = Path(args.output)
    output.write_text(json.dumps(payload, indent=2) + "\n")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
