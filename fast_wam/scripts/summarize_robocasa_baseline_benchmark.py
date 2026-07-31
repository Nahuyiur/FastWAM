#!/usr/bin/env python3
"""Summarize steady-state timing rows emitted by the baseline trainer."""

from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path


def percentile90(values: list[float]) -> float:
    ordered = sorted(values)
    return ordered[int(0.9 * (len(ordered) - 1))]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("profile_jsonl")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    source = Path(args.profile_jsonl)
    rows = [json.loads(line) for line in source.read_text().splitlines() if line.strip()]
    if len(rows) < 20:
        raise RuntimeError(f"Need at least 20 steady-state rows; found {len(rows)}")

    metric_names = sorted(
        key
        for key, value in rows[0].items()
        if isinstance(value, (int, float))
        and (key.endswith("_s_rank_avg") or key == "peak_memory_gb_rank_max")
    )
    payload: dict[str, object] = {
        "profile_jsonl": str(source.resolve()),
        "num_profiled_steps": len(rows),
        "first_step": int(rows[0]["step"]),
        "last_step": int(rows[-1]["step"]),
    }
    for name in metric_names:
        values = [float(row[name]) for row in rows]
        payload[name] = {
            "mean": statistics.fmean(values),
            "median": statistics.median(values),
            "p90": percentile90(values),
            "min": min(values),
            "max": max(values),
        }
    step_mean = payload["step_total_s_rank_avg"]["mean"]
    payload["optimizer_steps_per_sec"] = 1.0 / step_mean

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2) + "\n")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
