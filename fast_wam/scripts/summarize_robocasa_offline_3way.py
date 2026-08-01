#!/usr/bin/env python3
"""Aggregate repeated RoboCasa ordinary-online/offline/WDS-offline benchmarks."""

from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path


MODES = (
    "ordinary_online",
    "ordinary_offline",
    "webdataset_offline",
)


def percentile(values: list[float], fraction: float) -> float:
    ordered = sorted(values)
    return ordered[int(fraction * (len(ordered) - 1))]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root")
    parser.add_argument("--global-batch-size", type=int, default=4)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    root = Path(args.root).expanduser().resolve()
    payload = {
        "root": str(root),
        "global_batch_size": int(args.global_batch_size),
        "modes": {},
    }
    all_steps: dict[str, list[float]] = {}
    for mode in MODES:
        summaries = [
            json.loads(path.read_text(encoding="utf-8"))
            for path in sorted(root.glob(f"repeat_*/{mode}/summary.json"))
        ]
        if not summaries:
            raise RuntimeError(f"No benchmark summaries for {mode}")
        repeat_means = [float(value["step_seconds_mean"]) for value in summaries]
        steps = [
            float(seconds)
            for value in summaries
            for seconds in value["steady_step_seconds"]
        ]
        all_steps[mode] = steps
        payload["modes"][mode] = {
            "num_repeats": len(summaries),
            "num_steady_steps": len(steps),
            "repeat_step_seconds_mean": repeat_means,
            "step_seconds_mean": statistics.fmean(steps),
            "step_seconds_median": statistics.median(steps),
            "step_seconds_p90": percentile(steps, 0.9),
            "repeat_mean_stddev": (
                statistics.stdev(repeat_means) if len(repeat_means) > 1 else 0.0
            ),
        }
    baseline = statistics.fmean(all_steps["ordinary_online"])
    ordinary_offline = statistics.fmean(all_steps["ordinary_offline"])
    for mode in MODES:
        mean = float(payload["modes"][mode]["step_seconds_mean"])
        payload["modes"][mode]["speedup_vs_ordinary_online"] = baseline / mean
        payload["modes"][mode]["speedup_vs_ordinary_offline"] = (
            ordinary_offline / mean
        )
        payload["modes"][mode]["global_samples_per_second"] = (
            float(args.global_batch_size) / mean
        )
    output = Path(args.output)
    output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
