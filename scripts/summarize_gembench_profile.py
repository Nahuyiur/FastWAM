#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
from pathlib import Path
from statistics import mean, median


def _load_rows(path: Path) -> list[dict]:
    rows = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def _pct(values: list[float], q: float) -> float:
    if not values:
        return float("nan")
    ordered = sorted(values)
    idx = min(max(round((len(ordered) - 1) * q), 0), len(ordered) - 1)
    return float(ordered[idx])


def summarize(rows: list[dict]) -> dict:
    keys = [
        "step_total_s_rank_avg",
        "data_wait_s_rank_avg",
        "forward_loss_s_rank_avg",
        "backward_s_rank_avg",
        "grad_clip_s_rank_avg",
        "optimizer_step_s_rank_avg",
        "scheduler_step_s_rank_avg",
        "zero_grad_s_rank_avg",
        "log_gather_s_rank_avg",
        "peak_memory_gb_rank_max",
    ]
    out = {"num_profiled_steps": len(rows)}
    if rows:
        out["first_step"] = int(rows[0].get("step", -1))
        out["last_step"] = int(rows[-1].get("step", -1))
    for key in keys:
        values = [float(row[key]) for row in rows if key in row]
        if not values:
            continue
        out[key] = {
            "mean": mean(values),
            "median": median(values),
            "p90": _pct(values, 0.90),
            "min": min(values),
            "max": max(values),
        }
    if rows and "step_total_s_rank_avg" in rows[0]:
        total_mean = out["step_total_s_rank_avg"]["mean"]
        out["optimizer_steps_per_sec"] = 1.0 / total_mean if total_mean > 0 else 0.0
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description="Summarize FastWAM GEMBench profile JSONL.")
    parser.add_argument("profile_jsonl")
    parser.add_argument("--json-output")
    args = parser.parse_args()
    rows = _load_rows(Path(args.profile_jsonl))
    payload = summarize(rows)
    text = json.dumps(payload, indent=2, sort_keys=True)
    print(text)
    if args.json_output:
        Path(args.json_output).parent.mkdir(parents=True, exist_ok=True)
        Path(args.json_output).write_text(text + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
