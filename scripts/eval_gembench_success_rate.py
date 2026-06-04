#!/usr/bin/env python3
"""Run GEMBench official-style simulator success-rate evaluation for FastWAM.

This follows the robot-3dlotus GEMBench protocol:

* validation uses ``val_dataset/microsteps/seed100`` and the 31 train taskvars;
* test uses ``test_dataset/microsteps/seed{200,300,400,500,600}``;
* L1 reuses train taskvars, and L2/L3/L4 use the official taskvar lists;
* results are written as official-compatible ``seed*/results.jsonl`` rows with
  ``checkpoint/task/variation/num_demos/sr`` fields.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import re
import sys
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from fastwam.evaluation.gembench_success.policy import FastWAMGEMBenchPolicy
from fastwam.evaluation.gembench_success.simulator import evaluate_taskvar, set_eval_seed, split_taskvar
from fastwam.evaluation.gembench_success.splits import GEMBENCH_SPLITS


def _now_slug() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def _parse_taskvars(raw: list[str] | None) -> list[str] | None:
    if not raw:
        return None
    out: list[str] = []
    for item in raw:
        out.extend([part.strip() for part in item.split(",") if part.strip()])
    return out or None


def _load_existing(path: Path, checkpoint: str) -> set[str]:
    done: set[str] = set()
    if not path.exists():
        return done
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            if str(row.get("checkpoint")) == str(checkpoint):
                done.add(f"{row.get('task')}+{row.get('variation')}")
    return done


def _checkpoint_step(checkpoint: str) -> int:
    match = re.search(r"step_(\d+)\.pt$", Path(checkpoint).name)
    if not match:
        raise ValueError(
            f"Checkpoint basename must match step_XXXXXX.pt for official summarize compatibility, got: {checkpoint}"
        )
    return int(match.group(1))


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=True) + "\n", encoding="utf-8")


def _append_jsonl(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(payload, ensure_ascii=True) + "\n")
        f.flush()


def _read_result_rows(output_root: Path, checkpoint: str) -> list[dict[str, Any]]:
    rows = []
    for path in sorted(output_root.glob("seed*/results.jsonl")):
        with path.open("r", encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                row = json.loads(line)
                if str(row.get("checkpoint")) == str(checkpoint):
                    rows.append(row)
    return rows


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--splits", nargs="+", default=["val"], choices=sorted(GEMBENCH_SPLITS))
    parser.add_argument("--taskvars", nargs="*", default=None, help="Optional comma/space-separated taskvars for smoke runs.")
    parser.add_argument("--gembench-root", default=os.environ.get("GEMBENCH_ROOT", "/mnt/yuhan/datasets/GEMBench"))
    parser.add_argument("--output-root", default=None)
    parser.add_argument("--checkpoint", default=None, help="FastWAM weights checkpoint. Defaults to latest gembench run.")
    parser.add_argument("--task-name", default="gembench_keysteps_bbox_3cam224_1e-4")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--num-demos", type=int, default=20)
    parser.add_argument("--max-steps", type=int, default=25)
    parser.add_argument("--max-tries", type=int, default=10)
    parser.add_argument("--num-inference-steps", type=int, default=10)
    parser.add_argument("--replan-steps", type=int, default=1)
    parser.add_argument("--model-seed", type=int, default=-1)
    parser.add_argument("--image-size", type=int, nargs=2, default=[256, 256])
    parser.add_argument("--renderer", choices=["opengl", "opengl3"], default="opengl")
    parser.add_argument("--headless", action="store_true", default=True)
    parser.add_argument("--no-headless", dest="headless", action="store_false")
    parser.add_argument("--record-video", action="store_true")
    parser.add_argument("--video-fps", type=int, default=8)
    parser.add_argument("--no-skip-existing", dest="skip_existing", action="store_false", default=True)
    parser.add_argument("--min-z", type=float, default=None)
    parser.add_argument("--dry-run", action="store_true", help="Validate split/taskvar microstep coverage without loading FastWAM or RLBench.")
    parser.add_argument("--keep-going-on-error", action="store_true", help="Record unexpected policy/model errors as failed episodes instead of raising.")
    parser.add_argument("--log-level", default="INFO")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    logging.basicConfig(
        level=getattr(logging, str(args.log_level).upper()),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    output_root = Path(args.output_root) if args.output_root else PROJECT_ROOT / "runs" / "gembench_success_eval" / _now_slug()
    output_root.mkdir(parents=True, exist_ok=True)
    gembench_root = Path(args.gembench_root).expanduser().resolve()
    requested_taskvars = _parse_taskvars(args.taskvars)

    if args.dry_run:
        coverage = []
        for split_name in args.splits:
            split = GEMBENCH_SPLITS[split_name]
            taskvars = list(requested_taskvars or split.taskvars)
            for seed in split.seeds:
                base = gembench_root / split.dataset_split / "microsteps" / f"seed{seed}"
                missing = []
                for taskvar in taskvars:
                    task_str, variation = split_taskvar(taskvar)
                    if not (base / task_str / f"variation{variation}" / "episodes").exists():
                        missing.append(taskvar)
                coverage.append(
                    {
                        "split": split_name,
                        "seed": seed,
                        "microstep_dir": str(base),
                        "taskvars": len(taskvars),
                        "missing": missing,
                    }
                )
        payload = {"gembench_root": str(gembench_root), "coverage": coverage}
        _write_json(output_root / "dry_run.json", payload)
        print("[gembench-success-dry-run] " + json.dumps(payload, ensure_ascii=True), flush=True)
        return 1 if any(item["missing"] for item in coverage) else 0

    policy = FastWAMGEMBenchPolicy(
        checkpoint=args.checkpoint,
        task_name=args.task_name,
        device=args.device,
        num_inference_steps=args.num_inference_steps,
        replan_steps=args.replan_steps,
        model_seed=args.model_seed,
        min_z=args.min_z,
    )
    checkpoint_value = policy.actioner_checkpoint_value
    checkpoint_step = _checkpoint_step(checkpoint_value)
    config_payload = {
        "splits": args.splits,
        "taskvars": requested_taskvars,
        "gembench_root": str(gembench_root),
        "output_root": str(output_root),
        "checkpoint": checkpoint_value,
        "checkpoint_step": checkpoint_step,
        "num_demos": int(args.num_demos),
        "max_steps": int(args.max_steps),
        "max_tries": int(args.max_tries),
        "num_inference_steps": int(args.num_inference_steps),
        "replan_steps": int(args.replan_steps),
        "model_seed": int(args.model_seed),
        "image_size": [int(x) for x in args.image_size],
        "renderer": args.renderer,
        "headless": bool(args.headless),
    }
    _write_json(output_root / "eval_config.json", config_payload)

    all_rows = []
    for split_name in args.splits:
        split = GEMBENCH_SPLITS[split_name]
        taskvars = list(requested_taskvars or split.taskvars)
        for seed in split.seeds:
            microstep_data_dir = gembench_root / split.dataset_split / "microsteps" / f"seed{seed}"
            if not microstep_data_dir.exists():
                raise FileNotFoundError(
                    f"Missing GEMBench microsteps directory: {microstep_data_dir}. "
                    "Run scripts/extract_gembench_microsteps.sh first."
                )
            seed_dir = output_root / f"seed{seed}"
            results_path = seed_dir / "results.jsonl"
            detail_dir = seed_dir / "details"
            video_dir = seed_dir / "videos"
            done = _load_existing(results_path, checkpoint_value) if args.skip_existing else set()
            for taskvar in taskvars:
                task_str, variation = split_taskvar(taskvar)
                if taskvar in done:
                    logging.info("Skipping existing result: split=%s seed=%s taskvar=%s", split_name, seed, taskvar)
                    continue
                set_eval_seed(int(seed))
                if args.model_seed < 0 and hasattr(policy, "set_model_seed"):
                    policy.set_model_seed(int(seed))
                sr, details = evaluate_taskvar(
                    policy=policy,
                    taskvar=taskvar,
                    microstep_data_dir=microstep_data_dir,
                    num_demos=args.num_demos,
                    max_steps=args.max_steps,
                    max_tries=args.max_tries,
                    image_size=args.image_size,
                    cameras=policy.camera_order,
                    headless=args.headless,
                    renderer=args.renderer,
                    record_video=args.record_video,
                    video_dir=video_dir,
                    video_fps=args.video_fps,
                    seed=seed,
                    keep_going_on_error=args.keep_going_on_error,
                )
                if not details:
                    logging.warning(
                        "No evaluated episodes for split=%s seed=%s taskvar=%s; skipping official result row.",
                        split_name,
                        seed,
                        taskvar,
                    )
                    continue
                row = {
                    "checkpoint": checkpoint_value,
                    "checkpoint_step": int(checkpoint_step),
                    "task": task_str,
                    "variation": int(variation),
                    "num_demos": int(len(details)),
                    "sr": float(sr),
                    "split": split_name,
                    "official_level": split.official_level,
                    "seed": int(seed),
                }
                _append_jsonl(results_path, row)
                if details:
                    detail_path = detail_dir / f"{taskvar.replace('+', '_')}.jsonl"
                    for item in details:
                        _append_jsonl(detail_path, asdict(item))
                all_rows.append(row)
                logging.info("[summary-row] %s", json.dumps(row, ensure_ascii=True))

    result_rows = _read_result_rows(output_root, checkpoint_value)
    grouped: dict[str, list[float]] = {}
    for row in result_rows:
        grouped.setdefault(str(row["split"]), []).append(float(row["sr"]))
    summary = {
        "output_root": str(output_root),
        "checkpoint": checkpoint_value,
        "checkpoint_step": int(checkpoint_step),
        "num_new_rows": len(all_rows),
        "num_rows": len(result_rows),
        "split_success_rate": {k: float(sum(v) / max(len(v), 1)) for k, v in grouped.items()},
    }
    _write_json(output_root / "summary.json", summary)
    print("[gembench-success-summary] " + json.dumps(summary, ensure_ascii=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
