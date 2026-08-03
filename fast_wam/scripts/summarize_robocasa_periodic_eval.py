#!/usr/bin/env python3
"""Validate, merge, summarize, and optionally upload periodic RoboCasa evals."""

from __future__ import annotations

import argparse
import json
import os
from collections import defaultdict
from pathlib import Path


def read_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def summarize(root: Path, expected_episodes: int) -> tuple[dict, list[dict], list[Path]]:
    rows: list[dict] = []
    errors: list[dict] = []
    videos: list[Path] = []
    for shard in sorted(root.glob("shard_*")):
        rows.extend(read_jsonl(shard / "episode_results.jsonl"))
        errors.extend(read_jsonl(shard / "errors.jsonl"))
        videos.extend(path for path in (shard / "videos").rglob("*.mp4") if path.stat().st_size > 0)

    by_bucket: dict[str, list[bool]] = defaultdict(list)
    by_task: dict[str, list[bool]] = defaultdict(list)
    for row in rows:
        success = bool(row.get("success"))
        by_bucket[str(row["bucket"])].append(success)
        by_task[str(row["task"])].append(success)

    def rate(values: list[bool]) -> float:
        return sum(values) / len(values) if values else 0.0

    summary = {
        "expected_episodes": expected_episodes,
        "num_episodes": len(rows),
        "num_errors": len(errors),
        "num_videos": len(videos),
        "num_successes": sum(bool(row.get("success")) for row in rows),
        "success_rate": rate([bool(row.get("success")) for row in rows]),
        "by_bucket": {
            key: {"episodes": len(values), "success_rate": rate(values)}
            for key, values in sorted(by_bucket.items())
        },
        "by_task": {
            key: {"episodes": len(values), "success_rate": rate(values)}
            for key, values in sorted(by_task.items())
        },
    }
    return summary, rows, videos


def upload_wandb(
    *, root: Path, summary: dict, videos: list[Path], step: int, entity: str, project: str, run_id: str
) -> str:
    import wandb

    run = wandb.init(
        entity=entity,
        project=project,
        id=run_id,
        name=run_id,
        resume="allow",
        dir=str(root / "wandb"),
        job_type="periodic-eval",
        tags=["robocasa", "megatron", "periodic-eval"],
    )
    run.define_metric("checkpoint_step")
    run.define_metric("periodic_eval/*", step_metric="checkpoint_step")
    payload: dict[str, object] = {
        "checkpoint_step": step,
        "periodic_eval/success_rate": summary["success_rate"],
        "periodic_eval/num_successes": summary["num_successes"],
        "periodic_eval/num_episodes": summary["num_episodes"],
        "periodic_eval/num_errors": summary["num_errors"],
    }
    for bucket, metrics in summary["by_bucket"].items():
        payload[f"periodic_eval/bucket/{bucket}"] = metrics["success_rate"]
    for index, video in enumerate(videos):
        payload[f"periodic_eval/video_{index:02d}"] = wandb.Video(str(video), format="mp4")
    try:
        run.log(payload)
        return run.url
    finally:
        run.finish()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", required=True, type=Path)
    parser.add_argument("--checkpoint-step", required=True, type=int)
    parser.add_argument("--expected-episodes", type=int, default=16)
    parser.add_argument("--wandb-entity", default=None)
    parser.add_argument("--wandb-project", default="robocasa-acg-fastwam")
    parser.add_argument("--wandb-run-id", default=None)
    args = parser.parse_args()

    summary, rows, videos = summarize(args.root, args.expected_episodes)
    (args.root / "episode_results.jsonl").write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows)
    )
    (args.root / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n")
    valid = (
        summary["num_episodes"] == args.expected_episodes
        and summary["num_errors"] == 0
        and summary["num_videos"] == args.expected_episodes
    )
    if not valid:
        print(json.dumps(summary, indent=2))
        return 2

    if args.wandb_entity and args.wandb_run_id:
        try:
            (args.root / "wandb").mkdir(exist_ok=True)
            url = upload_wandb(
                root=args.root,
                summary=summary,
                videos=videos,
                step=args.checkpoint_step,
                entity=args.wandb_entity,
                project=args.wandb_project,
                run_id=args.wandb_run_id,
            )
            (args.root / "WANDB_DONE").write_text(url + "\n")
        except Exception as exc:
            (args.root / "WANDB_ERROR.txt").write_text(repr(exc) + "\n")
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
