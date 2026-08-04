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


def summarize(
    root: Path,
    expected_episodes: int,
    *,
    expected_replan_steps: int = 32,
    expected_inference_steps: int = 20,
    protocol_tag: str = "fastwam_formal_baseline_v1",
    expected_attention_backend: str = "structured_sdpa",
    expected_kernel_mode: str = "reference",
    expected_render_backend: str = "egl",
) -> tuple[dict, list[dict], list[Path]]:
    rows: list[dict] = []
    errors: list[dict] = []
    videos: list[Path] = []
    protocol_errors: list[str] = []
    for shard in sorted(path for path in root.glob("shard_*") if path.is_dir()):
        rows.extend(read_jsonl(shard / "episode_results.jsonl"))
        errors.extend(read_jsonl(shard / "errors.jsonl"))
        videos.extend(path for path in (shard / "videos").rglob("*.mp4") if path.stat().st_size > 0)
        config_path = shard / "eval_config.json"
        if not config_path.is_file():
            protocol_errors.append(f"{shard.name}: missing eval_config.json")
            continue
        config = json.loads(config_path.read_text())
        if int(config.get("replan_steps", -1)) != expected_replan_steps:
            protocol_errors.append(
                f"{shard.name}: replan_steps={config.get('replan_steps')} "
                f"expected={expected_replan_steps}"
            )
        if int(config.get("fastwam_num_inference_steps", -1)) != expected_inference_steps:
            protocol_errors.append(
                f"{shard.name}: inference_steps={config.get('fastwam_num_inference_steps')} "
                f"expected={expected_inference_steps}"
            )
        if config.get("render_backend") != expected_render_backend:
            protocol_errors.append(
                f"{shard.name}: render_backend={config.get('render_backend')} "
                f"expected={expected_render_backend}"
            )
        if config.get("validate_camera_integrity") is not True:
            protocol_errors.append(
                f"{shard.name}: camera integrity validation was disabled"
            )
        runtime = config.get("policy_runtime_contract") or {}
        if runtime.get("eval_num_inference_steps") != expected_inference_steps:
            protocol_errors.append(
                f"{shard.name}: runtime inference contract was not restored"
            )
        if runtime.get("checkpoint_joint_action_video_attention") is not True:
            protocol_errors.append(
                f"{shard.name}: checkpoint joint-attention contract is not true"
            )
        if runtime.get("checkpoint_training_attention_backend") != expected_attention_backend:
            protocol_errors.append(
                f"{shard.name}: checkpoint attention backend="
                f"{runtime.get('checkpoint_training_attention_backend')} "
                f"expected={expected_attention_backend}"
            )
        if runtime.get("checkpoint_training_kernel_mode") != expected_kernel_mode:
            protocol_errors.append(
                f"{shard.name}: checkpoint kernel mode="
                f"{runtime.get('checkpoint_training_kernel_mode')} "
                f"expected={expected_kernel_mode}"
            )

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
        "protocol_tag": protocol_tag,
        "protocol_errors": protocol_errors,
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
    parser.add_argument("--expected-replan-steps", type=int, default=32)
    parser.add_argument("--expected-inference-steps", type=int, default=20)
    parser.add_argument("--protocol-tag", default="fastwam_formal_baseline_v1")
    parser.add_argument("--expected-attention-backend", default="structured_sdpa")
    parser.add_argument("--expected-kernel-mode", default="reference")
    parser.add_argument("--expected-render-backend", default="egl")
    parser.add_argument("--wandb-entity", default=None)
    parser.add_argument("--wandb-project", default="robocasa-acg-fastwam")
    parser.add_argument("--wandb-run-id", default=None)
    args = parser.parse_args()

    summary, rows, videos = summarize(
        args.root,
        args.expected_episodes,
        expected_replan_steps=args.expected_replan_steps,
        expected_inference_steps=args.expected_inference_steps,
        protocol_tag=args.protocol_tag,
        expected_attention_backend=args.expected_attention_backend,
        expected_kernel_mode=args.expected_kernel_mode,
        expected_render_backend=args.expected_render_backend,
    )
    (args.root / "episode_results.jsonl").write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows)
    )
    (args.root / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n")
    valid = (
        summary["num_episodes"] == args.expected_episodes
        and summary["num_errors"] == 0
        and summary["num_videos"] == args.expected_episodes
        and not summary["protocol_errors"]
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
