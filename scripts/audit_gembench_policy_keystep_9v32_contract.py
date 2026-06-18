#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any

import torch

from fastwam.datasets.gembench.microsteps_9v32 import (
    DEFAULT_FRAME_OFFSETS,
    GEMBenchKeyStepPolicy9V32Dataset,
)


def _parse_csv(value: str | None) -> list[str] | None:
    if value is None:
        return None
    items = [item.strip() for item in str(value).split(",") if item.strip()]
    return items or None


def _shape(value: Any) -> list[int] | None:
    if not isinstance(value, torch.Tensor):
        return None
    return [int(dim) for dim in value.shape]


def _write_markdown(path: Path, payload: dict[str, Any]) -> None:
    lines = [
        "# GEMBench Policy Key-Step 9V32 Contract Audit",
        "",
        f"Status: `{payload['status']}`",
        f"Samples: `{payload['dataset_len']}`",
        f"Checked examples: `{len(payload['examples'])}`",
        "",
        "| check | passed | detail |",
        "|---|---:|---|",
    ]
    for check in payload["checks"]:
        detail = json.dumps(check["detail"], ensure_ascii=True, sort_keys=True)
        if len(detail) > 260:
            detail = detail[:257] + "..."
        lines.append(f"| `{check['name']}` | {check['passed']} | `{detail}` |")
    lines.extend(
        [
            "",
            "## Example Transitions",
            "",
            "| idx | taskvar | episode | current_key | next_key | delta | policy_action_raw |",
            "|---:|---|---|---:|---:|---:|---|",
        ]
    )
    for example in payload["examples"]:
        action = json.dumps(example["policy_action_raw"], ensure_ascii=True)
        lines.append(
            "| {idx} | `{taskvar}` | `{episode_key}` | {current_key} | {next_key} | {delta} | `{action}` |".format(
                idx=example["idx"],
                taskvar=example["taskvar"],
                episode_key=example["episode_key"],
                current_key=example["policy_current_key_idx"],
                next_key=example["policy_next_key_idx"],
                delta=example["policy_key_delta"],
                action=action,
            )
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def run(args: argparse.Namespace) -> dict[str, Any]:
    dataset = GEMBenchKeyStepPolicy9V32Dataset(
        manifest_path=args.manifest,
        rgb_cache_dir=args.rgb_cache_dir,
        keysteps_dir=args.keysteps_dir,
        split=args.split,
        seed=args.seed,
        frame_offsets=DEFAULT_FRAME_OFFSETS,
        action_horizon=32,
        video_size=(int(args.image_size), int(args.image_size) * len(_parse_csv(args.camera_order))),
        camera_order=_parse_csv(args.camera_order),
        cache_camera_order=_parse_csv(args.cache_camera_order),
        window_stride=1,
        max_windows_per_demo=args.max_windows_per_demo,
        taskvars=_parse_csv(args.taskvars),
        policy_max_index_demos=args.policy_max_index_demos,
        vae_latent_cache_dir=args.vae_latent_cache_dir,
        text_embedding_cache_dir=args.text_embedding_cache_dir,
        context_len=int(args.context_len),
        cache_text_embeddings=True,
        cache_gripper_arrays=True,
        allow_missing_text_embeds=bool(args.allow_missing_text_embeds),
        pretrained_norm_stats=args.pretrained_norm_stats,
        norm_default_mode=args.norm_default_mode,
        stats_scan_limit=0,
        allow_partial_cache=False,
        policy_include_final_key=False,
        policy_min_key_delta=int(args.policy_min_key_delta),
        processor={"action_output_dim": 8, "proprio_output_dim": 8},
    )

    sample_indices = list(range(min(int(args.max_samples), len(dataset))))
    examples: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []
    delta_values: list[int] = []
    taskvar_counts: Counter[str] = Counter()
    saw_video_latents = False
    saw_video = False
    for idx in sample_indices:
        try:
            sample = dataset[idx]
            policy_action = sample["policy_action"]
            dense_action = sample["action"]
            image_is_pad = sample["image_is_pad"]
            has_video_latents = "video_latents" in sample
            has_video = "video" in sample
            saw_video_latents = saw_video_latents or has_video_latents
            saw_video = saw_video or has_video
            current_key = int(sample["policy_current_key_idx"])
            next_key = int(sample["policy_next_key_idx"])
            delta = int(next_key - current_key)
            delta_values.append(delta)
            taskvar = str(sample["taskvar"])
            taskvar_counts[taskvar] += 1
            if list(policy_action.shape) != [1, 8]:
                failures.append({"idx": idx, "error": f"bad_policy_action_shape={tuple(policy_action.shape)}"})
            if list(dense_action.shape) != [32, 8]:
                failures.append({"idx": idx, "error": f"bad_dense_action_shape={tuple(dense_action.shape)}"})
            if list(image_is_pad.shape) != [9]:
                failures.append({"idx": idx, "error": f"bad_image_is_pad_shape={tuple(image_is_pad.shape)}"})
            if not has_video_latents and not has_video:
                failures.append({"idx": idx, "error": "missing_video_or_video_latents"})
            examples.append(
                {
                    "idx": int(idx),
                    "taskvar": taskvar,
                    "episode_key": str(sample["episode_key"]),
                    "policy_current_key_idx": current_key,
                    "policy_next_key_idx": next_key,
                    "policy_key_delta": delta,
                    "video_anchor_indices": [current_key + int(offset) for offset in DEFAULT_FRAME_OFFSETS],
                    "policy_action_shape": _shape(policy_action),
                    "dense_action_shape": _shape(dense_action),
                    "video_shape": _shape(sample.get("video")),
                    "video_latents_shape": _shape(sample.get("video_latents")),
                    "policy_action_raw": [
                        float(v) for v in sample["policy_action_raw"].reshape(-1).detach().cpu().tolist()
                    ],
                }
            )
        except Exception as exc:
            failures.append({"idx": idx, "error": f"{type(exc).__name__}: {exc}"})

    checks = [
        {"name": "dataset_nonempty", "passed": len(dataset) > 0, "detail": len(dataset)},
        {
            "name": "policy_action_shape_1x8",
            "passed": not any("bad_policy_action_shape" in item.get("error", "") for item in failures),
            "detail": [item for item in failures if "bad_policy_action_shape" in item.get("error", "")][:5],
        },
        {
            "name": "wam_aux_action_shape_32x8",
            "passed": not any("bad_dense_action_shape" in item.get("error", "") for item in failures),
            "detail": [item for item in failures if "bad_dense_action_shape" in item.get("error", "")][:5],
        },
        {
            "name": "video_or_latents_present",
            "passed": saw_video or saw_video_latents,
            "detail": {"video": saw_video, "video_latents": saw_video_latents},
        },
        {
            "name": "all_sample_loads_succeeded",
            "passed": not failures,
            "detail": failures[:10],
        },
    ]
    status = "passed" if all(check["passed"] for check in checks) else "failed"
    return {
        "status": status,
        "eval_type": "gembench_policy_keystep_9v32_contract_audit",
        "official_full_score": False,
        "manifest": str(Path(args.manifest).expanduser().resolve()),
        "rgb_cache_dir": str(Path(args.rgb_cache_dir).expanduser().resolve()),
        "vae_latent_cache_dir": None if args.vae_latent_cache_dir in (None, "", "null") else str(args.vae_latent_cache_dir),
        "dataset_len": int(len(dataset)),
        "policy_target_type": "next_key_step",
        "policy_action_horizon": 1,
        "wam_aux_action_horizon": 32,
        "num_video_frames": 9,
        "camera_order": list(_parse_csv(args.camera_order)),
        "cache_camera_order": list(_parse_csv(args.cache_camera_order)),
        "delta_stats": {
            "min": min(delta_values) if delta_values else None,
            "max": max(delta_values) if delta_values else None,
        },
        "taskvar_counts_head": dict(taskvar_counts.most_common(20)),
        "checks": checks,
        "examples": examples,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Audit FastWAM GEMBench key-step policy + 9V32 aux contract.")
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--rgb-cache-dir", required=True)
    parser.add_argument("--keysteps-dir", default="/mnt/yuhan/datasets/GEMBench/train_dataset/keysteps_bbox/seed0")
    parser.add_argument("--vae-latent-cache-dir", default=None)
    parser.add_argument("--split", default="train")
    parser.add_argument("--seed", default="seed0")
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--camera-order", default="left_shoulder,right_shoulder,wrist,front")
    parser.add_argument("--cache-camera-order", default="left_shoulder,right_shoulder,wrist,front")
    parser.add_argument("--taskvars", default=None)
    parser.add_argument("--max-windows-per-demo", type=int, default=None)
    parser.add_argument("--policy-max-index-demos", type=int, default=None)
    parser.add_argument("--max-samples", type=int, default=8)
    parser.add_argument("--policy-min-key-delta", type=int, default=1)
    parser.add_argument("--text-embedding-cache-dir", default="./data/text_embeds_cache/gembench_microsteps_9v32")
    parser.add_argument("--context-len", type=int, default=128)
    parser.add_argument("--allow-missing-text-embeds", action="store_true")
    parser.add_argument("--pretrained-norm-stats", default="./data/gembench_microsteps_9v32_dataset_stats.json")
    parser.add_argument("--norm-default-mode", default="-2.0/2.0")
    parser.add_argument("--output-json", required=True)
    parser.add_argument("--output-md", default=None)
    args = parser.parse_args()

    payload = run(args)
    output_json = Path(args.output_json)
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(payload, ensure_ascii=True, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    if args.output_md:
        _write_markdown(Path(args.output_md), payload)
    print(
        json.dumps(
            {
                "status": payload["status"],
                "dataset_len": payload["dataset_len"],
                "policy_action_horizon": payload["policy_action_horizon"],
                "wam_aux_action_horizon": payload["wam_aux_action_horizon"],
            },
            ensure_ascii=True,
            sort_keys=True,
        )
    )
    return 0 if payload["status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
