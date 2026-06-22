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
from fastwam.datasets.gembench.policy_local_frame import action_local_to_world


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
        key_frameids_path=args.key_frameids_path,
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
        text_encoder_id=args.text_encoder_id,
        cache_text_embeddings=True,
        cache_gripper_arrays=True,
        allow_missing_text_embeds=bool(args.allow_missing_text_embeds),
        pretrained_norm_stats=args.pretrained_norm_stats,
        norm_default_mode=args.norm_default_mode,
        stats_scan_limit=0,
        allow_partial_cache=False,
        policy_include_final_key=False,
        policy_min_key_delta=int(args.policy_min_key_delta),
        policy_target_frame=args.policy_target_frame,
        policy_pcd_data_dir=args.policy_pcd_data_dir,
        policy_local_xyz_shift=args.policy_local_xyz_shift,
        policy_local_xyz_norm=bool(args.policy_local_xyz_norm),
        policy_local_rm_table=bool(args.policy_local_rm_table),
        policy_local_rm_robot=args.policy_local_rm_robot,
        policy_local_num_points=int(args.policy_local_num_points),
        policy_local_sample_seed=int(args.policy_local_sample_seed),
        policy_local_train_voxel_size=float(args.policy_local_train_voxel_size),
        policy_local_require_open3d=bool(args.policy_local_require_open3d),
        robot_3dlotus_root=args.robot_3dlotus_root,
        processor={"action_output_dim": 8, "proprio_output_dim": 8},
    )

    sample_indices = list(range(min(int(args.max_samples), len(dataset))))
    examples: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []
    delta_values: list[int] = []
    taskvar_counts: Counter[str] = Counter()
    saw_video_latents = False
    saw_video = False
    roundtrip_errors: list[float] = []
    pcd_action_errors: list[float] = []
    pcd_key_mismatches: list[dict[str, Any]] = []
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
            local_roundtrip_error = None
            policy_local_centroid = sample.get("policy_local_centroid")
            policy_local_radius = sample.get("policy_local_radius")
            if args.policy_target_frame == "official_pcd_local":
                if policy_local_centroid is None or policy_local_radius is None:
                    failures.append({"idx": idx, "error": "missing_policy_local_frame_fields"})
                else:
                    frame = {
                        "centroid": policy_local_centroid.detach().cpu().numpy(),
                        "radius": float(policy_local_radius.reshape(-1)[0].item()),
                    }
                    recon = action_local_to_world(
                        sample["policy_action_raw"].detach().cpu().numpy(),
                        frame,
                    )
                    target = sample["policy_action_world_raw"].detach().cpu().numpy()
                    local_roundtrip_error = float(abs(recon[..., :3] - target[..., :3]).max())
                    roundtrip_errors.append(local_roundtrip_error)
                pcd_current = sample.get("policy_pcd_current_key_idx")
                pcd_next = sample.get("policy_pcd_next_key_idx")
                if pcd_current is None or int(pcd_current) != int(current_key):
                    pcd_key_mismatches.append(
                        {
                            "idx": idx,
                            "field": "current",
                            "pcd": None if pcd_current is None else int(pcd_current),
                            "dense": int(current_key),
                        }
                    )
                if pcd_next is None or int(pcd_next) != int(next_key):
                    pcd_key_mismatches.append(
                        {
                            "idx": idx,
                            "field": "next",
                            "pcd": None if pcd_next is None else int(pcd_next),
                            "dense": int(next_key),
                        }
                    )
                pcd_action = sample.get("policy_pcd_next_action_world")
                if pcd_action is None:
                    failures.append({"idx": idx, "error": "missing_policy_pcd_next_action_world"})
                else:
                    pcd_action_error = float(
                        abs(
                            pcd_action.reshape(-1).detach().cpu().numpy()
                            - sample["policy_action_world_raw"].reshape(-1).detach().cpu().numpy()
                        ).max()
                    )
                    pcd_action_errors.append(pcd_action_error)
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
                    "policy_action_world_raw": [
                        float(v) for v in sample["policy_action_world_raw"].reshape(-1).detach().cpu().tolist()
                    ],
                    "policy_target_frame": str(sample.get("policy_target_frame", "world")),
                    "policy_local_centroid": None
                    if policy_local_centroid is None
                    else [float(v) for v in policy_local_centroid.reshape(-1).detach().cpu().tolist()],
                    "policy_local_radius": None
                    if policy_local_radius is None
                    else float(policy_local_radius.reshape(-1)[0].item()),
                    "policy_local_roundtrip_max_abs_error": local_roundtrip_error,
                    "policy_pcd_current_key_idx": None
                    if sample.get("policy_pcd_current_key_idx") is None
                    else int(sample.get("policy_pcd_current_key_idx")),
                    "policy_pcd_next_key_idx": None
                    if sample.get("policy_pcd_next_key_idx") is None
                    else int(sample.get("policy_pcd_next_key_idx")),
                    "policy_pcd_next_action_world": None
                    if sample.get("policy_pcd_next_action_world") is None
                    else [
                        float(v)
                        for v in sample["policy_pcd_next_action_world"].reshape(-1).detach().cpu().tolist()
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
        {
            "name": "policy_target_frame_matches_requested",
            "passed": all(example.get("policy_target_frame") == args.policy_target_frame for example in examples),
            "detail": {
                "requested": args.policy_target_frame,
                "observed": sorted({example.get("policy_target_frame") for example in examples}),
            },
        },
    ]
    if args.policy_target_frame == "official_pcd_local":
        checks.append(
            {
                "name": "official_pcd_local_roundtrip",
                "passed": bool(roundtrip_errors) and max(roundtrip_errors) <= float(args.max_local_roundtrip_error),
                "detail": {
                    "max": None if not roundtrip_errors else max(roundtrip_errors),
                    "threshold": float(args.max_local_roundtrip_error),
                },
            }
        )
        checks.append(
            {
                "name": "pcd_key_frameids_match_dense_keyframes",
                "passed": not pcd_key_mismatches,
                "detail": pcd_key_mismatches[:10],
            }
        )
        checks.append(
            {
                "name": "pcd_next_action_matches_dense_policy_target",
                "passed": bool(pcd_action_errors) and max(pcd_action_errors) <= float(args.max_pcd_action_error),
                "detail": {
                    "max": None if not pcd_action_errors else max(pcd_action_errors),
                    "threshold": float(args.max_pcd_action_error),
                },
            }
        )
    status = "passed" if all(check["passed"] for check in checks) else "failed"
    return {
        "status": status,
        "eval_type": "gembench_policy_keystep_9v32_contract_audit",
        "official_full_score": False,
        "manifest": str(Path(args.manifest).expanduser().resolve()),
        "rgb_cache_dir": str(Path(args.rgb_cache_dir).expanduser().resolve()),
        "key_frameids_path": None if args.key_frameids_path in (None, "", "null") else str(args.key_frameids_path),
        "vae_latent_cache_dir": None if args.vae_latent_cache_dir in (None, "", "null") else str(args.vae_latent_cache_dir),
        "dataset_len": int(len(dataset)),
        "policy_target_type": "next_key_step",
        "policy_target_frame": args.policy_target_frame,
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
    parser.add_argument("--key-frameids-path", default=None)
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
    parser.add_argument("--policy-target-frame", choices=("world", "official_pcd_local"), default="world")
    parser.add_argument("--policy-pcd-data-dir", default=None)
    parser.add_argument("--policy-local-xyz-shift", choices=("none", "center", "gripper"), default="center")
    parser.add_argument("--policy-local-xyz-norm", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--policy-local-rm-table", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--policy-local-rm-robot", choices=("none", "box", "box_keep_gripper"), default="none")
    parser.add_argument("--policy-local-num-points", type=int, default=4096)
    parser.add_argument("--policy-local-sample-seed", type=int, default=0)
    parser.add_argument("--policy-local-train-voxel-size", type=float, default=0.0)
    parser.add_argument("--policy-local-require-open3d", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--robot-3dlotus-root", default=None)
    parser.add_argument("--max-local-roundtrip-error", type=float, default=1e-5)
    parser.add_argument("--max-pcd-action-error", type=float, default=1e-5)
    parser.add_argument("--text-embedding-cache-dir", default="./data/text_embeds_cache/gembench_microsteps_9v32")
    parser.add_argument("--context-len", type=int, default=128)
    parser.add_argument("--text-encoder-id", default="wan22ti2v5b")
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
