#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import subprocess
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np

from fastwam.datasets.gembench.microsteps_9v32 import (
    DEFAULT_FRAME_OFFSETS,
    GEMBenchKeyStepPolicy9V32Dataset,
)
from fastwam.datasets.gembench.policy_local_frame import action_world_to_local


def _parse_csv(value: str | None) -> list[str] | None:
    if value is None:
        return None
    items = [item.strip() for item in str(value).split(",") if item.strip()]
    return items or None


def _field_stats(rows: np.ndarray) -> dict[str, list[float]]:
    rows = np.asarray(rows, dtype=np.float32).reshape(-1, 8)
    if rows.size == 0:
        raise ValueError("Cannot compute stats from zero rows.")
    return {
        "global_min": rows.min(axis=0).astype(float).tolist(),
        "global_max": rows.max(axis=0).astype(float).tolist(),
        "global_mean": rows.mean(axis=0).astype(float).tolist(),
        "global_std": (rows.std(axis=0) + 1e-8).astype(float).tolist(),
        "global_q01": np.quantile(rows, 0.01, axis=0).astype(float).tolist(),
        "global_q99": np.quantile(rows, 0.99, axis=0).astype(float).tolist(),
    }


def _write_json_atomic(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    with tmp_path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=True, indent=2)
        f.write("\n")
    os.replace(tmp_path, path)


def _git_provenance() -> dict[str, Any]:
    root = Path(__file__).resolve().parents[1]

    def _run(args: list[str]) -> str | None:
        try:
            return subprocess.check_output(args, cwd=str(root), text=True, stderr=subprocess.DEVNULL).strip()
        except Exception:
            return None

    return {
        "commit": _run(["git", "rev-parse", "HEAD"]),
        "branch": _run(["git", "branch", "--show-current"]),
        "status_short": _run(["git", "status", "--short"]),
    }


def build_dataset(args: argparse.Namespace) -> GEMBenchKeyStepPolicy9V32Dataset:
    camera_order = _parse_csv(args.camera_order)
    if not camera_order:
        raise ValueError("--camera-order must contain at least one camera.")
    cache_camera_order = _parse_csv(args.cache_camera_order) or camera_order
    image_size = int(args.image_size)
    return GEMBenchKeyStepPolicy9V32Dataset(
        manifest_path=args.manifest,
        rgb_cache_dir=args.rgb_cache_dir,
        keysteps_dir=args.keysteps_dir,
        key_frameids_path=args.key_frameids_path,
        split=args.split,
        seed=args.seed,
        frame_offsets=DEFAULT_FRAME_OFFSETS,
        action_horizon=32,
        video_size=(image_size, image_size * len(camera_order)),
        camera_order=camera_order,
        cache_camera_order=cache_camera_order,
        window_stride=1,
        max_windows_per_demo=args.max_windows_per_demo,
        taskvars=_parse_csv(args.taskvars),
        policy_max_index_demos=args.policy_max_index_demos,
        vae_latent_cache_dir=None,
        text_embedding_cache_dir=args.text_embedding_cache_dir,
        context_len=int(args.context_len),
        text_encoder_id=args.text_encoder_id,
        cache_text_embeddings=False,
        cache_gripper_arrays=True,
        allow_missing_text_embeds=True,
        pretrained_norm_stats=None,
        norm_default_mode=args.norm_default_mode,
        stats_scan_limit=0,
        allow_partial_cache=bool(args.allow_partial_cache),
        policy_include_final_key=bool(args.policy_include_final_key),
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


def compute_stats(args: argparse.Namespace) -> tuple[dict[str, Any], dict[str, Any]]:
    dataset = build_dataset(args)
    max_samples = None if args.max_samples is None else int(args.max_samples)
    rows = dataset.index if max_samples is None else dataset.index[:max_samples]
    if not rows:
        raise ValueError("No key-step policy rows selected for stats.")

    actions: list[np.ndarray] = []
    states: list[np.ndarray] = []
    examples: list[dict[str, Any]] = []
    taskvar_counts: Counter[str] = Counter()
    key_delta_values: list[int] = []

    for sample_idx, (row_idx, current_key_idx, next_key_idx, key_position) in enumerate(rows):
        row = dataset.demo_rows[int(row_idx)]
        taskvar = str(row["taskvar"])
        episode_key = str(row["episode_key"])
        gripper = dataset._load_gripper(row, dataset._cache_path(row))
        action_world = gripper[int(next_key_idx) : int(next_key_idx) + 1].astype(np.float32, copy=True)
        state_world = gripper[int(current_key_idx) : int(current_key_idx) + 1].astype(np.float32, copy=True)
        action_model = action_world
        local_frame: dict[str, Any] | None = None
        if dataset.policy_target_frame == "official_pcd_local":
            local_frame = dataset._policy_local_frame(row, int(current_key_idx), int(key_position))
            if "pcd_next_action_world" in local_frame:
                action_world = np.asarray(local_frame["pcd_next_action_world"], dtype=np.float32).reshape(1, -1)
            action_model = action_world_to_local(action_world, local_frame).astype(np.float32, copy=False)

        actions.append(action_model)
        states.append(state_world)
        taskvar_counts[taskvar] += 1
        key_delta_values.append(int(next_key_idx) - int(current_key_idx))
        if len(examples) < int(args.num_examples):
            example = {
                "sample_idx": int(sample_idx),
                "taskvar": taskvar,
                "episode_key": episode_key,
                "current_key_idx": int(current_key_idx),
                "next_key_idx": int(next_key_idx),
                "key_delta": int(next_key_idx) - int(current_key_idx),
                "policy_target_frame": dataset.policy_target_frame,
                "policy_action_model_raw": action_model.reshape(-1).astype(float).tolist(),
                "policy_action_world_raw": action_world.reshape(-1).astype(float).tolist(),
                "policy_state_world_raw": state_world.reshape(-1).astype(float).tolist(),
            }
            if local_frame is not None:
                example["policy_local_centroid"] = np.asarray(local_frame["centroid"]).reshape(-1).astype(float).tolist()
                example["policy_local_radius"] = float(local_frame["radius"])
            examples.append(example)

    action_arr = np.concatenate(actions, axis=0)
    state_arr = np.concatenate(states, axis=0)
    stats = {
        "action": {"default": _field_stats(action_arr)},
        "state": {"default": _field_stats(state_arr)},
    }
    provenance = {
        "script": "scripts/precompute_gembench_policy_keystep_norm_stats.py",
        "eval_type": "gembench_policy_keystep_norm_stats",
        "official_full_score": False,
        "write_official_preds": False,
        "policy_action_horizon": 1,
        "wam_aux_action_horizon": 32,
        "policy_target_type": "next_key_step",
        "policy_target_frame": dataset.policy_target_frame,
        "num_rows": int(action_arr.shape[0]),
        "dataset_len": int(len(dataset)),
        "manifest_path": str(Path(args.manifest).expanduser()),
        "rgb_cache_dir": str(Path(args.rgb_cache_dir).expanduser()),
        "keysteps_dir": args.keysteps_dir,
        "key_frameids_path": args.key_frameids_path,
        "taskvars": _parse_csv(args.taskvars),
        "taskvar_counts": dict(sorted(taskvar_counts.items())),
        "policy_max_index_demos": args.policy_max_index_demos,
        "max_samples": max_samples,
        "max_windows_per_demo": args.max_windows_per_demo,
        "norm_default_mode": args.norm_default_mode,
        "action_min": stats["action"]["default"]["global_min"],
        "action_max": stats["action"]["default"]["global_max"],
        "state_min": stats["state"]["default"]["global_min"],
        "state_max": stats["state"]["default"]["global_max"],
        "key_delta_min": int(min(key_delta_values)),
        "key_delta_max": int(max(key_delta_values)),
        "key_delta_mean": float(np.mean(np.asarray(key_delta_values, dtype=np.float32))),
        "examples": examples,
        "git": _git_provenance(),
    }
    return stats, provenance


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Precompute normalization stats for GEMBench key-step policy targets. "
            "Unlike precompute_gembench_norm_stats.py, this follows "
            "GEMBenchKeyStepPolicy9V32Dataset.index and aggregates policy_action_raw, "
            "not dense microstep 9V32 auxiliary actions."
        )
    )
    parser.add_argument("--manifest", default=os.environ.get("GEMBENCH_9V32_4CAM_MANIFEST", "/mnt/yuhan/datasets/GEMBench/fastwam_cache/microsteps_9v32_4cam224_manifest.json"))
    parser.add_argument("--rgb-cache-dir", default=os.environ.get("GEMBENCH_9V32_4CAM_RGB_CACHE_DIR", "/mnt/yuhan/datasets/GEMBench/fastwam_cache/microsteps_9v32_4cam224_rgb"))
    parser.add_argument("--keysteps-dir", default=os.environ.get("GEMBENCH_KEYSTEPS_BBOX_DIR", "/mnt/yuhan/datasets/GEMBench/train_dataset/keysteps_bbox/seed0"))
    parser.add_argument("--key-frameids-path", default=os.environ.get("GEMBENCH_KEY_FRAMEIDS_CACHE", "/mnt/yuhan/datasets/GEMBench/fastwam_cache/microsteps_9v32_seed0_key_frameids.json"))
    parser.add_argument("--output", default=os.environ.get("GEMBENCH_POLICY_KEYSTEP_NORM_STATS", "data/gembench_policy_keystep_9v32_4cam224_stats.json"))
    parser.add_argument("--provenance-output", default=None)
    parser.add_argument("--split", default="train")
    parser.add_argument("--seed", default="seed0")
    parser.add_argument("--taskvars", default=None)
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--camera-order", default="left_shoulder,right_shoulder,wrist,front")
    parser.add_argument("--cache-camera-order", default="left_shoulder,right_shoulder,wrist,front")
    parser.add_argument("--max-windows-per-demo", type=int, default=None)
    parser.add_argument("--policy-max-index-demos", type=int, default=None)
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--policy-min-key-delta", type=int, default=1)
    parser.add_argument("--policy-include-final-key", action="store_true")
    parser.add_argument("--policy-target-frame", choices=("world", "official_pcd_local"), default="world")
    parser.add_argument("--policy-pcd-data-dir", default=None)
    parser.add_argument("--policy-local-xyz-shift", default="center")
    parser.add_argument("--policy-local-xyz-norm", action="store_true")
    parser.add_argument("--policy-local-rm-table", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--policy-local-rm-robot", default="none")
    parser.add_argument("--policy-local-num-points", type=int, default=4096)
    parser.add_argument("--policy-local-sample-seed", type=int, default=0)
    parser.add_argument("--policy-local-train-voxel-size", type=float, default=0.0)
    parser.add_argument("--policy-local-require-open3d", action="store_true")
    parser.add_argument("--robot-3dlotus-root", default=None)
    parser.add_argument("--text-embedding-cache-dir", default=os.environ.get("GEMBENCH_TEXT_EMBED_CACHE", "data/text_embeds_cache/gembench_microsteps_9v32"))
    parser.add_argument("--text-encoder-id", default=os.environ.get("GEMBENCH_TEXT_ENCODER_ID", "wan22ti2v5b"))
    parser.add_argument("--context-len", type=int, default=128)
    parser.add_argument("--norm-default-mode", default="min/max")
    parser.add_argument("--allow-partial-cache", action="store_true")
    parser.add_argument("--num-examples", type=int, default=5)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    stats, provenance = compute_stats(args)
    output = Path(args.output).expanduser()
    provenance_output = (
        Path(args.provenance_output).expanduser()
        if args.provenance_output
        else output.with_suffix(output.suffix + ".provenance.json")
    )
    if args.dry_run:
        print(json.dumps(provenance, ensure_ascii=True, indent=2))
        return
    _write_json_atomic(output, stats)
    _write_json_atomic(provenance_output, provenance)
    print(
        "wrote GEMBench key-step policy stats: "
        f"rows={provenance['num_rows']} policy_target_frame={provenance['policy_target_frame']} "
        f"stats={output} provenance={provenance_output}"
    )


if __name__ == "__main__":
    main()
