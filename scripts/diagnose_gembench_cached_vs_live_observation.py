#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import torch
from hydra.utils import instantiate
from omegaconf import OmegaConf
from PIL import Image

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
SCRIPTS_ROOT = REPO_ROOT / "scripts"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))
if str(SCRIPTS_ROOT) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_ROOT))

from fastwam.evaluation.gembench_official.common import git_provenance, utc_now, write_json
from fastwam.evaluation.gembench_official.runner import OFFICIAL_CAMERA_NAMES, _official_imports
from replay_gembench_policy_keystep_targets import _gripper_action


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=True) + "\n")


def _as_uint8_rgb(image: np.ndarray) -> np.ndarray:
    arr = np.asarray(image)
    if arr.dtype != np.uint8:
        if np.issubdtype(arr.dtype, np.floating) and float(np.nanmax(arr)) <= 1.0:
            arr = arr * 255.0
        arr = np.clip(arr, 0, 255).astype(np.uint8)
    if arr.ndim != 3 or arr.shape[-1] != 3:
        raise ValueError(f"Expected RGB image [H,W,3], got {arr.shape}")
    return arr


def _obs_image_tensor(
    obs_state_dict: dict[str, Any],
    *,
    video_size: list[int],
    camera_order: list[str],
    observation_camera_names: tuple[str, ...] = OFFICIAL_CAMERA_NAMES,
) -> torch.Tensor:
    rgb = np.asarray(obs_state_dict.get("rgb"))
    if rgb.ndim != 4:
        raise ValueError(f"Official observation must contain rgb [C,H,W,3], got {rgb.shape}")
    camera_h = int(video_size[0])
    camera_w = int(video_size[1]) // len(camera_order)
    frames = []
    for camera in camera_order:
        index = observation_camera_names.index(camera)
        image = _as_uint8_rgb(rgb[index])
        pil = Image.fromarray(image, mode="RGB").resize((camera_w, camera_h), resample=Image.BILINEAR)
        frames.append(np.asarray(pil, dtype=np.uint8))
    cat = np.concatenate(frames, axis=1)
    tensor = torch.from_numpy(cat).permute(2, 0, 1).to(dtype=torch.float32)
    return tensor * (2.0 / 255.0) - 1.0


def _metric_summary(values: list[float]) -> dict[str, float | None]:
    if not values:
        return {"mean": None, "median": None, "p90": None, "max": None}
    arr = np.asarray(values, dtype=np.float64)
    return {
        "mean": float(arr.mean()),
        "median": float(np.median(arr)),
        "p90": float(np.percentile(arr, 90.0)),
        "max": float(arr.max()),
    }


def _summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "num_rows": len(rows),
        "image_l1": _metric_summary([float(row["image_l1"]) for row in rows]),
        "image_rmse": _metric_summary([float(row["image_rmse"]) for row in rows]),
        "image_max_abs": _metric_summary([float(row["image_max_abs"]) for row in rows]),
        "gripper_l2": _metric_summary([float(row["gripper_l2"]) for row in rows]),
        "gripper_xyz_l2": _metric_summary([float(row["gripper_xyz_l2"]) for row in rows]),
        "target_max_abs_diff": _metric_summary([float(row["target_max_abs_diff"]) for row in rows]),
        "worst_image_rows": sorted(
            [
                {
                    "sample_index": int(row["sample_index"]),
                    "taskvar": row["taskvar"],
                    "episode_key": row["episode_key"],
                    "image_rmse": float(row["image_rmse"]),
                    "gripper_xyz_l2": float(row["gripper_xyz_l2"]),
                }
                for row in rows
            ],
            key=lambda item: item["image_rmse"],
            reverse=True,
        )[:10],
        "worst_gripper_rows": sorted(
            [
                {
                    "sample_index": int(row["sample_index"]),
                    "taskvar": row["taskvar"],
                    "episode_key": row["episode_key"],
                    "gripper_l2": float(row["gripper_l2"]),
                    "gripper_xyz_l2": float(row["gripper_xyz_l2"]),
                    "gripper_abs": row["gripper_abs"],
                }
                for row in rows
            ],
            key=lambda item: item["gripper_l2"],
            reverse=True,
        )[:10],
    }


def _cache_raw_for_sample(dataset: Any, sample_index: int) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    row_idx, current_key_idx, next_key_idx, key_position = dataset.index[int(sample_index)]
    row = dataset.demo_rows[int(row_idx)]
    cache_path = dataset._cache_path(row)
    payload = np.load(cache_path, allow_pickle=False)
    try:
        gripper = np.asarray(payload["gripper"], dtype=np.float32)
        cached_current = gripper[int(current_key_idx)].copy()
        cached_target = gripper[int(next_key_idx)].copy()
    finally:
        payload.close()
    meta = {
        "row_idx": int(row_idx),
        "current_key_idx": int(current_key_idx),
        "next_key_idx": int(next_key_idx),
        "key_position": int(key_position),
        "taskvar": str(row["taskvar"]),
        "task": str(row["task"]),
        "variation": int(row["variation"]),
        "episode_key": str(row["episode_key"]),
        "demo_id": int(str(row["episode_key"]).replace("episode", "")),
        "length": int(row["length"]),
    }
    return cached_current, cached_target, meta


def _run_task_group(
    *,
    args: argparse.Namespace,
    dataset: Any,
    sample_indices: list[int],
    modules: dict[str, Any],
) -> list[dict[str, Any]]:
    task_file_to_task_class = modules["task_file_to_task_class"]
    RLBenchEnv = modules["RLBenchEnv"]
    Mover = modules["Mover"]
    official_exceptions = modules["exceptions"]

    video_size = [int(v) for v in dataset.video_size]
    camera_order = [str(v) for v in dataset.camera_order]
    microstep_data_dir = Path(args.gembench_root).expanduser() / "train_dataset" / "microsteps" / str(args.seed)
    env = RLBenchEnv(
        data_path=str(microstep_data_dir),
        apply_rgb=True,
        apply_pc=True,
        apply_mask=True,
        headless=True,
        image_size=[int(args.image_size), int(args.image_size)],
        cam_rand_factor=0,
        apply_cameras=list(OFFICIAL_CAMERA_NAMES),
    )
    rows: list[dict[str, Any]] = []
    try:
        env.env.launch()
        grouped: dict[str, list[tuple[int, np.ndarray, np.ndarray, dict[str, Any]]]] = defaultdict(list)
        for sample_index in sample_indices:
            cached_current, cached_target, meta = _cache_raw_for_sample(dataset, sample_index)
            grouped[f"{meta['task']}+{meta['variation']}"].append((sample_index, cached_current, cached_target, meta))

        for _, items in grouped.items():
            first_meta = items[0][3]
            task_type = task_file_to_task_class(first_meta["task"])
            task = env.env.get_task(task_type)
            task.set_variation(int(first_meta["variation"]))
            move = Mover(task, max_tries=int(args.max_tries))
            for sample_index, cached_current, cached_target, meta in items:
                demo = env.get_demo(meta["task"], int(meta["variation"]), int(meta["demo_id"]), load_images=False)
                demo_obs = list(getattr(demo, "_observations", demo))
                instructions, obs = task.reset_to_demo(demo)
                obs_state = env.get_observation(obs)
                move.reset(obs_state["gripper"])
                if int(meta["key_position"]) > 0:
                    key_frameids = dataset._normalized_key_frameids(dataset.demo_rows[int(meta["row_idx"])], length=int(meta["length"]))
                    prefix_pairs = list(zip(key_frameids[:-1], key_frameids[1:]))[: int(meta["key_position"])]
                    for _, prefix_next_key_idx in prefix_pairs:
                        try:
                            move_out = move(_gripper_action(demo_obs[int(prefix_next_key_idx)]), verbose=False)
                            obs = move_out[0]
                            obs_state = env.get_observation(obs)
                        except official_exceptions as exc:
                            raise RuntimeError(
                                f"GT prefix failed for {meta['taskvar']}/{meta['episode_key']} "
                                f"key_position={meta['key_position']}: {type(exc).__name__}: {exc}"
                            ) from exc
                sample = dataset[int(sample_index)]
                cached_image = sample["video"][:, 0].detach().cpu().float()
                live_image = _obs_image_tensor(
                    obs_state,
                    video_size=video_size,
                    camera_order=camera_order,
                ).cpu().float()
                image_diff = live_image - cached_image
                live_gripper = np.asarray(obs_state["gripper"], dtype=np.float64).reshape(8)
                cached_current = np.asarray(cached_current, dtype=np.float64).reshape(8)
                cached_target = np.asarray(cached_target, dtype=np.float64).reshape(8)
                sample_target = np.asarray(sample["policy_action_world_raw"], dtype=np.float64).reshape(8)
                gripper_abs = np.abs(live_gripper - cached_current)
                row = {
                    "sample_index": int(sample_index),
                    **meta,
                    "instruction": str(instructions[0] if isinstance(instructions, (list, tuple)) else instructions),
                    "image_l1": float(image_diff.abs().mean().item()),
                    "image_rmse": float(torch.sqrt(image_diff.pow(2).mean()).item()),
                    "image_max_abs": float(image_diff.abs().max().item()),
                    "live_gripper": [float(v) for v in live_gripper.tolist()],
                    "cached_current_gripper": [float(v) for v in cached_current.tolist()],
                    "gripper_abs": [float(v) for v in gripper_abs.tolist()],
                    "gripper_l2": float(np.linalg.norm(live_gripper - cached_current)),
                    "gripper_xyz_l2": float(np.linalg.norm(live_gripper[:3] - cached_current[:3])),
                    "cached_target": [float(v) for v in cached_target.tolist()],
                    "sample_policy_target": [float(v) for v in sample_target.tolist()],
                    "target_max_abs_diff": float(np.max(np.abs(cached_target - sample_target))),
                }
                rows.append(row)
                print(
                    "[cached-vs-live-obs] "
                    f"sample={sample_index} taskvar={meta['taskvar']} episode={meta['episode_key']} "
                    f"image_rmse={row['image_rmse']:.6f} gripper_xyz_l2={row['gripper_xyz_l2']:.6f}",
                    flush=True,
                )
    finally:
        try:
            env.env.shutdown()
        except Exception:
            pass
    return rows


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Compare cached GEMBench key-step observations with live RLBench reset/prefix observations."
    )
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--dataset-key", default="val", choices=("train", "val"))
    parser.add_argument("--gembench-root", default="/mnt/yuhan/datasets/GEMBench")
    parser.add_argument("--robot-3dlotus-root", default=None)
    parser.add_argument("--seed", default="seed0")
    parser.add_argument("--max-samples", type=int, default=50)
    parser.add_argument("--sample-offset", type=int, default=0)
    parser.add_argument("--image-size", type=int, default=256)
    parser.add_argument("--max-tries", type=int, default=10)
    parser.add_argument("--output-root", default=None)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    run_dir = Path(args.run_dir).expanduser().resolve()
    cfg = OmegaConf.load(run_dir / "config.yaml")
    dataset_cfg = cfg.data.get(str(args.dataset_key))
    if dataset_cfg is None:
        raise ValueError(f"config.yaml has no data.{args.dataset_key}")
    dataset = instantiate(dataset_cfg)
    end = min(len(dataset), int(args.sample_offset) + int(args.max_samples))
    sample_indices = list(range(int(args.sample_offset), end))
    output_root = Path(
        args.output_root
        or run_dir / "diagnostics" / f"cached_vs_live_observation_{args.dataset_key}"
    ).expanduser()
    output_root.mkdir(parents=True, exist_ok=True)

    manifest = {
        "eval_type": "gembench_cached_vs_live_observation_diagnostic",
        "official_full_score": False,
        "write_official_preds": False,
        "generated_at": utc_now(),
        "run_dir": str(run_dir),
        "dataset_key": str(args.dataset_key),
        "dataset_class": type(dataset).__name__,
        "dataset_len": int(len(dataset)),
        "sample_indices": sample_indices,
        "gembench_root": str(Path(args.gembench_root).expanduser()),
        "robot_3dlotus_root": args.robot_3dlotus_root,
        "seed": str(args.seed),
        "note": "This diagnostic compares inputs only; it does not run the model or official scoring.",
        "git": git_provenance(),
    }
    write_json(output_root / "diagnostic_manifest.json", manifest)
    if args.dry_run:
        print(json.dumps(manifest, ensure_ascii=True, indent=2), flush=True)
        return 0

    modules = _official_imports(args.robot_3dlotus_root)
    rows = _run_task_group(args=args, dataset=dataset, sample_indices=sample_indices, modules=modules)
    _write_jsonl(output_root / "cached_vs_live_observation.jsonl", rows)
    summary = {
        **manifest,
        "output_root": str(output_root),
        "summary": _summarize(rows),
        "rows": rows,
    }
    write_json(output_root / "summary.json", summary)
    print(json.dumps(summary, ensure_ascii=True, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
