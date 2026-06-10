#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import tarfile
import time
from pathlib import Path
from typing import Any

import numpy as np

from fastwam.datasets.gembench.microsteps_9v32 import (
    DEFAULT_CACHE_CAMERA_ORDER,
    SCHEMA_VERSION,
    cache_episode_path,
    load_manifest,
    manifest_demo_rows,
)


def _load_sim_backend(robot_3dlotus_root: str | None) -> dict[str, Any]:
    try:
        from fastwam.evaluation.gembench_official.runner import _official_imports

        modules = _official_imports(robot_3dlotus_root)
        modules["kind"] = "official_runner"
        return modules
    except Exception:
        from fastwam.evaluation.gembench_success.simulator import GEMBenchSimulator, Mover

        return {"kind": "gembench_success", "GEMBenchSimulator": GEMBenchSimulator, "Mover": Mover}


def _ensure_microsteps_dir(root: Path, microsteps_tar: Path, seed: str, *, extract_if_missing: bool) -> Path:
    data_dir = root / "train_dataset" / "microsteps" / seed
    if data_dir.is_dir():
        return data_dir
    if not extract_if_missing:
        raise FileNotFoundError(
            f"Missing extracted train microsteps dir: {data_dir}. "
            "Run with --extract-if-missing to extract train_dataset/microsteps.tar.gz first."
        )
    if not microsteps_tar.is_file():
        raise FileNotFoundError(f"Missing microsteps tar: {microsteps_tar}")
    target = root / "train_dataset"
    print(f"[gembench-9v32-render] extracting {microsteps_tar} -> {target}", flush=True)
    with tarfile.open(microsteps_tar, "r:gz") as tf:
        tf.extractall(target)
    if not data_dir.is_dir():
        raise FileNotFoundError(f"Extraction did not create expected dir: {data_dir}")
    return data_dir


def _gripper_action(obs: Any) -> np.ndarray:
    pose = np.asarray(obs.gripper_pose, dtype=np.float32).reshape(-1)
    if pose.shape != (7,):
        raise ValueError(f"Expected 7D gripper_pose, got {pose.shape}")
    return np.concatenate([pose, np.asarray([float(obs.gripper_open)], dtype=np.float32)]).astype(np.float32)


def _selected_rows(manifest: dict[str, Any], *, max_demos: int | None, taskvars: set[str] | None) -> list[dict[str, Any]]:
    rows = []
    for row in manifest_demo_rows(manifest):
        if taskvars is not None and str(row["taskvar"]) not in taskvars:
            continue
        rows.append(row)
        if max_demos is not None and len(rows) >= int(max_demos):
            break
    return rows


def _cache_path(row: dict[str, Any], cache_dir: Path) -> Path:
    if row.get("cache_path"):
        path = Path(str(row["cache_path"])).expanduser()
        return path if path.is_absolute() else cache_dir / path
    return cache_episode_path(
        cache_dir,
        seed=str(row["seed"]),
        taskvar=str(row["taskvar"]),
        episode_key=str(row["episode_key"]),
    )


def _save_episode_cache(
    *,
    output_path: Path,
    row: dict[str, Any],
    rgb: np.ndarray,
    gripper: np.ndarray,
    replay_reward: np.ndarray,
    replay_terminate: np.ndarray,
    camera_order: tuple[str, ...],
    image_size: tuple[int, int],
) -> None:
    if rgb.ndim != 5 or rgb.shape[-1] != 3:
        raise ValueError(f"Expected rgb [L,C,H,W,3], got {rgb.shape}")
    if gripper.shape != (rgb.shape[0], 8):
        raise ValueError(f"Expected gripper [{rgb.shape[0]},8], got {gripper.shape}")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = output_path.with_name(f".{output_path.stem}.{os.getpid()}.tmp.npz")
    np.savez(
        tmp,
        schema_version=np.asarray(SCHEMA_VERSION),
        taskvar=np.asarray(str(row["taskvar"])),
        task=np.asarray(str(row["task"])),
        variation=np.asarray(int(row["variation"]), dtype=np.int64),
        episode_key=np.asarray(str(row["episode_key"])),
        seed=np.asarray(str(row["seed"])),
        camera_order=np.asarray(camera_order),
        image_size=np.asarray(image_size, dtype=np.int64),
        rgb=rgb.astype(np.uint8, copy=False),
        gripper=gripper.astype(np.float32, copy=False),
        replay_reward=replay_reward.astype(np.float32, copy=False),
        replay_terminate=replay_terminate.astype(np.bool_, copy=False),
    )
    os.replace(tmp, output_path)


def render_rows(args: argparse.Namespace) -> dict[str, Any]:
    root = Path(args.root).expanduser().resolve()
    manifest_path = Path(args.manifest).expanduser().resolve()
    manifest = load_manifest(manifest_path)
    if manifest.get("schema_version") != SCHEMA_VERSION:
        raise ValueError(f"Manifest schema mismatch: {manifest_path}")
    cache_dir = Path(args.rgb_cache_dir or manifest["rgb_cache_dir"]).expanduser().resolve()
    seed = str(args.seed)
    microsteps_tar = Path(args.microsteps_tar).expanduser().resolve() if args.microsteps_tar else root / "train_dataset" / "microsteps.tar.gz"
    data_dir_candidate = root / "train_dataset" / "microsteps" / seed
    if args.dry_run and not data_dir_candidate.is_dir() and not args.extract_if_missing:
        data_dir = data_dir_candidate
    else:
        data_dir = _ensure_microsteps_dir(root, microsteps_tar, seed, extract_if_missing=bool(args.extract_if_missing))
    taskvars = None if args.taskvars is None else {item.strip() for item in args.taskvars.split(",") if item.strip()}
    rows = _selected_rows(manifest, max_demos=args.max_demos, taskvars=taskvars)
    camera_order = tuple(args.cache_camera_order.split(","))
    image_size = (int(args.image_size), int(args.image_size))

    dry_rows = []
    for row in rows:
        output_path = _cache_path(row, cache_dir)
        dry_rows.append({**row, "output_path": str(output_path), "exists": output_path.is_file()})
    if args.dry_run:
        return {
            "status": "dry_run",
            "manifest": str(manifest_path),
            "data_dir": str(data_dir),
            "data_dir_exists": data_dir.is_dir(),
            "requires_extract_for_render": not data_dir.is_dir(),
            "rgb_cache_dir": str(cache_dir),
            "rows": dry_rows,
            "official_full_score": False,
        }

    modules = _load_sim_backend(args.robot_3dlotus_root)
    Mover = modules["Mover"]

    completed = []
    failures = []
    grouped: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        grouped.setdefault(str(row["taskvar"]), []).append(row)

    for taskvar, group in grouped.items():
        first = group[0]
        env = None
        try:
            print(f"[gembench-9v32-render] launch taskvar={taskvar} demos={len(group)}", flush=True)
            if modules["kind"] == "official_runner":
                RLBenchEnv = modules["RLBenchEnv"]
                task_file_to_task_class = modules["task_file_to_task_class"]
                env = RLBenchEnv(
                    data_path=str(data_dir),
                    apply_rgb=True,
                    apply_depth=False,
                    apply_pc=False,
                    apply_mask=False,
                    headless=True,
                    image_size=list(image_size),
                    cam_rand_factor=0,
                    apply_cameras=camera_order,
                )
                env.env.launch()
                task_type = task_file_to_task_class(str(first["task"]))
                task = env.env.get_task(task_type)
                get_demo = lambda row: env.get_demo(str(row["task"]), int(row["variation"]), int(str(row["episode_key"]).replace("episode", "")), load_images=False)
                get_observation = env.get_observation
                shutdown = env.env.shutdown
            else:
                GEMBenchSimulator = modules["GEMBenchSimulator"]
                env = GEMBenchSimulator(
                    microstep_data_dir=str(data_dir),
                    image_size=list(image_size),
                    cameras=camera_order,
                    headless=True,
                ).launch()
                task = env.get_task(str(first["task"]))
                get_demo = lambda row: env.get_demo(str(row["task"]), int(row["variation"]), int(str(row["episode_key"]).replace("episode", "")))
                get_observation = env.observation_dict
                shutdown = env.shutdown
            task.set_variation(int(first["variation"]))
            move = Mover(task, max_tries=int(args.max_tries))

            for row in group:
                output_path = _cache_path(row, cache_dir)
                if output_path.is_file() and not args.overwrite:
                    completed.append({**row, "output_path": str(output_path), "status": "exists"})
                    continue
                start = time.time()
                try:
                    demo = get_demo(row)
                    instructions, obs = task.reset_to_demo(demo)
                    obs_state = get_observation(obs)
                    move.reset(obs_state["gripper"])
                    demo_obs = list(getattr(demo, "_observations", demo))
                    expected_len = int(row["length"])
                    if len(demo_obs) != expected_len:
                        raise ValueError(f"demo length changed: manifest={expected_len} actual={len(demo_obs)}")
                    rgb_frames = [np.asarray(obs_state["rgb"], dtype=np.uint8)]
                    gripper = [_gripper_action(demo_obs[0])]
                    rewards = [0.0]
                    terminates = [False]
                    for step_idx in range(expected_len - 1):
                        target = _gripper_action(demo_obs[step_idx + 1])
                        move_out = move(target, verbose=False)
                        if len(move_out) == 3:
                            obs, reward, terminate = move_out
                        else:
                            obs, reward, terminate, _ = move_out
                        obs_state = get_observation(obs)
                        rgb_frames.append(np.asarray(obs_state["rgb"], dtype=np.uint8))
                        gripper.append(target)
                        rewards.append(float(reward))
                        terminates.append(bool(terminate))
                    rgb = np.stack(rgb_frames, axis=0)
                    grip = np.stack(gripper, axis=0)
                    _save_episode_cache(
                        output_path=output_path,
                        row=row,
                        rgb=rgb,
                        gripper=grip,
                        replay_reward=np.asarray(rewards, dtype=np.float32),
                        replay_terminate=np.asarray(terminates, dtype=np.bool_),
                        camera_order=camera_order,
                        image_size=image_size,
                    )
                    completed.append(
                        {
                            **row,
                            "output_path": str(output_path),
                            "status": "rendered",
                            "seconds": time.time() - start,
                            "rgb_shape": list(rgb.shape),
                            "gripper_shape": list(grip.shape),
                        }
                    )
                    print(f"[gembench-9v32-render] rendered {taskvar}/{row['episode_key']} seconds={time.time() - start:.1f}", flush=True)
                except Exception as exc:
                    failure = {**row, "status": "failed", "error": f"{type(exc).__name__}: {exc}"}
                    failures.append(failure)
                    print(f"[gembench-9v32-render] failed {taskvar}/{row['episode_key']} error={failure['error']}", flush=True)
                    if not args.keep_going:
                        raise
        finally:
            try:
                if env is not None:
                    shutdown()
            except Exception as exc:
                print(f"[gembench-9v32-render] env shutdown warning: {type(exc).__name__}: {exc}", flush=True)

    return {
        "status": "completed" if not failures else "failed",
        "manifest": str(manifest_path),
        "data_dir": str(data_dir),
        "rgb_cache_dir": str(cache_dir),
        "completed": completed,
        "failures": failures,
        "official_full_score": False,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Render dense GEMBench train microsteps RGB cache for 9V/32A WAM training.")
    parser.add_argument("--root", default="/mnt/yuhan/datasets/GEMBench")
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--rgb-cache-dir", default=None)
    parser.add_argument("--microsteps-tar", default=None)
    parser.add_argument("--seed", default="seed0")
    parser.add_argument("--robot-3dlotus-root", default=None)
    parser.add_argument("--cache-camera-order", default=",".join(DEFAULT_CACHE_CAMERA_ORDER))
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--max-demos", type=int, default=2)
    parser.add_argument("--taskvars", default=None)
    parser.add_argument("--max-tries", type=int, default=10)
    parser.add_argument("--extract-if-missing", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--keep-going", action="store_true")
    parser.add_argument("--output-json", required=True)
    args = parser.parse_args()
    payload = render_rows(args)
    output_json = Path(args.output_json)
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(payload, ensure_ascii=True, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"status": payload["status"], "output_json": str(output_json)}, ensure_ascii=True, sort_keys=True))
    return 0 if payload["status"] in {"completed", "dry_run"} else 1


if __name__ == "__main__":
    raise SystemExit(main())
