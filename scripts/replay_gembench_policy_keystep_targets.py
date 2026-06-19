#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import time
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from fastwam.datasets.gembench.microsteps_9v32 import load_manifest, manifest_demo_rows, parse_taskvar
from fastwam.evaluation.gembench_official.common import git_provenance, safe_token, utc_now, write_json
from fastwam.evaluation.gembench_official.runner import (
    OFFICIAL_CAMERA_NAMES,
    _SafeTaskRecorderCallback,
    _action_list,
    _append_video_error,
    _build_task_recorder,
    _clear_task_recorder,
    _convert_recorder_avi_to_mp4,
    _official_imports,
    _recorder_imports,
    _task_recorder_frame_counts,
    _task_success_state,
    _video_frame,
    _write_task_recorder_snaps_mp4,
    _write_video,
)


@dataclass(frozen=True)
class ReplaySpec:
    row: dict[str, Any]
    key_frameids: list[int]

    @property
    def taskvar(self) -> str:
        return str(self.row["taskvar"])

    @property
    def task(self) -> str:
        return str(self.row["task"])

    @property
    def variation(self) -> int:
        return int(self.row["variation"])

    @property
    def episode_key(self) -> str:
        return str(self.row["episode_key"])

    @property
    def demo_id(self) -> int:
        text = self.episode_key
        if text.startswith("episode"):
            text = text[len("episode") :]
        return int(text)


def _gripper_action(obs: Any) -> np.ndarray:
    pose = np.asarray(obs.gripper_pose, dtype=np.float32).reshape(-1)
    if pose.shape != (7,):
        raise ValueError(f"Expected 7D gripper_pose, got {pose.shape}")
    return np.concatenate([pose, np.asarray([float(obs.gripper_open)], dtype=np.float32)]).astype(np.float32)


def _parse_csv(value: str | None) -> list[str] | None:
    if value is None or str(value).strip() == "":
        return None
    return [item.strip() for item in str(value).split(",") if item.strip()]


def _load_key_frameids(path: Path) -> dict[str, list[int]]:
    payload = json.loads(path.expanduser().read_text(encoding="utf-8"))
    entries = payload.get("entries")
    if not isinstance(entries, dict):
        raise ValueError(f"Key-frame sidecar is missing `entries`: {path}")
    out: dict[str, list[int]] = {}
    for key, value in entries.items():
        if not isinstance(value, list):
            raise ValueError(f"Invalid key_frameids entry for {key!r}: {type(value).__name__}")
        out[str(key)] = [int(item) for item in value]
    return out


def _select_specs(args: argparse.Namespace) -> list[ReplaySpec]:
    manifest = load_manifest(args.manifest)
    key_frameids_by_demo = _load_key_frameids(Path(args.key_frameids_path))
    requested_taskvars = set(_parse_csv(args.taskvars) or [])
    requested_episodes = set(_parse_csv(args.episodes) or [])

    grouped: dict[str, list[ReplaySpec]] = defaultdict(list)
    missing_key_frameids: list[str] = []
    for row in manifest_demo_rows(manifest):
        if str(row.get("seed", args.seed)) != str(args.seed):
            continue
        taskvar = str(row.get("taskvar"))
        if requested_taskvars and taskvar not in requested_taskvars:
            continue
        episode_key = str(row.get("episode_key"))
        if requested_episodes and episode_key not in requested_episodes:
            continue
        demo_key = f"{taskvar}/{episode_key}"
        key_frameids = key_frameids_by_demo.get(demo_key)
        if key_frameids is None:
            missing_key_frameids.append(demo_key)
            continue
        length = int(row["length"])
        normalized = sorted({int(v) for v in key_frameids if 0 <= int(v) < length})
        if not normalized:
            continue
        if normalized[0] != 0:
            normalized.insert(0, 0)
        if len(normalized) < 2:
            continue
        grouped[taskvar].append(ReplaySpec(row=dict(row), key_frameids=normalized))

    selected: list[ReplaySpec] = []
    for taskvar in sorted(grouped):
        selected.extend(grouped[taskvar][: int(args.episodes_per_taskvar)])
    if args.max_trials is not None:
        selected = selected[: int(args.max_trials)]
    if not selected:
        preview = ", ".join(missing_key_frameids[:5])
        raise RuntimeError(
            "No replay specs selected. Check --taskvars/--episodes and key-frame sidecar coverage. "
            f"Missing key-frame preview: {preview}"
        )
    return selected


def _write_jsonl(path: Path, row: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, ensure_ascii=True) + "\n")


def _save_recorder_video(
    *,
    output_root: Path,
    task_recorder: Any | None,
    task_recorder_callback: _SafeTaskRecorderCallback | None,
    frames: list[np.ndarray],
    taskvar: str,
    demo_id: int,
    success: bool,
    fps: int,
    video_mode: str,
) -> tuple[list[str], dict[str, int], str | None]:
    video_error = task_recorder_callback.error if task_recorder_callback is not None else None
    video_mp4_paths: list[str] = []
    video_frame_counts: dict[str, int] = {}
    if video_mode == "official_recorder" and task_recorder is not None:
        video_frame_counts = _task_recorder_frame_counts(task_recorder)
        if any(video_frame_counts.values()):
            video_path = (
                output_root
                / "videos"
                / safe_token(taskvar)
                / f"demo_{int(demo_id):03d}_SR{int(success)}"
            )
            try:
                video_path.parent.mkdir(parents=True, exist_ok=True)
                task_recorder.save(str(video_path))
                _, video_mp4_paths, convert_error = _convert_recorder_avi_to_mp4(video_path)
                video_error = _append_video_error(video_error, convert_error)
            except Exception as exc:
                try:
                    video_mp4_paths = _write_task_recorder_snaps_mp4(video_path, task_recorder, fps=int(fps))
                except Exception as fallback_exc:
                    video_error = _append_video_error(
                        video_error,
                        f"{type(exc).__name__}: {exc}; fallback {type(fallback_exc).__name__}: {fallback_exc}",
                    )
        else:
            video_error = _append_video_error(video_error, "TaskRecorder captured 0 frames.")
    if not video_mp4_paths and frames:
        fallback_path = output_root / "videos" / safe_token(taskvar) / f"demo_{int(demo_id):03d}_observation.mp4"
        _write_video(fallback_path, frames, fps=int(fps))
        video_mp4_paths = [str(fallback_path)]
        video_frame_counts = {"observation": len(frames)}
    return video_mp4_paths, video_frame_counts, video_error


def _run_task_group(
    *,
    args: argparse.Namespace,
    taskvar: str,
    specs: list[ReplaySpec],
    modules: dict[str, Any],
    output_root: Path,
) -> list[dict[str, Any]]:
    task_file_to_task_class = modules["task_file_to_task_class"]
    RLBenchEnv = modules["RLBenchEnv"]
    Mover = modules["Mover"]
    official_exceptions = modules["exceptions"]

    first = specs[0]
    microstep_data_dir = Path(args.gembench_root).expanduser() / "train_dataset" / "microsteps" / str(args.seed)
    env = RLBenchEnv(
        data_path=str(microstep_data_dir),
        apply_rgb=True,
        apply_pc=False,
        apply_mask=False,
        headless=True,
        image_size=[int(args.image_size), int(args.image_size)],
        cam_rand_factor=0,
        apply_cameras=list(OFFICIAL_CAMERA_NAMES),
    )
    rows: list[dict[str, Any]] = []
    task_recorder = None
    task_recorder_callback = None
    try:
        print(f"[keystep-replay] env_launch taskvar={taskvar} data={microstep_data_dir}", flush=True)
        env.env.launch()
        task_type = task_file_to_task_class(first.task)
        task = env.env.get_task(task_type)
        task.set_variation(first.variation)
        move = Mover(task, max_tries=int(args.max_tries))
        recorder_setup_error = None
        video_mode = "none"
        if args.record_video:
            if args.video_mode == "official_recorder":
                try:
                    task_recorder = _build_task_recorder(
                        task,
                        _recorder_imports(args.robot_3dlotus_root),
                        resolution=int(args.video_resolution),
                        include_robot_cameras=bool(args.video_include_robot_cameras),
                        rotate_cam=bool(args.video_rotate_cam),
                        fps=int(args.video_fps),
                    )
                    task_recorder_callback = _SafeTaskRecorderCallback(task_recorder)
                    task._scene.register_step_callback(task_recorder_callback)
                    video_mode = "official_recorder"
                except Exception as exc:
                    recorder_setup_error = f"{type(exc).__name__}: {exc}"
                    if args.video_recorder_required:
                        raise
                    print(
                        "[keystep-replay] official recorder unavailable; falling back to observation video: "
                        f"{recorder_setup_error}",
                        flush=True,
                    )
                    video_mode = "observation"
            else:
                video_mode = "observation"

        for spec in specs:
            action_path = output_root / "actions" / safe_token(spec.taskvar) / f"demo_{spec.demo_id:03d}.jsonl"
            if action_path.exists():
                action_path.unlink()
            frames: list[np.ndarray] = []
            error = None
            reward = 0.0
            terminate = False
            success = False
            steps = 0
            initial_success = False
            initial_success_error = None
            if task_recorder is not None:
                _clear_task_recorder(task_recorder)
            if task_recorder_callback is not None:
                task_recorder_callback.reset()

            start_time = time.time()
            try:
                demo = env.get_demo(spec.task, spec.variation, spec.demo_id, load_images=False)
                demo_obs = list(getattr(demo, "_observations", demo))
                if len(demo_obs) != int(spec.row["length"]):
                    raise ValueError(f"demo length changed: manifest={spec.row['length']} actual={len(demo_obs)}")
                instructions, obs = task.reset_to_demo(demo)
                obs_state = env.get_observation(obs)
                move.reset(obs_state["gripper"])
                initial_success, _, initial_success_error = _task_success_state(task)
                if task_recorder_callback is not None and args.video_initial_snap:
                    task_recorder_callback()
                if args.record_video and video_mode == "observation":
                    frames.append(_video_frame(obs_state))

                transition_pairs = list(zip(spec.key_frameids[:-1], spec.key_frameids[1:]))
                transition_pairs = transition_pairs[: int(args.max_key_transitions)]
                for step_id, (current_key_idx, next_key_idx) in enumerate(transition_pairs):
                    action = _gripper_action(demo_obs[int(next_key_idx)])
                    step_row = {
                        "step_id": int(step_id),
                        "taskvar": spec.taskvar,
                        "task": spec.task,
                        "variation": int(spec.variation),
                        "demo_id": int(spec.demo_id),
                        "episode_key": spec.episode_key,
                        "current_key_idx": int(current_key_idx),
                        "next_key_idx": int(next_key_idx),
                        "key_delta": int(next_key_idx) - int(current_key_idx),
                        "target_action": _action_list(action),
                        "target_source": "demo_obs[next_key_idx].gripper_pose_plus_open",
                        "instruction": instructions[0] if isinstance(instructions, list) and instructions else instructions,
                    }
                    try:
                        move_out = move(action, verbose=False)
                        if len(move_out) == 3:
                            obs, reward, terminate = move_out
                        else:
                            obs, reward, terminate, _ = move_out
                        steps = int(step_id) + 1
                        step_row.update({"reward": float(reward), "terminate": bool(terminate), "error": None})
                        if float(reward) == 1.0:
                            success = True
                        obs_state = env.get_observation(obs)
                        if args.record_video and video_mode == "observation":
                            frames.append(_video_frame(obs_state))
                        _write_jsonl(action_path, step_row)
                        if success:
                            break
                    except official_exceptions as exc:
                        error = f"{type(exc).__name__}: {exc}"
                        step_row.update({"reward": 0.0, "terminate": False, "error": error})
                        _write_jsonl(action_path, step_row)
                        break

            except Exception as exc:
                error = f"{type(exc).__name__}: {exc}"

            video_paths, video_frame_counts, video_error = _save_recorder_video(
                output_root=output_root,
                task_recorder=task_recorder,
                task_recorder_callback=task_recorder_callback,
                frames=frames,
                taskvar=spec.taskvar,
                demo_id=spec.demo_id,
                success=success,
                fps=int(args.video_fps),
                video_mode=video_mode,
            )
            row = {
                "taskvar": spec.taskvar,
                "task": spec.task,
                "variation": int(spec.variation),
                "demo_id": int(spec.demo_id),
                "episode_key": spec.episode_key,
                "seed": str(args.seed),
                "split": "train",
                "key_frameids": [int(v) for v in spec.key_frameids],
                "num_key_frameids": len(spec.key_frameids),
                "max_key_transitions": int(args.max_key_transitions),
                "steps_executed": int(steps),
                "reward": float(reward),
                "success": bool(success),
                "terminate": bool(terminate),
                "initial_success": bool(initial_success),
                "initial_success_error": initial_success_error,
                "error": error,
                "action_log": str(action_path),
                "video_mode": video_mode,
                "video_mp4_paths": video_paths,
                "video_frame_counts": video_frame_counts,
                "video_error": _append_video_error(recorder_setup_error, video_error),
                "duration_seconds": round(time.time() - start_time, 3),
            }
            _write_jsonl(output_root / "results.jsonl", row)
            rows.append(row)
            print(
                f"[keystep-replay] trial_done taskvar={spec.taskvar} demo={spec.demo_id} "
                f"success={success} steps={steps} error={error}",
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
        description=(
            "Replay the exact GEMBench key-step policy training targets in RLBench. "
            "This validates obs_key -> next-key 8D gripper target labels without loading a model."
        )
    )
    parser.add_argument("--manifest", default="/mnt/yuhan/datasets/GEMBench/fastwam_cache/microsteps_9v32_4cam224_manifest.json")
    parser.add_argument("--key-frameids-path", default="/mnt/yuhan/datasets/GEMBench/fastwam_cache/microsteps_9v32_seed0_key_frameids.json")
    parser.add_argument("--gembench-root", default="/mnt/yuhan/datasets/GEMBench")
    parser.add_argument("--robot-3dlotus-root", default=None)
    parser.add_argument("--seed", default="seed0")
    parser.add_argument("--taskvars", default=None, help="Comma-separated taskvars, e.g. close_fridge+0,open_box+0.")
    parser.add_argument("--episodes", default=None, help="Comma-separated episode keys, e.g. episode0,episode7.")
    parser.add_argument("--episodes-per-taskvar", type=int, default=1)
    parser.add_argument("--max-trials", type=int, default=4)
    parser.add_argument("--max-key-transitions", type=int, default=12)
    parser.add_argument("--max-tries", type=int, default=10)
    parser.add_argument("--image-size", type=int, default=256)
    parser.add_argument("--output-root", default=None)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--record-video", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--video-mode", choices=("official_recorder", "observation"), default="official_recorder")
    parser.add_argument("--video-fps", type=int, default=30)
    parser.add_argument("--video-resolution", type=int, default=480)
    parser.add_argument("--video-include-robot-cameras", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--video-rotate-cam", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--video-initial-snap", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--video-recorder-required", action="store_true")
    args = parser.parse_args()

    specs = _select_specs(args)
    output_root = Path(
        args.output_root
        or f"runs/gembench_policy_keystep_target_replay/replay_{time.strftime('%Y%m%d_%H%M%S')}"
    ).expanduser()
    output_root.mkdir(parents=True, exist_ok=True)
    manifest_payload = {
        "eval_type": "gembench_policy_keystep_training_target_replay",
        "official_full_score": False,
        "write_official_preds": False,
        "target_source": "training policy_action_raw = gripper[next_key_idx]",
        "generated_at": utc_now(),
        "manifest": str(Path(args.manifest).expanduser()),
        "key_frameids_path": str(Path(args.key_frameids_path).expanduser()),
        "gembench_root": str(Path(args.gembench_root).expanduser()),
        "robot_3dlotus_root": args.robot_3dlotus_root,
        "seed": str(args.seed),
        "max_tries": int(args.max_tries),
        "max_key_transitions": int(args.max_key_transitions),
        "record_video": bool(args.record_video),
        "video_mode": args.video_mode,
        "selected_trials": [
            {
                "taskvar": spec.taskvar,
                "task": spec.task,
                "variation": int(spec.variation),
                "episode_key": spec.episode_key,
                "demo_id": int(spec.demo_id),
                "key_frameids": [int(v) for v in spec.key_frameids],
            }
            for spec in specs
        ],
        "git": git_provenance(),
    }
    write_json(output_root / "replay_manifest.json", manifest_payload)
    if args.dry_run:
        print(json.dumps(manifest_payload, ensure_ascii=True, indent=2), flush=True)
        return 0

    modules = _official_imports(args.robot_3dlotus_root)
    grouped: dict[str, list[ReplaySpec]] = defaultdict(list)
    for spec in specs:
        grouped[spec.taskvar].append(spec)

    all_rows: list[dict[str, Any]] = []
    for taskvar, group in grouped.items():
        all_rows.extend(
            _run_task_group(args=args, taskvar=taskvar, specs=group, modules=modules, output_root=output_root)
        )
    successes = sum(1 for row in all_rows if row.get("success"))
    summary = {
        "eval_type": "gembench_policy_keystep_training_target_replay",
        "official_full_score": False,
        "write_official_preds": False,
        "num_trials": len(all_rows),
        "successes": int(successes),
        "sr": float(successes / max(len(all_rows), 1)),
        "rows": all_rows,
        "output_root": str(output_root),
    }
    write_json(output_root / "summary.json", summary)
    print(json.dumps(summary, ensure_ascii=True, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
