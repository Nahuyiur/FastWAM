from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import time
import traceback
from collections import defaultdict
from dataclasses import asdict
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np

from .common import (
    OFFICIAL_CAMERA_NAMES,
    PROJECT_ROOT,
    TrialSpec,
    append_jsonl,
    git_provenance,
    safe_token,
    utc_now,
    write_json,
)

if TYPE_CHECKING:
    from .policy import GEMBenchOfficialActioner


def add_robot_3dlotus_path(robot_3dlotus_root: str | None) -> None:
    roots = []
    if robot_3dlotus_root:
        roots.append(Path(robot_3dlotus_root).expanduser())
    if os.environ.get("ROBOT_3DLOTUS_ROOT"):
        roots.append(Path(os.environ["ROBOT_3DLOTUS_ROOT"]).expanduser())
    for root in roots:
        root = root.resolve()
        if root.exists() and str(root) not in sys.path:
            sys.path.insert(0, str(root))


def _official_imports(robot_3dlotus_root: str | None):
    add_robot_3dlotus_path(robot_3dlotus_root)
    try:
        from genrobo3d.rlbench.environments import Mover, RLBenchEnv
        from pyrep.errors import ConfigurationPathError, IKError
        from rlbench.backend.exceptions import InvalidActionError
        from rlbench.backend.utils import task_file_to_task_class
    except Exception as exc:
        raise RuntimeError(
            "Official GEMBench eval requires robot-3dlotus/RLBench/PyRep on PYTHONPATH. "
            "Pass --robot-3dlotus-root or set ROBOT_3DLOTUS_ROOT in the simulator environment."
        ) from exc
    return {
        "Mover": Mover,
        "RLBenchEnv": RLBenchEnv,
        "task_file_to_task_class": task_file_to_task_class,
        "exceptions": (IKError, ConfigurationPathError, InvalidActionError),
    }


def _recorder_imports(robot_3dlotus_root: str | None):
    add_robot_3dlotus_path(robot_3dlotus_root)
    try:
        from genrobo3d.rlbench.recorder import (
            AttachedCameraMotion,
            CircleCameraMotion,
            StaticCameraMotion,
            TaskRecorder,
        )
        from pyrep.objects.dummy import Dummy
        from pyrep.objects.vision_sensor import VisionSensor
    except Exception as exc:
        raise RuntimeError(
            "Official GEMBench video recorder requires robot-3dlotus recorder and PyRep camera objects. "
            "Pass --robot-3dlotus-root or set ROBOT_3DLOTUS_ROOT in the simulator environment; "
            "or use --video-mode observation for action-step videos."
        ) from exc
    return {
        "AttachedCameraMotion": AttachedCameraMotion,
        "CircleCameraMotion": CircleCameraMotion,
        "Dummy": Dummy,
        "StaticCameraMotion": StaticCameraMotion,
        "TaskRecorder": TaskRecorder,
        "VisionSensor": VisionSensor,
    }


def _build_task_recorder(
    task: Any,
    modules: dict[str, Any],
    *,
    resolution: int,
    include_robot_cameras: bool,
    rotate_cam: bool,
    fps: int,
) -> Any:
    Dummy = modules["Dummy"]
    VisionSensor = modules["VisionSensor"]
    TaskRecorder = modules["TaskRecorder"]
    StaticCameraMotion = modules["StaticCameraMotion"]
    CircleCameraMotion = modules["CircleCameraMotion"]
    AttachedCameraMotion = modules["AttachedCameraMotion"]

    cam_resolution = [int(resolution), int(resolution)]
    cam_placeholder = Dummy("cam_cinematic_placeholder")
    cam = VisionSensor.create(cam_resolution)
    cam.set_pose(cam_placeholder.get_pose())
    cam.set_parent(cam_placeholder)

    if rotate_cam:
        global_cam_motion = CircleCameraMotion(cam, Dummy("cam_cinematic_base"), 0.005)
    else:
        global_cam_motion = StaticCameraMotion(cam)

    cams_motion = {"global": global_cam_motion}
    if include_robot_cameras:
        scene = task._scene
        for camera_name, scene_attr in (
            ("left", "_cam_over_shoulder_left"),
            ("right", "_cam_over_shoulder_right"),
            ("wrist", "_cam_wrist"),
        ):
            attached_cam = VisionSensor.create(cam_resolution)
            cams_motion[camera_name] = AttachedCameraMotion(attached_cam, getattr(scene, scene_attr))

    return TaskRecorder(cams_motion, fps=int(fps))


def _clear_task_recorder(task_recorder: Any | None) -> None:
    if task_recorder is None:
        return
    snaps = getattr(task_recorder, "_snaps", None)
    if isinstance(snaps, dict):
        task_recorder._snaps = {cam_name: [] for cam_name in snaps.keys()}


def _task_recorder_frame_counts(task_recorder: Any | None) -> dict[str, int]:
    if task_recorder is None:
        return {}
    snaps = getattr(task_recorder, "_snaps", None)
    if not isinstance(snaps, dict):
        return {}
    return {str(cam_name): len(images) for cam_name, images in snaps.items()}


def _write_task_recorder_snaps_mp4(path: Path, task_recorder: Any, *, fps: int) -> list[str]:
    snaps = getattr(task_recorder, "_snaps", None)
    if not isinstance(snaps, dict):
        raise RuntimeError("TaskRecorder fallback requires a _snaps dictionary.")
    try:
        import imageio.v2 as imageio
    except Exception as exc:
        raise RuntimeError("TaskRecorder fallback requires imageio when cv2 is unavailable.") from exc

    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    mp4_paths: list[str] = []
    for cam_name, images in snaps.items():
        if not images:
            continue
        frames = [np.asarray(image, dtype=np.uint8) for image in images]
        mp4_path = path / f"{safe_token(cam_name)}.mp4"
        try:
            imageio.mimwrite(str(mp4_path), frames, fps=int(fps), macro_block_size=1)
        except TypeError:
            imageio.mimwrite(str(mp4_path), frames, fps=int(fps))
        mp4_paths.append(str(mp4_path))
    if not mp4_paths:
        raise RuntimeError("TaskRecorder fallback found no frames to write.")
    return mp4_paths


class _SafeTaskRecorderCallback:
    def __init__(self, task_recorder: Any):
        self.task_recorder = task_recorder
        self.errors: list[str] = []
        self.disabled = False

    def reset(self) -> None:
        self.errors = []
        self.disabled = False

    def __call__(self) -> None:
        if self.disabled:
            return
        try:
            self.task_recorder.take_snap()
        except Exception as exc:
            self.errors.append(f"{type(exc).__name__}: {exc}")
            self.disabled = True

    @property
    def error(self) -> str | None:
        if not self.errors:
            return None
        return "; ".join(self.errors[:3])


def _video_frame(obs_state_dict: dict[str, Any]) -> np.ndarray:
    rgb = np.asarray(obs_state_dict.get("rgb"))
    if rgb.ndim != 4:
        raise ValueError(f"Expected official rgb [C,H,W,3], got {rgb.shape}")
    frames = []
    for image in rgb:
        arr = np.asarray(image)
        if arr.dtype != np.uint8:
            if np.issubdtype(arr.dtype, np.floating) and float(np.nanmax(arr)) <= 1.0:
                arr = arr * 255.0
            arr = np.clip(arr, 0, 255).astype(np.uint8)
        frames.append(arr)
    return np.concatenate(frames, axis=1)


def _frame_to_rgb_array(frame: Any) -> np.ndarray:
    if isinstance(frame, dict):
        images = [_frame_to_rgb_array(value) for value in frame.values()]
        return np.concatenate(images, axis=1)
    if hasattr(frame, "convert"):
        return np.asarray(frame.convert("RGB"), dtype=np.uint8)
    arr = np.asarray(frame)
    if arr.dtype != np.uint8:
        if np.issubdtype(arr.dtype, np.floating) and arr.size and float(np.nanmax(arr)) <= 1.0:
            arr = arr * 255.0
        arr = np.clip(arr, 0, 255).astype(np.uint8)
    if arr.ndim == 3 and arr.shape[0] == 3 and arr.shape[-1] != 3:
        arr = np.transpose(arr, (1, 2, 0))
    if arr.ndim != 3 or arr.shape[-1] != 3:
        raise ValueError(f"Expected RGB frame [H,W,3], got {arr.shape}")
    return arr


def _predicted_video_frames(value: Any) -> list[np.ndarray]:
    if value is None:
        return []
    if isinstance(value, (list, tuple)):
        return [_frame_to_rgb_array(frame) for frame in value]
    return [_frame_to_rgb_array(value)]


def _write_video(path: Path, frames: list[np.ndarray], fps: int) -> None:
    if not frames:
        return
    import imageio.v2 as imageio

    path.parent.mkdir(parents=True, exist_ok=True)
    imageio.mimwrite(str(path), frames, fps=int(fps), macro_block_size=1)


def _convert_recorder_avi_to_mp4(video_dir: Path) -> tuple[list[str], list[str], str | None]:
    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg is None:
        return [], [], "ffmpeg not found for recorder AVI -> MP4 conversion."
    avi_paths = sorted(Path(video_dir).glob("*.avi"))
    mp4_paths: list[str] = []
    errors: list[str] = []
    for avi_path in avi_paths:
        mp4_path = avi_path.with_suffix(".mp4")
        cmd = [
            ffmpeg,
            "-y",
            "-loglevel",
            "error",
            "-i",
            str(avi_path),
            "-c:v",
            "libx264",
            "-pix_fmt",
            "yuv420p",
            "-movflags",
            "+faststart",
            str(mp4_path),
        ]
        try:
            subprocess.run(cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
            mp4_paths.append(str(mp4_path))
        except subprocess.CalledProcessError as exc:
            errors.append(f"{avi_path.name}: {exc.stderr.strip() or exc}")
    return [str(path) for path in avi_paths], mp4_paths, "; ".join(errors) if errors else None


def _append_video_error(existing: str | None, new_error: str | None) -> str | None:
    if not new_error:
        return existing
    if existing:
        return f"{existing}; {new_error}"
    return new_error


def _task_success_state(task: Any) -> tuple[bool, bool, str | None]:
    try:
        success_fn = getattr(getattr(task, "_task", None), "success", None)
        if not callable(success_fn):
            return False, False, "task._task.success is not callable"
        value = success_fn()
        if isinstance(value, tuple):
            success = bool(value[0]) if len(value) > 0 else False
            terminate = bool(value[1]) if len(value) > 1 else False
        else:
            success = bool(value)
            terminate = False
        return success, terminate, None
    except Exception as exc:
        return False, False, f"{type(exc).__name__}: {exc}"


def _write_step_record(path: Path, row: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=True) + "\n")


def _action_list(value: Any) -> Any:
    if value is None:
        return None
    arr = np.asarray(value)
    return arr.astype(float).tolist()


def _official_result_row(row: dict[str, Any]) -> dict[str, Any]:
    return {
        "checkpoint": row["checkpoint"],
        "task": row["task"],
        "variation": row["variation"],
        "num_demos": row["num_demos"],
        "sr": row["sr"],
    }


class GEMBenchOfficialRunner:
    def __init__(
        self,
        *,
        actioner: GEMBenchOfficialActioner,
        output_root: Path,
        robot_3dlotus_root: str | None,
        image_size: tuple[int, int] = (256, 256),
        cam_ids: tuple[int, ...] = (0, 1, 2, 3),
        max_steps: int = 25,
        max_tries: int = 10,
        record_video: bool = False,
        video_fps: int = 8,
        video_stride: int = 1,
        video_mode: str = "observation",
        video_resolution: int = 480,
        video_include_robot_cameras: bool = True,
        video_rotate_cam: bool = False,
        video_recorder_required: bool = False,
        video_initial_snap: bool = False,
        min_video_frames: int = 0,
        initial_success_policy: str = "record_only",
        write_official_preds: bool = True,
        eval_protocol: str = "official_one_step",
        chunk_replan_steps: int = 1,
        chunk_predict_video: bool = False,
    ):
        self.actioner = actioner
        self.output_root = Path(output_root).resolve()
        self.robot_3dlotus_root = robot_3dlotus_root
        self.image_size = [int(image_size[0]), int(image_size[1])]
        self.cam_ids = tuple(int(idx) for idx in cam_ids)
        self.max_steps = int(max_steps)
        self.max_tries = int(max_tries)
        self.record_video = bool(record_video)
        self.video_fps = int(video_fps)
        self.video_stride = max(int(video_stride), 1)
        if video_mode not in {"observation", "official_recorder"}:
            raise ValueError(f"Unsupported video_mode={video_mode!r}")
        self.video_mode = str(video_mode)
        self.video_resolution = int(video_resolution)
        self.video_include_robot_cameras = bool(video_include_robot_cameras)
        self.video_rotate_cam = bool(video_rotate_cam)
        self.video_recorder_required = bool(video_recorder_required)
        self.video_initial_snap = bool(video_initial_snap)
        self.min_video_frames = max(int(min_video_frames), 0)
        if initial_success_policy not in {"record_only", "mark_invalid", "fail"}:
            raise ValueError(f"Unsupported initial_success_policy={initial_success_policy!r}")
        self.initial_success_policy = str(initial_success_policy)
        if bool(write_official_preds) and str(actioner.relation_mode) == "noop_smoke":
            raise ValueError("noop_smoke relation mode may not write official preds/results.jsonl.")
        if eval_protocol in {"fastwam_chunk_replan", "trace_chunk_replan"}:
            eval_protocol = "chunk_replan"
        if eval_protocol not in {"official_one_step", "chunk_replan"}:
            raise ValueError(f"Unsupported eval_protocol={eval_protocol!r}")
        if eval_protocol != "official_one_step" and bool(write_official_preds):
            raise ValueError("Non-official eval protocols must use write_official_preds=false.")
        self.write_official_preds = bool(write_official_preds)
        self.eval_protocol = str(eval_protocol)
        self.chunk_replan_steps = 1 if self.eval_protocol == "official_one_step" else max(int(chunk_replan_steps), 1)
        self.chunk_predict_video = bool(chunk_predict_video)

    def dry_run_payload(self, *, trials: list[TrialSpec], skipped_taskvars: list[str] | None = None) -> dict[str, Any]:
        return {
            "evidence_type": "gembench_official_dry_run",
            "status": "dry_run",
            "generated_at": utc_now(),
            "provenance": git_provenance(),
            "checkpoint": str(self.actioner.checkpoint),
            "run_dir": str(self.actioner.run_dir),
            "output_root": str(self.output_root),
            "relation_mode": str(self.actioner.relation_mode),
            "has_relation_expert": bool(self.actioner.has_relation),
            "action_horizon": int(self.actioner.action_horizon),
            "chunk_action_horizon": int(getattr(self.actioner, "chunk_action_horizon", self.actioner.action_horizon)),
            "training_action_horizon": int(getattr(self.actioner, "training_action_horizon", self.actioner.action_horizon)),
            "executed_action_index": int(getattr(self.actioner, "executed_action_index", 0)),
            "policy_vgm_auxiliary_action_horizon": getattr(self.actioner, "policy_vgm_auxiliary_action_horizon", None),
            "num_inference_steps": int(self.actioner.num_inference_steps),
            "model_seed": int(self.actioner.model_seed),
            "observation_camera_names": list(self.actioner.observation_camera_names),
            "max_steps": self.max_steps,
            "max_tries": self.max_tries,
            "record_video": self.record_video,
            "video_mode": self.video_mode,
            "video_fps": self.video_fps,
            "video_stride": self.video_stride,
            "video_resolution": self.video_resolution,
            "video_include_robot_cameras": self.video_include_robot_cameras,
            "video_rotate_cam": self.video_rotate_cam,
            "video_recorder_required": self.video_recorder_required,
            "video_initial_snap": self.video_initial_snap,
            "min_video_frames": self.min_video_frames,
            "initial_success_policy": self.initial_success_policy,
            "write_official_preds": self.write_official_preds,
            "eval_protocol": self.eval_protocol,
            "chunk_replan_steps": self.chunk_replan_steps,
            "chunk_predict_video": self.chunk_predict_video,
            "chunk_replan_contract": {
                "predicts_action_chunk": True,
                "requested_chunk_action_horizon": int(
                    getattr(self.actioner, "chunk_action_horizon", self.actioner.action_horizon)
                ),
                "executes_first_k_actions": int(self.chunk_replan_steps),
                "reward_check_each_action": True,
                "stop_on_success": True,
                "reobserve_after_k_actions": self.eval_protocol == "chunk_replan",
                "official_style_equivalent": (
                    self.eval_protocol == "chunk_replan"
                    and int(self.chunk_replan_steps) == 1
                    and int(getattr(self.actioner, "chunk_action_horizon", self.actioner.action_horizon))
                    == int(self.actioner.action_horizon)
                ),
            },
            "camera_names": [OFFICIAL_CAMERA_NAMES[idx] for idx in self.cam_ids],
            "trials": [asdict(trial) for trial in trials],
            "skipped_taskvars": skipped_taskvars or [],
        }

    def run(self, *, trials: list[TrialSpec], skipped_taskvars: list[str] | None = None) -> dict[str, Any]:
        modules = _official_imports(self.robot_3dlotus_root)
        trial_results_path = self.output_root / "trials" / "results.jsonl"
        if trial_results_path.exists():
            trial_results_path.unlink()
        grouped: dict[tuple[int, str, str], list[TrialSpec]] = defaultdict(list)
        for trial in trials:
            grouped[(int(trial.seed), trial.microstep_data_dir, trial.taskvar)].append(trial)

        trial_rows: list[dict[str, Any]] = []
        official_rows: list[dict[str, Any]] = []
        for (_, _, taskvar), group in grouped.items():
            row = self._run_taskvar(modules, taskvar=taskvar, trials=group, trial_rows=trial_rows)
            official_rows.append(row)
            if self.write_official_preds and int(row["num_demos"]) > 0:
                append_jsonl(
                    self.output_root / "preds" / f"seed{int(row['seed'])}" / "results.jsonl",
                    [_official_result_row(row)],
                )

        valid_rows = [row for row in trial_rows if row.get("valid_for_model_success", True)]
        valid_successes = [row for row in valid_rows if row["success"]]
        raw_successes = [row for row in trial_rows if row["success"]]
        visual_valid_rows = [row for row in trial_rows if row.get("visual_rollout_valid", False)]
        visual_too_short_rows = [row for row in trial_rows if row.get("video_too_short", False)]
        chunk_records = [
            record
            for row in trial_rows
            for record in row.get("chunk_replan_records", [])
            if isinstance(record, dict)
        ]
        chunk_replan_steps_used = sorted({int(record.get("replan_steps", 0)) for record in chunk_records})
        chunk_replan_horizons = [int(record.get("chunk_horizon", 0)) for record in chunk_records]
        chunk_replan_executed_actions = sum(
            len(record.get("selected_chunk_indices") or []) for record in chunk_records
        )
        summary = {
            "evidence_type": "gembench_official_success_rate",
            "status": "completed",
            "generated_at": utc_now(),
            "provenance": git_provenance(),
            "checkpoint": str(self.actioner.checkpoint),
            "run_dir": str(self.actioner.run_dir),
            "output_root": str(self.output_root),
            "relation_mode": str(self.actioner.relation_mode),
            "has_relation_expert": bool(self.actioner.has_relation),
            "action_horizon": int(self.actioner.action_horizon),
            "chunk_action_horizon": int(getattr(self.actioner, "chunk_action_horizon", self.actioner.action_horizon)),
            "training_action_horizon": int(getattr(self.actioner, "training_action_horizon", self.actioner.action_horizon)),
            "executed_action_index": int(getattr(self.actioner, "executed_action_index", 0)),
            "policy_vgm_auxiliary_action_horizon": getattr(self.actioner, "policy_vgm_auxiliary_action_horizon", None),
            "num_inference_steps": int(self.actioner.num_inference_steps),
            "model_seed": int(self.actioner.model_seed),
            "camera_names": [OFFICIAL_CAMERA_NAMES[idx] for idx in self.cam_ids],
            "observation_camera_names": list(self.actioner.observation_camera_names),
            "image_size": list(self.image_size),
            "max_steps": self.max_steps,
            "max_tries": self.max_tries,
            "record_video": self.record_video,
            "video_mode": self.video_mode,
            "video_fps": self.video_fps,
            "video_stride": self.video_stride,
            "video_resolution": self.video_resolution,
            "video_include_robot_cameras": self.video_include_robot_cameras,
            "video_rotate_cam": self.video_rotate_cam,
            "video_recorder_required": self.video_recorder_required,
            "video_initial_snap": self.video_initial_snap,
            "min_video_frames": self.min_video_frames,
            "initial_success_policy": self.initial_success_policy,
            "write_official_preds": self.write_official_preds,
            "eval_protocol": self.eval_protocol,
            "chunk_replan_steps": self.chunk_replan_steps,
            "chunk_predict_video": self.chunk_predict_video,
            "chunk_replan_contract": {
                "predicts_action_chunk": True,
                "requested_chunk_action_horizon": int(
                    getattr(self.actioner, "chunk_action_horizon", self.actioner.action_horizon)
                ),
                "executes_first_k_actions": int(self.chunk_replan_steps),
                "reward_check_each_action": True,
                "stop_on_success": True,
                "reobserve_after_k_actions": self.eval_protocol == "chunk_replan",
                "official_style_equivalent": (
                    self.eval_protocol == "chunk_replan"
                    and int(self.chunk_replan_steps) == 1
                    and int(getattr(self.actioner, "chunk_action_horizon", self.actioner.action_horizon))
                    == int(self.actioner.action_horizon)
                ),
            },
            "chunk_replan_total_replans": int(len(chunk_records)),
            "chunk_replan_total_executed_actions": int(chunk_replan_executed_actions),
            "chunk_replan_steps_used": chunk_replan_steps_used,
            "chunk_replan_max_chunk_horizon": int(max(chunk_replan_horizons) if chunk_replan_horizons else 0),
            "official_score_candidate": bool(
                self.eval_protocol == "official_one_step"
                and self.write_official_preds
                and self.actioner.relation_mode != "noop_smoke"
            ),
            "success_rate_scope": (
                "chunk_replan_visual_diagnostic_not_official_score"
                if self.eval_protocol == "chunk_replan"
                else
                "eval10_model_diagnostic_excludes_initial_success"
                if self.initial_success_policy == "mark_invalid"
                else "official_reward_raw_records_initial_success_diagnostic"
            ),
            "trials": len(trial_rows),
            "successes": int(len(raw_successes)),
            "success_rate": float(len(raw_successes) / max(len(trial_rows), 1)),
            "initial_successes": int(sum(1 for row in trial_rows if row.get("initial_success"))),
            "valid_trials": int(len(valid_rows)),
            "valid_successes": int(len(valid_successes)),
            "valid_success_rate": float(len(valid_successes) / max(len(valid_rows), 1)),
            "visual_rollout_valid_trials": int(len(visual_valid_rows)),
            "visual_rollout_too_short_trials": int(len(visual_too_short_rows)),
            "visual_rollout_valid_fraction": float(len(visual_valid_rows) / max(len(trial_rows), 1)),
            "skipped_taskvars": skipped_taskvars or [],
            "official_results": official_rows,
            "trial_results_path": str(trial_results_path),
        }
        append_jsonl(trial_results_path, trial_rows)
        write_json(self.output_root / "summary.json", summary)
        return summary

    def _run_taskvar(
        self,
        modules: dict[str, Any],
        *,
        taskvar: str,
        trials: list[TrialSpec],
        trial_rows: list[dict[str, Any]],
    ) -> dict[str, Any]:
        task_file_to_task_class = modules["task_file_to_task_class"]
        RLBenchEnv = modules["RLBenchEnv"]
        Mover = modules["Mover"]
        official_exceptions = modules["exceptions"]

        first = trials[0]
        apply_cameras = [OFFICIAL_CAMERA_NAMES[idx] for idx in self.cam_ids]
        env = RLBenchEnv(
            data_path=first.microstep_data_dir,
            apply_rgb=True,
            apply_pc=True,
            apply_mask=True,
            headless=True,
            image_size=self.image_size,
            cam_rand_factor=0,
            apply_cameras=apply_cameras,
        )
        successes = 0
        try:
            launch_start = time.time()
            print(
                f"[gembench-official] env_launch start taskvar={taskvar} data={first.microstep_data_dir}",
                flush=True,
            )
            env.env.launch()
            print(f"[gembench-official] env_launch done seconds={time.time() - launch_start:.1f}", flush=True)
            task_type = task_file_to_task_class(first.task)
            task = env.env.get_task(task_type)
            task.set_variation(first.variation)
            move = Mover(task, max_tries=self.max_tries)

            task_recorder = None
            task_recorder_callback = None
            video_mode_effective = "none"
            recorder_setup_error = None
            if self.record_video:
                if self.video_mode == "official_recorder":
                    try:
                        recorder_start = time.time()
                        print("[gembench-official] recorder_setup start", flush=True)
                        task_recorder = _build_task_recorder(
                            task,
                            _recorder_imports(self.robot_3dlotus_root),
                            resolution=self.video_resolution,
                            include_robot_cameras=self.video_include_robot_cameras,
                            rotate_cam=self.video_rotate_cam,
                            fps=self.video_fps,
                        )
                        task_recorder_callback = _SafeTaskRecorderCallback(task_recorder)
                        task._scene.register_step_callback(task_recorder_callback)
                        video_mode_effective = "official_recorder"
                        print(
                            f"[gembench-official] recorder_setup done seconds={time.time() - recorder_start:.1f}",
                            flush=True,
                        )
                    except Exception as exc:
                        recorder_setup_error = f"{type(exc).__name__}: {exc}"
                        if self.video_recorder_required:
                            raise
                        print(
                            "[gembench-official] recorder unavailable; falling back to observation-step video: "
                            f"{recorder_setup_error}",
                            flush=True,
                        )
                        video_mode_effective = "observation_fallback"
                else:
                    video_mode_effective = "observation"

            loaded_trials: list[tuple[TrialSpec, Any]] = []
            demo_load_failures: list[dict[str, Any]] = []
            for trial in trials:
                try:
                    demo = env.get_demo(trial.task, trial.variation, trial.demo_id, load_images=False)
                    loaded_trials.append((trial, demo))
                except Exception as exc:
                    failure = {**asdict(trial), "error": f"{type(exc).__name__}: {exc}"}
                    demo_load_failures.append(failure)
                    print(
                        f"[gembench-official] demo_load_failure taskvar={trial.taskvar} "
                        f"demo={trial.demo_id} error={failure['error']}",
                        flush=True,
                    )

            for trial, demo in loaded_trials:
                row = self._run_trial(
                    env=env,
                    task=task,
                    move=move,
                    trial=trial,
                    demo=demo,
                    official_exceptions=official_exceptions,
                    task_recorder=task_recorder,
                    task_recorder_callback=task_recorder_callback,
                    video_mode_effective=video_mode_effective,
                    recorder_setup_error=recorder_setup_error,
                )
                trial_rows.append(row)
                successes += int(row["success"])
                print(
                    "[gembench-official] "
                    f"taskvar={taskvar} demo={trial.demo_id} success={row['success']} "
                    f"steps={row['steps']} rate={successes}/{len(trial_rows)} error={row['error']}",
                    flush=True,
                )
        finally:
            try:
                env.env.shutdown()
            except Exception as exc:
                print(f"[gembench-official] env shutdown warning: {type(exc).__name__}: {exc}", flush=True)

        return {
            "checkpoint": str(self.actioner.checkpoint),
            "seed": int(first.seed),
            "split": first.split,
            "taskvar_file": first.taskvar_file,
            "microstep_data_dir": first.microstep_data_dir,
            "taskvar": taskvar,
            "task": first.task,
            "variation": int(first.variation),
            "num_demos": len(loaded_trials),
            "successes": int(successes),
            "sr": float(successes / max(len(loaded_trials), 1)),
            "requested_num_demos": len(trials),
            "demo_load_failures": demo_load_failures,
        }

    def _run_trial(
        self,
        *,
        env: Any,
        task: Any,
        move: Any,
        trial: TrialSpec,
        demo: Any,
        official_exceptions: tuple[type[BaseException], ...],
        task_recorder: Any | None,
        task_recorder_callback: _SafeTaskRecorderCallback | None,
        video_mode_effective: str,
        recorder_setup_error: str | None,
    ) -> dict[str, Any]:
        start = time.time()
        action_path = (
            self.output_root
            / "actions"
            / f"seed{trial.seed}"
            / safe_token(trial.taskvar)
            / f"demo_{trial.demo_id:03d}.jsonl"
        )
        if action_path.exists():
            action_path.unlink()

        frames: list[np.ndarray] = []
        reward = 0
        terminate = False
        success = False
        initial_success = False
        initial_terminate = False
        initial_success_error = None
        valid_for_model_success = True
        error = None
        error_traceback = None
        video_mode = video_mode_effective if self.record_video else "none"
        video_error = recorder_setup_error if video_mode == "observation_fallback" else None
        video_frame_counts: dict[str, int] = {}
        steps = 0
        instructions: Any = []
        video_path: Path | None = None
        video_avi_paths: list[str] = []
        video_mp4_paths: list[str] = []
        video_max_frame_count = 0
        video_duration_seconds = 0.0
        video_too_short = False
        visual_rollout_valid = False
        predicted_full_video_paths: list[str] = []
        predicted_prefix_video_paths: list[str] = []
        predicted_full_timeline_path: str | None = None
        predicted_prefix_timeline_path: str | None = None
        predicted_full_timeline_frames: list[np.ndarray] = []
        predicted_prefix_timeline_frames: list[np.ndarray] = []
        chunk_replan_records: list[dict[str, Any]] = []
        record_observation_video = self.record_video and video_mode in {"observation", "observation_fallback"}
        if self.record_video and task_recorder is not None:
            _clear_task_recorder(task_recorder)
        if task_recorder_callback is not None:
            task_recorder_callback.reset()
        try:
            print(
                f"[gembench-official] trial_start taskvar={trial.taskvar} demo={trial.demo_id} split={trial.split}",
                flush=True,
            )
            reset_start = time.time()
            instructions, obs = task.reset_to_demo(demo)
            obs_state_dict = env.get_observation(obs)
            move.reset(obs_state_dict["gripper"])
            print(f"[gembench-official] reset_to_demo done seconds={time.time() - reset_start:.1f}", flush=True)
            initial_success, initial_terminate, initial_success_error = _task_success_state(task)
            valid_for_model_success = not initial_success
            if initial_success:
                print(
                    f"[gembench-official] initial_success taskvar={trial.taskvar} demo={trial.demo_id} "
                    f"policy={self.initial_success_policy}",
                    flush=True,
                )
                if self.initial_success_policy == "fail":
                    raise RuntimeError("InitialSuccess: task is already successful immediately after reset_to_demo.")
                if self.initial_success_policy == "mark_invalid":
                    error = "InitialSuccess: task is already successful immediately after reset_to_demo."
            if task_recorder_callback is not None and self.video_initial_snap:
                task_recorder_callback()
                print("[gembench-official] recorder_initial_snap done", flush=True)
            if record_observation_video:
                try:
                    frames.append(_video_frame(obs_state_dict))
                except Exception as exc:
                    video_error = _append_video_error(video_error, f"{type(exc).__name__}: {exc}")

            if error is None and self.eval_protocol == "official_one_step":
                for step_id in range(self.max_steps):
                    predict_start = time.time()
                    print(
                        f"[gembench-official] predict_start taskvar={trial.taskvar} demo={trial.demo_id} step={step_id}",
                        flush=True,
                    )
                    output = self.actioner.predict(
                        task_str=trial.task,
                        variation=trial.variation,
                        step_id=step_id,
                        obs_state_dict=obs_state_dict,
                        episode_id=trial.demo_id,
                        instructions=instructions,
                    )
                    print(
                        f"[gembench-official] predict_done taskvar={trial.taskvar} demo={trial.demo_id} "
                        f"step={step_id} seconds={time.time() - predict_start:.1f}",
                        flush=True,
                    )
                    action = np.asarray(output["action"], dtype=np.float32)
                    if action.shape != (8,):
                        raise ValueError(f"Official GEMBench Mover requires one 8D action, got {action.shape}")
                    step_row = {
                        "step_id": int(step_id),
                        "taskvar": trial.taskvar,
                        "demo_id": int(trial.demo_id),
                        "instruction": output.get("instruction"),
                        "action": _action_list(action),
                        "normalized_action": _action_list(output.get("normalized_action")),
                        "denormalized_action": _action_list(output.get("denormalized_action")),
                        "model_denormalized_action": _action_list(output.get("model_denormalized_action")),
                        "policy_target_frame": output.get("policy_target_frame"),
                        "policy_local_frame": output.get("policy_local_frame"),
                        "normalization": output.get("normalization"),
                        "relation": output.get("relation"),
                        "num_inference_steps": output.get("num_inference_steps"),
                    }
                    try:
                        move_start = time.time()
                        obs, reward, terminate, _ = move(action, verbose=False)
                        steps = step_id + 1
                        print(
                            f"[gembench-official] move_done taskvar={trial.taskvar} demo={trial.demo_id} "
                            f"step={step_id} reward={float(reward)} terminate={bool(terminate)} "
                            f"seconds={time.time() - move_start:.1f}",
                            flush=True,
                        )
                        step_row.update({"reward": float(reward), "terminate": bool(terminate), "error": None})
                        if reward == 1:
                            success = True
                        obs_state_dict = env.get_observation(obs)
                        if record_observation_video and (step_id % self.video_stride == 0):
                            try:
                                frames.append(_video_frame(obs_state_dict))
                            except Exception as exc:
                                video_error = _append_video_error(video_error, f"{type(exc).__name__}: {exc}")
                        _write_step_record(action_path, step_row)
                        if reward == 1:
                            break
                    except official_exceptions as exc:
                        error = f"{type(exc).__name__}: {exc}"
                        step_row.update({"reward": 0.0, "terminate": False, "error": error})
                        _write_step_record(action_path, step_row)
                        break
            elif error is None:
                pending_actions: list[dict[str, Any]] = []
                replan_id = -1
                step_id = 0
                while step_id < self.max_steps:
                    if not pending_actions:
                        replan_id += 1
                        predict_start = time.time()
                        print(
                            f"[gembench-official] chunk_replan_predict_start taskvar={trial.taskvar} "
                            f"demo={trial.demo_id} step={step_id} replan={replan_id}",
                            flush=True,
                        )
                        output = self.actioner.predict_chunk(
                            task_str=trial.task,
                            variation=trial.variation,
                            step_id=step_id,
                            obs_state_dict=obs_state_dict,
                            episode_id=trial.demo_id,
                            instructions=instructions,
                            return_predicted_video=self.chunk_predict_video,
                        )
                        print(
                            f"[gembench-official] chunk_replan_predict_done taskvar={trial.taskvar} "
                            f"demo={trial.demo_id} step={step_id} replan={replan_id} "
                            f"seconds={time.time() - predict_start:.1f}",
                            flush=True,
                        )
                        action_chunk = np.asarray(output["action_chunk"], dtype=np.float32)
                        normalized_chunk = np.asarray(output["normalized_action_chunk"], dtype=np.float32)
                        denormalized_chunk = np.asarray(output["denormalized_action_chunk"], dtype=np.float32)
                        model_denormalized_chunk = np.asarray(
                            output.get("model_denormalized_action_chunk", denormalized_chunk),
                            dtype=np.float32,
                        )
                        if action_chunk.ndim != 2 or action_chunk.shape[-1] != 8:
                            raise ValueError(f"Chunk-replan requires action_chunk [T,8], got {action_chunk.shape}")
                        n_exec = min(
                            int(self.chunk_replan_steps),
                            int(action_chunk.shape[0]),
                            int(self.max_steps - step_id),
                        )
                        if n_exec <= 0:
                            break

                        predicted_full_paths_for_replan: list[str] = []
                        predicted_prefix_paths_for_replan: list[str] = []
                        predicted_frames = _predicted_video_frames(output.get("predicted_video"))
                        if predicted_frames:
                            pred_dir = (
                                self.output_root
                                / "predicted_videos"
                                / f"seed{trial.seed}"
                                / safe_token(trial.taskvar)
                                / f"demo_{trial.demo_id:03d}"
                            )
                            full_path = pred_dir / f"replan_{replan_id:03d}_full_imagination.mp4"
                            try:
                                _write_video(full_path, predicted_frames, fps=self.video_fps)
                                predicted_full_video_paths.append(str(full_path))
                                predicted_full_paths_for_replan.append(str(full_path))
                                predicted_full_timeline_frames.extend(predicted_frames)
                            except Exception as exc:
                                video_error = _append_video_error(
                                    video_error,
                                    f"PredictedFullVideoWriteError: {type(exc).__name__}: {exc}",
                                )

                            ratio = max(int(output.get("action_video_freq_ratio") or 4), 1)
                            prefix_frames = [
                                frame for frame_idx, frame in enumerate(predicted_frames) if int(frame_idx) * ratio <= n_exec
                            ]
                            if prefix_frames:
                                prefix_path = pred_dir / f"replan_{replan_id:03d}_executed_prefix.mp4"
                                try:
                                    _write_video(prefix_path, prefix_frames, fps=self.video_fps)
                                    predicted_prefix_video_paths.append(str(prefix_path))
                                    predicted_prefix_paths_for_replan.append(str(prefix_path))
                                    if predicted_prefix_timeline_frames:
                                        predicted_prefix_timeline_frames.extend(prefix_frames[1:] or prefix_frames)
                                    else:
                                        predicted_prefix_timeline_frames.extend(prefix_frames)
                                except Exception as exc:
                                    video_error = _append_video_error(
                                        video_error,
                                        f"PredictedPrefixVideoWriteError: {type(exc).__name__}: {exc}",
                                    )

                        selected_indices = list(range(n_exec))
                        chunk_replan_records.append(
                            {
                                "replan_id": int(replan_id),
                                "start_step_id": int(step_id),
                                "selected_chunk_indices": selected_indices,
                                "chunk_horizon": int(action_chunk.shape[0]),
                                "chunk_action_horizon": int(output.get("chunk_action_horizon", action_chunk.shape[0])),
                                "replan_steps": int(n_exec),
                                "num_video_frames": output.get("num_video_frames"),
                                "action_video_freq_ratio": output.get("action_video_freq_ratio"),
                                "predicted_full_video_paths": predicted_full_paths_for_replan,
                                "predicted_prefix_video_paths": predicted_prefix_paths_for_replan,
                            }
                        )
                        for chunk_action_index in selected_indices:
                            pending_actions.append(
                                {
                                    "action": action_chunk[chunk_action_index],
                                    "normalized_action": normalized_chunk[chunk_action_index],
                                    "denormalized_action": denormalized_chunk[chunk_action_index],
                                    "model_denormalized_action": model_denormalized_chunk[chunk_action_index],
                                    "instruction": output.get("instruction"),
                                    "relation": output.get("relation"),
                                    "policy_target_frame": output.get("policy_target_frame"),
                                    "policy_local_frame": output.get("policy_local_frame"),
                                    "normalization": output.get("normalization"),
                                    "num_inference_steps": output.get("num_inference_steps"),
                                    "chunk_horizon": int(action_chunk.shape[0]),
                                    "chunk_action_horizon": int(output.get("chunk_action_horizon", action_chunk.shape[0])),
                                    "replan_id": int(replan_id),
                                    "chunk_action_index": int(chunk_action_index),
                                    "selected_chunk_indices": selected_indices,
                                    "predicted_full_video_paths": predicted_full_paths_for_replan,
                                    "predicted_prefix_video_paths": predicted_prefix_paths_for_replan,
                                    "normalized_action_chunk": normalized_chunk if chunk_action_index == 0 else None,
                                    "denormalized_action_chunk": denormalized_chunk if chunk_action_index == 0 else None,
                                    "model_denormalized_action_chunk": model_denormalized_chunk
                                    if chunk_action_index == 0
                                    else None,
                                }
                            )

                    item = pending_actions.pop(0)
                    action = np.asarray(item["action"], dtype=np.float32)
                    if action.shape != (8,):
                        raise ValueError(f"Official GEMBench Mover requires one 8D action, got {action.shape}")
                    step_row = {
                        "step_id": int(step_id),
                        "taskvar": trial.taskvar,
                        "demo_id": int(trial.demo_id),
                        "instruction": item.get("instruction"),
                        "action": _action_list(action),
                        "normalized_action": _action_list(item.get("normalized_action")),
                        "denormalized_action": _action_list(item.get("denormalized_action")),
                        "model_denormalized_action": _action_list(item.get("model_denormalized_action")),
                        "policy_target_frame": item.get("policy_target_frame"),
                        "policy_local_frame": item.get("policy_local_frame"),
                        "normalization": item.get("normalization"),
                        "relation": item.get("relation"),
                        "num_inference_steps": item.get("num_inference_steps"),
                        "eval_protocol": self.eval_protocol,
                        "official_full_score": False,
                        "write_official_preds": bool(self.write_official_preds),
                        "replan_id": int(item.get("replan_id")),
                        "chunk_action_index": int(item.get("chunk_action_index")),
                        "selected_chunk_indices": item.get("selected_chunk_indices"),
                        "chunk_horizon": int(item.get("chunk_horizon")),
                        "chunk_action_horizon": int(item.get("chunk_action_horizon", item.get("chunk_horizon"))),
                        "chunk_replan_steps": int(self.chunk_replan_steps),
                        "predicted_full_video_paths": item.get("predicted_full_video_paths"),
                        "predicted_prefix_video_paths": item.get("predicted_prefix_video_paths"),
                    }
                    if item.get("normalized_action_chunk") is not None:
                        step_row["normalized_action_chunk"] = _action_list(item.get("normalized_action_chunk"))
                    if item.get("denormalized_action_chunk") is not None:
                        step_row["denormalized_action_chunk"] = _action_list(item.get("denormalized_action_chunk"))
                    if item.get("model_denormalized_action_chunk") is not None:
                        step_row["model_denormalized_action_chunk"] = _action_list(
                            item.get("model_denormalized_action_chunk")
                        )
                    try:
                        move_start = time.time()
                        obs, reward, terminate, _ = move(action, verbose=False)
                        steps = step_id + 1
                        print(
                            f"[gembench-official] chunk_replan_move_done taskvar={trial.taskvar} "
                            f"demo={trial.demo_id} step={step_id} replan={item.get('replan_id')} "
                            f"chunk_idx={item.get('chunk_action_index')} reward={float(reward)} "
                            f"terminate={bool(terminate)} seconds={time.time() - move_start:.1f}",
                            flush=True,
                        )
                        step_row.update({"reward": float(reward), "terminate": bool(terminate), "error": None})
                        if reward == 1:
                            success = True
                        obs_state_dict = env.get_observation(obs)
                        if record_observation_video and (step_id % self.video_stride == 0):
                            try:
                                frames.append(_video_frame(obs_state_dict))
                            except Exception as exc:
                                video_error = _append_video_error(video_error, f"{type(exc).__name__}: {exc}")
                        _write_step_record(action_path, step_row)
                        step_id += 1
                        if reward == 1:
                            break
                    except official_exceptions as exc:
                        error = f"{type(exc).__name__}: {exc}"
                        step_row.update({"reward": 0.0, "terminate": False, "error": error})
                        _write_step_record(action_path, step_row)
                        break
        except Exception as exc:
            error = f"{type(exc).__name__}: {exc}"
            error_traceback = traceback.format_exc()
            print(f"[gembench-official] trial exception:\n{error_traceback}", flush=True)

        if predicted_full_timeline_frames:
            try:
                pred_dir = self.output_root / "predicted_videos" / f"seed{trial.seed}" / safe_token(trial.taskvar)
                full_timeline = pred_dir / f"demo_{trial.demo_id:03d}_full_imagination_timeline.mp4"
                _write_video(full_timeline, predicted_full_timeline_frames, fps=self.video_fps)
                predicted_full_timeline_path = str(full_timeline)
            except Exception as exc:
                video_error = _append_video_error(
                    video_error,
                    f"PredictedFullTimelineWriteError: {type(exc).__name__}: {exc}",
                )
        if predicted_prefix_timeline_frames:
            try:
                pred_dir = self.output_root / "predicted_videos" / f"seed{trial.seed}" / safe_token(trial.taskvar)
                prefix_timeline = pred_dir / f"demo_{trial.demo_id:03d}_executed_prefix_timeline.mp4"
                _write_video(prefix_timeline, predicted_prefix_timeline_frames, fps=self.video_fps)
                predicted_prefix_timeline_path = str(prefix_timeline)
            except Exception as exc:
                video_error = _append_video_error(
                    video_error,
                    f"PredictedPrefixTimelineWriteError: {type(exc).__name__}: {exc}",
                )

        if self.record_video and task_recorder is not None:
            if task_recorder_callback is not None:
                video_error = _append_video_error(video_error, task_recorder_callback.error)
            video_frame_counts = _task_recorder_frame_counts(task_recorder)
            if any(video_frame_counts.values()):
                reward_token = int(float(reward or 0))
                video_path = (
                    self.output_root
                    / "videos"
                    / f"seed{trial.seed}"
                    / safe_token(trial.taskvar)
                    / f"demo_{trial.demo_id:03d}_SR{reward_token}"
                )
                try:
                    save_start = time.time()
                    print(
                        f"[gembench-official] recorder_save start path={video_path} frames={video_frame_counts}",
                        flush=True,
                    )
                    video_path.parent.mkdir(parents=True, exist_ok=True)
                    task_recorder.save(str(video_path))
                    print(f"[gembench-official] recorder_save done seconds={time.time() - save_start:.1f}", flush=True)
                    convert_start = time.time()
                    print(f"[gembench-official] recorder_mp4_convert start path={video_path}", flush=True)
                    video_avi_paths, video_mp4_paths, convert_error = _convert_recorder_avi_to_mp4(video_path)
                    video_error = _append_video_error(video_error, convert_error)
                    print(
                        f"[gembench-official] recorder_mp4_convert done seconds={time.time() - convert_start:.1f} "
                        f"mp4s={len(video_mp4_paths)}",
                        flush=True,
                    )
                except Exception as exc:
                    print(
                        f"[gembench-official] recorder_save failed; trying imageio mp4 fallback: "
                        f"{type(exc).__name__}: {exc}",
                        flush=True,
                    )
                    try:
                        fallback_start = time.time()
                        video_avi_paths = []
                        video_mp4_paths = _write_task_recorder_snaps_mp4(
                            video_path,
                            task_recorder,
                            fps=self.video_fps,
                        )
                        print(
                            f"[gembench-official] recorder_imageio_fallback done "
                            f"seconds={time.time() - fallback_start:.1f} mp4s={len(video_mp4_paths)}",
                            flush=True,
                        )
                    except Exception as fallback_exc:
                        video_error = _append_video_error(
                            video_error,
                            f"{type(exc).__name__}: {exc}; fallback {type(fallback_exc).__name__}: {fallback_exc}",
                        )
                        video_path = None
                finally:
                    _clear_task_recorder(task_recorder)
            else:
                video_error = _append_video_error(video_error, "TaskRecorder captured 0 frames.")
        elif self.record_video and frames:
            video_path = (
                self.output_root
                / "videos"
                / f"seed{trial.seed}"
                / safe_token(trial.taskvar)
                / f"demo_{trial.demo_id:03d}.mp4"
            )
            try:
                _write_video(video_path, frames, fps=self.video_fps)
                video_frame_counts = {"observation": len(frames)}
                video_mp4_paths = [str(video_path)]
            except Exception as exc:
                video_error = _append_video_error(video_error, f"{type(exc).__name__}: {exc}")
                video_path = None
        elif self.record_video and recorder_setup_error:
            video_error = _append_video_error(video_error, recorder_setup_error)

        if self.record_video and self.video_recorder_required and (video_error or video_path is None):
            raise RuntimeError(f"Required GEMBench video artifact was not produced: {video_error or 'missing video_path'}")

        if self.record_video:
            if video_frame_counts:
                video_max_frame_count = int(max(video_frame_counts.values()))
            elif frames:
                video_max_frame_count = int(len(frames))
            video_duration_seconds = float(video_max_frame_count / max(int(self.video_fps), 1))
            if self.min_video_frames > 0:
                video_too_short = video_max_frame_count < self.min_video_frames
                visual_rollout_valid = bool(video_path is not None and not video_too_short)
                if video_too_short:
                    video_error = _append_video_error(
                        video_error,
                        f"VideoTooShort: max_frames={video_max_frame_count} min_video_frames={self.min_video_frames}",
                    )
            else:
                visual_rollout_valid = bool(video_path is not None)

        return {
            **asdict(trial),
            "success": bool(success),
            "reward": float(reward or 0),
            "terminate": bool(terminate),
            "initial_success": bool(initial_success),
            "initial_terminate": bool(initial_terminate),
            "initial_success_error": initial_success_error,
            "valid_for_model_success": bool(valid_for_model_success),
            "initial_success_policy": self.initial_success_policy,
            "steps": int(steps),
            "seconds": float(time.time() - start),
            "instructions": instructions,
            "actions_path": str(action_path) if action_path.exists() else None,
            "video_path": str(video_path) if video_path is not None else None,
            "video_avi_paths": video_avi_paths,
            "video_mp4_paths": video_mp4_paths,
            "video_mode": video_mode,
            "video_frame_counts": video_frame_counts,
            "video_max_frame_count": int(video_max_frame_count),
            "video_duration_seconds": float(video_duration_seconds),
            "min_video_frames": int(self.min_video_frames),
            "video_too_short": bool(video_too_short),
            "visual_rollout_valid": bool(visual_rollout_valid),
            "eval_protocol": self.eval_protocol,
            "official_full_score": False,
            "write_official_preds": bool(self.write_official_preds),
            "chunk_replan_steps": int(self.chunk_replan_steps),
            "chunk_action_horizon": int(getattr(self.actioner, "chunk_action_horizon", self.actioner.action_horizon)),
            "chunk_predict_video": bool(self.chunk_predict_video),
            "reward_check_each_action": True,
            "stop_on_success": True,
            "chunk_replan_records": chunk_replan_records,
            "predicted_full_video_paths": predicted_full_video_paths,
            "predicted_prefix_video_paths": predicted_prefix_video_paths,
            "predicted_full_timeline_path": predicted_full_timeline_path,
            "predicted_prefix_timeline_path": predicted_prefix_timeline_path,
            "predicted_full_timeline_frame_count": int(len(predicted_full_timeline_frames)),
            "predicted_prefix_timeline_frame_count": int(len(predicted_prefix_timeline_frames)),
            "video_error": video_error,
            "error": error,
            "error_traceback": error_traceback,
        }
