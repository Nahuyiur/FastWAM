#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
SCRIPTS_ROOT = REPO_ROOT / "scripts"
if SRC_ROOT.is_dir() and str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))
if SCRIPTS_ROOT.is_dir() and str(SCRIPTS_ROOT) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_ROOT))

from fastwam.evaluation.gembench_official.common import git_provenance, safe_token, utc_now, write_json
from fastwam.evaluation.gembench_official.policy import GEMBenchOfficialActioner
from fastwam.evaluation.gembench_official.runner import (
    OFFICIAL_CAMERA_NAMES,
    _action_list,
    _official_imports,
    add_robot_3dlotus_path,
)

from replay_gembench_policy_keystep_targets import ReplaySpec, _gripper_action, _select_specs


ACTION_DIM_NAMES = ("x", "y", "z", "qx", "qy", "qz", "qw", "gripper")


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=True) + "\n")


def _normalize_quat(quat: np.ndarray) -> np.ndarray:
    q = np.asarray(quat, dtype=np.float64).reshape(4)
    norm = float(np.linalg.norm(q))
    if not np.isfinite(norm) or norm <= 1.0e-8:
        return np.asarray([0.0, 0.0, 0.0, 1.0], dtype=np.float64)
    return q / norm


def _quat_angle_deg(pred: np.ndarray, target: np.ndarray) -> float:
    qp = _normalize_quat(pred)
    qt = _normalize_quat(target)
    dot = float(abs(np.dot(qp, qt)))
    dot = min(max(dot, 0.0), 1.0)
    return float(2.0 * math.acos(dot) * 180.0 / math.pi)


def _gripper_bin(value: float) -> int:
    return 1 if float(value) > 0.5 else 0


def _failure_class(error: str | None) -> str:
    if not error:
        return "none"
    lower = str(error).lower()
    if "outside of workspace" in lower:
        return "outside_workspace"
    if "collision" in lower or "inaccessible" in lower or "path could not be found" in lower:
        return "ik_or_collision_or_inaccessible"
    if "initialsuccess" in lower:
        return "initial_success_invalid"
    return "other"


def _official_workspace(robot_3dlotus_root: str | None) -> dict[str, Any]:
    add_robot_3dlotus_path(robot_3dlotus_root)
    try:
        from genrobo3d.configs.rlbench.constants import get_robot_workspace

        workspace = dict(get_robot_workspace(real_robot=False))
        workspace["_source"] = "robot_3dlotus.genrobo3d.configs.rlbench.constants"
        return workspace
    except Exception as exc:
        return {
            "TABLE_HEIGHT": 0.7505,
            "X_BBOX": (-0.5, 1.5),
            "Y_BBOX": (-1.0, 1.0),
            "Z_BBOX": (0.2, 2.0),
            "_source": f"fallback_verified_robot_3dlotus_constants_import_failed:{type(exc).__name__}:{exc}",
        }


def _maybe_voxel_downsample(
    xyz: np.ndarray,
    *,
    voxel_size: float,
) -> tuple[np.ndarray, str]:
    if float(voxel_size) <= 0:
        return xyz, "disabled"
    try:
        import open3d as o3d

        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(xyz)
        pcd, _, _ = pcd.voxel_down_sample_and_trace(float(voxel_size), np.min(xyz, 0), np.max(xyz, 0))
        return np.asarray(pcd.points), "open3d_voxel_down_sample_and_trace"
    except Exception as exc:
        return xyz, f"voxel_downsample_unavailable:{type(exc).__name__}:{exc}"


def _compute_official_local_frame(
    obs_state: dict[str, Any],
    *,
    workspace: dict[str, Any],
    xyz_shift: str,
    xyz_norm: bool,
    rm_table: bool,
    voxel_size: float,
    num_points: int,
    sample_seed: int,
) -> dict[str, Any]:
    """Compute the same xyz frame family used by robot-3dlotus simple policy.

    Official simple_policy_ptv3 uses xyz_shift=center and xyz_norm=False for
    GEMBench, then converts model xyz back by `pred_xyz + pc_centroid`.
    This diagnostic mirrors that geometry path without running the official
    policy model. It records provenance because sampling/voxelization can affect
    centroids.
    """
    xyz = np.asarray(obs_state.get("pc"), dtype=np.float64)
    if xyz.ndim != 4 or xyz.shape[-1] != 3:
        raise ValueError(f"Expected obs_state['pc'] with shape [C,H,W,3], got {xyz.shape}")
    xyz = xyz.reshape(-1, 3)

    before_workspace = int(xyz.shape[0])
    in_mask = (
        (xyz[:, 0] > float(workspace["X_BBOX"][0]))
        & (xyz[:, 0] < float(workspace["X_BBOX"][1]))
        & (xyz[:, 1] > float(workspace["Y_BBOX"][0]))
        & (xyz[:, 1] < float(workspace["Y_BBOX"][1]))
        & (xyz[:, 2] > float(workspace["Z_BBOX"][0]))
        & (xyz[:, 2] < float(workspace["Z_BBOX"][1]))
    )
    if rm_table:
        in_mask = in_mask & (xyz[:, 2] > float(workspace["TABLE_HEIGHT"]))
    xyz = xyz[in_mask]
    after_workspace = int(xyz.shape[0])
    if after_workspace <= 0:
        raise ValueError("Official-local point cloud is empty after workspace/table filtering.")

    xyz, voxel_note = _maybe_voxel_downsample(xyz, voxel_size=float(voxel_size))
    after_voxel = int(xyz.shape[0])
    if after_voxel <= 0:
        raise ValueError("Official-local point cloud is empty after voxel downsample.")

    sampled = False
    if int(num_points) > 0 and len(xyz) > int(num_points):
        rng = np.random.default_rng(int(sample_seed))
        point_idxs = rng.choice(len(xyz), int(num_points), replace=False)
        xyz = xyz[point_idxs]
        sampled = True

    if xyz_shift == "none":
        centroid = np.zeros((3,), dtype=np.float64)
    elif xyz_shift == "center":
        centroid = np.mean(xyz, axis=0)
    elif xyz_shift == "gripper":
        gripper = np.asarray(obs_state.get("gripper"), dtype=np.float64).reshape(-1)
        if gripper.shape[0] < 3:
            raise ValueError(f"Expected obs_state['gripper'] to contain xyz, got {gripper.shape}")
        centroid = gripper[:3].copy()
    else:
        raise ValueError(f"Unsupported xyz_shift={xyz_shift!r}")

    if xyz_norm:
        radius = float(np.max(np.sqrt(np.sum((xyz - centroid) ** 2, axis=1))))
        if not np.isfinite(radius) or radius <= 1.0e-8:
            radius = 1.0
    else:
        radius = 1.0

    return {
        "centroid": centroid,
        "radius": float(radius),
        "table_height": float(workspace["TABLE_HEIGHT"]),
        "min_action_z": float(workspace["TABLE_HEIGHT"]) + 0.005,
        "num_points_before_workspace_filter": before_workspace,
        "num_points_after_workspace_table_filter": after_workspace,
        "num_points_after_voxel": after_voxel,
        "num_points_used": int(xyz.shape[0]),
        "sampled_to_num_points": bool(sampled),
        "voxel_note": voxel_note,
        "xyz_shift": str(xyz_shift),
        "xyz_norm": bool(xyz_norm),
        "rm_table": bool(rm_table),
        "voxel_size": float(voxel_size),
        "num_points_target": int(num_points),
    }


def _to_local_xyz(action: np.ndarray, frame: dict[str, Any]) -> np.ndarray:
    arr = np.asarray(action, dtype=np.float64).reshape(8)
    return (arr[:3] - np.asarray(frame["centroid"], dtype=np.float64)) / float(frame["radius"])


def _load_action_stats(path: str | None) -> dict[str, Any] | None:
    if path is None or str(path).strip() in {"", "none", "null"}:
        return None
    stats_path = Path(path).expanduser()
    if not stats_path.exists():
        return None
    payload = json.loads(stats_path.read_text(encoding="utf-8"))
    action = payload.get("action", {}).get("default")
    if not isinstance(action, dict):
        return None
    out = {}
    for key in ("global_min", "global_max", "global_q01", "global_q99"):
        value = action.get(key)
        if isinstance(value, list) and len(value) == 8:
            out[key] = np.asarray(value, dtype=np.float64)
    if "global_min" not in out or "global_max" not in out:
        return None
    # The current GEMBench stats file can be a dummy 0/1 placeholder for the
    # normalizer mode. Do not report it as a real raw-action distribution.
    if np.allclose(out["global_min"], 0.0) and np.allclose(out["global_max"], 1.0):
        out["looks_like_dummy_0_1"] = True
    return out


def _bounds_flags(action: np.ndarray, low: np.ndarray, high: np.ndarray, *, margin: float = 0.0) -> tuple[bool, list[str]]:
    arr = np.asarray(action, dtype=np.float64).reshape(8)
    mask = (arr < (low - margin)) | (arr > (high + margin))
    return bool(mask.any()), [ACTION_DIM_NAMES[i] for i, flag in enumerate(mask.tolist()) if flag]


def _predict_and_probe_transition(
    *,
    args: argparse.Namespace,
    actioner: GEMBenchOfficialActioner,
    env: Any,
    task: Any,
    move: Any,
    official_exceptions: tuple[type[BaseException], ...],
    spec: ReplaySpec,
    demo: Any,
    demo_obs: list[Any],
    transition_pairs: list[tuple[int, int]],
    step_id: int,
    current_key_idx: int,
    next_key_idx: int,
) -> dict[str, Any]:
    instructions, obs = task.reset_to_demo(demo)
    obs_state = env.get_observation(obs)
    move.reset(obs_state["gripper"])

    prefix_error = None
    prefix_reward = 0.0
    prefix_terminate = False
    for prefix_step, (_, prefix_next_key_idx) in enumerate(transition_pairs[:step_id]):
        prefix_action = _gripper_action(demo_obs[int(prefix_next_key_idx)])
        try:
            move_out = move(prefix_action, verbose=False)
            if len(move_out) == 3:
                obs, prefix_reward, prefix_terminate = move_out
            else:
                obs, prefix_reward, prefix_terminate, _ = move_out
            obs_state = env.get_observation(obs)
        except official_exceptions as exc:
            prefix_error = f"{type(exc).__name__}: {exc}"
            break

    current_gt_action = _gripper_action(demo_obs[int(current_key_idx)])
    gt_action = _gripper_action(demo_obs[int(next_key_idx)])
    pred_error = None
    pred_output: dict[str, Any] | None = None
    try:
        pred_output = actioner.predict(
            task_str=spec.task,
            variation=spec.variation,
            step_id=int(step_id),
            obs_state_dict=obs_state,
            episode_id=spec.demo_id,
            instructions=instructions,
        )
    except Exception as exc:
        pred_error = f"{type(exc).__name__}: {exc}"

    if pred_output is None:
        return {
            "taskvar": spec.taskvar,
            "task": spec.task,
            "variation": int(spec.variation),
            "episode_key": spec.episode_key,
            "demo_id": int(spec.demo_id),
            "step_id": int(step_id),
            "current_key_idx": int(current_key_idx),
            "next_key_idx": int(next_key_idx),
            "key_delta": int(next_key_idx) - int(current_key_idx),
            "current_gt_action": _action_list(current_gt_action),
            "gt_next_key_action": _action_list(gt_action),
            "prefix_error": prefix_error,
            "model_predict_error": pred_error,
            "pred_mover_error": None,
            "pred_mover_failure_class": "no_prediction",
        }

    pred_denorm = np.asarray(pred_output["denormalized_action"], dtype=np.float64).reshape(8)
    pred_exec = np.asarray(pred_output["action"], dtype=np.float64).reshape(8)
    pred_norm = np.asarray(pred_output["normalized_action"], dtype=np.float64).reshape(8)

    pred_mover_error = None
    pred_reward = 0.0
    pred_terminate = False
    if not prefix_error and bool(args.probe_predicted_mover):
        try:
            move_out = move(pred_exec.astype(np.float32), verbose=False)
            if len(move_out) == 3:
                _, pred_reward, pred_terminate = move_out
            else:
                _, pred_reward, pred_terminate, _ = move_out
        except official_exceptions as exc:
            pred_mover_error = f"{type(exc).__name__}: {exc}"

    official_local_error: str | None = None
    official_frame: dict[str, Any] | None = None
    official_local_payload: dict[str, Any] | None = None
    try:
        official_frame = _compute_official_local_frame(
            obs_state,
            workspace=args._official_workspace,
            xyz_shift=str(args.official_local_xyz_shift),
            xyz_norm=bool(args.official_local_xyz_norm),
            rm_table=bool(args.official_local_rm_table),
            voxel_size=float(args.official_local_voxel_size),
            num_points=int(args.official_local_num_points),
            sample_seed=int(args.official_local_sample_seed) + int(spec.demo_id) * 1000 + int(step_id),
        )
        gt_local = _to_local_xyz(gt_action, official_frame)
        pred_denorm_local = _to_local_xyz(pred_denorm, official_frame)
        pred_exec_local = _to_local_xyz(pred_exec, official_frame)
        centroid = np.asarray(official_frame["centroid"], dtype=np.float64)
        radius = float(official_frame["radius"])
        pred_denorm_interpreted_local_to_world = pred_denorm[:3] * radius + centroid
        pred_exec_interpreted_local_to_world = pred_exec[:3] * radius + centroid
        official_local_payload = {
            "mode": "robot_3dlotus_simple_policy_xyz_frame",
            "centroid": [float(v) for v in official_frame["centroid"].tolist()],
            "radius": float(official_frame["radius"]),
            "gt_next_key_xyz_local": [float(v) for v in gt_local.tolist()],
            "pred_denormalized_xyz_local": [float(v) for v in pred_denorm_local.tolist()],
            "pred_executed_xyz_local": [float(v) for v in pred_exec_local.tolist()],
            "pred_denormalized_raw_vs_gt_local_xyz_error": float(np.linalg.norm(pred_denorm[:3] - gt_local)),
            "pred_denormalized_raw_vs_gt_local_xyz_abs_error": [
                float(v) for v in np.abs(pred_denorm[:3] - gt_local).tolist()
            ],
            "pred_executed_raw_vs_gt_local_xyz_error": float(np.linalg.norm(pred_exec[:3] - gt_local)),
            "pred_denormalized_interpreted_local_to_world_xyz": [
                float(v) for v in pred_denorm_interpreted_local_to_world.tolist()
            ],
            "pred_executed_interpreted_local_to_world_xyz": [
                float(v) for v in pred_exec_interpreted_local_to_world.tolist()
            ],
            "pred_denormalized_interpreted_local_to_world_xyz_error": float(
                np.linalg.norm(pred_denorm_interpreted_local_to_world - gt_action[:3])
            ),
            "pred_executed_interpreted_local_to_world_xyz_error": float(
                np.linalg.norm(pred_exec_interpreted_local_to_world - gt_action[:3])
            ),
            "denormalized_local_xyz_error": float(np.linalg.norm(pred_denorm_local - gt_local)),
            "denormalized_local_xyz_abs_error": [float(v) for v in np.abs(pred_denorm_local - gt_local).tolist()],
            "executed_local_xyz_error": float(np.linalg.norm(pred_exec_local - gt_local)),
            "executed_local_xyz_abs_error": [float(v) for v in np.abs(pred_exec_local - gt_local).tolist()],
            "frame_provenance": {
                key: value
                for key, value in official_frame.items()
                if key not in {"centroid"}
            },
        }
    except Exception as exc:
        official_local_error = f"{type(exc).__name__}: {exc}"

    xyz_abs = np.abs(pred_denorm[:3] - gt_action[:3])
    xyz_exec_abs = np.abs(pred_exec[:3] - gt_action[:3])
    row = {
        "taskvar": spec.taskvar,
        "task": spec.task,
        "variation": int(spec.variation),
        "episode_key": spec.episode_key,
        "demo_id": int(spec.demo_id),
        "step_id": int(step_id),
        "current_key_idx": int(current_key_idx),
        "next_key_idx": int(next_key_idx),
        "key_delta": int(next_key_idx) - int(current_key_idx),
        "instruction": pred_output.get("instruction"),
        "current_gt_action": _action_list(current_gt_action),
        "gt_next_key_action": _action_list(gt_action),
        "pred_denormalized_action": _action_list(pred_denorm),
        "pred_executed_action": _action_list(pred_exec),
        "pred_normalized_action": _action_list(pred_norm),
        "xyz_error": float(np.linalg.norm(pred_denorm[:3] - gt_action[:3])),
        "xyz_abs_error": [float(v) for v in xyz_abs.tolist()],
        "executed_xyz_error": float(np.linalg.norm(pred_exec[:3] - gt_action[:3])),
        "executed_xyz_abs_error": [float(v) for v in xyz_exec_abs.tolist()],
        "official_local": official_local_payload,
        "official_local_error": official_local_error,
        "quat_angle_deg": _quat_angle_deg(pred_denorm[3:7], gt_action[3:7]),
        "pred_gripper_bin": _gripper_bin(pred_denorm[7]),
        "gt_gripper_bin": _gripper_bin(gt_action[7]),
        "gripper_mismatch": bool(_gripper_bin(pred_denorm[7]) != _gripper_bin(gt_action[7])),
        "pred_normalized_abs_max": float(np.max(np.abs(pred_norm))),
        "pred_normalized_outside_abs1": bool(np.any(np.abs(pred_norm) > 1.0)),
        "pred_normalized_outside_abs2": bool(np.any(np.abs(pred_norm) > 2.0)),
        "prefix_reward": float(prefix_reward),
        "prefix_terminate": bool(prefix_terminate),
        "prefix_error": prefix_error,
        "model_predict_error": pred_error,
        "pred_mover_reward": float(pred_reward),
        "pred_mover_terminate": bool(pred_terminate),
        "pred_mover_error": pred_mover_error,
        "pred_mover_failure_class": _failure_class(pred_mover_error),
        "chunk_horizon": int(pred_output.get("chunk_horizon", 1)),
        "chunk_action_horizon": int(pred_output.get("chunk_action_horizon", 1)),
        "policy_action_horizon": int(pred_output.get("policy_action_horizon", 1)),
        "training_action_horizon": int(pred_output.get("training_action_horizon", 1)),
        "policy_vgm_auxiliary_action_horizon": pred_output.get("policy_vgm_auxiliary_action_horizon"),
        "normalization": pred_output.get("normalization"),
        "postprocess": pred_output.get("postprocess"),
    }
    return row


def _run_task_group(
    *,
    args: argparse.Namespace,
    actioner: GEMBenchOfficialActioner,
    taskvar: str,
    specs: list[ReplaySpec],
    modules: dict[str, Any],
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
        apply_pc=True,
        apply_mask=True,
        headless=True,
        image_size=[int(args.image_size), int(args.image_size)],
        cam_rand_factor=0,
        apply_cameras=list(OFFICIAL_CAMERA_NAMES),
    )
    rows: list[dict[str, Any]] = []
    try:
        print(f"[keystep-action-diagnosis] env_launch taskvar={taskvar} data={microstep_data_dir}", flush=True)
        env.env.launch()
        task_type = task_file_to_task_class(first.task)
        task = env.env.get_task(task_type)
        task.set_variation(first.variation)
        move = Mover(task, max_tries=int(args.max_tries))

        for spec in specs:
            demo = env.get_demo(spec.task, spec.variation, spec.demo_id, load_images=False)
            demo_obs = list(getattr(demo, "_observations", demo))
            if len(demo_obs) != int(spec.row["length"]):
                raise ValueError(f"demo length changed: manifest={spec.row['length']} actual={len(demo_obs)}")
            transition_pairs = list(zip(spec.key_frameids[:-1], spec.key_frameids[1:]))
            transition_pairs = transition_pairs[: int(args.max_key_transitions)]
            for step_id, (current_key_idx, next_key_idx) in enumerate(transition_pairs):
                row = _predict_and_probe_transition(
                    args=args,
                    actioner=actioner,
                    env=env,
                    task=task,
                    move=move,
                    official_exceptions=official_exceptions,
                    spec=spec,
                    demo=demo,
                    demo_obs=demo_obs,
                    transition_pairs=transition_pairs,
                    step_id=step_id,
                    current_key_idx=int(current_key_idx),
                    next_key_idx=int(next_key_idx),
                )
                rows.append(row)
                print(
                    "[keystep-action-diagnosis] "
                    f"taskvar={spec.taskvar} demo={spec.demo_id} step={step_id} "
                    f"xyz_error={row.get('xyz_error')} gripper_mismatch={row.get('gripper_mismatch')} "
                    f"pred_error={row.get('pred_mover_error')}",
                    flush=True,
                )
    finally:
        try:
            env.env.shutdown()
        except Exception:
            pass
    return rows


def _annotate_distribution(rows: list[dict[str, Any]], action_stats: dict[str, Any] | None) -> dict[str, Any]:
    valid_rows = [row for row in rows if row.get("pred_denormalized_action") is not None]
    if not valid_rows:
        return {"num_valid_action_rows": 0}

    gt = np.asarray([row["gt_next_key_action"] for row in valid_rows], dtype=np.float64)
    pred = np.asarray([row["pred_denormalized_action"] for row in valid_rows], dtype=np.float64)
    selected_low = gt.min(axis=0)
    selected_high = gt.max(axis=0)

    for row in valid_rows:
        arr = np.asarray(row["pred_denormalized_action"], dtype=np.float64)
        flag, dims = _bounds_flags(arr, selected_low, selected_high, margin=float(0.02))
        row["outside_selected_gt_minmax_margin002"] = flag
        row["outside_selected_gt_dims_margin002"] = dims
        if action_stats is not None and not action_stats.get("looks_like_dummy_0_1"):
            flag, dims = _bounds_flags(arr, action_stats["global_min"], action_stats["global_max"])
            row["outside_train_stats_minmax"] = flag
            row["outside_train_stats_dims_minmax"] = dims
            if "global_q01" in action_stats and "global_q99" in action_stats:
                flag, dims = _bounds_flags(arr, action_stats["global_q01"], action_stats["global_q99"])
                row["outside_train_stats_q01q99"] = flag
                row["outside_train_stats_dims_q01q99"] = dims
        else:
            row["outside_train_stats_minmax"] = None
            row["outside_train_stats_dims_minmax"] = []
            row["outside_train_stats_q01q99"] = None
            row["outside_train_stats_dims_q01q99"] = []

    summary = {
        "num_valid_action_rows": len(valid_rows),
        "selected_gt_min": _action_list(selected_low),
        "selected_gt_max": _action_list(selected_high),
        "pred_min": _action_list(pred.min(axis=0)),
        "pred_max": _action_list(pred.max(axis=0)),
        "train_stats_available": bool(action_stats is not None and not action_stats.get("looks_like_dummy_0_1")),
        "train_stats_note": (
            "stats_unavailable_or_dummy_0_1"
            if action_stats is None or action_stats.get("looks_like_dummy_0_1")
            else "raw_action_stats_used"
        ),
    }
    return summary


def _mean_metric(rows: list[dict[str, Any]], key: str) -> float | None:
    values = [float(row[key]) for row in rows if row.get(key) is not None]
    return float(np.mean(values)) if values else None


def _median_metric(rows: list[dict[str, Any]], key: str) -> float | None:
    values = [float(row[key]) for row in rows if row.get(key) is not None]
    return float(np.median(values)) if values else None


def _max_metric(rows: list[dict[str, Any]], key: str) -> float | None:
    values = [float(row[key]) for row in rows if row.get(key) is not None]
    return float(np.max(values)) if values else None


def _percentile_metric(rows: list[dict[str, Any]], key: str, percentile: float) -> float | None:
    values = [float(row[key]) for row in rows if row.get(key) is not None]
    return float(np.percentile(values, float(percentile))) if values else None


def _official_local_summary(valid_rows: list[dict[str, Any]]) -> dict[str, Any]:
    rows = [row for row in valid_rows if isinstance(row.get("official_local"), dict)]
    if not rows:
        return {
            "num_rows": 0,
            "num_errors": len([row for row in valid_rows if row.get("official_local_error")]),
        }

    def collect(name: str) -> list[float]:
        return [float(row["official_local"][name]) for row in rows if row["official_local"].get(name) is not None]

    out: dict[str, Any] = {
        "num_rows": len(rows),
        "num_errors": len([row for row in valid_rows if row.get("official_local_error")]),
    }
    metric_names = (
        "denormalized_local_xyz_error",
        "executed_local_xyz_error",
        "pred_denormalized_raw_vs_gt_local_xyz_error",
        "pred_executed_raw_vs_gt_local_xyz_error",
        "pred_denormalized_interpreted_local_to_world_xyz_error",
        "pred_executed_interpreted_local_to_world_xyz_error",
    )
    for metric_name in metric_names:
        values = collect(metric_name)
        out[f"mean_{metric_name}"] = float(np.mean(values)) if values else None
        out[f"median_{metric_name}"] = float(np.median(values)) if values else None
        out[f"max_{metric_name}"] = float(np.max(values)) if values else None

    centroids = np.asarray([row["official_local"]["centroid"] for row in rows], dtype=np.float64)
    radii = np.asarray([row["official_local"]["radius"] for row in rows], dtype=np.float64)
    out["centroid_mean"] = [float(v) for v in centroids.mean(axis=0).tolist()]
    out["centroid_min"] = [float(v) for v in centroids.min(axis=0).tolist()]
    out["centroid_max"] = [float(v) for v in centroids.max(axis=0).tolist()]
    out["radius_mean"] = float(radii.mean())
    out["radius_min"] = float(radii.min())
    out["radius_max"] = float(radii.max())
    return out


def _gate_summary(args: argparse.Namespace, summary: dict[str, Any]) -> dict[str, Any]:
    checks = {
        "all_predictions_valid": {
            "value": int(summary["num_valid_action_rows"]),
            "threshold": int(summary["num_rows"]),
            "passed": int(summary["num_valid_action_rows"]) == int(summary["num_rows"]),
        },
        "mean_executed_xyz_error_m": {
            "value": summary.get("mean_executed_xyz_error"),
            "threshold": float(args.gate_max_mean_xyz_error),
            "passed": summary.get("mean_executed_xyz_error") is not None
            and float(summary["mean_executed_xyz_error"]) <= float(args.gate_max_mean_xyz_error),
        },
        "median_executed_xyz_error_m": {
            "value": summary.get("median_executed_xyz_error"),
            "threshold": float(args.gate_max_median_xyz_error),
            "passed": summary.get("median_executed_xyz_error") is not None
            and float(summary["median_executed_xyz_error"]) <= float(args.gate_max_median_xyz_error),
        },
        "p90_executed_xyz_error_m": {
            "value": summary.get("p90_executed_xyz_error"),
            "threshold": float(args.gate_max_p90_xyz_error),
            "passed": summary.get("p90_executed_xyz_error") is not None
            and float(summary["p90_executed_xyz_error"]) <= float(args.gate_max_p90_xyz_error),
        },
        "max_executed_xyz_error_m": {
            "value": summary.get("max_executed_xyz_error"),
            "threshold": float(args.gate_max_max_xyz_error),
            "passed": summary.get("max_executed_xyz_error") is not None
            and float(summary["max_executed_xyz_error"]) <= float(args.gate_max_max_xyz_error),
        },
        "pred_mover_error_rate": {
            "value": summary.get("pred_mover_error_rate"),
            "threshold": float(args.gate_max_mover_error_rate),
            "passed": summary.get("pred_mover_error_rate") is not None
            and float(summary["pred_mover_error_rate"]) <= float(args.gate_max_mover_error_rate),
        },
        "gripper_mismatch_rate": {
            "value": summary.get("gripper_mismatch_rate"),
            "threshold": float(args.gate_max_gripper_mismatch_rate),
            "passed": summary.get("gripper_mismatch_rate") is not None
            and float(summary["gripper_mismatch_rate"]) <= float(args.gate_max_gripper_mismatch_rate),
        },
        "median_quat_angle_deg": {
            "value": summary.get("median_quat_angle_deg"),
            "threshold": float(args.gate_max_median_quat_angle_deg),
            "passed": summary.get("median_quat_angle_deg") is not None
            and float(summary["median_quat_angle_deg"]) <= float(args.gate_max_median_quat_angle_deg),
        },
    }
    failed = [name for name, item in checks.items() if not bool(item["passed"])]
    return {
        "gate_name": "pre_full_train_gt_keystate_action_precision",
        "note": (
            "This is a diagnostic gate for deciding whether a checkpoint is precise enough "
            "to justify full GEMBench policy training/evaluation. It is not an official score."
        ),
        "passed": not failed,
        "failed_checks": failed,
        "checks": checks,
    }


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Compare FastWAM predicted actions against GEMBench key-step training targets "
            "on the same GT key states."
        )
    )
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--manifest", default="/mnt/yuhan/datasets/GEMBench/fastwam_cache/microsteps_9v32_4cam224_manifest.json")
    parser.add_argument("--key-frameids-path", default="/mnt/yuhan/datasets/GEMBench/fastwam_cache/microsteps_9v32_seed0_key_frameids.json")
    parser.add_argument("--gembench-root", default="/mnt/yuhan/datasets/GEMBench")
    parser.add_argument("--robot-3dlotus-root", default=None)
    parser.add_argument("--seed", default="seed0")
    parser.add_argument("--taskvars", default=None)
    parser.add_argument("--episodes", default=None)
    parser.add_argument("--episodes-per-taskvar", type=int, default=1)
    parser.add_argument("--max-trials", type=int, default=4)
    parser.add_argument("--max-key-transitions", type=int, default=8)
    parser.add_argument("--max-tries", type=int, default=10)
    parser.add_argument("--image-size", type=int, default=256)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--mixed-precision", default="bf16", choices=("no", "fp16", "bf16"))
    parser.add_argument("--num-inference-steps", type=int, default=10)
    parser.add_argument("--rand-device", default="cpu")
    parser.add_argument("--model-seed", type=int, default=-1)
    parser.add_argument("--tiled", action="store_true")
    parser.add_argument("--probe-predicted-mover", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--action-stats-json", default=None)
    parser.add_argument("--official-local-xyz-shift", default="center", choices=("none", "center", "gripper"))
    parser.add_argument("--official-local-xyz-norm", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--official-local-rm-table", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--official-local-voxel-size", type=float, default=0.01)
    parser.add_argument("--official-local-num-points", type=int, default=4096)
    parser.add_argument("--official-local-sample-seed", type=int, default=0)
    parser.add_argument("--gate-max-mean-xyz-error", type=float, default=0.05)
    parser.add_argument("--gate-max-median-xyz-error", type=float, default=0.03)
    parser.add_argument("--gate-max-p90-xyz-error", type=float, default=0.10)
    parser.add_argument("--gate-max-max-xyz-error", type=float, default=0.20)
    parser.add_argument("--gate-max-mover-error-rate", type=float, default=0.0)
    parser.add_argument("--gate-max-gripper-mismatch-rate", type=float, default=0.05)
    parser.add_argument("--gate-max-median-quat-angle-deg", type=float, default=30.0)
    parser.add_argument("--fail-on-gate", action="store_true")
    parser.add_argument("--output-root", default=None)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    args._official_workspace = _official_workspace(args.robot_3dlotus_root)

    specs = _select_specs(args)
    output_root = Path(
        args.output_root
        or f"runs/gembench_policy_keystep_action_diagnostics/diagnosis_{time.strftime('%Y%m%d_%H%M%S')}"
    ).expanduser()
    output_root.mkdir(parents=True, exist_ok=True)

    manifest_payload = {
        "eval_type": "gembench_policy_keystep_pred_vs_gt_action_diagnostic",
        "official_full_score": False,
        "write_official_preds": False,
        "target_source": "gt_next_key_action = demo_obs[next_key_idx].gripper_pose_plus_open",
        "generated_at": utc_now(),
        "run_dir": str(Path(args.run_dir).expanduser()),
        "checkpoint": str(Path(args.checkpoint).expanduser()),
        "manifest": str(Path(args.manifest).expanduser()),
        "key_frameids_path": str(Path(args.key_frameids_path).expanduser()),
        "gembench_root": str(Path(args.gembench_root).expanduser()),
        "robot_3dlotus_root": args.robot_3dlotus_root,
        "seed": str(args.seed),
        "max_tries": int(args.max_tries),
        "max_key_transitions": int(args.max_key_transitions),
        "probe_predicted_mover": bool(args.probe_predicted_mover),
        "official_local_frame": {
            "note": (
                "robot-3dlotus simple_policy_ptv3 defaults use xyz_shift=center, xyz_norm=False, "
                "rm_table=True, voxel_size=0.01, num_points=4096. Final Mover action is still world 8D; "
                "this frame diagnoses the internal policy target representation."
            ),
            "xyz_shift": str(args.official_local_xyz_shift),
            "xyz_norm": bool(args.official_local_xyz_norm),
            "rm_table": bool(args.official_local_rm_table),
            "voxel_size": float(args.official_local_voxel_size),
            "num_points": int(args.official_local_num_points),
            "sample_seed": int(args.official_local_sample_seed),
            "workspace": args._official_workspace,
        },
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
    write_json(output_root / "diagnostic_manifest.json", manifest_payload)
    if args.dry_run:
        print(json.dumps(manifest_payload, ensure_ascii=True, indent=2), flush=True)
        return 0

    actioner = GEMBenchOfficialActioner.from_run_dir(
        run_dir=Path(args.run_dir),
        checkpoint=Path(args.checkpoint),
        device=str(args.device),
        mixed_precision=str(args.mixed_precision),
        num_inference_steps=int(args.num_inference_steps),
        relation_mode="none",
        observation_camera_names=OFFICIAL_CAMERA_NAMES,
        rand_device=str(args.rand_device),
        model_seed=int(args.model_seed),
        tiled=bool(args.tiled),
        chunk_action_horizon=1,
        min_chunk_action_horizon=None,
    )

    modules = _official_imports(args.robot_3dlotus_root)
    grouped: dict[str, list[ReplaySpec]] = defaultdict(list)
    for spec in specs:
        grouped[spec.taskvar].append(spec)

    all_rows: list[dict[str, Any]] = []
    for taskvar, group in grouped.items():
        all_rows.extend(
            _run_task_group(
                args=args,
                actioner=actioner,
                taskvar=taskvar,
                specs=group,
                modules=modules,
            )
        )

    stats_path = args.action_stats_json or getattr(actioner, "stats_path", None)
    action_stats = _load_action_stats(str(stats_path) if stats_path is not None else None)
    distribution_summary = _annotate_distribution(all_rows, action_stats)
    _write_jsonl(output_root / "action_diagnostics.jsonl", all_rows)

    valid_rows = [row for row in all_rows if row.get("pred_denormalized_action") is not None]
    pred_errors = [row for row in valid_rows if row.get("pred_mover_error")]
    prefix_success_rows = [
        row for row in valid_rows if float(row.get("prefix_reward", 0.0)) == 1.0 or bool(row.get("prefix_terminate"))
    ]
    summary = {
        "eval_type": "gembench_policy_keystep_pred_vs_gt_action_diagnostic",
        "official_full_score": False,
        "write_official_preds": False,
        "output_root": str(output_root),
        "num_rows": len(all_rows),
        "num_valid_action_rows": len(valid_rows),
        "num_pred_mover_errors": len(pred_errors),
        "pred_mover_error_rate": float(len(pred_errors) / max(len(valid_rows), 1)),
        "mover_failure_classes": dict(Counter(str(row.get("pred_mover_failure_class", "none")) for row in valid_rows)),
        "mean_xyz_error": float(np.mean([row["xyz_error"] for row in valid_rows])) if valid_rows else None,
        "median_xyz_error": float(np.median([row["xyz_error"] for row in valid_rows])) if valid_rows else None,
        "max_xyz_error": float(np.max([row["xyz_error"] for row in valid_rows])) if valid_rows else None,
        "mean_executed_xyz_error": _mean_metric(valid_rows, "executed_xyz_error"),
        "median_executed_xyz_error": _median_metric(valid_rows, "executed_xyz_error"),
        "max_executed_xyz_error": _max_metric(valid_rows, "executed_xyz_error"),
        "p90_executed_xyz_error": _percentile_metric(valid_rows, "executed_xyz_error", 90.0),
        "official_local": _official_local_summary(valid_rows),
        "mean_quat_angle_deg": float(np.mean([row["quat_angle_deg"] for row in valid_rows])) if valid_rows else None,
        "median_quat_angle_deg": float(np.median([row["quat_angle_deg"] for row in valid_rows])) if valid_rows else None,
        "gripper_mismatches": int(sum(bool(row["gripper_mismatch"]) for row in valid_rows)),
        "gripper_mismatch_rate": float(
            sum(bool(row["gripper_mismatch"]) for row in valid_rows) / max(len(valid_rows), 1)
        ),
        "outside_selected_gt_minmax_margin002": int(
            sum(bool(row.get("outside_selected_gt_minmax_margin002")) for row in valid_rows)
        ),
        "outside_selected_gt_minmax_margin002_rate": float(
            sum(bool(row.get("outside_selected_gt_minmax_margin002")) for row in valid_rows) / max(len(valid_rows), 1)
        ),
        "prefix_already_success_rows": len(prefix_success_rows),
        "prefix_already_success_note": (
            "Rows with prefix_reward=1 or prefix_terminate=true should not be used as a success-rate estimate. "
            "They are still valid for pred-vs-GT action error because the comparison is against the next key target."
        ),
        "distribution": distribution_summary,
        "rows": all_rows,
    }
    summary["gate"] = _gate_summary(args, summary)
    write_json(output_root / "summary.json", summary)
    print(json.dumps(summary, ensure_ascii=True, indent=2), flush=True)
    if bool(args.fail_on_gate) and not bool(summary["gate"]["passed"]):
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
