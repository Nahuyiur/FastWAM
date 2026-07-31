#!/usr/bin/env python3
"""Enhanced RoboCasa online evaluator for RoboCasa-ACG.

This wraps the official RoboCasa openpi rollout logic and keeps the official
`info["success"]` metric unchanged. Additional metrics are written as separate
fields so downstream tables can use them without changing official SR.

The rollout / metric / video logic is shared across policy backends. Use
``--policy-backend websocket`` for the pi0.5 policy server path and
``--policy-backend fastwam`` for in-process FastWAM checkpoints.
"""

from __future__ import annotations

import argparse
import collections
import csv
import dataclasses
import json
import math
import os
import pathlib
import re
import statistics
import time
import traceback
from typing import Any

import imageio
import numpy as np
import tqdm

from robocasa_acg_policy_backends import (
    FastWAMPolicyClient,
    MegatronFastWAMPolicyClient,
    WebsocketPolicyClient,
    convert_to_uint8,
    preprocess_image,
)

try:
    import gymnasium as gym
except Exception:  # pragma: no cover - setup script installs this before eval
    gym = None

try:
    from robocasa.utils.dataset_registry_utils import get_task_horizon
except Exception:  # pragma: no cover - setup script installs this before eval
    get_task_horizon = None

try:
    from robocasa.utils.env_utils import detect_robot_collision
except Exception:  # pragma: no cover - robust to upstream changes
    detect_robot_collision = None


DEFAULT_PLAN = {
    "id_pretrain_online": {
        "split": "pretrain",
        "num_trials": 10,
        "tasks": [
            "OpenBlenderLid",
            "OpenCabinet",
            "OpenDishwasher",
            "OpenDrawer",
            "OpenElectricKettleLid",
            "OpenFridge",
            "OpenFridgeDrawer",
            "OpenMicrowave",
            "OpenOven",
            "OpenToasterOvenDoor",
            "CloseCabinet",
            "CloseDishwasher",
            "CloseDrawer",
            "CloseElectricKettleLid",
            "CloseFridgeDrawer",
            "CloseMicrowave",
            "CloseOven",
            "CloseStandMixerHead",
            "PickPlaceCabinetToCounter",
            "PickPlaceCounterToBlender",
            "PickPlaceCounterToDrawer",
            "PickPlaceCounterToMicrowave",
            "PickPlaceCounterToOven",
            "PickPlaceCounterToSink",
            "PickPlaceCounterToStandMixer",
            "PickPlaceCounterToToasterOven",
            "PickPlaceFridgeDrawerToShelf",
            "PickPlaceFridgeShelfToDrawer",
            "PickPlaceMicrowaveToCounter",
            "PickPlaceStoveToCounter",
            "PickPlaceToasterOvenToCounter",
            "TurnOffMicrowave",
            "TurnOffSinkFaucet",
            "TurnOnBlender",
            "TurnOnElectricKettle",
            "TurnOnSinkFaucet",
            "TurnOnStove",
            "TurnOnToaster",
            "TurnOnToasterOven",
            "SlideOvenRack",
            "SlideToasterOvenRack",
            "AdjustWaterTemperature",
            "AdjustToasterOvenTemperature",
        ],
    },
    "target_id_sanity": {
        "split": "target",
        "num_trials": 25,
        "tasks": [
            "OpenCabinet",
            "OpenDrawer",
            "TurnOnElectricKettle",
            "TurnOnSinkFaucet",
        ],
    },
    "ood_pair_strict": {
        "split": "target",
        "num_trials": 50,
        "tasks": [
            "CloseBlenderLid",
            "CloseFridge",
            "CloseToasterOvenDoor",
            "OpenStandMixerHead",
            "PickPlaceCounterToCabinet",
            "PickPlaceCounterToStove",
            "PickPlaceDrawerToCounter",
            "PickPlaceSinkToCounter",
            "TurnOffStove",
            "TurnOnMicrowave",
        ],
    },
    "ood_pair_probe": {
        "split": "target",
        "num_trials": 25,
        "tasks": [
            "PickPlaceToasterToCounter",
            "SlideDishwasherRack",
        ],
    },
}


def json_default(obj: Any) -> Any:
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, np.generic):
        return obj.item()
    if isinstance(obj, pathlib.Path):
        return str(obj)
    return str(obj)


def append_jsonl(path: pathlib.Path, row: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False, default=json_default) + "\n")


def load_plan(path: str | None) -> dict[str, dict[str, Any]]:
    if path is None:
        return DEFAULT_PLAN
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def parse_pair_key(task: str) -> dict[str, Any]:
    if task.startswith("PickPlace"):
        body = task[len("PickPlace") :]
        if "To" in body:
            src, dst = body.split("To", 1)
            return {
                "primitive": "PickPlace",
                "source_slot": src,
                "target_slot": dst,
                "pair_key": f"PickPlace:{src}->{dst}",
            }
    for primitive in ("TurnOff", "TurnOn", "Adjust", "Close", "Open", "Slide"):
        if task.startswith(primitive):
            slot = task[len(primitive) :]
            return {
                "primitive": primitive,
                "target_slot": slot,
                "pair_key": f"{primitive}:{slot}",
            }
    return {"primitive": None, "target_slot": None, "pair_key": task}


def wilson_ci(k: int, n: int, z: float = 1.96) -> tuple[float | None, float | None]:
    if n <= 0:
        return None, None
    p = k / n
    denom = 1 + z * z / n
    centre = p + z * z / (2 * n)
    margin = z * math.sqrt((p * (1 - p) + z * z / (4 * n)) / n)
    return max(0.0, (centre - margin) / denom), min(1.0, (centre + margin) / denom)


def finite_mean(values: list[float | int | None]) -> float | None:
    xs = [float(v) for v in values if v is not None and math.isfinite(float(v))]
    return float(statistics.fmean(xs)) if xs else None


def safe_bool(value: Any) -> bool | None:
    if value is None:
        return None
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    if isinstance(value, (int, float, np.integer, np.floating)):
        return bool(value)
    return None


def get_base_env(env: Any) -> Any:
    cur = env
    for attr in ("env", "unwrapped"):
        nxt = getattr(cur, attr, None)
        if nxt is not None and nxt is not cur:
            cur = nxt
    return cur


def get_obs_state(obs: dict[str, Any]) -> np.ndarray:
    # Match RoboCasa365 LeRobot v3 observation.state:
    # base_pos(3) + base_quat(4) + eef_pos_rel(3) + eef_quat_rel(4) + gripper_qpos(2).
    return np.concatenate(
        (
            np.asarray(obs["state.base_position"]),
            np.asarray(obs["state.base_rotation"]),
            np.asarray(obs["state.end_effector_position_relative"]),
            np.asarray(obs["state.end_effector_rotation_relative"]),
            np.asarray(obs["state.gripper_qpos"]),
        ),
        axis=0,
    )


def resolve_action_layout(layout: str, policy_backend: str) -> str:
    if layout != "auto":
        return layout
    # FastWAM is trained directly on RoboCasa365 LeRobot parquet actions:
    # base_motion(4), control_mode(1), eef_delta_pos(3), eef_delta_aa(3), gripper(1).
    # The websocket/pi0.5 path keeps the existing eef-first bridge because that
    # policy server was trained/exported with the OpenPI transform contract.
    return "base_first" if policy_backend in {"fastwam", "fastwam_megatron"} else "eef_first"


def convert_lerobot_action(action: np.ndarray, action_layout: str) -> dict[str, np.ndarray]:
    """Convert policy action to the gym dict expected by RoboCasa."""
    action = np.asarray(action, dtype=np.float32).copy()
    if action.shape[-1] < 12:
        raise ValueError(f"Expected at least 12 action dims, got shape={action.shape}")
    if action_layout == "eef_first":
        return {
            "action.end_effector_position": action[0:3],
            "action.end_effector_rotation": action[3:6],
            "action.gripper_close": action[6:7],
            "action.base_motion": action[7:11],
            "action.control_mode": action[11:12],
        }
    if action_layout == "base_first":
        return {
            "action.base_motion": action[0:4],
            "action.control_mode": action[4:5],
            "action.end_effector_position": action[5:8],
            "action.end_effector_rotation": action[8:11],
            "action.gripper_close": action[11:12],
        }
    raise ValueError(f"Unknown action_layout={action_layout!r}")


def get_ee_pos(obs: dict[str, Any]) -> np.ndarray | None:
    for key in (
        "state.end_effector_position",
        "state.end_effector_position_relative",
        "robot0_eef_pos",
    ):
        if key in obs:
            arr = np.asarray(obs[key], dtype=float).reshape(-1)
            if arr.size >= 3:
                return arr[:3]
    return None


def sparc_from_positions(positions: list[np.ndarray], dt: float = 1.0) -> float | None:
    if len(positions) < 8:
        return None
    xyz = np.asarray(positions, dtype=float)
    speed = np.linalg.norm(np.diff(xyz, axis=0), axis=1) / max(dt, 1e-8)
    if speed.size < 8 or float(np.max(speed)) <= 1e-8:
        return None
    speed = speed - float(np.mean(speed))
    spec = np.abs(np.fft.rfft(speed))
    if spec.size < 3 or float(np.max(spec)) <= 1e-8:
        return None
    spec = spec / float(np.max(spec))
    freq = np.fft.rfftfreq(speed.size, d=dt)
    keep = freq <= min(float(np.max(freq)), 10.0)
    freq = freq[keep]
    spec = spec[keep]
    if len(freq) < 3:
        return None
    df = np.diff(freq)
    ds = np.diff(spec)
    arc = np.sum(np.sqrt(df * df + ds * ds))
    return -float(arc)


def path_metrics(positions: list[np.ndarray]) -> dict[str, Any]:
    if len(positions) < 2:
        return {
            "ee_path_length": None,
            "ee_speed_mean": None,
            "ee_speed_max": None,
            "ee_sparc": None,
        }
    xyz = np.asarray(positions, dtype=float)
    d = np.linalg.norm(np.diff(xyz, axis=0), axis=1)
    return {
        "ee_path_length": float(np.sum(d)),
        "ee_speed_mean": float(np.mean(d)),
        "ee_speed_max": float(np.max(d)),
        "ee_sparc": sparc_from_positions(positions),
    }


def numeric_leaf_items(obj: Any, prefix: str = "") -> list[tuple[str, Any]]:
    out: list[tuple[str, Any]] = []
    if isinstance(obj, dict):
        for key, val in obj.items():
            out.extend(numeric_leaf_items(val, f"{prefix}.{key}" if prefix else str(key)))
    else:
        out.append((prefix, obj))
    return out


SAFETY_PATTERNS = (
    "collision",
    "collide",
    "dropped",
    "drop",
    "disturb",
    "bumped",
    "moved",
    "out_of_scene",
    "tipped",
    "multiple",
)
WRONG_OBJECT_PATTERNS = ("wrong_object", "wrong object", "wrong_grasp", "wrong grasp")
WRONG_TARGET_PATTERNS = ("wrong_target", "wrong target", "wrong_receptacle", "wrong receptacle")


def scan_info_events(info: dict[str, Any]) -> dict[str, Any]:
    safety_events = 0
    wrong_object = None
    wrong_target = None
    matched_keys: list[str] = []
    for key, val in numeric_leaf_items(info):
        lower = key.lower()
        b = safe_bool(val)
        if b is None:
            continue
        if any(pat in lower for pat in SAFETY_PATTERNS) and lower not in {"success", "is_success"}:
            if b:
                safety_events += 1
                matched_keys.append(key)
        if any(pat in lower for pat in WRONG_OBJECT_PATTERNS):
            wrong_object = bool(wrong_object or b)
        if any(pat in lower for pat in WRONG_TARGET_PATTERNS):
            wrong_target = bool(wrong_target or b)
    return {
        "safety_events_from_info": safety_events,
        "safety_info_keys": matched_keys,
        "wrong_object": wrong_object,
        "wrong_target": wrong_target,
        "wrong_object_available": wrong_object is not None,
        "wrong_target_available": wrong_target is not None,
    }


def detect_collision_event(env: Any) -> bool | None:
    if detect_robot_collision is None:
        return None
    try:
        return bool(detect_robot_collision(get_base_env(env)))
    except Exception:
        return None


def capture_object_positions(env: Any) -> dict[str, list[float]]:
    base = get_base_env(env)
    sim = getattr(base, "sim", None)
    objects = getattr(base, "objects", {}) or {}
    out: dict[str, list[float]] = {}
    if sim is None:
        return out
    for name, obj in objects.items():
        body_name = getattr(obj, "root_body", None)
        if body_name is None:
            continue
        try:
            pos = np.asarray(sim.data.get_body_xpos(body_name), dtype=float).reshape(-1)[:3]
        except Exception:
            continue
        out[str(name)] = [float(x) for x in pos]
    return out


def disturbance_metric(
    initial: dict[str, list[float]], final: dict[str, list[float]], threshold: float
) -> dict[str, Any]:
    if not initial or not final:
        return {"disturbance_available": False, "disturbed_objects": [], "disturbance_rate": None}
    disturbed = []
    for name, p0 in initial.items():
        if name not in final:
            continue
        dist = float(np.linalg.norm(np.asarray(final[name]) - np.asarray(p0)))
        if dist >= threshold:
            disturbed.append({"name": name, "delta": dist})
    return {
        "disturbance_available": True,
        "disturbed_objects": disturbed,
        "disturbance_rate": 1.0 if disturbed else 0.0,
    }


def write_csv(path: pathlib.Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    keys = sorted({k for row in rows for k in row.keys()})
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k) for k in keys})


def aggregate(rows: list[dict[str, Any]], group_keys: tuple[str, ...]) -> list[dict[str, Any]]:
    groups: dict[tuple[Any, ...], list[dict[str, Any]]] = collections.defaultdict(list)
    for row in rows:
        groups[tuple(row.get(k) for k in group_keys)].append(row)

    out = []
    for key, vals in sorted(groups.items(), key=lambda kv: tuple(str(x) for x in kv[0])):
        n = len(vals)
        successes = sum(1 for v in vals if v.get("success") is True)
        lo, hi = wilson_ci(successes, n)
        out_row = {k: v for k, v in zip(group_keys, key, strict=True)}
        out_row.update(
            {
                "num_episodes": n,
                "success_count": successes,
                "success_rate": successes / n if n else None,
                "success_ci95_low": lo,
                "success_ci95_high": hi,
                "timeout_rate": finite_mean([1.0 if v.get("timeout") else 0.0 for v in vals]),
                "avg_steps": finite_mean([v.get("steps") for v in vals]),
                "avg_success_steps": finite_mean(
                    [v.get("success_step") for v in vals if v.get("success") is True]
                ),
                "avg_return": finite_mean([v.get("return") for v in vals]),
                "avg_policy_inference_ms": finite_mean(
                    [v.get("timing", {}).get("policy_inference_avg_ms") for v in vals]
                ),
                "avg_env_step_ms": finite_mean([v.get("timing", {}).get("env_step_avg_ms") for v in vals]),
                "avg_video_write_ms": finite_mean(
                    [v.get("timing", {}).get("video_write_total_ms") for v in vals]
                ),
                "collision_event_rate": finite_mean(
                    [1.0 if v.get("safety", {}).get("collision_event") else 0.0 for v in vals]
                ),
                "safety_event_rate": finite_mean(
                    [1.0 if v.get("safety", {}).get("safety_event") else 0.0 for v in vals]
                ),
                "wrong_object_rate": finite_mean(
                    [
                        1.0 if v.get("object_grounding", {}).get("wrong_object") else 0.0
                        for v in vals
                        if v.get("object_grounding", {}).get("wrong_object_available")
                    ]
                ),
                "wrong_target_rate": finite_mean(
                    [
                        1.0 if v.get("object_grounding", {}).get("wrong_target") else 0.0
                        for v in vals
                        if v.get("object_grounding", {}).get("wrong_target_available")
                    ]
                ),
                "avg_ee_path_length": finite_mean([v.get("trajectory", {}).get("ee_path_length") for v in vals]),
                "avg_ee_sparc": finite_mean([v.get("trajectory", {}).get("ee_sparc") for v in vals]),
            }
        )
        out.append(out_row)
    return out


@dataclasses.dataclass
class EpisodeResult:
    eval_protocol: str
    gt_matched: bool
    gt_episode_index: int | None
    gt_window_start: int | None
    gt_video_path: str | None
    protocol_note: str
    bucket: str
    split: str
    task: str
    episode_idx: int
    seed: int
    success: bool
    success_step: int | None
    steps: int
    horizon: int
    timeout: bool
    return_value: float
    video_path: str | None
    prompt: str | None
    pair: dict[str, Any]
    timing: dict[str, Any]
    trajectory: dict[str, Any]
    safety: dict[str, Any]
    object_grounding: dict[str, Any]
    env_info_final: dict[str, Any]

    def to_row(self) -> dict[str, Any]:
        row = dataclasses.asdict(self)
        row["return"] = row.pop("return_value")
        return row


def run_episode(
    env: Any,
    client: Any,
    *,
    bucket: str,
    split: str,
    task: str,
    episode_idx: int,
    seed: int,
    horizon: int,
    resize_size: int,
    replan_steps: int,
    render_every: int,
    action_layout: str,
    save_video: bool,
    video_path: pathlib.Path | None,
    disturbance_threshold: float,
) -> EpisodeResult:
    obs, info = env.reset(seed=seed)
    task_lang = obs.get("annotation.human.task_description")
    action_plan: collections.deque[np.ndarray] = collections.deque()
    replay_images: list[np.ndarray] = []
    ee_positions: list[np.ndarray] = []
    policy_times: list[float] = []
    env_times: list[float] = []
    video_write_s = 0.0
    rewards = []
    collision_seen = False
    info_safety_events = 0
    wrong_object = None
    wrong_target = None
    initial_positions = capture_object_positions(env)
    final_info: dict[str, Any] = {}

    done = False
    success_step = None
    steps = 0
    for t in range(horizon):
        ee = get_ee_pos(obs)
        if ee is not None:
            ee_positions.append(ee)

        if not action_plan:
            img = preprocess_image(obs["video.robot0_agentview_left"], resize_size)
            wrist_img = preprocess_image(obs["video.robot0_eye_in_hand"], resize_size)
            img_right = preprocess_image(obs["video.robot0_agentview_right"], resize_size)
            element = {
                "observation/base_image": img,
                "observation/wrist_image": wrist_img,
                "observation/right_image": img_right,
                "observation/state": get_obs_state(obs),
                "prompt": task_lang,
            }
            t0 = time.perf_counter()
            action_chunk = client.infer(element)["actions"]
            policy_times.append(time.perf_counter() - t0)
            if len(action_chunk) < replan_steps:
                raise RuntimeError(
                    f"Policy returned horizon {len(action_chunk)}, but replan_steps={replan_steps}"
                )
            action_plan.extend(action_chunk[:replan_steps])

        action = convert_lerobot_action(np.asarray(action_plan.popleft()), action_layout)
        t0 = time.perf_counter()
        obs, reward, terminated, truncated, info = env.step(action)
        env_times.append(time.perf_counter() - t0)
        rewards.append(float(reward))
        final_info = dict(info)
        done = bool(info.get("success", False))
        steps = t + 1

        scanned = scan_info_events(final_info)
        info_safety_events += int(scanned["safety_events_from_info"])
        if scanned["wrong_object_available"]:
            wrong_object = bool(wrong_object or scanned["wrong_object"])
        if scanned["wrong_target_available"]:
            wrong_target = bool(wrong_target or scanned["wrong_target"])

        collision = detect_collision_event(env)
        if collision is True:
            collision_seen = True

        if save_video and video_path is not None and (t % render_every == 0 or done or t == horizon - 1):
            t_video = time.perf_counter()
            frame = convert_to_uint8(np.ascontiguousarray(env.render()))
            replay_images.append(frame)
            video_write_s += time.perf_counter() - t_video

        if done:
            success_step = steps
            break

    if save_video and video_path is not None:
        video_path.parent.mkdir(parents=True, exist_ok=True)
        if replay_images:
            t_video = time.perf_counter()
            imageio.mimwrite(video_path, [np.asarray(x) for x in replay_images], fps=20)
            video_write_s += time.perf_counter() - t_video

    final_positions = capture_object_positions(env)
    disturbance = disturbance_metric(initial_positions, final_positions, disturbance_threshold)
    safety_event = bool(collision_seen or info_safety_events > 0 or disturbance.get("disturbance_rate") == 1.0)

    timing = {
        "policy_inference_avg_ms": 1000 * finite_mean(policy_times) if policy_times else None,
        "policy_inference_count": len(policy_times),
        "env_step_avg_ms": 1000 * finite_mean(env_times) if env_times else None,
        "video_write_total_ms": 1000 * video_write_s,
    }
    safety = {
        "collision_event": collision_seen,
        "info_safety_event_count": info_safety_events,
        "safety_event": safety_event,
        **disturbance,
    }
    object_grounding = {
        "wrong_object": wrong_object,
        "wrong_object_available": wrong_object is not None,
        "wrong_target": wrong_target,
        "wrong_target_available": wrong_target is not None,
    }
    return EpisodeResult(
        eval_protocol="online_task_success_rate",
        gt_matched=False,
        gt_episode_index=None,
        gt_window_start=None,
        gt_video_path=None,
        protocol_note=(
            "Online RoboCasa rollout sampled from task/split/seed. "
            "This episode is not tied to a RoboCasa365 dataset episode; "
            "use eval_robocasa_acg_open_loop_wam_smoke.py for GT-matched video diagnostics."
        ),
        bucket=bucket,
        split=split,
        task=task,
        episode_idx=episode_idx,
        seed=seed,
        success=bool(done),
        success_step=success_step,
        steps=steps,
        horizon=horizon,
        timeout=not bool(done),
        return_value=float(np.sum(rewards)) if rewards else 0.0,
        video_path=str(video_path) if save_video and video_path is not None else None,
        prompt=task_lang,
        pair=parse_pair_key(task),
        timing=timing,
        trajectory=path_metrics(ee_positions),
        safety=safety,
        object_grounding=object_grounding,
        env_info_final=final_info,
    )


def save_summaries(output_dir: pathlib.Path, rows: list[dict[str, Any]], config: dict[str, Any]) -> None:
    by_bucket = aggregate(rows, ("bucket",))
    by_task = aggregate(rows, ("bucket", "task"))
    by_cell = aggregate(
        [
            {
                **r,
                "pair_key": (r.get("pair") or {}).get("pair_key"),
                "primitive": (r.get("pair") or {}).get("primitive"),
            }
            for r in rows
        ],
        ("bucket", "primitive", "pair_key"),
    )

    id_pretrain = next((r for r in by_bucket if r["bucket"] == "id_pretrain_online"), None)
    target_id = next((r for r in by_bucket if r["bucket"] == "target_id_sanity"), None)
    for row in by_bucket:
        if id_pretrain and row["bucket"] != "id_pretrain_online":
            row["generalization_gap_vs_id_pretrain_online"] = (
                id_pretrain["success_rate"] - row["success_rate"]
                if id_pretrain["success_rate"] is not None and row["success_rate"] is not None
                else None
            )
        else:
            row["generalization_gap_vs_id_pretrain_online"] = None
        if target_id and row["bucket"] != "target_id_sanity":
            row["generalization_gap_vs_target_id_sanity"] = (
                target_id["success_rate"] - row["success_rate"]
                if target_id["success_rate"] is not None and row["success_rate"] is not None
                else None
            )
        else:
            row["generalization_gap_vs_target_id_sanity"] = None

    summary = {
        "config": config,
        "by_bucket": by_bucket,
        "by_task": by_task,
        "by_cell": by_cell,
        "metric_notes": {
            "official_success": "Unchanged RoboCasa info['success'] per episode.",
            "wrong_object_rate": "Only aggregated over episodes where RoboCasa info exposes wrong-object style keys; otherwise null.",
            "wrong_target_rate": "Only aggregated over episodes where RoboCasa info exposes wrong-target style keys; otherwise null.",
            "safety_event_rate": "Union of available robot collision, info safety keys, and coarse object displacement diagnostics.",
            "progress_score": "Not emitted by default because RoboCasa atomic tasks do not expose a uniform subpredicate-progress API.",
        },
    }
    (output_dir / "summary_metrics.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False, default=json_default),
        encoding="utf-8",
    )
    write_csv(output_dir / "summary_by_bucket.csv", by_bucket)
    write_csv(output_dir / "per_task_metrics.csv", by_task)
    write_csv(output_dir / "cell_metrics.csv", by_cell)


def write_video_index(output_dir: pathlib.Path, rows: list[dict[str, Any]]) -> None:
    items = [r for r in rows if r.get("video_path")]
    html = [
        "<html><head><meta charset='utf-8'><title>RoboCasa ACG Videos</title>",
        "<style>body{font-family:sans-serif} video{width:360px} .card{margin:12px 0}</style>",
        "</head><body><h1>RoboCasa ACG eval videos</h1>",
    ]
    for r in items:
        rel = os.path.relpath(r["video_path"], output_dir)
        html.append("<div class='card'>")
        html.append(
            f"<h3>{r['bucket']} / {r['task']} / ep {r['episode_idx']} / success={r['success']}</h3>"
        )
        html.append(f"<video controls src='{rel}'></video>")
        html.append("</div>")
    html.append("</body></html>")
    (output_dir / "video_index.html").write_text("\n".join(html), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--plan", default=None)
    parser.add_argument("--bucket", action="append", default=None)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=18080)
    parser.add_argument(
        "--policy-backend",
        choices=["websocket", "fastwam", "fastwam_megatron"],
        default="websocket",
    )
    parser.add_argument("--action-layout", choices=["auto", "eef_first", "base_first"], default="auto")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--num-trials", type=int, default=5)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--max-horizon", type=int, default=None)
    parser.add_argument("--resize-size", type=int, default=224)
    parser.add_argument("--replan-steps", type=int, default=5)
    parser.add_argument("--render-every", type=int, default=2)
    parser.add_argument("--video-policy", choices=["all", "success_failure_sample", "none"], default="all")
    parser.add_argument("--disturbance-threshold", type=float, default=0.05)
    parser.add_argument("--fastwam-repo", default="/mnt/yuhan/FastWAM_robocasa_acg_8gpu")
    parser.add_argument("--fastwam-task-config", default="robocasa_acg_v1_fastwam_8gpu")
    parser.add_argument("--fastwam-checkpoint", default=None)
    parser.add_argument(
        "--fastwam-norm-stats",
        default="/mnt/yuhan/experiments/robocasa_acg_v1/fastwam/cache/norm_stats/robocasa_acg_v1_train_id_dataset_stats.json",
    )
    parser.add_argument(
        "--fastwam-text-cache",
        default="/mnt/yuhan/experiments/robocasa_acg_v1/fastwam/cache/text_embeds/robocasa_acg_v1",
    )
    parser.add_argument("--fastwam-device", default="cuda:0")
    parser.add_argument("--fastwam-mixed-precision", default="bf16")
    parser.add_argument("--fastwam-num-video-frames", type=int, default=9)
    parser.add_argument("--fastwam-action-horizon", type=int, default=32)
    parser.add_argument("--fastwam-num-inference-steps", type=int, default=10)
    parser.add_argument("--fastwam-rand-device", default="cpu")
    parser.add_argument("--fastwam-vae-checkpoint", default=None)
    parser.add_argument("--fastwam-action-dim", type=int, default=12)
    parser.add_argument("--fastwam-proprio-dim", type=int, default=16)
    args = parser.parse_args()

    if gym is None or get_task_horizon is None:
        raise ImportError(
            "RoboCasa online eval dependencies are missing. Run "
            "`bash scripts/setup_robocasa_eval_env.sh` in the FastWAM repo first."
        )

    output_dir = pathlib.Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    errors_path = output_dir / "errors.jsonl"
    episode_path = output_dir / "episode_results.jsonl"
    eval_manifest_path = output_dir / "eval_manifest.csv"
    plan = load_plan(args.plan)
    buckets = args.bucket or list(plan.keys())
    resolved_action_layout = resolve_action_layout(args.action_layout, args.policy_backend)
    run_config = vars(args) | {
        "eval_protocol": "online_task_success_rate",
        "gt_matched": False,
        "protocol_note": (
            "This evaluator measures closed-loop online RoboCasa success rate. "
            "It does not have a one-to-one RoboCasa365 GT demo for each rollout."
        ),
        "buckets": buckets,
        "plan": {k: plan[k] for k in buckets},
        "resolved_action_layout": resolved_action_layout,
    }
    (output_dir / "eval_config.json").write_text(
        json.dumps(run_config, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    if args.policy_backend == "websocket":
        client = WebsocketPolicyClient(args.host, args.port)
    elif args.policy_backend == "fastwam":
        if not args.fastwam_checkpoint:
            raise ValueError("--fastwam-checkpoint is required for --policy-backend fastwam")
        client = FastWAMPolicyClient(
            repo=args.fastwam_repo,
            task_config=args.fastwam_task_config,
            checkpoint=args.fastwam_checkpoint,
            norm_stats=args.fastwam_norm_stats,
            text_cache=args.fastwam_text_cache,
            device=args.fastwam_device,
            mixed_precision=args.fastwam_mixed_precision,
            num_video_frames=args.fastwam_num_video_frames,
            action_horizon=args.fastwam_action_horizon,
            num_inference_steps=args.fastwam_num_inference_steps,
            seed=args.seed,
            rand_device=args.fastwam_rand_device,
        )
    else:
        if not args.fastwam_checkpoint or not args.fastwam_vae_checkpoint:
            raise ValueError(
                "--fastwam-checkpoint and --fastwam-vae-checkpoint are required "
                "for --policy-backend fastwam_megatron"
            )
        client = MegatronFastWAMPolicyClient(
            repo=args.fastwam_repo,
            checkpoint=args.fastwam_checkpoint,
            vae_checkpoint=args.fastwam_vae_checkpoint,
            norm_stats=args.fastwam_norm_stats,
            text_cache=args.fastwam_text_cache,
            device=args.fastwam_device,
            mixed_precision=args.fastwam_mixed_precision,
            action_dim=args.fastwam_action_dim,
            proprio_dim=args.fastwam_proprio_dim,
            action_horizon=args.fastwam_action_horizon,
            num_inference_steps=args.fastwam_num_inference_steps,
            seed=args.seed,
        )
    rows: list[dict[str, Any]] = []
    manifest_rows: list[dict[str, Any]] = []

    for bucket in buckets:
        if bucket not in plan:
            raise KeyError(f"Unknown bucket {bucket}; available={sorted(plan)}")
        split = plan[bucket]["split"]
        tasks = plan[bucket]["tasks"]
        num_trials = int(plan[bucket].get("num_trials", args.num_trials))
        for task in tasks:
            pair = parse_pair_key(task)
            manifest_rows.append(
                {
                    "eval_protocol": "online_task_success_rate",
                    "gt_matched": False,
                    "bucket": bucket,
                    "split": split,
                    "task": task,
                    **pair,
                }
            )
            horizon = int(get_task_horizon(task))
            if args.max_horizon is not None:
                horizon = min(horizon, int(args.max_horizon))
            env = None
            try:
                env = gym.make(f"robocasa/{task}", split=split, seed=args.seed)
                for episode_idx in tqdm.tqdm(
                    range(num_trials), desc=f"{bucket}/{task}", dynamic_ncols=True
                ):
                    seed = args.seed + episode_idx
                    suffix = "unknown"
                    video_path = output_dir / "videos" / bucket / task / f"rollout_{episode_idx:04d}_{seed}.mp4"
                    result = run_episode(
                        env,
                        client,
                        bucket=bucket,
                        split=split,
                        task=task,
                        episode_idx=episode_idx,
                        seed=seed,
                        horizon=horizon,
                        resize_size=args.resize_size,
                        replan_steps=args.replan_steps,
                        render_every=args.render_every,
                        action_layout=resolved_action_layout,
                        save_video=args.video_policy != "none",
                        video_path=video_path,
                        disturbance_threshold=args.disturbance_threshold,
                    )
                    suffix = "success" if result.success else "failure"
                    if result.video_path:
                        old_path = pathlib.Path(result.video_path)
                        new_path = old_path.with_name(old_path.stem + f"_{suffix}.mp4")
                        if old_path.exists():
                            old_path.rename(new_path)
                        result.video_path = str(new_path)
                    row = result.to_row()
                    rows.append(row)
                    append_jsonl(episode_path, row)
                    save_summaries(output_dir, rows, run_config)
                    write_video_index(output_dir, rows)
            except Exception as exc:
                append_jsonl(
                    errors_path,
                    {
                        "bucket": bucket,
                        "split": split,
                        "task": task,
                        "error": repr(exc),
                        "traceback": traceback.format_exc(),
                    },
                )
            finally:
                if env is not None:
                    try:
                        env.close()
                    except Exception:
                        pass

    write_csv(eval_manifest_path, manifest_rows)
    save_summaries(output_dir, rows, run_config)
    write_video_index(output_dir, rows)
    close_client = getattr(client, "close", None)
    if callable(close_client):
        close_client()


if __name__ == "__main__":
    main()
