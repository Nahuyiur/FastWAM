#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import sys
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if SRC_ROOT.is_dir() and str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from fastwam.datasets.gembench.lmdb_reader import LMDBEpisodeStore


DEFAULT_PCD_DATA_DIR = "/mnt/yuhan/datasets/GEMBench/train_dataset/keysteps_bbox_pcd/seed0/voxel1cm"
DEFAULT_FASTWAM_9V32_MANIFEST = "/mnt/yuhan/datasets/GEMBench/fastwam_cache/microsteps_9v32_4cam224_manifest.json"
DEFAULT_TASKVAR_FILE = "/mnt/yuhan/gembench_sim/robot-3dlotus/assets/taskvars_train.json"
DEFAULT_ROBOT_3DLOTUS_ROOT = "/mnt/yuhan/gembench_sim/robot-3dlotus"


def _add_robot_3dlotus_path(root: str | None) -> None:
    if not root:
        return
    path = Path(root).expanduser().resolve()
    if path.exists() and str(path) not in sys.path:
        sys.path.insert(0, str(path))


def _official_workspace(robot_3dlotus_root: str | None) -> dict[str, Any]:
    _add_robot_3dlotus_path(robot_3dlotus_root)
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


def _load_taskvars(path: str | None) -> list[str] | None:
    if path in (None, "", "none", "null"):
        return None
    taskvar_path = Path(str(path)).expanduser().resolve()
    payload = json.loads(taskvar_path.read_text(encoding="utf-8"))
    if not isinstance(payload, list):
        raise ValueError(f"Expected taskvar file to contain a list: {taskvar_path}")
    return [str(item) for item in payload]


def _load_manifest(path: str | None) -> dict[str, Any] | None:
    if path in (None, "", "none", "null"):
        return None
    manifest_path = Path(str(path)).expanduser().resolve()
    return json.loads(manifest_path.read_text(encoding="utf-8"))


def _episode_sort_key(key: bytes) -> tuple[int, str]:
    text = key.decode("ascii", errors="ignore")
    if text.startswith("episode") and text[len("episode") :].isdigit():
        return int(text[len("episode") :]), text
    return 10**12, text


def _list_lmdb_taskvars(data_dir: Path) -> list[str]:
    if not data_dir.is_dir():
        raise FileNotFoundError(f"PCD LMDB directory not found: {data_dir}")
    return [
        path.name
        for path in sorted(data_dir.iterdir())
        if path.is_dir() and (path / "data.mdb").is_file()
    ]


def _manifest_demo_sets(manifest: dict[str, Any] | None) -> tuple[set[str], set[tuple[str, str]]]:
    if not manifest:
        return set(), set()
    demos = manifest.get("demos") or []
    taskvars: set[str] = set()
    demo_keys: set[tuple[str, str]] = set()
    for row in demos:
        taskvar = str(row.get("taskvar"))
        episode_key = str(row.get("episode_key"))
        taskvars.add(taskvar)
        demo_keys.add((taskvar, episode_key))
    return taskvars, demo_keys


def _to_array_list(value: Any) -> list[np.ndarray]:
    if isinstance(value, list):
        return [np.asarray(item) for item in value]
    arr = np.asarray(value)
    if arr.dtype == object:
        return [np.asarray(item) for item in arr.tolist()]
    return [np.asarray(item) for item in arr]


def _robot_keep_mask(
    *,
    xyz: np.ndarray,
    bbox_info: dict[str, Any],
    pose_info: dict[str, Any],
    step: int,
    rm_robot: str,
    robot_3dlotus_root: str | None,
) -> tuple[np.ndarray, str]:
    if rm_robot == "none":
        return np.ones((xyz.shape[0],), dtype=bool), "disabled"
    _add_robot_3dlotus_path(robot_3dlotus_root)
    try:
        from genrobo3d.utils.robot_box import RobotBox

        arm_links_info = (
            {key: np.asarray(value)[int(step)] for key, value in bbox_info.items()},
            {key: np.asarray(value)[int(step)] for key, value in pose_info.items()},
        )
        robot_box = RobotBox(
            arm_links_info,
            keep_gripper=(rm_robot == "box_keep_gripper"),
            env_name="rlbench",
        )
        _, robot_point_ids = robot_box.get_pc_overlap_ratio(xyz=xyz, return_indices=True)
        mask = np.ones((xyz.shape[0],), dtype=bool)
        ids = np.asarray(list(robot_point_ids), dtype=np.int64)
        if ids.size:
            mask[ids] = False
        return mask, "robot_3dlotus.RobotBox"
    except Exception as exc:
        raise RuntimeError(f"Could not apply rm_robot={rm_robot}: {type(exc).__name__}: {exc}") from exc


def _sample_points(xyz: np.ndarray, rgb: np.ndarray, *, num_points: int, seed: int) -> tuple[np.ndarray, np.ndarray, bool]:
    if int(num_points) <= 0 or xyz.shape[0] <= int(num_points):
        return xyz, rgb, False
    rng = np.random.default_rng(int(seed))
    idx = rng.choice(xyz.shape[0], int(num_points), replace=False)
    return xyz[idx], rgb[idx], True


def _compute_policy_frame(
    *,
    episode: dict[str, Any],
    step: int,
    workspace: dict[str, Any],
    robot_3dlotus_root: str | None,
    xyz_shift: str,
    xyz_norm: bool,
    rm_table: bool,
    rm_robot: str,
    num_points: int,
    sample_seed: int,
) -> dict[str, Any]:
    xyz_seq = _to_array_list(episode["xyz"])
    rgb_seq = _to_array_list(episode["rgb"])
    action = np.asarray(episode["action"], dtype=np.float64)
    xyz = np.asarray(xyz_seq[int(step)], dtype=np.float64)
    rgb = np.asarray(rgb_seq[int(step)])
    if xyz.ndim != 2 or xyz.shape[-1] != 3:
        raise ValueError(f"Expected xyz step array [N,3], got {xyz.shape}")
    if rgb.ndim != 2 or rgb.shape[0] != xyz.shape[0]:
        raise ValueError(f"Expected rgb step array [N,C] aligned with xyz, got xyz={xyz.shape} rgb={rgb.shape}")

    before = int(xyz.shape[0])
    if bool(rm_table):
        keep = xyz[:, 2] > float(workspace["TABLE_HEIGHT"])
        xyz = xyz[keep]
        rgb = rgb[keep]
    after_table = int(xyz.shape[0])
    if after_table <= 0:
        raise ValueError("No points remain after rm_table.")

    robot_note = "disabled"
    if rm_robot != "none":
        keep, robot_note = _robot_keep_mask(
            xyz=xyz,
            bbox_info=episode["bbox_info"],
            pose_info=episode["pose_info"],
            step=int(step),
            rm_robot=rm_robot,
            robot_3dlotus_root=robot_3dlotus_root,
        )
        xyz = xyz[keep]
        rgb = rgb[keep]
    after_robot = int(xyz.shape[0])
    if after_robot <= 0:
        raise ValueError("No points remain after rm_robot.")

    xyz, rgb, sampled = _sample_points(
        xyz,
        rgb,
        num_points=int(num_points),
        seed=int(sample_seed) + int(step) * 1009,
    )
    ee_pose = action[int(step)].copy()
    gt_action_world = action[int(step) + 1].copy()
    if xyz_shift == "none":
        centroid = np.zeros((3,), dtype=np.float64)
    elif xyz_shift == "center":
        centroid = np.mean(xyz, axis=0)
    elif xyz_shift == "gripper":
        centroid = ee_pose[:3].copy()
    else:
        raise ValueError(f"Unsupported xyz_shift={xyz_shift!r}")

    if bool(xyz_norm):
        radius = float(np.max(np.sqrt(np.sum((xyz - centroid) ** 2, axis=1))))
        if not np.isfinite(radius) or radius <= 1.0e-8:
            radius = 1.0
    else:
        radius = 1.0

    gt_local = (gt_action_world[:3] - centroid) / radius
    ee_local = (ee_pose[:3] - centroid) / radius
    recon = gt_local * radius + centroid
    local_dist = float(np.linalg.norm(gt_local))
    world_step_dist = float(np.linalg.norm(gt_action_world[:3] - ee_pose[:3]))
    return {
        "step": int(step),
        "points_before_filter": before,
        "points_after_table": after_table,
        "points_after_robot": after_robot,
        "points_used": int(xyz.shape[0]),
        "sampled_to_num_points": bool(sampled),
        "robot_filter": robot_note,
        "centroid": [float(v) for v in centroid.tolist()],
        "radius": float(radius),
        "ee_xyz_world": [float(v) for v in ee_pose[:3].tolist()],
        "gt_next_xyz_world": [float(v) for v in gt_action_world[:3].tolist()],
        "ee_xyz_local": [float(v) for v in ee_local.tolist()],
        "gt_next_xyz_local": [float(v) for v in gt_local.tolist()],
        "local_target_l2": local_dist,
        "world_current_to_next_l2": world_step_dist,
        "roundtrip_max_abs_error": float(np.max(np.abs(recon - gt_action_world[:3]))),
    }


def _percentiles(values: list[float]) -> dict[str, float | None]:
    if not values:
        return {"min": None, "p50": None, "p90": None, "p95": None, "max": None}
    arr = np.asarray(values, dtype=np.float64)
    return {
        "min": float(np.min(arr)),
        "p50": float(np.percentile(arr, 50)),
        "p90": float(np.percentile(arr, 90)),
        "p95": float(np.percentile(arr, 95)),
        "max": float(np.max(arr)),
    }


def _write_markdown(path: Path, payload: dict[str, Any]) -> None:
    lines = [
        "# GEMBench Official PCD Policy Contract Audit",
        "",
        f"Status: `{payload['status']}`",
        f"PCD taskvars: `{payload['coverage']['pcd_taskvars']}`",
        f"Expected taskvars: `{payload['coverage']['expected_taskvars']}`",
        f"Sampled transitions: `{payload['summary']['sampled_transitions']}`",
        "",
        "| check | passed | detail |",
        "|---|---:|---|",
    ]
    for check in payload["checks"]:
        detail = json.dumps(check["detail"], ensure_ascii=True, sort_keys=True)
        if len(detail) > 260:
            detail = detail[:257] + "..."
        lines.append(f"| `{check['name']}` | {check['passed']} | `{detail}` |")
    lines.extend(["", "## Coverage", "", "```json"])
    lines.append(json.dumps(payload["coverage"], ensure_ascii=True, indent=2, sort_keys=True))
    lines.extend(["```", "", "## Examples", "", "```json"])
    lines.append(json.dumps(payload["examples"][: int(payload["args"]["max_examples"])], ensure_ascii=True, indent=2))
    lines.append("```")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def run(args: argparse.Namespace) -> dict[str, Any]:
    pcd_data_dir = Path(args.pcd_data_dir).expanduser().resolve()
    expected_taskvars = _load_taskvars(args.taskvar_file)
    manifest = _load_manifest(args.fastwam_9v32_manifest)
    manifest_taskvars, manifest_demo_keys = _manifest_demo_sets(manifest)
    pcd_taskvars = set(_list_lmdb_taskvars(pcd_data_dir))
    selected_taskvars = expected_taskvars or sorted(pcd_taskvars)
    if args.taskvars:
        wanted = [item.strip() for item in str(args.taskvars).split(",") if item.strip()]
        selected_taskvars = [item for item in selected_taskvars if item in set(wanted)]
    selected_taskvars = selected_taskvars[: int(args.max_taskvars)] if args.max_taskvars else selected_taskvars

    store = LMDBEpisodeStore(pcd_data_dir)
    workspace = _official_workspace(args.robot_3dlotus_root)
    examples: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []
    roundtrip_errors: list[float] = []
    local_l2: list[float] = []
    world_step_l2: list[float] = []
    point_counts: list[float] = []
    episode_counts: Counter[str] = Counter()
    transition_counts: Counter[str] = Counter()
    manifest_overlap = 0
    sampled_demo_keys: set[tuple[str, str]] = set()

    for taskvar in selected_taskvars:
        if taskvar not in pcd_taskvars:
            failures.append({"taskvar": taskvar, "error": "taskvar_missing_from_pcd_data_dir"})
            continue
        keys = sorted(store.list_episode_keys(taskvar), key=_episode_sort_key)
        episode_counts[taskvar] = len(keys)
        for episode_idx, key in enumerate(keys[: int(args.episodes_per_taskvar)]):
            episode_key = key.decode("ascii", errors="ignore")
            sampled_demo_keys.add((taskvar, episode_key))
            if (taskvar, episode_key) in manifest_demo_keys:
                manifest_overlap += 1
            try:
                episode = store.get(taskvar, key)
                action = np.asarray(episode["action"], dtype=np.float64)
                xyz_seq = _to_array_list(episode["xyz"])
                rgb_seq = _to_array_list(episode["rgb"])
                key_frameids = [int(v) for v in episode.get("key_frameids", [])]
                if action.ndim != 2 or action.shape[-1] != 8:
                    raise ValueError(f"bad action shape: {action.shape}")
                if len(xyz_seq) != action.shape[0] or len(rgb_seq) != action.shape[0]:
                    raise ValueError(
                        f"step length mismatch: xyz={len(xyz_seq)} rgb={len(rgb_seq)} action={action.shape[0]}"
                    )
                if key_frameids and len(key_frameids) != action.shape[0]:
                    raise ValueError(f"key_frameids/action length mismatch: {len(key_frameids)} vs {action.shape[0]}")
                max_steps = min(int(args.max_steps_per_episode), max(0, int(action.shape[0]) - 1))
                for step in range(max_steps):
                    try:
                        frame = _compute_policy_frame(
                            episode=episode,
                            step=step,
                            workspace=workspace,
                            robot_3dlotus_root=args.robot_3dlotus_root,
                            xyz_shift=args.xyz_shift,
                            xyz_norm=bool(args.xyz_norm),
                            rm_table=bool(args.rm_table),
                            rm_robot=str(args.rm_robot),
                            num_points=int(args.num_points),
                            sample_seed=int(args.sample_seed) + episode_idx * 10007,
                        )
                    except Exception as exc:
                        failures.append(
                            {
                                "taskvar": taskvar,
                                "episode_key": episode_key,
                                "step": int(step),
                                "error": f"{type(exc).__name__}: {exc}",
                            }
                        )
                        continue
                    frame.update(
                        {
                            "taskvar": taskvar,
                            "episode_key": episode_key,
                            "key_frameid": key_frameids[step] if step < len(key_frameids) else None,
                            "next_key_frameid": key_frameids[step + 1] if step + 1 < len(key_frameids) else None,
                            "in_fastwam_9v32_manifest": (taskvar, episode_key) in manifest_demo_keys,
                        }
                    )
                    examples.append(frame)
                    transition_counts[taskvar] += 1
                    roundtrip_errors.append(float(frame["roundtrip_max_abs_error"]))
                    local_l2.append(float(frame["local_target_l2"]))
                    world_step_l2.append(float(frame["world_current_to_next_l2"]))
                    point_counts.append(float(frame["points_used"]))
            except Exception as exc:
                failures.append({"taskvar": taskvar, "episode_key": episode_key, "error": f"{type(exc).__name__}: {exc}"})

    store.close()

    missing_expected_from_pcd = sorted(set(expected_taskvars or []) - pcd_taskvars)
    missing_expected_from_manifest = sorted(set(expected_taskvars or []) - manifest_taskvars) if manifest else []
    pcd_not_manifest = sorted(pcd_taskvars - manifest_taskvars) if manifest else []
    sampled_manifest_overlap_fraction = float(manifest_overlap / max(len(sampled_demo_keys), 1))
    max_roundtrip = max(roundtrip_errors) if roundtrip_errors else math.inf
    checks = [
        {
            "name": "expected_taskvars_present_in_pcd",
            "passed": not missing_expected_from_pcd,
            "detail": {"missing": missing_expected_from_pcd[:20], "count": len(missing_expected_from_pcd)},
        },
        {
            "name": "sampled_transitions_nonempty",
            "passed": len(examples) > 0,
            "detail": len(examples),
        },
        {
            "name": "sampled_transitions_load_without_error",
            "passed": not failures,
            "detail": failures[:20],
        },
        {
            "name": "local_target_roundtrip",
            "passed": max_roundtrip <= float(args.max_roundtrip_error),
            "detail": {"max": None if not roundtrip_errors else max_roundtrip, "threshold": float(args.max_roundtrip_error)},
        },
        {
            "name": "points_after_filter_nonempty",
            "passed": bool(point_counts) and min(point_counts) > 0,
            "detail": _percentiles(point_counts),
        },
    ]
    if bool(args.require_manifest_covers_expected_taskvars):
        checks.append(
            {
                "name": "fastwam_9v32_manifest_covers_expected_taskvars",
                "passed": not missing_expected_from_manifest,
                "detail": {"missing": missing_expected_from_manifest[:20], "count": len(missing_expected_from_manifest)},
            }
        )
    status = "passed" if all(bool(check["passed"]) for check in checks) else "failed"
    return {
        "eval_type": "gembench_official_pcd_policy_contract_audit",
        "official_full_score": False,
        "status": status,
        "args": {
            "pcd_data_dir": str(pcd_data_dir),
            "taskvar_file": args.taskvar_file,
            "fastwam_9v32_manifest": args.fastwam_9v32_manifest,
            "robot_3dlotus_root": args.robot_3dlotus_root,
            "xyz_shift": args.xyz_shift,
            "xyz_norm": bool(args.xyz_norm),
            "rm_table": bool(args.rm_table),
            "rm_robot": str(args.rm_robot),
            "num_points": int(args.num_points),
            "episodes_per_taskvar": int(args.episodes_per_taskvar),
            "max_steps_per_episode": int(args.max_steps_per_episode),
            "max_examples": int(args.max_examples),
        },
        "official_training_contract": {
            "data_dir_family": "train_dataset/keysteps_bbox_pcd/seed0/voxel1cm",
            "sample": "current processed key-step point cloud/action[t] -> action[t+1]",
            "target_xyz_local": "(gt_action[:3] - pc_centroid) / pc_radius",
            "eval_world_xyz": "pred_xyz_local * pc_radius + pc_centroid, then z clamp before Mover",
            "default_checked_contract": "xyz_shift=center, xyz_norm=False, rm_table=True, rm_robot=box_keep_gripper",
            "rotation_note": "official job_scripts/train_3dlotus_policy.sh uses rot_type=euler_disc; this audit checks xyz frame/source, not model rotation head.",
        },
        "workspace": workspace,
        "coverage": {
            "expected_taskvars": len(expected_taskvars or []),
            "pcd_taskvars": len(pcd_taskvars),
            "manifest_taskvars": len(manifest_taskvars) if manifest else None,
            "missing_expected_from_pcd": missing_expected_from_pcd,
            "missing_expected_from_fastwam_9v32_manifest": missing_expected_from_manifest,
            "pcd_taskvars_not_in_fastwam_9v32_manifest": pcd_not_manifest,
            "sampled_demo_keys": len(sampled_demo_keys),
            "sampled_manifest_overlap": manifest_overlap,
            "sampled_manifest_overlap_fraction": sampled_manifest_overlap_fraction,
            "episode_counts_head": dict(episode_counts.most_common(20)),
            "transition_counts_head": dict(transition_counts.most_common(20)),
        },
        "summary": {
            "sampled_transitions": len(examples),
            "failures": len(failures),
            "roundtrip_max_abs_error": None if not roundtrip_errors else max_roundtrip,
            "local_target_l2": _percentiles(local_l2),
            "world_current_to_next_l2": _percentiles(world_step_l2),
            "points_used": _percentiles(point_counts),
        },
        "checks": checks,
        "examples": examples[: int(args.max_examples)],
        "failures": failures[: int(args.max_failures)],
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Audit official GEMBench PCD/local policy contract for FastWAM.")
    parser.add_argument("--pcd-data-dir", default=DEFAULT_PCD_DATA_DIR)
    parser.add_argument("--taskvar-file", default=DEFAULT_TASKVAR_FILE)
    parser.add_argument("--fastwam-9v32-manifest", default=DEFAULT_FASTWAM_9V32_MANIFEST)
    parser.add_argument("--robot-3dlotus-root", default=DEFAULT_ROBOT_3DLOTUS_ROOT)
    parser.add_argument("--taskvars", default=None)
    parser.add_argument("--max-taskvars", type=int, default=0)
    parser.add_argument("--episodes-per-taskvar", type=int, default=2)
    parser.add_argument("--max-steps-per-episode", type=int, default=4)
    parser.add_argument("--xyz-shift", choices=("none", "center", "gripper"), default="center")
    parser.add_argument("--xyz-norm", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--rm-table", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--rm-robot", choices=("none", "box", "box_keep_gripper"), default="box_keep_gripper")
    parser.add_argument("--num-points", type=int, default=4096)
    parser.add_argument("--sample-seed", type=int, default=0)
    parser.add_argument("--max-roundtrip-error", type=float, default=1e-6)
    parser.add_argument("--require-manifest-covers-expected-taskvars", action="store_true")
    parser.add_argument("--max-examples", type=int, default=12)
    parser.add_argument("--max-failures", type=int, default=20)
    parser.add_argument("--output-json", required=True)
    parser.add_argument("--output-md", default=None)
    args = parser.parse_args()

    payload = run(args)
    out_json = Path(args.output_json).expanduser()
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(payload, ensure_ascii=True, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    if args.output_md:
        _write_markdown(Path(args.output_md).expanduser(), payload)
    print(
        json.dumps(
            {
                "status": payload["status"],
                "sampled_transitions": payload["summary"]["sampled_transitions"],
                "failures": payload["summary"]["failures"],
                "pcd_taskvars": payload["coverage"]["pcd_taskvars"],
                "manifest_taskvars": payload["coverage"]["manifest_taskvars"],
            },
            ensure_ascii=True,
            sort_keys=True,
        )
    )
    return 0 if payload["status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
