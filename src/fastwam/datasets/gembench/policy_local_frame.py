from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np


OFFICIAL_RLBENCH_WORKSPACE = {
    "TABLE_HEIGHT": 0.7505,
    "X_BBOX": (-0.5, 1.5),
    "Y_BBOX": (-1.0, 1.0),
    "Z_BBOX": (0.2, 2.0),
}


@dataclass(frozen=True)
class PolicyLocalFrameConfig:
    enabled: bool = False
    xyz_shift: str = "center"
    xyz_norm: bool = False
    rm_table: bool = True
    rm_robot: str = "none"
    num_points: int = 4096
    sample_seed: int = 0
    voxel_size: float = 0.0
    require_open3d: bool = False


def _as_workspace(workspace: dict[str, Any] | None = None) -> dict[str, Any]:
    out = dict(OFFICIAL_RLBENCH_WORKSPACE)
    if workspace:
        out.update(workspace)
    return out


def _maybe_voxel_downsample(
    xyz: np.ndarray,
    rgb: np.ndarray | None = None,
    *,
    voxel_size: float,
    require_open3d: bool,
) -> tuple[np.ndarray, np.ndarray | None, str]:
    if float(voxel_size) <= 0:
        return xyz, rgb, "disabled"
    try:
        import open3d as o3d

        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(xyz)
        pcd, _, trace = pcd.voxel_down_sample_and_trace(float(voxel_size), np.min(xyz, 0), np.max(xyz, 0))
        out_xyz = np.asarray(pcd.points)
        out_rgb = None
        if rgb is not None:
            trace_idx = np.asarray([int(v[0]) for v in trace], dtype=np.int64)
            out_rgb = rgb[trace_idx]
        return out_xyz, out_rgb, "open3d_voxel_down_sample_and_trace"
    except Exception as exc:
        if require_open3d:
            raise RuntimeError(f"open3d voxel downsample failed: {type(exc).__name__}: {exc}") from exc
        return xyz, rgb, f"unavailable:{type(exc).__name__}:{exc}"


def _robot_keep_mask(
    *,
    xyz: np.ndarray,
    arm_links_info: Any,
    rm_robot: str,
    robot_3dlotus_root: str | None = None,
) -> tuple[np.ndarray, str]:
    if rm_robot == "none":
        return np.ones((xyz.shape[0],), dtype=bool), "disabled"
    if arm_links_info is None:
        raise ValueError(f"rm_robot={rm_robot!r} requires arm_links_info.")
    if robot_3dlotus_root:
        import sys
        from pathlib import Path

        root = Path(robot_3dlotus_root).expanduser().resolve()
        if str(root) not in sys.path:
            sys.path.insert(0, str(root))
    try:
        from genrobo3d.utils.robot_box import RobotBox

        robot_box = RobotBox(
            arm_links_info,
            keep_gripper=(rm_robot == "box_keep_gripper"),
            env_name="rlbench",
        )
        _, robot_point_ids = robot_box.get_pc_overlap_ratio(xyz=xyz, return_indices=True)
        keep = np.ones((xyz.shape[0],), dtype=bool)
        ids = np.asarray(list(robot_point_ids), dtype=np.int64)
        if ids.size:
            keep[ids] = False
        return keep, "robot_3dlotus.RobotBox"
    except Exception as exc:
        raise RuntimeError(f"RobotBox filter failed for rm_robot={rm_robot}: {type(exc).__name__}: {exc}") from exc


def compute_policy_local_frame(
    *,
    xyz: np.ndarray,
    rgb: np.ndarray | None = None,
    ee_pose: np.ndarray | None = None,
    arm_links_info: Any = None,
    workspace: dict[str, Any] | None = None,
    config: PolicyLocalFrameConfig | None = None,
    sample_seed: int | None = None,
    robot_3dlotus_root: str | None = None,
) -> dict[str, Any]:
    cfg = config or PolicyLocalFrameConfig(enabled=True)
    ws = _as_workspace(workspace)
    xyz_arr = np.asarray(xyz, dtype=np.float64).reshape(-1, 3)
    rgb_arr = None if rgb is None else np.asarray(rgb).reshape(-1, np.asarray(rgb).shape[-1])
    if rgb_arr is not None and rgb_arr.shape[0] != xyz_arr.shape[0]:
        raise ValueError(f"RGB/xyz length mismatch: xyz={xyz_arr.shape} rgb={rgb_arr.shape}")

    before_workspace = int(xyz_arr.shape[0])
    keep = (
        (xyz_arr[:, 0] > float(ws["X_BBOX"][0]))
        & (xyz_arr[:, 0] < float(ws["X_BBOX"][1]))
        & (xyz_arr[:, 1] > float(ws["Y_BBOX"][0]))
        & (xyz_arr[:, 1] < float(ws["Y_BBOX"][1]))
        & (xyz_arr[:, 2] > float(ws["Z_BBOX"][0]))
        & (xyz_arr[:, 2] < float(ws["Z_BBOX"][1]))
    )
    if bool(cfg.rm_table):
        keep = keep & (xyz_arr[:, 2] > float(ws["TABLE_HEIGHT"]))
    xyz_arr = xyz_arr[keep]
    if rgb_arr is not None:
        rgb_arr = rgb_arr[keep]
    after_workspace = int(xyz_arr.shape[0])
    if after_workspace <= 0:
        raise ValueError("Policy local frame point cloud is empty after workspace/table filtering.")

    xyz_arr, rgb_arr, voxel_note = _maybe_voxel_downsample(
        xyz_arr,
        rgb_arr,
        voxel_size=float(cfg.voxel_size),
        require_open3d=bool(cfg.require_open3d),
    )
    after_voxel = int(xyz_arr.shape[0])
    if after_voxel <= 0:
        raise ValueError("Policy local frame point cloud is empty after voxel downsample.")

    robot_note = "disabled"
    if cfg.rm_robot != "none":
        keep_robot, robot_note = _robot_keep_mask(
            xyz=xyz_arr,
            arm_links_info=arm_links_info,
            rm_robot=str(cfg.rm_robot),
            robot_3dlotus_root=robot_3dlotus_root,
        )
        xyz_arr = xyz_arr[keep_robot]
        if rgb_arr is not None:
            rgb_arr = rgb_arr[keep_robot]
    after_robot = int(xyz_arr.shape[0])
    if after_robot <= 0:
        raise ValueError("Policy local frame point cloud is empty after robot filtering.")

    sampled = False
    if int(cfg.num_points) > 0 and xyz_arr.shape[0] > int(cfg.num_points):
        rng = np.random.default_rng(int(cfg.sample_seed if sample_seed is None else sample_seed))
        point_idxs = rng.choice(xyz_arr.shape[0], int(cfg.num_points), replace=False)
        xyz_arr = xyz_arr[point_idxs]
        if rgb_arr is not None:
            rgb_arr = rgb_arr[point_idxs]
        sampled = True

    if cfg.xyz_shift == "none":
        centroid = np.zeros((3,), dtype=np.float64)
    elif cfg.xyz_shift == "center":
        centroid = np.mean(xyz_arr, axis=0)
    elif cfg.xyz_shift == "gripper":
        if ee_pose is None:
            raise ValueError("xyz_shift='gripper' requires ee_pose.")
        centroid = np.asarray(ee_pose, dtype=np.float64).reshape(-1)[:3].copy()
    else:
        raise ValueError(f"Unsupported xyz_shift={cfg.xyz_shift!r}")

    if bool(cfg.xyz_norm):
        radius = float(np.max(np.sqrt(np.sum((xyz_arr - centroid) ** 2, axis=1))))
        if not np.isfinite(radius) or radius <= 1.0e-8:
            radius = 1.0
    else:
        radius = 1.0

    return {
        "centroid": centroid.astype(np.float32),
        "radius": float(radius),
        "xyz_shift": str(cfg.xyz_shift),
        "xyz_norm": bool(cfg.xyz_norm),
        "rm_table": bool(cfg.rm_table),
        "rm_robot": str(cfg.rm_robot),
        "voxel_size": float(cfg.voxel_size),
        "voxel_filter": voxel_note,
        "robot_filter": robot_note,
        "points_before_workspace": before_workspace,
        "points_after_workspace": after_workspace,
        "points_after_voxel": after_voxel,
        "points_after_robot": after_robot,
        "points_used": int(xyz_arr.shape[0]),
        "sampled_to_num_points": bool(sampled),
    }


def action_world_to_local(action: np.ndarray, frame: dict[str, Any]) -> np.ndarray:
    out = np.asarray(action, dtype=np.float32).copy()
    centroid = np.asarray(frame["centroid"], dtype=np.float32).reshape(3)
    radius = float(frame["radius"])
    out[..., :3] = (out[..., :3] - centroid) / radius
    return out


def action_local_to_world(action: np.ndarray, frame: dict[str, Any]) -> np.ndarray:
    out = np.asarray(action, dtype=np.float32).copy()
    centroid = np.asarray(frame["centroid"], dtype=np.float32).reshape(3)
    radius = float(frame["radius"])
    out[..., :3] = out[..., :3] * radius + centroid
    return out
