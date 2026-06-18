#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import pickle
import tarfile
from pathlib import Path
from typing import Any

import numpy as np


def _is_stopped(demo: Any, i: int, obs: Any, stopped_buffer: int) -> bool:
    next_is_not_final = i < (len(demo) - 2)
    gripper_state_no_change = i < (len(demo) - 2) and (
        obs.gripper_open == demo[i + 1].gripper_open
        and obs.gripper_open == demo[max(0, i - 1)].gripper_open
        and demo[max(0, i - 2)].gripper_open == demo[max(0, i - 1)].gripper_open
    )
    small_delta = np.allclose(obs.joint_velocities, 0, atol=0.1)
    return bool(stopped_buffer <= 0 and small_delta and next_is_not_final and gripper_state_no_change)


def keypoint_discovery(demo: Any) -> list[int]:
    """GEMBench/3D-LOTUS keypoint discovery on low-dim RLBench demos."""
    episode_keypoints: list[int] = []
    prev_gripper_open = demo[0].gripper_open
    stopped_buffer = 0
    for i, obs in enumerate(demo):
        stopped = _is_stopped(demo, i, obs, stopped_buffer)
        stopped_buffer = 4 if stopped else stopped_buffer - 1
        last = i == (len(demo) - 1)
        if i != 0 and (obs.gripper_open != prev_gripper_open or last or stopped):
            episode_keypoints.append(int(i))
        prev_gripper_open = obs.gripper_open
    if len(episode_keypoints) > 1 and (episode_keypoints[-1] - 1) == episode_keypoints[-2]:
        episode_keypoints.pop(-2)
    if not episode_keypoints or episode_keypoints[0] != 0:
        episode_keypoints.insert(0, 0)
    return episode_keypoints


def _load_manifest(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    demos = payload.get("demos")
    if not isinstance(demos, list):
        raise ValueError(f"Invalid GEMBench 9V32 manifest, missing demos: {path}")
    return payload


def _member_for_row(row: dict[str, Any]) -> str:
    member = row.get("member")
    if member:
        return str(member)
    task = str(row["task"])
    variation = int(row["variation"])
    episode_key = str(row["episode_key"])
    return f"microsteps/seed0/{task}/variation{variation}/episodes/{episode_key}/low_dim_obs.pkl"


def build(args: argparse.Namespace) -> dict[str, Any]:
    manifest_path = Path(args.manifest).expanduser().resolve()
    microsteps_tar = Path(args.microsteps_tar).expanduser().resolve()
    manifest = _load_manifest(manifest_path)
    demos = manifest["demos"]

    wanted: dict[str, dict[str, Any]] = {}
    for row in demos:
        if str(row.get("seed", args.seed)) != str(args.seed):
            continue
        member = _member_for_row(row)
        wanted[member] = row

    entries: dict[str, list[int]] = {}
    lengths: dict[str, int] = {}
    errors: dict[str, str] = {}
    seen_members = 0
    with tarfile.open(microsteps_tar, "r:gz") as tar:
        for member in tar:
            if not member.isfile() or member.name not in wanted:
                continue
            row = wanted[member.name]
            demo_key = f"{row['taskvar']}/{row['episode_key']}"
            try:
                file_obj = tar.extractfile(member)
                if file_obj is None:
                    raise FileNotFoundError(member.name)
                demo = pickle.load(file_obj)
                key_frameids = keypoint_discovery(demo)
                length = int(len(demo))
                key_frameids = [int(v) for v in key_frameids if 0 <= int(v) < length]
                if not key_frameids:
                    raise ValueError("empty key_frameids")
                if key_frameids[0] != 0:
                    key_frameids.insert(0, 0)
                entries[demo_key] = key_frameids
                lengths[demo_key] = length
            except Exception as exc:
                errors[demo_key] = f"{type(exc).__name__}: {exc}"
            seen_members += 1

    missing = []
    for member, row in wanted.items():
        demo_key = f"{row['taskvar']}/{row['episode_key']}"
        if demo_key not in entries and demo_key not in errors:
            missing.append(member)

    payload: dict[str, Any] = {
        "schema_version": "gembench_microsteps_key_frameids_v1",
        "method": "robot_3dlotus_keypoint_discovery_lowdim",
        "source_manifest": str(manifest_path),
        "source_microsteps_tar": str(microsteps_tar),
        "seed": str(args.seed),
        "num_manifest_demos": len(demos),
        "num_requested_demos": len(wanted),
        "num_seen_members": seen_members,
        "num_entries": len(entries),
        "num_missing_members": len(missing),
        "num_errors": len(errors),
        "entries": entries,
        "lengths": lengths,
        "missing_members_preview": missing[:20],
        "errors_preview": dict(list(errors.items())[:20]),
    }
    if args.fail_on_missing and (missing or errors):
        raise SystemExit(
            "failed to build complete key-frame sidecar: "
            f"entries={len(entries)} missing={len(missing)} errors={len(errors)}"
        )
    return payload


def main() -> int:
    parser = argparse.ArgumentParser(description="Build a fast GEMBench microsteps key_frameids sidecar.")
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--microsteps-tar", required=True)
    parser.add_argument("--seed", default="seed0")
    parser.add_argument("--output-json", required=True)
    parser.add_argument("--fail-on-missing", action="store_true")
    args = parser.parse_args()

    payload = build(args)
    output = Path(args.output_json).expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, ensure_ascii=True, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {
                "status": "passed" if payload["num_errors"] == 0 and payload["num_missing_members"] == 0 else "partial",
                "entries": payload["num_entries"],
                "missing": payload["num_missing_members"],
                "errors": payload["num_errors"],
            },
            ensure_ascii=True,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
