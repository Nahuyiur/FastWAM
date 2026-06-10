#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import pickle
import tarfile
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np

from fastwam.datasets.gembench.lmdb_reader import LMDBEpisodeStore
from fastwam.datasets.gembench.microsteps_9v32 import (
    DEFAULT_CACHE_CAMERA_ORDER,
    DEFAULT_CAMERA_ORDER,
    DEFAULT_FRAME_OFFSETS,
    OFFICIAL_GEMBENCH_CAMERA_ORDER,
    SCHEMA_VERSION,
    cache_episode_path,
    make_taskvar,
)


class _Dummy:
    def __init__(self, *args, **kwargs):
        pass

    def __setstate__(self, state):
        self.__dict__.update(state if isinstance(state, dict) else {"state": state})

    def __len__(self):
        return len(getattr(self, "_observations", getattr(self, "observations", getattr(self, "state", []))))


class _LengthOnlyUnpickler(pickle.Unpickler):
    def find_class(self, module: str, name: str):
        return _Dummy


def _demo_len_from_tar_member(tf: tarfile.TarFile, member: tarfile.TarInfo) -> int:
    handle = tf.extractfile(member)
    if handle is None:
        raise ValueError(f"Could not extract tar member: {member.name}")
    obj = _LengthOnlyUnpickler(handle).load()
    return int(len(obj))


def _parse_member_name(name: str) -> dict[str, Any] | None:
    parts = name.split("/")
    # microsteps/seed0/task/variation0/episodes/episode66/low_dim_obs.pkl
    if len(parts) != 7 or parts[0] != "microsteps" or parts[4] != "episodes" or parts[-1] != "low_dim_obs.pkl":
        return None
    seed = parts[1]
    task = parts[2]
    variation_text = parts[3]
    episode_key = parts[5]
    if not variation_text.startswith("variation"):
        return None
    variation = int(variation_text[len("variation") :])
    return {
        "seed": seed,
        "task": task,
        "variation": variation,
        "taskvar": make_taskvar(task, variation),
        "episode_key": episode_key,
        "member": name,
    }


def _percentiles(values: list[int]) -> dict[str, float | None]:
    if not values:
        return {"min": None, "p25": None, "p50": None, "p75": None, "p90": None, "p95": None, "max": None}
    arr = np.asarray(values, dtype=np.float64)
    return {
        "min": float(np.min(arr)),
        "p25": float(np.percentile(arr, 25)),
        "p50": float(np.percentile(arr, 50)),
        "p75": float(np.percentile(arr, 75)),
        "p90": float(np.percentile(arr, 90)),
        "p95": float(np.percentile(arr, 95)),
        "max": float(np.max(arr)),
    }


def _load_key_frameids(store: LMDBEpisodeStore, taskvar: str, episode_key: str) -> tuple[list[int] | None, str | None]:
    try:
        return store.key_frameids(taskvar, episode_key), None
    except Exception as exc:
        return None, f"{type(exc).__name__}: {exc}"


def _parse_camera_order(value: str | list[str] | tuple[str, ...]) -> tuple[str, ...]:
    if isinstance(value, str):
        items = [item.strip() for item in value.split(",")]
    else:
        items = [str(item).strip() for item in value]
    order = tuple(item for item in items if item)
    if not order:
        raise ValueError("camera order must contain at least one camera")
    return order


def _write_markdown(path: Path, payload: dict[str, Any]) -> None:
    lines = [
        "# GEMBench Microsteps 9V32 Contract Audit",
        "",
        f"Status: `{payload['status']}`",
        f"Recommendation: `{payload['recommendation']}`",
        f"Demos: `{payload['summary']['demos']}`",
        f"Eligible 32-step windows: `{payload['summary']['eligible_32_start_count']}`",
        "",
        "| check | passed | detail |",
        "|---|---:|---|",
    ]
    for check in payload["checks"]:
        detail = json.dumps(check["detail"], ensure_ascii=True, sort_keys=True)
        if len(detail) > 260:
            detail = detail[:257] + "..."
        lines.append(f"| `{check['name']}` | {check['passed']} | `{detail}` |")
    lines.extend(["", "## Summary", "", "```json"])
    lines.append(json.dumps(payload["summary"], ensure_ascii=True, indent=2, sort_keys=True))
    lines.append("```")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def run(args: argparse.Namespace) -> dict[str, Any]:
    root = Path(args.root).expanduser().resolve()
    microsteps_tar = Path(args.microsteps_tar).expanduser().resolve() if args.microsteps_tar else root / "train_dataset" / "microsteps.tar.gz"
    keysteps_dir = Path(args.keysteps_dir).expanduser().resolve() if args.keysteps_dir else root / "train_dataset" / "keysteps_bbox" / "seed0"
    cache_dir = Path(args.rgb_cache_dir).expanduser().resolve()
    if not microsteps_tar.is_file():
        raise FileNotFoundError(f"Missing GEMBench train microsteps tar: {microsteps_tar}")
    if not keysteps_dir.is_dir():
        raise FileNotFoundError(f"Missing GEMBench train keysteps LMDB dir: {keysteps_dir}")
    camera_order = _parse_camera_order(args.camera_order)
    cache_camera_order = _parse_camera_order(args.cache_camera_order)
    missing_cameras = [camera for camera in camera_order if camera not in cache_camera_order]
    if missing_cameras:
        raise ValueError(f"camera_order contains cameras absent from cache_camera_order: {missing_cameras}")
    image_size = int(args.image_size)
    if image_size <= 0:
        raise ValueError(f"image_size must be positive, got {image_size}")

    demos: list[dict[str, Any]] = []
    task_counts: Counter[str] = Counter()
    suffix_counts: Counter[str] = Counter()
    with tarfile.open(microsteps_tar, "r:gz") as tf:
        for member in tf:
            if not member.isfile():
                continue
            parsed = _parse_member_name(member.name)
            if parsed is None:
                continue
            if args.max_demos is not None and len(demos) >= int(args.max_demos):
                break
            suffix = member.name.rsplit(".", 1)[-1] if "." in member.name else ""
            suffix_counts[suffix] += 1
            if args.length_source == "pkl":
                parsed["length"] = int(_demo_len_from_tar_member(tf, member))
            parsed["cache_path"] = str(
                cache_episode_path(
                    cache_dir,
                    seed=str(parsed["seed"]),
                    taskvar=str(parsed["taskvar"]),
                    episode_key=str(parsed["episode_key"]),
                ).relative_to(cache_dir)
            )
            demos.append(parsed)
            task_counts[str(parsed["taskvar"])] += 1

    store = LMDBEpisodeStore(keysteps_dir)
    alignment_failures: list[dict[str, Any]] = []
    try:
        for row in demos:
            key_frameids, error = _load_key_frameids(store, str(row["taskvar"]), str(row["episode_key"]))
            if error is not None:
                row["key_frameids"] = []
                row["keyframe_alignment_ok"] = False
                row["keyframe_alignment_error"] = error
                alignment_failures.append(row)
                continue
            assert key_frameids is not None
            row["key_frameids"] = key_frameids
            if args.length_source == "key_frameids":
                row["length"] = int(max(key_frameids) + 1) if key_frameids else 0
            ok = bool(key_frameids) and min(key_frameids) >= 0 and max(key_frameids) < int(row["length"])
            row["keyframe_alignment_ok"] = ok
            if not ok:
                row["keyframe_alignment_error"] = "key_frameids outside microsteps length"
                alignment_failures.append(row)
    finally:
        store.close()

    for row in demos:
        row["available_32_starts"] = int(max(0, int(row["length"]) - 32))

    lengths = [int(row["length"]) for row in demos]
    eligible_episodes = sum(1 for length in lengths if length >= 33)
    eligible_starts = sum(max(0, length - 32) for length in lengths)
    policy_starts = sum(max(0, length - 1) for length in lengths)
    eligible_transition_fraction = float(eligible_starts / policy_starts) if policy_starts else 0.0
    eligible_episode_fraction = float(eligible_episodes / len(demos)) if demos else 0.0
    cache_required_bytes_estimate = sum(length * len(cache_camera_order) * image_size * image_size * 3 for length in lengths)

    checks = [
        {"name": "microsteps_demos_nonempty", "passed": len(demos) > 0, "detail": len(demos)},
        {
            "name": "all_demos_have_33_observations",
            "passed": eligible_episode_fraction >= float(args.min_episode_fraction),
            "detail": {"value": eligible_episode_fraction, "threshold": float(args.min_episode_fraction)},
        },
        {
            "name": "eligible_32_transition_fraction",
            "passed": eligible_transition_fraction >= float(args.min_transition_fraction),
            "detail": {"value": eligible_transition_fraction, "threshold": float(args.min_transition_fraction)},
        },
        {
            "name": "keysteps_key_frameids_align",
            "passed": len(alignment_failures) == 0,
            "detail": {"failures": len(alignment_failures), "first": alignment_failures[:5]},
        },
        {
            "name": "train_microsteps_are_low_dim_only",
            "passed": dict(suffix_counts) == {"pkl": len(demos)},
            "detail": dict(suffix_counts),
        },
    ]
    status = "passed" if all(check["passed"] for check in checks) else "failed"
    payload = {
        "schema_version": SCHEMA_VERSION,
        "status": status,
        "recommendation": "render_dense_rgb_cache_then_train_9v32" if status == "passed" else "do_not_train_9v32",
        "root": str(root),
        "microsteps_tar": str(microsteps_tar),
        "keysteps_dir": str(keysteps_dir),
        "rgb_cache_dir": str(cache_dir),
        "frame_offsets": list(DEFAULT_FRAME_OFFSETS),
        "action_horizon": 32,
        "num_video_frames": 9,
        "camera_order": list(camera_order),
        "cache_camera_order": list(cache_camera_order),
        "image_size": image_size,
        "summary": {
            "demos": len(demos),
            "length_source": str(args.length_source),
            "taskvars": len(task_counts),
            "length": _percentiles(lengths),
            "eligible_episodes": eligible_episodes,
            "eligible_episode_fraction": eligible_episode_fraction,
            "eligible_32_start_count": int(eligible_starts),
            "policy_start_count": int(policy_starts),
            "eligible_32_transition_fraction": eligible_transition_fraction,
            "cache_required_bytes_estimate_uint8": int(cache_required_bytes_estimate),
            "cache_required_gib_estimate_uint8": float(cache_required_bytes_estimate / (1024**3)),
            "task_counts": dict(sorted(task_counts.items())),
            "suffix_counts": dict(suffix_counts),
        },
        "checks": checks,
        "demos": demos,
        "official_full_score": False,
    }
    return payload


def main() -> int:
    parser = argparse.ArgumentParser(description="Audit and manifest GEMBench train microsteps for FastWAM-style 9V/32A training.")
    parser.add_argument("--root", default="/mnt/yuhan/datasets/GEMBench")
    parser.add_argument("--microsteps-tar", default=None)
    parser.add_argument("--keysteps-dir", default=None)
    parser.add_argument("--rgb-cache-dir", default="/mnt/yuhan/datasets/GEMBench/fastwam_cache/microsteps_9v32_rgb")
    parser.add_argument("--camera-order", default=",".join(DEFAULT_CAMERA_ORDER))
    parser.add_argument("--cache-camera-order", default=",".join(DEFAULT_CACHE_CAMERA_ORDER))
    parser.add_argument("--official-camera-order", action="store_const", dest="camera_order", const=",".join(OFFICIAL_GEMBENCH_CAMERA_ORDER))
    parser.add_argument("--official-cache-camera-order", action="store_const", dest="cache_camera_order", const=",".join(OFFICIAL_GEMBENCH_CAMERA_ORDER))
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--max-demos", type=int, default=None)
    parser.add_argument(
        "--length-source",
        choices=("pkl", "key_frameids"),
        default="pkl",
        help=(
            "Use `pkl` for strict demo length loading, or `key_frameids` for a fast full-dataset gate. "
            "The render step later checks the real demo length before writing cache."
        ),
    )
    parser.add_argument("--min-episode-fraction", type=float, default=1.0)
    parser.add_argument("--min-transition-fraction", type=float, default=0.70)
    parser.add_argument("--output-json", required=True)
    parser.add_argument("--output-md", default=None)
    args = parser.parse_args()
    payload = run(args)
    output_json = Path(args.output_json)
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(payload, ensure_ascii=True, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    if args.output_md:
        _write_markdown(Path(args.output_md), payload)
    print(
        json.dumps(
            {
                "status": payload["status"],
                "recommendation": payload["recommendation"],
                "summary": payload["summary"],
            },
            ensure_ascii=True,
            sort_keys=True,
        )
    )
    return 0 if payload["status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
