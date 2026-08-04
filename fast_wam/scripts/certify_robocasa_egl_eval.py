#!/usr/bin/env python3
"""Certify a completed EGL rollout artifact before formal training launch."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--eval-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--min-episodes", type=int, default=16)
    parser.add_argument("--expected-checkpoint", type=Path, default=None)
    args = parser.parse_args()

    root = args.eval_dir.resolve()
    summary_path = root / "summary.json"
    if not summary_path.is_file():
        raise FileNotFoundError(summary_path)
    summary = json.loads(summary_path.read_text())
    videos = sorted(
        path
        for shard in root.glob("shard_*")
        if shard.is_dir()
        for path in (shard / "videos").rglob("*.mp4")
    )
    shard_configs = []
    for shard in sorted(path for path in root.glob("shard_*") if path.is_dir()):
        config_path = shard / "eval_config.json"
        if not config_path.is_file():
            shard_configs.append({"shard": shard.name, "missing": True})
            continue
        config = json.loads(config_path.read_text())
        shard_configs.append(
            {
                "shard": shard.name,
                "render_backend": config.get("render_backend"),
                "validate_camera_integrity": config.get("validate_camera_integrity"),
                "checkpoint": config.get("fastwam_checkpoint"),
            }
        )
    checks = {
        "done": (root / "DONE").is_file(),
        "episodes": int(summary.get("num_episodes", -1)) >= args.min_episodes,
        "errors": int(summary.get("num_errors", -1)) == 0,
        "videos": len(videos) >= args.min_episodes and all(path.stat().st_size > 0 for path in videos),
        "protocol": summary.get("protocol_tag") == "fastwam_formal_baseline_v1",
        "protocol_errors": summary.get("protocol_errors") == [],
        "egl": bool(shard_configs)
        and all(item.get("render_backend") == "egl" for item in shard_configs),
        "camera_integrity": bool(shard_configs)
        and all(item.get("validate_camera_integrity") is True for item in shard_configs),
    }
    expected_checkpoint = (
        str(args.expected_checkpoint.resolve()) if args.expected_checkpoint else None
    )
    if expected_checkpoint is not None:
        checks["checkpoint"] = bool(shard_configs) and all(
            item.get("checkpoint") == expected_checkpoint for item in shard_configs
        )
    video_probes = []
    for video in videos:
        capture = cv2.VideoCapture(str(video))
        frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
        width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
        decoded = 0
        first_shape = None
        while True:
            ok, frame = capture.read()
            if not ok:
                break
            decoded += 1
            if first_shape is None:
                first_shape = list(frame.shape)
        capture.release()
        valid = (
            frame_count > 0
            and decoded == frame_count
            and width == 256
            and height == 256
            and first_shape == [256, 256, 3]
        )
        video_probes.append(
            {
                "path": str(video),
                "valid": valid,
                "width": width,
                "height": height,
                "declared_frames": frame_count,
                "decoded_frames": decoded,
                "first_frame_shape": first_shape,
            }
        )
    checks["video_streams"] = bool(video_probes) and all(item["valid"] for item in video_probes)
    passed = all(checks.values())
    payload = {
        "status": "PASS" if passed else "FAIL",
        "eval_dir": str(root),
        "source_checkpoint": expected_checkpoint,
        "render_backend": "egl",
        "checks": checks,
        "shard_configs": shard_configs,
        "summary": summary,
        "video_probes": video_probes,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n")
    print(json.dumps(payload, indent=2))
    if not passed:
        raise SystemExit("EGL rollout certification failed")


if __name__ == "__main__":
    main()
