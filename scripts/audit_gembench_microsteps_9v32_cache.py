#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import zipfile
from pathlib import Path
from typing import Any

import numpy as np

from fastwam.datasets.gembench.microsteps_9v32 import (
    DEFAULT_CACHE_CAMERA_ORDER,
    DEFAULT_FRAME_OFFSETS,
    SCHEMA_VERSION,
    cache_episode_path,
    load_manifest,
    manifest_demo_rows,
)


def _npz_scalar_str(payload: Any, key: str) -> str:
    value = np.asarray(payload[key])
    return str(value.item() if value.shape == () else value.tolist())


def _write_markdown(path: Path, payload: dict[str, Any]) -> None:
    lines = [
        "# GEMBench Microsteps 9V32 Cache Audit",
        "",
        f"Status: `{payload['status']}`",
        f"Checked demos: `{payload['checked_demos']}`",
        f"Usable windows: `{payload['usable_windows']}`",
        "",
        "| check | passed | detail |",
        "|---|---:|---|",
    ]
    for check in payload["checks"]:
        detail = json.dumps(check["detail"], ensure_ascii=True, sort_keys=True)
        if len(detail) > 260:
            detail = detail[:257] + "..."
        lines.append(f"| `{check['name']}` | {check['passed']} | `{detail}` |")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


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


def _array_metadata(path: Path, key: str) -> tuple[tuple[int, ...], np.dtype]:
    with zipfile.ZipFile(path, "r") as zf:
        with zf.open(f"{key}.npy", "r") as handle:
            version = np.lib.format.read_magic(handle)
            if version == (1, 0):
                shape, _, dtype = np.lib.format.read_array_header_1_0(handle)
            elif version == (2, 0):
                shape, _, dtype = np.lib.format.read_array_header_2_0(handle)
            elif version == (3, 0):
                shape, _, dtype = np.lib.format.read_array_header_3_0(handle)
            else:
                raise ValueError(f"Unsupported npy version for {key!r}: {version}")
    return tuple(int(v) for v in shape), np.dtype(dtype)


def _audit_file(row: dict[str, Any], path: Path, *, expected_camera_order: tuple[str, ...]) -> tuple[int, dict[str, Any] | None]:
    if not path.is_file():
        return 0, {"row": row, "path": str(path), "error": "missing_cache"}
    try:
        payload = np.load(path, allow_pickle=False)
        try:
            required = {"rgb", "gripper", "schema_version", "taskvar", "episode_key", "seed", "camera_order", "image_size"}
            missing = sorted(required.difference(payload.files))
            if missing:
                return 0, {"row": row, "path": str(path), "error": f"missing_keys={missing}"}
            rgb_shape, rgb_dtype = _array_metadata(path, "rgb")
            gripper_shape, _ = _array_metadata(path, "gripper")
            if _npz_scalar_str(payload, "schema_version") != SCHEMA_VERSION:
                return 0, {"row": row, "path": str(path), "error": f"bad_schema_version={_npz_scalar_str(payload, 'schema_version')}"}
            for key in ("taskvar", "episode_key", "seed"):
                expected = str(row[key])
                actual = _npz_scalar_str(payload, key)
                if actual != expected:
                    return 0, {"row": row, "path": str(path), "error": f"bad_{key}={actual}"}
            camera_order = tuple(str(value) for value in np.asarray(payload["camera_order"]).tolist())
            image_size = tuple(int(value) for value in np.asarray(payload["image_size"]).reshape(-1).tolist())
            length = int(row["length"])
            if len(rgb_shape) != 5 or rgb_shape[0] != length or rgb_shape[-1] != 3:
                return 0, {"row": row, "path": str(path), "error": f"bad_rgb_shape={rgb_shape}"}
            if image_size != tuple(int(v) for v in rgb_shape[2:4]):
                return 0, {"row": row, "path": str(path), "error": f"bad_image_size={image_size}"}
            if int(rgb_shape[1]) != len(expected_camera_order):
                return 0, {"row": row, "path": str(path), "error": f"bad_camera_count={rgb_shape[1]}"}
            if camera_order != expected_camera_order:
                return 0, {"row": row, "path": str(path), "error": f"bad_camera_order={camera_order}"}
            if gripper_shape != (length, 8):
                return 0, {"row": row, "path": str(path), "error": f"bad_gripper_shape={gripper_shape}"}
            if rgb_dtype != np.dtype(np.uint8):
                return 0, {"row": row, "path": str(path), "error": f"bad_rgb_dtype={rgb_dtype}"}
            windows = max(0, length - 32)
            if windows <= 0:
                return 0, {"row": row, "path": str(path), "error": "no_32_step_window"}
            return windows, None
        finally:
            payload.close()
    except Exception as exc:
        return 0, {"row": row, "path": str(path), "error": f"{type(exc).__name__}: {exc}"}


def run(args: argparse.Namespace) -> dict[str, Any]:
    manifest = load_manifest(args.manifest)
    if manifest.get("schema_version") != SCHEMA_VERSION:
        raise ValueError(f"Manifest schema mismatch: {args.manifest}")
    cache_dir = Path(args.rgb_cache_dir or manifest["rgb_cache_dir"]).expanduser().resolve()
    rows = manifest_demo_rows(manifest)
    if args.taskvars is not None:
        taskvars = {item.strip() for item in args.taskvars.split(",") if item.strip()}
        rows = [row for row in rows if str(row["taskvar"]) in taskvars]
    if args.max_demos is not None:
        rows = rows[: int(args.max_demos)]
    expected_camera_order = tuple(str(value) for value in manifest.get("cache_camera_order", DEFAULT_CACHE_CAMERA_ORDER))

    failures: list[dict[str, Any]] = []
    usable_windows = 0
    present = 0
    for row in rows:
        path = _cache_path(row, cache_dir)
        windows, failure = _audit_file(row, path, expected_camera_order=expected_camera_order)
        if failure is None:
            present += 1
            usable_windows += int(windows)
        else:
            failures.append(failure)

    required = len(rows)
    min_present_fraction = float(args.min_present_fraction)
    present_fraction = float(present / required) if required else 0.0
    checks = [
        {"name": "rows_nonempty", "passed": required > 0, "detail": required},
        {
            "name": "cache_present_fraction",
            "passed": present_fraction >= min_present_fraction,
            "detail": {"present": present, "required": required, "value": present_fraction, "threshold": min_present_fraction},
        },
        {"name": "cache_schema_and_shape", "passed": not failures, "detail": failures[:10]},
        {"name": "usable_32_windows_nonempty", "passed": usable_windows > 0, "detail": usable_windows},
    ]
    status = "passed" if all(check["passed"] for check in checks) else "failed"
    return {
        "status": status,
        "manifest": str(Path(args.manifest).expanduser().resolve()),
        "rgb_cache_dir": str(cache_dir),
        "checked_demos": required,
        "present_demos": present,
        "present_fraction": present_fraction,
        "usable_windows": int(usable_windows),
        "failures": failures,
        "checks": checks,
        "official_full_score": False,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Audit rendered GEMBench 9V/32A RGB cache before training.")
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--rgb-cache-dir", default=None)
    parser.add_argument("--taskvars", default=None)
    parser.add_argument("--max-demos", type=int, default=None)
    parser.add_argument("--min-present-fraction", type=float, default=1.0)
    parser.add_argument("--output-json", required=True)
    parser.add_argument("--output-md", default=None)
    args = parser.parse_args()
    payload = run(args)
    output_json = Path(args.output_json)
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(payload, ensure_ascii=True, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    if args.output_md:
        _write_markdown(Path(args.output_md), payload)
    print(json.dumps({"status": payload["status"], "usable_windows": payload["usable_windows"]}, ensure_ascii=True, sort_keys=True))
    return 0 if payload["status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
