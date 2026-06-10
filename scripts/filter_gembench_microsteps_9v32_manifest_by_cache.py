#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

from fastwam.datasets.gembench.microsteps_9v32 import (
    DEFAULT_CACHE_CAMERA_ORDER,
    SCHEMA_VERSION,
    cache_episode_path,
    load_manifest,
    manifest_demo_rows,
)


def _npz_scalar_str(payload: Any, key: str) -> str:
    value = np.asarray(payload[key])
    return str(value.item() if value.shape == () else value.tolist())


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


def _relative_cache_path(path: Path, cache_dir: Path) -> str:
    try:
        return str(path.relative_to(cache_dir))
    except ValueError:
        return str(path)


def _is_valid_cache(row: dict[str, Any], path: Path, *, expected_camera_order: tuple[str, ...]) -> tuple[bool, str | None]:
    if not path.is_file():
        return False, "missing_cache"
    try:
        payload = np.load(path, allow_pickle=False)
        try:
            required = {"rgb", "gripper", "schema_version", "taskvar", "episode_key", "seed", "camera_order", "image_size"}
            missing = sorted(required.difference(payload.files))
            if missing:
                return False, f"missing_keys={missing}"
            if _npz_scalar_str(payload, "schema_version") != SCHEMA_VERSION:
                return False, f"bad_schema={_npz_scalar_str(payload, 'schema_version')}"
            for key in ("taskvar", "episode_key", "seed"):
                if _npz_scalar_str(payload, key) != str(row[key]):
                    return False, f"bad_{key}={_npz_scalar_str(payload, key)}"
            camera_order = tuple(str(value) for value in np.asarray(payload["camera_order"]).tolist())
            if camera_order != expected_camera_order:
                return False, f"bad_camera_order={camera_order}"
            rgb = payload["rgb"]
            gripper = payload["gripper"]
            length = int(row["length"])
            if rgb.ndim != 5 or rgb.shape[0] != length or rgb.shape[1] != len(expected_camera_order) or rgb.shape[-1] != 3:
                return False, f"bad_rgb_shape={tuple(rgb.shape)}"
            if gripper.shape != (length, 8):
                return False, f"bad_gripper_shape={tuple(gripper.shape)}"
            return True, None
        finally:
            payload.close()
    except Exception as exc:
        return False, f"{type(exc).__name__}: {exc}"


def main() -> int:
    parser = argparse.ArgumentParser(description="Create a GEMBench 9V/32A manifest containing only successfully rendered cache files.")
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--rgb-cache-dir", default=None)
    parser.add_argument("--output-json", required=True)
    parser.add_argument("--output-md", default=None)
    parser.add_argument("--min-present-fraction", type=float, default=1.0)
    args = parser.parse_args()

    payload = load_manifest(args.manifest)
    if payload.get("schema_version") != SCHEMA_VERSION:
        raise ValueError(f"Manifest schema mismatch: {args.manifest}")
    cache_dir = Path(args.rgb_cache_dir or payload["rgb_cache_dir"]).expanduser().resolve()
    expected_camera_order = tuple(str(value) for value in payload.get("cache_camera_order", DEFAULT_CACHE_CAMERA_ORDER))

    kept: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []
    for row in manifest_demo_rows(payload):
        path = _cache_path(row, cache_dir)
        ok, error = _is_valid_cache(row, path, expected_camera_order=expected_camera_order)
        if ok:
            new_row = dict(row)
            new_row["cache_path"] = _relative_cache_path(path, cache_dir)
            kept.append(new_row)
        else:
            failures.append({"row": row, "path": str(path), "error": error})

    total = len(kept) + len(failures)
    present_fraction = float(len(kept) / total) if total else 0.0
    status = "passed" if kept and present_fraction >= float(args.min_present_fraction) else "failed"
    lengths = [int(row["length"]) for row in kept]
    kept_windows = sum(max(0, length - 32) for length in lengths)
    out = dict(payload)
    out["status"] = status
    out["recommendation"] = "train_on_filtered_rendered_cache" if status == "passed" else "do_not_train_filtered_cache"
    out["rgb_cache_dir"] = str(cache_dir)
    out["demos"] = kept
    summary = dict(out.get("summary", {}))
    summary.update(
        {
            "source_manifest": str(Path(args.manifest).expanduser().resolve()),
            "source_demos": total,
            "filtered_demos": len(kept),
            "filtered_failures": len(failures),
            "filtered_present_fraction": present_fraction,
            "eligible_32_start_count": int(kept_windows),
        }
    )
    out["summary"] = summary
    out["filter_failures_preview"] = failures[:20]
    out["official_full_score"] = False

    output_json = Path(args.output_json).expanduser()
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(out, ensure_ascii=True, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    if args.output_md:
        lines = [
            "# GEMBench Microsteps 9V32 Filtered Manifest",
            "",
            f"Status: `{status}`",
            f"Source demos: `{total}`",
            f"Filtered demos: `{len(kept)}`",
            f"Present fraction: `{present_fraction}`",
            f"Eligible windows: `{kept_windows}`",
            "",
            "## Failure Preview",
            "",
        ]
        for failure in failures[:20]:
            row = failure["row"]
            lines.append(f"- `{row.get('taskvar')}/{row.get('episode_key')}`: `{failure['error']}`")
        output_md = Path(args.output_md).expanduser()
        output_md.parent.mkdir(parents=True, exist_ok=True)
        output_md.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({"status": status, "filtered_demos": len(kept), "eligible_windows": kept_windows}, sort_keys=True))
    return 0 if status == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
