#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import pickle
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np

from fastwam.datasets.gembench.lmdb_reader import LMDBEpisodeStore
from fastwam.datasets.gembench.microsteps_9v32 import (
    DEFAULT_CACHE_CAMERA_ORDER,
    DEFAULT_CAMERA_ORDER,
    DEFAULT_FRAME_OFFSETS,
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


def _demo_len(path: Path) -> int:
    with path.open("rb") as handle:
        return int(len(_LengthOnlyUnpickler(handle).load()))


def _parse_low_dim_path(path: Path, microsteps_dir: Path, seed: str) -> dict[str, Any] | None:
    rel = path.relative_to(microsteps_dir)
    parts = rel.parts
    # task/variation0/episodes/episode66/low_dim_obs.pkl
    if len(parts) != 5 or parts[2] != "episodes" or parts[-1] != "low_dim_obs.pkl":
        return None
    task = parts[0]
    variation_text = parts[1]
    episode_key = parts[3]
    if not variation_text.startswith("variation"):
        return None
    variation = int(variation_text[len("variation") :])
    return {
        "seed": seed,
        "task": task,
        "variation": variation,
        "taskvar": make_taskvar(task, variation),
        "episode_key": episode_key,
        "member": f"microsteps/{seed}/{task}/{variation_text}/episodes/{episode_key}/low_dim_obs.pkl",
        "low_dim_path": str(path),
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


def _manifest_payload(
    *,
    root: Path,
    microsteps_dir: Path,
    keysteps_dir: Path,
    cache_dir: Path,
    demos: list[dict[str, Any]],
    shard_index: int | None,
    num_shards: int | None,
) -> dict[str, Any]:
    lengths = [int(row["length"]) for row in demos]
    eligible_episodes = sum(1 for length in lengths if length >= 33)
    eligible_starts = sum(max(0, length - 32) for length in lengths)
    policy_starts = sum(max(0, length - 1) for length in lengths)
    task_counts = Counter(str(row["taskvar"]) for row in demos)
    missing_keysteps = [row for row in demos if not bool(row.get("keyframe_alignment_ok"))]
    status = "passed" if not missing_keysteps else "passed_with_keyframe_audit_gaps"
    return {
        "schema_version": SCHEMA_VERSION,
        "status": status,
        "recommendation": "render_dense_rgb_cache_then_train_9v32",
        "root": str(root),
        "microsteps_dir": str(microsteps_dir),
        "keysteps_dir": str(keysteps_dir),
        "rgb_cache_dir": str(cache_dir),
        "frame_offsets": list(DEFAULT_FRAME_OFFSETS),
        "action_horizon": 32,
        "num_video_frames": 9,
        "camera_order": list(DEFAULT_CAMERA_ORDER),
        "cache_camera_order": list(DEFAULT_CACHE_CAMERA_ORDER),
        "summary": {
            "demos": len(demos),
            "shard_index": shard_index,
            "num_shards": num_shards,
            "length_source": "extracted_low_dim_pkl",
            "taskvars": len(task_counts),
            "length": _percentiles(lengths),
            "eligible_episodes": eligible_episodes,
            "eligible_episode_fraction": float(eligible_episodes / len(demos)) if demos else 0.0,
            "eligible_32_start_count": int(eligible_starts),
            "policy_start_count": int(policy_starts),
            "eligible_32_transition_fraction": float(eligible_starts / policy_starts) if policy_starts else 0.0,
            "cache_required_bytes_estimate_uint8": int(sum(length * len(DEFAULT_CAMERA_ORDER) * 224 * 224 * 3 for length in lengths)),
            "cache_required_gib_estimate_uint8": float(
                sum(length * len(DEFAULT_CAMERA_ORDER) * 224 * 224 * 3 for length in lengths) / (1024**3)
            ),
            "task_counts": dict(sorted(task_counts.items())),
            "missing_keysteps_key_count": len(missing_keysteps),
            "missing_keysteps_taskvars": sorted({str(row["taskvar"]) for row in missing_keysteps}),
        },
        "checks": [
            {"name": "microsteps_demos_nonempty", "passed": len(demos) > 0, "detail": len(demos)},
            {
                "name": "all_demos_have_33_observations",
                "passed": all(length >= 33 for length in lengths),
                "detail": {"eligible": eligible_episodes, "total": len(demos)},
            },
            {
                "name": "keysteps_episode_keys_exist",
                "passed": len(missing_keysteps) == 0,
                "detail": {"checked": len(demos), "missing": len(missing_keysteps), "first": missing_keysteps[:5]},
            },
        ],
        "demos": demos,
        "official_full_score": False,
    }


def build(args: argparse.Namespace) -> dict[str, Any]:
    root = Path(args.root).expanduser().resolve()
    seed = str(args.seed)
    microsteps_dir = Path(args.microsteps_dir).expanduser().resolve() if args.microsteps_dir else root / "train_dataset" / "microsteps" / seed
    keysteps_dir = Path(args.keysteps_dir).expanduser().resolve() if args.keysteps_dir else root / "train_dataset" / "keysteps_bbox" / seed
    cache_dir = Path(args.rgb_cache_dir).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve()
    if not microsteps_dir.is_dir():
        raise FileNotFoundError(f"Missing extracted microsteps dir: {microsteps_dir}")
    if not keysteps_dir.is_dir():
        raise FileNotFoundError(f"Missing keysteps LMDB dir: {keysteps_dir}")

    store = LMDBEpisodeStore(keysteps_dir)
    keysets: dict[str, set[str]] = {}
    rows: list[dict[str, Any]] = []
    for path in sorted(microsteps_dir.glob("*/variation*/episodes/episode*/low_dim_obs.pkl")):
        parsed = _parse_low_dim_path(path, microsteps_dir, seed)
        if parsed is None:
            continue
        taskvar = str(parsed["taskvar"])
        if taskvar not in keysets:
            if store.has_taskvar(taskvar):
                keysets[taskvar] = {key.decode("ascii", errors="ignore") for key in store.list_episode_keys(taskvar)}
            elif args.missing_keysteps_policy == "fail":
                raise FileNotFoundError(f"Missing keysteps taskvar directory for {taskvar}")
            else:
                keysets[taskvar] = set()
        exists = str(parsed["episode_key"]) in keysets[taskvar]
        if args.missing_keysteps_policy == "fail" and not exists:
            raise KeyError(f"Missing keysteps episode key for {taskvar}/{parsed['episode_key']}")
        if args.missing_keysteps_policy == "skip" and not exists:
            continue
        length = _demo_len(path)
        parsed["length"] = int(length)
        parsed["available_32_starts"] = int(max(0, length - 32))
        parsed["key_frameids"] = []
        parsed["keyframe_alignment_ok"] = bool(exists)
        parsed["cache_path"] = str(
            cache_episode_path(
                cache_dir,
                seed=str(parsed["seed"]),
                taskvar=str(parsed["taskvar"]),
                episode_key=str(parsed["episode_key"]),
            ).relative_to(cache_dir)
        )
        rows.append(parsed)
        if args.max_demos is not None and len(rows) >= int(args.max_demos):
            break
    store.close()

    if not rows:
        raise RuntimeError(f"No low_dim_obs.pkl demos found under {microsteps_dir}")
    rows = [row for row in rows if int(row["length"]) >= 33]
    if not rows:
        raise RuntimeError("No demos are long enough for 9V/32A windows.")

    num_shards = int(args.num_shards)
    shards: list[list[dict[str, Any]]] = [[] for _ in range(num_shards)]
    shard_lengths = [0 for _ in range(num_shards)]
    for row in sorted(rows, key=lambda item: int(item["length"]), reverse=True):
        shard_idx = min(range(num_shards), key=lambda idx: shard_lengths[idx])
        shards[shard_idx].append(row)
        shard_lengths[shard_idx] += int(row["length"])
    for shard in shards:
        shard.sort(key=lambda item: (str(item["taskvar"]), str(item["episode_key"])))

    output_dir.mkdir(parents=True, exist_ok=True)
    shard_payloads = []
    for idx, demos in enumerate(shards):
        payload = _manifest_payload(
            root=root,
            microsteps_dir=microsteps_dir,
            keysteps_dir=keysteps_dir,
            cache_dir=cache_dir,
            demos=demos,
            shard_index=idx,
            num_shards=num_shards,
        )
        path = output_dir / f"manifest_shard_{idx:03d}_of_{num_shards:03d}.json"
        path.write_text(json.dumps(payload, ensure_ascii=True, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        shard_payloads.append(
            {
                "path": str(path),
                "shard_index": idx,
                "demos": len(demos),
                "length_sum": int(sum(int(row["length"]) for row in demos)),
                "cache_required_gib_estimate_uint8": payload["summary"]["cache_required_gib_estimate_uint8"],
            }
        )

    full_payload = _manifest_payload(
        root=root,
        microsteps_dir=microsteps_dir,
        keysteps_dir=keysteps_dir,
        cache_dir=cache_dir,
        demos=sorted(rows, key=lambda item: (str(item["taskvar"]), str(item["episode_key"]))),
        shard_index=None,
        num_shards=num_shards,
    )
    full_path = output_dir / "manifest_full.json"
    full_path.write_text(json.dumps(full_payload, ensure_ascii=True, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    index_payload = {
        "schema_version": SCHEMA_VERSION,
        "status": full_payload["status"],
        "root": str(root),
        "rgb_cache_dir": str(cache_dir),
        "output_dir": str(output_dir),
        "full_manifest": str(full_path),
        "num_shards": num_shards,
        "shards": shard_payloads,
        "summary": full_payload["summary"],
        "official_full_score": False,
    }
    index_path = output_dir / "manifest_shards_index.json"
    index_path.write_text(json.dumps(index_payload, ensure_ascii=True, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return index_payload


def main() -> int:
    parser = argparse.ArgumentParser(description="Build balanced GEMBench 9V/32A render shard manifests from extracted microsteps.")
    parser.add_argument("--root", default="/mnt/yuhan/datasets/GEMBench")
    parser.add_argument("--seed", default="seed0")
    parser.add_argument("--microsteps-dir", default=None)
    parser.add_argument("--keysteps-dir", default=None)
    parser.add_argument("--rgb-cache-dir", default="/mnt/yuhan/datasets/GEMBench/fastwam_cache/microsteps_9v32_rgb")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--num-shards", type=int, required=True)
    parser.add_argument("--max-demos", type=int, default=None)
    parser.add_argument("--missing-keysteps-policy", choices=("fail", "allow", "skip"), default="fail")
    args = parser.parse_args()
    payload = build(args)
    print(json.dumps({"status": payload["status"], "summary": payload["summary"], "num_shards": payload["num_shards"]}, ensure_ascii=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
