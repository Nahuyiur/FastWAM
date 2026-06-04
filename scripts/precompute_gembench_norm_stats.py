#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import os
import shutil
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import lmdb
import msgpack
import numpy as np


def _episode_sort_key(key: bytes) -> tuple[int, str]:
    text = key.decode("ascii", errors="ignore")
    if text.startswith("episode"):
        suffix = text[len("episode") :]
        if suffix.isdigit():
            return int(suffix), text
    return 10**12, text


def _dict_get(d: dict, key: bytes):
    if key in d:
        return d[key]
    text_key = key.decode("ascii")
    return d[text_key]


def _unpack_action_only(value: bytes) -> np.ndarray:
    unpacker = msgpack.Unpacker(raw=False, max_buffer_size=max(len(value) + 1024, 1024))
    unpacker.feed(value)
    map_len = unpacker.read_map_header()
    action_obj = None
    for _ in range(map_len):
        key = unpacker.unpack()
        if key == "action":
            action_obj = unpacker.unpack()
        else:
            unpacker.skip()
    if action_obj is None:
        raise KeyError("GEMBench episode is missing 'action'")
    dtype = np.dtype(_dict_get(action_obj, b"type"))
    shape = tuple(int(x) for x in _dict_get(action_obj, b"shape"))
    data = _dict_get(action_obj, b"data")
    return np.frombuffer(data, dtype=dtype).reshape(shape)


def _frame_indices(length: int, target_length: int) -> np.ndarray:
    if length <= 0:
        raise ValueError("Episode has no action rows")
    if target_length <= 1:
        return np.zeros((target_length,), dtype=np.int64)
    return np.rint(np.linspace(0, length - 1, target_length)).astype(np.int64)


def _action_proprio_rows(action: np.ndarray, *, num_video_frames: int, action_horizon: int) -> tuple[np.ndarray, np.ndarray]:
    if action.ndim != 2 or action.shape[1] != 8:
        raise ValueError(f"Expected action [T,8], got {action.shape}")
    frame_idx = _frame_indices(action.shape[0], num_video_frames)
    proprio_idx = frame_idx[:-1]
    next_idx = frame_idx[1:]
    if action_horizon != len(next_idx):
        positions = np.linspace(0, len(next_idx) - 1, action_horizon)
        take = np.rint(positions).astype(np.int64)
        next_idx = next_idx[take]
        proprio_idx = proprio_idx[take]
    return action[next_idx].astype(np.float32, copy=True), action[proprio_idx].astype(np.float32, copy=True)


def _list_episode_keys(taskvar_dir: Path) -> list[bytes]:
    env = lmdb.open(str(taskvar_dir), readonly=True, lock=False, readahead=False, subdir=True)
    try:
        with env.begin(write=False) as txn:
            keys = list(txn.cursor().iternext(values=False))
    finally:
        env.close()
    return sorted(keys, key=_episode_sort_key)


def _resolve_taskvars(data_dir: Path, requested: Sequence[str] | None) -> list[str]:
    if requested:
        taskvars = [str(x) for x in requested]
    else:
        taskvars = sorted(path.name for path in data_dir.iterdir() if path.is_dir())
    out = []
    missing = []
    for taskvar in taskvars:
        taskvar_dir = data_dir / taskvar
        if taskvar_dir.is_dir() and (taskvar_dir / "data.mdb").is_file() and (taskvar_dir / "results.json").is_file():
            out.append(taskvar)
        else:
            missing.append(taskvar)
    if missing:
        raise FileNotFoundError(f"Missing/incomplete GEMBench taskvars: {missing[:10]}")
    if not out:
        raise ValueError(f"No complete GEMBench taskvars found under {data_dir}")
    return out


def _select_split_keys(
    data_dir: Path,
    taskvars: Sequence[str],
    *,
    val_set_proportion: float,
    is_training_set: bool,
    split_seed: int,
    max_episodes_per_taskvar: int | None,
) -> list[tuple[str, list[bytes]]]:
    rng = np.random.default_rng(split_seed)
    jobs: list[tuple[str, list[bytes]]] = []
    for taskvar in taskvars:
        keys = _list_episode_keys(data_dir / taskvar)
        if val_set_proportion > 0:
            order = np.arange(len(keys))
            rng.shuffle(order)
            split_idx = int(len(order) * (1.0 - val_set_proportion))
            selected_idx = order[:split_idx] if is_training_set else order[split_idx:]
            keys = [keys[int(i)] for i in sorted(selected_idx.tolist())]
        if max_episodes_per_taskvar is not None:
            keys = keys[:max_episodes_per_taskvar]
        jobs.append((taskvar, keys))
    return jobs


@dataclass
class WorkerResult:
    taskvar: str
    episodes: int
    action_rows: np.ndarray
    state_rows: np.ndarray
    elapsed_s: float


def _scan_taskvar(args: tuple[str, str, list[bytes], int, int, bool]) -> WorkerResult:
    data_dir_s, taskvar, keys, num_video_frames, action_horizon, readahead = args
    data_dir = Path(data_dir_s)
    taskvar_dir = data_dir / taskvar
    actions = []
    states = []
    t0 = time.time()
    env = lmdb.open(str(taskvar_dir), readonly=True, lock=False, readahead=readahead, subdir=True)
    try:
        with env.begin(write=False) as txn:
            for key in keys:
                value = txn.get(key)
                if value is None:
                    raise KeyError(f"Missing episode key {key!r} in {taskvar}")
                action = _unpack_action_only(value)
                action_rows, state_rows = _action_proprio_rows(
                    action,
                    num_video_frames=num_video_frames,
                    action_horizon=action_horizon,
                )
                actions.append(action_rows)
                states.append(state_rows)
    finally:
        env.close()
    action_arr = np.concatenate(actions, axis=0) if actions else np.empty((0, 8), dtype=np.float32)
    state_arr = np.concatenate(states, axis=0) if states else np.empty((0, 8), dtype=np.float32)
    return WorkerResult(
        taskvar=taskvar,
        episodes=len(keys),
        action_rows=action_arr,
        state_rows=state_arr,
        elapsed_s=time.time() - t0,
    )


def _field_stats(rows: np.ndarray) -> dict[str, list[float]]:
    rows = np.asarray(rows, dtype=np.float32).reshape(-1, 8)
    if rows.size == 0:
        raise ValueError("Cannot compute stats from zero rows")
    return {
        "global_min": rows.min(axis=0).astype(float).tolist(),
        "global_max": rows.max(axis=0).astype(float).tolist(),
        "global_mean": rows.mean(axis=0).astype(float).tolist(),
        "global_std": (rows.std(axis=0) + 1e-8).astype(float).tolist(),
        "global_q01": np.quantile(rows, 0.01, axis=0).astype(float).tolist(),
        "global_q99": np.quantile(rows, 0.99, axis=0).astype(float).tolist(),
    }


def _write_json_atomic(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    with tmp_path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
        f.write("\n")
    os.replace(tmp_path, path)


def _parse_taskvars(value: str | None) -> list[str] | None:
    if value is None or value.strip().lower() in {"", "all"}:
        return None
    return [item.strip() for item in value.split(",") if item.strip()]


def main() -> None:
    parser = argparse.ArgumentParser(description="Precompute GEMBench action/proprio normalization stats for FastWAM.")
    parser.add_argument("--root", default=os.environ.get("GEMBENCH_ROOT", "/mnt/yuhan/datasets/GEMBench"))
    parser.add_argument("--split", default="train")
    parser.add_argument("--subset", default="keysteps_bbox")
    parser.add_argument("--seed", default="seed0")
    parser.add_argument("--taskvars", default=None, help="Comma-separated taskvars. Defaults to all complete taskvars.")
    parser.add_argument("--output", default="data/gembench_keysteps_bbox_dataset_stats.json")
    parser.add_argument("--num-video-frames", type=int, default=9)
    parser.add_argument("--action-horizon", type=int, default=8)
    parser.add_argument("--val-set-proportion", type=float, default=0.02)
    parser.add_argument("--split-seed", type=int, default=42)
    parser.add_argument("--include-val-split", action="store_true", help="Use the val split instead of the train split.")
    parser.add_argument("--max-episodes-per-taskvar", type=int, default=None)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--readahead", action="store_true")
    parser.add_argument("--no-backup", action="store_true")
    args = parser.parse_args()

    data_dir = Path(args.root).expanduser().resolve() / f"{args.split}_dataset" / args.subset / args.seed
    output = Path(args.output).expanduser()
    taskvars = _resolve_taskvars(data_dir, _parse_taskvars(args.taskvars))
    jobs = _select_split_keys(
        data_dir,
        taskvars,
        val_set_proportion=args.val_set_proportion,
        is_training_set=not args.include_val_split,
        split_seed=args.split_seed,
        max_episodes_per_taskvar=args.max_episodes_per_taskvar,
    )
    total_episodes = sum(len(keys) for _, keys in jobs)
    if total_episodes <= 0:
        raise ValueError("Selected split has zero episodes")

    if output.exists() and not args.no_backup:
        backup = output.with_suffix(output.suffix + f".bak_{time.strftime('%Y%m%d_%H%M%S')}")
        shutil.copy2(output, backup)
        print(f"[gembench-stats] backed_up_existing={backup}", flush=True)

    print(
        "[gembench-stats]",
        f"data_dir={data_dir}",
        f"taskvars={len(taskvars)}",
        f"episodes={total_episodes}",
        f"workers={args.workers}",
        f"output={output}",
        flush=True,
    )
    t0 = time.time()
    worker_args = [
        (str(data_dir), taskvar, keys, args.num_video_frames, args.action_horizon, bool(args.readahead))
        for taskvar, keys in jobs
        if keys
    ]
    action_parts = []
    state_parts = []
    done_episodes = 0
    with ProcessPoolExecutor(max_workers=max(1, int(args.workers))) as executor:
        futures = [executor.submit(_scan_taskvar, item) for item in worker_args]
        for future in as_completed(futures):
            result = future.result()
            done_episodes += result.episodes
            action_parts.append(result.action_rows)
            state_parts.append(result.state_rows)
            print(
                "[gembench-stats]",
                f"done={done_episodes}/{total_episodes}",
                f"taskvar={result.taskvar}",
                f"episodes={result.episodes}",
                f"rows={result.action_rows.shape[0]}",
                f"task_elapsed={result.elapsed_s:.1f}s",
                f"total_elapsed={time.time() - t0:.1f}s",
                flush=True,
            )

    actions = np.concatenate(action_parts, axis=0)
    states = np.concatenate(state_parts, axis=0)
    payload = {
        "action": {"default": _field_stats(actions)},
        "state": {"default": _field_stats(states)},
    }
    _write_json_atomic(output, payload)
    print(
        "[gembench-stats] wrote",
        f"output={output}",
        f"action_rows={actions.shape[0]}",
        f"state_rows={states.shape[0]}",
        f"elapsed={time.time() - t0:.1f}s",
        flush=True,
    )


if __name__ == "__main__":
    main()
