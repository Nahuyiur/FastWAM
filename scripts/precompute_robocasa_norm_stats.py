#!/usr/bin/env python
import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq
from tqdm import tqdm


def read_selected_episodes(manifest: Path, split: str, repo_name: str) -> set[int]:
    selected = set()
    with manifest.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row["split"] == split and row["repo"] == repo_name:
                selected.add(int(row["episode_index"]))
    if not selected:
        raise ValueError(f"No episodes selected for split={split!r}, repo={repo_name!r}")
    return selected


def stack_list_column(column) -> np.ndarray:
    values = column.to_numpy(zero_copy_only=False)
    return np.asarray(list(values), dtype=np.float32)


def field_stats(x: np.ndarray) -> dict[str, list[float]]:
    if x.ndim != 2:
        raise ValueError(f"Expected [N,D] array, got {x.shape}")
    q01 = np.quantile(x, 0.01, axis=0)
    q99 = np.quantile(x, 0.99, axis=0)
    stats = {
        "global_min": x.min(axis=0),
        "global_max": x.max(axis=0),
        "global_q01": q01,
        "global_q99": q99,
        "global_mean": x.mean(axis=0),
        "global_std": x.std(axis=0),
    }
    # Stepwise keys are included for compatibility. This run uses
    # use_stepwise_action_norm=false, so only global_* is consumed.
    for key in list(stats):
        stats[key.replace("global_", "stepwise_")] = stats[key]
    return {k: np.asarray(v, dtype=np.float32).tolist() for k, v in stats.items()}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", default="/mnt/pub_dataset/RoboCasa365")
    parser.add_argument("--repo", default="robocasa365-pretrain-atomic")
    parser.add_argument("--manifest", default=None)
    parser.add_argument("--split", default="train_id")
    parser.add_argument(
        "--output",
        default="/mnt/yuhan/experiments/robocasa_acg_v1/fastwam/cache/norm_stats/robocasa_acg_v1_train_id_dataset_stats.json",
    )
    args = parser.parse_args()

    data_root = Path(args.data_root)
    repo_root = data_root / "repos" / args.repo
    manifest = Path(args.manifest) if args.manifest else data_root / "splits" / "robocasa_acg_v1_episode_manifest.csv"
    selected = read_selected_episodes(manifest, args.split, args.repo)

    by_file: dict[Path, set[int]] = defaultdict(set)
    episode_meta_files = sorted((repo_root / "meta" / "episodes").glob("chunk-*/file-*.parquet"))
    for meta_path in episode_meta_files:
        table = pq.read_table(meta_path, columns=["episode_index", "data/chunk_index", "data/file_index"])
        df = table.to_pandas()
        for _, row in df.iterrows():
            ep = int(row["episode_index"])
            if ep not in selected:
                continue
            chunk_idx = int(row["data/chunk_index"])
            file_idx = int(row["data/file_index"])
            by_file[repo_root / "data" / f"chunk-{chunk_idx:03d}" / f"file-{file_idx:03d}.parquet"].add(ep)

    actions = []
    states = []
    num_frames = 0
    for data_path, episodes in tqdm(sorted(by_file.items()), desc="Scanning RoboCasa train_id parquet"):
        table = pq.read_table(data_path, columns=["episode_index", "observation.state", "action"])
        mask = pc.is_in(table["episode_index"], value_set=pa.array(sorted(episodes), type=pa.int64()))
        table = table.filter(mask)
        num_frames += table.num_rows
        states.append(stack_list_column(table["observation.state"]))
        actions.append(stack_list_column(table["action"]))

    action_arr = np.concatenate(actions, axis=0)
    state_arr = np.concatenate(states, axis=0)
    if action_arr.shape[1] != 12:
        raise ValueError(f"Expected action dim 12, got {action_arr.shape}")
    if state_arr.shape[1] != 16:
        raise ValueError(f"Expected state dim 16, got {state_arr.shape}")

    payload = {
        "action": {"default": field_stats(action_arr)},
        "state": {"default": field_stats(state_arr)},
        "num_episodes": len(selected),
        "num_transition": int(num_frames),
        "source": {
            "data_root": str(data_root),
            "repo": args.repo,
            "manifest": str(manifest),
            "split": args.split,
        },
    }
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps({"output": str(out), "num_episodes": len(selected), "num_frames": int(num_frames)}, indent=2))


if __name__ == "__main__":
    main()
