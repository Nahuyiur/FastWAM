#!/usr/bin/env python
import argparse
import csv
import json
from collections import Counter, defaultdict
from pathlib import Path

import pandas as pd


EXPECTED_COUNTS = {
    "train_id": 3440,
    "val_id": 430,
    "test_id_offline": 430,
    "excluded_pretrain_ood_pair": 1279,
    "target_id_sanity": 200,
    "test_ood_pair_strict": 500,
    "test_ood_pair_probe": 100,
    "reserve_id": 316,
}


def read_manifest(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        required = {"split", "repo", "episode_index", "canonical_task", "task_text"}
        missing = required - set(reader.fieldnames or [])
        if missing:
            raise KeyError(f"{path} missing required columns: {sorted(missing)}")
        return list(reader)


def load_episode_indices(repo_root: Path) -> set[int]:
    files = sorted((repo_root / "meta" / "episodes").glob("chunk-*/file-*.parquet"))
    if not files:
        raise FileNotFoundError(f"No LeRobot v3 episode metadata under {repo_root}")
    indices: set[int] = set()
    for path in files:
        df = pd.read_parquet(path, columns=["episode_index"])
        indices.update(int(v) for v in df["episode_index"].tolist())
    return indices


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default="/mnt/pub_dataset/RoboCasa365")
    parser.add_argument("--manifest", default=None)
    parser.add_argument("--strict-counts", action="store_true")
    parser.add_argument("--json-out", default=None)
    args = parser.parse_args()

    root = Path(args.root)
    manifest = Path(args.manifest) if args.manifest else root / "splits" / "robocasa_acg_v1_episode_manifest.csv"
    rows = read_manifest(manifest)

    split_counts = Counter(row["split"] for row in rows)
    duplicate_keys = [
        key for key, count in Counter((row["repo"], row["episode_index"]) for row in rows).items() if count > 1
    ]
    if duplicate_keys:
        raise ValueError(f"Duplicate repo/episode rows in manifest: first={duplicate_keys[:5]}")

    if args.strict_counts:
        for split, expected in EXPECTED_COUNTS.items():
            actual = split_counts.get(split, 0)
            if actual != expected:
                raise AssertionError(f"Split count mismatch for {split}: actual={actual} expected={expected}")

    by_repo_split = defaultdict(Counter)
    for row in rows:
        by_repo_split[row["repo"]][row["split"]] += 1

    train_keys = {(row["repo"], row["episode_index"]) for row in rows if row["split"] == "train_id"}
    forbidden_splits = {"excluded_pretrain_ood_pair", "test_ood_pair_strict", "test_ood_pair_probe"}
    forbidden_keys = {
        (row["repo"], row["episode_index"]) for row in rows if row["split"] in forbidden_splits
    }
    leakage = sorted(train_keys & forbidden_keys)
    if leakage:
        raise AssertionError(f"train_id overlaps forbidden OOD splits: first={leakage[:5]}")

    missing_files = []
    for repo in sorted({row["repo"] for row in rows}):
        repo_root = root / "repos" / repo
        if not repo_root.exists():
            missing_files.append(str(repo_root))
            continue
        available = load_episode_indices(repo_root)
        wanted = {int(row["episode_index"]) for row in rows if row["repo"] == repo}
        missing = sorted(wanted - available)
        if missing:
            raise AssertionError(f"Manifest references missing episodes in {repo}: first={missing[:10]}")

    task_counts = Counter(row["canonical_task"] for row in rows if row["split"] == "train_id")
    summary = {
        "root": str(root),
        "manifest": str(manifest),
        "split_counts": dict(sorted(split_counts.items())),
        "by_repo_split": {repo: dict(counter) for repo, counter in sorted(by_repo_split.items())},
        "train_tasks": len(task_counts),
        "train_min_episodes_per_task": min(task_counts.values()) if task_counts else 0,
        "train_max_episodes_per_task": max(task_counts.values()) if task_counts else 0,
        "missing_files": missing_files,
    }
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    if args.json_out:
        out = Path(args.json_out)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")


if __name__ == "__main__":
    main()
