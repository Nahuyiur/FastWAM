#!/usr/bin/env python3
"""Verify WebDataset samples against the ordinary RoboCasa input path."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader

REPO_ROOT = Path(__file__).resolve().parents[2]
for path in (REPO_ROOT, REPO_ROOT / "src"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from fast_wam.train.robocasa_data import RoboCasaLatentDataset, build_robocasa_datasets
from fast_wam.train.robocasa_webdataset import (
    RoboCasaIndexedSubset,
    RoboCasaWebDataset,
    load_source_indices,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", required=True)
    parser.add_argument("--webdataset", required=True)
    parser.add_argument("--latent-cache", default=None)
    parser.add_argument("--num-samples", type=int, default=8)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--output", default=None)
    args = parser.parse_args()

    webdataset = RoboCasaWebDataset(args.webdataset)
    if (webdataset.mode == "offline") != bool(args.latent_cache):
        parser.error("offline verification requires --latent-cache; online forbids it")
    train_dataset, _, _ = build_robocasa_datasets(
        args.repo_root,
        "robocasa_acg_v1_fastwam_8gpu",
    )
    source_indices = load_source_indices(
        Path(args.webdataset) / webdataset.manifest["source_index_file"],
        len(train_dataset),
    )
    assert source_indices is not None
    ordinary = (
        RoboCasaLatentDataset(train_dataset, args.latent_cache, source_indices)
        if webdataset.mode == "offline"
        else RoboCasaIndexedSubset(train_dataset, source_indices)
    )
    checked = min(len(webdataset), int(args.num_samples))
    tensor_keys = []
    for index in range(checked):
        expected = ordinary[index]
        actual = webdataset[index]
        expected_keys = {key for key, value in expected.items() if torch.is_tensor(value)}
        actual_keys = {key for key, value in actual.items() if torch.is_tensor(value)}
        if expected_keys != actual_keys:
            raise AssertionError(
                f"tensor keys differ at {index}: {expected_keys ^ actual_keys}"
            )
        for key in sorted(expected_keys):
            torch.testing.assert_close(actual[key], expected[key], rtol=0, atol=0)
        if actual["prompt"] != expected["prompt"]:
            raise AssertionError(f"prompt differs at {index}")
        tensor_keys = sorted(actual_keys)

    loader = DataLoader(
        webdataset,
        batch_size=2,
        num_workers=int(args.num_workers),
        persistent_workers=int(args.num_workers) > 0,
    )
    batch = next(iter(loader))
    payload = {
        "passed": True,
        "mode": webdataset.mode,
        "num_samples": len(webdataset),
        "samples_checked_exact": checked,
        "tensor_keys": tensor_keys,
        "multiworker_batch_size": int(batch["idx"].shape[0]),
        "num_workers": int(args.num_workers),
    }
    if args.output:
        Path(args.output).write_text(json.dumps(payload, indent=2) + "\n")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
