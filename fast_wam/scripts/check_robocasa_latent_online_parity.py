#!/usr/bin/env python3
"""Compare real cached RoboCasa latents with fresh online Wan-VAE encoding."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import torch

os.environ.setdefault("GIT_PYTHON_REFRESH", "quiet")

from fast_wam.train.robocasa_data import RoboCasaLatentCache, build_robocasa_datasets
from fast_wam.train.vae import WanVideoVAE38Encoder


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, required=True)
    parser.add_argument("--task-config", default="robocasa_acg_v1_fastwam_8gpu")
    parser.add_argument("--latent-cache", type=Path, required=True)
    parser.add_argument("--vae-checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--indices", type=int, nargs="+", default=[0, 1023, 65537, 286100])
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("Latent parity requires one CUDA GPU")
    dataset, _, _ = build_robocasa_datasets(args.repo_root, args.task_config)
    cache = RoboCasaLatentCache(args.latent_cache, len(dataset))
    indices = [int(index) for index in args.indices]
    if any(index < 0 or index >= len(dataset) for index in indices):
        raise IndexError(f"indices={indices}, dataset_size={len(dataset)}")

    device = torch.device("cuda", 0)
    encoder = WanVideoVAE38Encoder.from_pretrained(
        args.vae_checkpoint,
        device=device,
        dtype=torch.bfloat16,
    )
    samples = []
    for index in indices:
        online_sample = dataset[index]
        video = online_sample["video"].unsqueeze(0).to(device=device)
        actual = encoder.encode_normalized_video(video).squeeze(0).cpu().to(torch.bfloat16)
        expected = cache[index].clone()
        difference = (actual.float() - expected.float()).abs()
        samples.append(
            {
                "index": index,
                "episode_index": int(online_sample["episode_index"]),
                "window_start": int(online_sample["window_start"]),
                "shape": list(actual.shape),
                "exact": torch.equal(actual, expected),
                "max_absolute_error": float(difference.max()),
                "mean_absolute_error": float(difference.mean()),
            }
        )

    passed = all(sample["exact"] for sample in samples)
    payload = {
        "status": "PASS" if passed else "FAIL",
        "task_config": args.task_config,
        "latent_cache": str(args.latent_cache.resolve()),
        "vae_checkpoint": str(args.vae_checkpoint.resolve()),
        "dataset_size": len(dataset),
        "dtype": "bfloat16",
        "samples": samples,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n")
    print(json.dumps(payload, indent=2), flush=True)
    if not passed:
        raise SystemExit("RoboCasa cached-vs-online VAE parity failed")


if __name__ == "__main__":
    main()
