"""Run one real RoboCasa sample through the Megatron FastWAM eval backend."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

os.environ.setdefault("GIT_PYTHON_REFRESH", "quiet")

import numpy as np
import torch

from fast_wam.train.robocasa_data import build_robocasa_datasets
from scripts.robocasa_acg_policy_backends import MegatronFastWAMPolicyClient


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--vae-checkpoint", type=Path, required=True)
    parser.add_argument("--norm-stats", type=Path, required=True)
    parser.add_argument("--text-cache", type=Path, required=True)
    parser.add_argument(
        "--task-config",
        default="robocasa_acg_v1_fastwam_8gpu",
    )
    parser.add_argument("--sample-index", type=int, default=0)
    parser.add_argument("--num-inference-steps", type=int, default=2)
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args()


def _to_uint8(frame: torch.Tensor) -> np.ndarray:
    value = ((frame.to(torch.float32) + 1.0) * 127.5).round().clamp(0, 255)
    return value.permute(1, 2, 0).to(torch.uint8).cpu().numpy()


def main() -> None:
    args = parse_args()
    train_dataset, _, _ = build_robocasa_datasets(
        args.repo,
        args.task_config,
    )
    sample_index = int(args.sample_index)
    sample = train_dataset[sample_index]
    video = sample["video"]
    if video.ndim != 4 or video.shape[0] != 3 or video.shape[-1] % 2:
        raise ValueError(f"Expected horizontal two-camera [3,T,H,2W], got {tuple(video.shape)}")
    camera_width = video.shape[-1] // 2
    first_frame = video[:, 0]

    episode_pos, start = train_dataset.windows[sample_index]
    episode = train_dataset.episodes[episode_pos]
    states, _, _ = train_dataset._load_episode_arrays(episode)
    raw_state = np.asarray(states[start], dtype=np.float32)

    client = MegatronFastWAMPolicyClient(
        repo=str(args.repo),
        checkpoint=str(args.checkpoint),
        vae_checkpoint=str(args.vae_checkpoint),
        norm_stats=str(args.norm_stats),
        text_cache=str(args.text_cache),
        device="cuda:0",
        mixed_precision="bf16",
        action_dim=12,
        proprio_dim=16,
        action_horizon=32,
        num_inference_steps=int(args.num_inference_steps),
        seed=int(args.seed),
    )
    try:
        output = client.infer(
            {
                "observation/base_image": _to_uint8(first_frame[:, :, :camera_width]),
                "observation/wrist_image": _to_uint8(first_frame[:, :, camera_width:]),
                "observation/state": raw_state,
                "prompt": episode.task_text,
            }
        )
    finally:
        client.close()

    actions = np.asarray(output["actions"], dtype=np.float32)
    if actions.shape != (32, 12):
        raise ValueError(f"Unexpected action shape: {actions.shape}")
    if not np.isfinite(actions).all():
        raise FloatingPointError("Eval smoke produced non-finite actions")
    print(
        json.dumps(
            {
                "status": "MEGATRON_ROBOCASA_EVAL_SMOKE_OK",
                "sample_index": sample_index,
                "episode_index": int(episode.episode_index),
                "task": episode.task_text,
                "action_shape": list(actions.shape),
                "action_min": float(actions.min()),
                "action_max": float(actions.max()),
                "action_mean": float(actions.mean()),
                "process_group_initialized_after_close": bool(
                    torch.distributed.is_available()
                    and torch.distributed.is_initialized()
                ),
            },
            ensure_ascii=False,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
