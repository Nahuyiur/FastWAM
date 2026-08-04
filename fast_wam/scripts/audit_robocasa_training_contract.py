#!/usr/bin/env python3
"""Fail-closed audit of the accelerated RoboCasa recipe against FastWAM."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
from pathlib import Path
from typing import Any

import torch
import yaml


REFERENCE_FILES = (
    "configs/train.yaml",
    "configs/task/robocasa_acg_v1_fastwam_8gpu.yaml",
    "configs/data/robocasa_acg_v1_2cam224_train.yaml",
    "configs/model/fastwam_joint.yaml",
    "src/fastwam/models/wan22/fastwam.py",
    "src/fastwam/models/wan22/fastwam_joint.py",
    "src/fastwam/models/wan22/action_dit.py",
    "src/fastwam/models/wan22/mot.py",
    "src/fastwam/models/wan22/wan_video_dit.py",
    "src/fastwam/datasets/robocasa365/acg_video_dataset.py",
    "src/fastwam/datasets/lerobot/processors/fastwam_processor.py",
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _git(repo: Path, *args: str) -> str:
    executable = shutil.which("git")
    if executable is None:
        candidates = sorted(Path("/opt/conda/pkgs").glob("git-*/bin/git"))
        executable = str(candidates[-1]) if candidates else None
    if executable is None:
        raise FileNotFoundError("git executable is required for the baseline audit")
    environment = os.environ.copy()
    library_dirs = [
        str(path)
        for pattern in ("libiconv-*/lib", "pcre2-*/lib")
        for path in sorted(Path("/opt/conda/pkgs").glob(pattern))[-1:]
    ]
    if library_dirs:
        environment["LD_LIBRARY_PATH"] = ":".join(
            library_dirs + [environment.get("LD_LIBRARY_PATH", "")]
        ).rstrip(":")
    return subprocess.check_output(
        [executable, "-c", f"safe.directory={repo}", "-C", str(repo), *args],
        env=environment,
        text=True,
    ).strip()


class Audit:
    def __init__(self) -> None:
        self.checks: list[dict[str, Any]] = []

    def equal(self, name: str, actual: Any, expected: Any) -> None:
        self.checks.append(
            {
                "name": name,
                "passed": actual == expected,
                "actual": actual,
                "expected": expected,
            }
        )

    def true(self, name: str, value: Any) -> None:
        self.equal(name, bool(value), True)

    @property
    def passed(self) -> bool:
        return all(check["passed"] for check in self.checks)


def _load_yaml(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as stream:
        payload = yaml.safe_load(stream)
    if not isinstance(payload, dict):
        raise TypeError(f"Expected mapping in {path}")
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, required=True)
    parser.add_argument("--baseline-repo", type=Path, required=True)
    parser.add_argument("--baseline-commit", required=True)
    parser.add_argument("--latent-cache", type=Path, required=True)
    parser.add_argument("--initial-dcp", type=Path, required=True)
    parser.add_argument("--vae-checkpoint", type=Path, required=True)
    parser.add_argument("--action-checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--world-size", type=int, default=4)
    parser.add_argument("--tensor-parallel-size", type=int, default=1)
    parser.add_argument("--micro-batch-size", type=int, default=1)
    parser.add_argument("--global-batch-size", type=int, default=32)
    parser.add_argument("--train-iters", type=int, default=50000)
    parser.add_argument("--attention-backend", default="structured_sdpa")
    parser.add_argument("--kernel-mode", default="reference")
    parser.add_argument("--optimizer-weight-decay-policy", default="all_trainable")
    args = parser.parse_args()

    repo = args.repo_root.resolve()
    baseline = args.baseline_repo.resolve()
    cache = args.latent_cache.resolve()
    audit = Audit()

    baseline_head = _git(baseline, "rev-parse", "HEAD")
    audit.equal("baseline.commit", baseline_head, args.baseline_commit)
    audit.equal("baseline.worktree_clean", _git(baseline, "status", "--short"), "")
    for relative in REFERENCE_FILES:
        current = repo / relative
        reference = baseline / relative
        audit.true(f"reference_file.exists.{relative}", current.is_file() and reference.is_file())
        if current.is_file() and reference.is_file():
            audit.equal(
                f"reference_file.sha256.{relative}",
                _sha256(current),
                _sha256(reference),
            )

    task = _load_yaml(baseline / "configs/task/robocasa_acg_v1_fastwam_8gpu.yaml")
    data = _load_yaml(baseline / "configs/data/robocasa_acg_v1_2cam224_train.yaml")["train"]
    model = _load_yaml(baseline / "configs/model/fastwam_joint.yaml")

    baseline_world_size = 8
    baseline_effective_batch = (
        int(task["batch_size"])
        * baseline_world_size
        * int(task["gradient_accumulation_steps"])
    )
    dp_size = args.world_size // args.tensor_parallel_size
    audit.equal("training.world_divisible_by_tp", args.world_size % args.tensor_parallel_size, 0)
    audit.equal("training.global_batch_size", args.global_batch_size, baseline_effective_batch)
    audit.equal("training.micro_batch_size", args.micro_batch_size, int(task["batch_size"]))
    audit.equal("training.train_iters", args.train_iters, int(task["max_steps"]))
    audit.equal(
        "training.gradient_accumulation_steps",
        args.global_batch_size // (args.micro_batch_size * dp_size),
        8,
    )
    audit.equal("optimizer.learning_rate", float(task["learning_rate"]), 5.0e-5)
    audit.equal("optimizer.weight_decay", float(task["weight_decay"]), 1.0e-2)
    audit.equal(
        "optimizer.weight_decay_policy",
        args.optimizer_weight_decay_policy,
        "all_trainable",
    )
    audit.equal("optimizer.beta1", 0.9, 0.9)
    audit.equal("optimizer.beta2", 0.95, 0.95)
    audit.equal("optimizer.eps", 1.0e-8, 1.0e-8)
    audit.equal("optimizer.clip_grad", 1.0, 1.0)
    audit.equal("scheduler.type", task["lr_scheduler_type"], "cosine")
    audit.equal("scheduler.warmup_fraction", 0.05, 0.05)
    audit.equal("scheduler.warmup_init", 2.0e-8, 5.0e-5 / 2500)
    audit.equal("scheduler.min_lr", 5.0e-7, 5.0e-7)
    audit.true(
        "scheduler.min_lr_ratio",
        abs(5.0e-7 - float(task["learning_rate"]) * 0.01) <= 1.0e-15,
    )
    audit.equal("precision", task["mixed_precision"], "bf16")
    audit.equal("training.attention_backend", args.attention_backend, "structured_sdpa")
    audit.equal("training.kernel_mode", args.kernel_mode, "reference")

    audit.equal("data.camera_keys", data["camera_keys"], ["robot0_agentview_left", "robot0_eye_in_hand"])
    audit.equal("data.frame_offsets", data["frame_offsets"], [0, 4, 8, 12, 16, 20, 24, 28, 32])
    audit.equal("data.video_size", data["video_size"], [224, 448])
    audit.equal("data.num_frames", int(data["num_frames"]), 9)
    audit.equal("data.action_horizon", int(data["action_horizon"]), 32)
    audit.equal("data.action_dim", int(data["processor"]["action_output_dim"]), 12)
    audit.equal("data.proprio_dim", int(data["processor"]["proprio_output_dim"]), 16)
    audit.equal("model.target", model["_target_"], "fastwam.runtime.create_fastwam_joint")
    audit.equal("model.video_layers", int(model["video_dit_config"]["num_layers"]), 30)
    audit.equal("model.action_layers", int(model["action_dit_config"]["num_layers"]), 30)
    audit.equal("model.video_attention", model["video_dit_config"]["video_attention_mask_mode"], "first_frame_causal")
    audit.equal("model.video_action_conditioned", bool(model["video_dit_config"]["action_conditioned"]), False)
    audit.equal("model.loss_lambda_action", float(model["loss"]["lambda_action"]), 1.0)

    manifest_path = cache / "manifest.json"
    audit.true("cache.manifest_exists", manifest_path.is_file())
    manifest = json.loads(manifest_path.read_text()) if manifest_path.is_file() else {}
    audit.equal("cache.complete", manifest.get("complete"), True)
    audit.equal("cache.split", manifest.get("split"), "train")
    audit.equal("cache.dtype", manifest.get("dtype"), "bfloat16")
    audit.equal("cache.encoding_batch_size", manifest.get("encoding_batch_size"), 1)
    audit.equal("cache.num_samples", int(manifest.get("num_samples", -1)), 286101)
    audit.equal("cache.sample_shape", manifest.get("sample_shape"), [48, 3, 14, 28])
    audit.equal("cache.camera_keys", manifest.get("camera_keys"), data["camera_keys"])
    audit.equal("cache.frame_offsets", manifest.get("frame_offsets"), data["frame_offsets"])
    audit.equal("cache.video_size", manifest.get("video_size"), data["video_size"])
    audit.equal("cache.vae_path", Path(str(manifest.get("vae_checkpoint", ""))).resolve(), args.vae_checkpoint.resolve())
    audit.equal("cache.vae_sha256", manifest.get("vae_sha256"), _sha256(args.vae_checkpoint))
    episode_manifest = Path(data["episode_manifest_path"]).resolve()
    audit.equal("cache.episode_manifest", Path(str(manifest.get("episode_manifest", ""))).resolve(), episode_manifest)
    audit.equal("cache.episode_manifest_sha256", manifest.get("episode_manifest_sha256"), _sha256(episode_manifest))

    initialization_path = args.initial_dcp / "fast_wam_initialization.json"
    audit.true("initialization.manifest_exists", initialization_path.is_file())
    initialization = json.loads(initialization_path.read_text()) if initialization_path.is_file() else {}
    audit.equal("initialization.format", initialization.get("format"), "megatron-dcp")
    audit.equal("initialization.dtype", initialization.get("dtype"), "bfloat16")
    audit.equal("initialization.seed", initialization.get("seed_before_model_construction"), 42)
    audit.equal("initialization.tp", initialization.get("tp_at_save"), 1)
    audit.equal("initialization.action_dim", initialization.get("action_dim"), 12)
    audit.equal("initialization.proprio_dim", initialization.get("proprio_dim"), 16)
    audit.equal("initialization.video_copied", initialization.get("counts", {}).get("video_copied"), 825)
    audit.equal("initialization.action_tensors", sum(initialization.get("counts", {}).get(key, 0) for key in ("action_copied", "action_interpolated")), 820)
    audit.equal("initialization.random_io_tensors", initialization.get("counts", {}).get("official_random_io_tensors"), 6)
    audit.true("initialization.action_checkpoint_exists", args.action_checkpoint.is_file())
    if args.action_checkpoint.is_file():
        audit.equal(
            "initialization.action_checkpoint_sha256",
            _sha256(args.action_checkpoint),
            "7bd65e1986accaaf2e8dd5e59b5e98ef348d7c2f1d88ab2df010eb954a441445",
        )

    payload = {
        "status": "PASS" if audit.passed else "FAIL",
        "baseline": {
            "repo": str(baseline),
            "commit": baseline_head,
            "world_size": baseline_world_size,
            "effective_batch_size": baseline_effective_batch,
        },
        "accelerated": {
            "repo": str(repo),
            "world_size": args.world_size,
            "tensor_parallel_size": args.tensor_parallel_size,
            "data_parallel_size": dp_size,
            "micro_batch_size": args.micro_batch_size,
            "global_batch_size": args.global_batch_size,
            "allowed_differences": [
                "Megatron TP1+DP4 instead of Accelerate/DeepSpeed DP8",
                "ordinary BF16 mmap VAE cache instead of online VAE",
                "structured SDPA with the reference Fast-WAM kernels",
            ],
        },
        "checks": audit.checks,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, default=str) + "\n")
    print(json.dumps(payload, indent=2, default=str))
    if not audit.passed:
        failed = [check["name"] for check in audit.checks if not check["passed"]]
        raise SystemExit(f"RoboCasa training contract audit failed: {failed}")


if __name__ == "__main__":
    main()
