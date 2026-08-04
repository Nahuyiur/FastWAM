from __future__ import annotations

import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from fast_wam.config import FastWAMConfig


ROOT = Path(__file__).resolve().parents[2]


def _load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


policy_backends = _load_module(
    "robocasa_eval_policy_backends_test",
    ROOT / "scripts" / "robocasa_acg_policy_backends.py",
)
periodic_summary = _load_module(
    "robocasa_periodic_summary_test",
    ROOT / "fast_wam" / "scripts" / "summarize_robocasa_periodic_eval.py",
)


def test_megatron_checkpoint_restores_training_model_contract(tmp_path):
    args = SimpleNamespace(
        fast_wam_action_dim=12,
        fast_wam_proprio_dim=16,
        fast_wam_attention_backend="structured_sdpa",
        fast_wam_kernel_mode="optimized",
        fast_wam_joint_action_video_attention=True,
    )
    torch.save({"args": args, "iteration": 5000}, tmp_path / "common.pt")

    config = FastWAMConfig.from_megatron_checkpoint(tmp_path)

    assert config.action.action_dim == 12
    assert config.proprio_dim == 16
    assert config.training_attention_backend == "structured_sdpa"
    assert config.training_kernel_mode == "optimized"
    assert config.joint_action_video_attention is True


def test_megatron_checkpoint_contract_fails_closed_when_metadata_is_missing(tmp_path):
    torch.save({"args": {"fast_wam_action_dim": 12}}, tmp_path / "common.pt")
    with pytest.raises(ValueError, match="fast_wam_proprio_dim"):
        FastWAMConfig.from_megatron_checkpoint(tmp_path)


def test_camera_integrity_accepts_smooth_scene_and_rejects_known_corruption_shapes():
    y, x = np.mgrid[:224, :224]
    smooth = np.stack(
        [
            60 + 0.35 * x,
            40 + 0.30 * y,
            30 + 0.15 * x + 0.20 * y,
        ],
        axis=-1,
    ).clip(0, 255).astype(np.uint8)
    policy_backends.validate_camera_frame(smooth, "smooth")

    detailed = np.stack(
        [
            128 + 35 * np.sin(2 * np.pi * x / 8 + phase)
            + 21 * np.sin(2 * np.pi * y / 11 + phase)
            for phase in (0.0, 1.1, 2.2)
        ],
        axis=-1,
    ).clip(0, 255).astype(np.uint8)
    metrics = policy_backends.camera_frame_integrity_metrics(detailed)
    assert metrics["edge_to_std_ratio"] > 0.35
    policy_backends.validate_camera_frame(detailed, "normal_high_detail")

    rng = np.random.default_rng(7)
    noise = rng.integers(0, 256, size=(224, 224, 3), dtype=np.uint8)
    with pytest.raises(policy_backends.CameraFrameIntegrityError):
        policy_backends.validate_camera_frame(noise, "noise")

    stripes = smooth.copy()
    stripes[1::2] = 255 - stripes[1::2]
    with pytest.raises(policy_backends.CameraFrameIntegrityError):
        policy_backends.validate_camera_frame(stripes, "stripes")

    blank = np.zeros((224, 224, 3), dtype=np.uint8)
    with pytest.raises(policy_backends.CameraFrameIntegrityError):
        policy_backends.validate_camera_frame(blank, "blank")


def _write_periodic_shard(
    root: Path,
    index: int,
    *,
    replan_steps: int = 32,
    inference_steps: int = 20,
    disable_camera_check: bool = False,
) -> None:
    shard = root / f"shard_{index:02d}"
    video = shard / "videos" / "bucket" / "task" / f"{index}.mp4"
    video.parent.mkdir(parents=True, exist_ok=True)
    video.write_bytes(b"video")
    (shard / "episode_results.jsonl").write_text(
        json.dumps({"bucket": "bucket", "task": "task", "success": index == 0}) + "\n"
    )
    (shard / "errors.jsonl").write_text("")
    (shard / "eval_config.json").write_text(
        json.dumps(
            {
                "replan_steps": replan_steps,
                "fastwam_num_inference_steps": inference_steps,
                "render_backend": "egl",
                "validate_camera_integrity": not disable_camera_check,
                "policy_runtime_contract": {
                    "eval_num_inference_steps": inference_steps,
                    "checkpoint_joint_action_video_attention": True,
                    "checkpoint_training_attention_backend": "structured_sdpa",
                    "checkpoint_training_kernel_mode": "optimized",
                },
            }
        )
    )


def test_periodic_summary_requires_baseline_protocol_contract(tmp_path):
    for index in range(4):
        _write_periodic_shard(tmp_path, index)
    (tmp_path / "shard_03_retry.pid").write_text("12345\n")
    summary, rows, videos = periodic_summary.summarize(
        tmp_path,
        4,
        expected_replan_steps=32,
        expected_inference_steps=20,
        protocol_tag="fastwam_formal_baseline_v1",
        expected_attention_backend="structured_sdpa",
        expected_kernel_mode="optimized",
        expected_render_backend="egl",
    )
    assert len(rows) == len(videos) == 4
    assert summary["protocol_errors"] == []

    _write_periodic_shard(tmp_path, 3, replan_steps=5)
    summary, _, _ = periodic_summary.summarize(
        tmp_path,
        4,
        expected_replan_steps=32,
        expected_inference_steps=20,
        protocol_tag="fastwam_formal_baseline_v1",
        expected_attention_backend="structured_sdpa",
        expected_kernel_mode="optimized",
        expected_render_backend="egl",
    )
    assert any("replan_steps=5" in error for error in summary["protocol_errors"])

    _write_periodic_shard(tmp_path, 3, disable_camera_check=True)
    summary, _, _ = periodic_summary.summarize(
        tmp_path,
        4,
        expected_replan_steps=32,
        expected_inference_steps=20,
        protocol_tag="fastwam_formal_baseline_v1",
        expected_attention_backend="structured_sdpa",
        expected_kernel_mode="optimized",
        expected_render_backend="egl",
    )
    assert any("camera integrity validation was disabled" in error for error in summary["protocol_errors"])


def test_periodic_watcher_defaults_match_fastwam_baseline_protocol():
    script = (ROOT / "fast_wam" / "scripts" / "watch_robocasa_megatron_periodic_eval.sh").read_text()
    assert 'FASTWAM_REPLAN_STEPS="${FASTWAM_REPLAN_STEPS:-32}"' in script
    assert 'FASTWAM_INFER_STEPS="${FASTWAM_INFER_STEPS:-20}"' in script
    assert "__EGL_VENDOR_LIBRARY_FILENAMES" in script
    assert "NVIDIA_EGL_ROOT" in script
    vendor = json.loads((ROOT / "fast_wam" / "runtime" / "10_nvidia.json").read_text())
    assert vendor["ICD"]["library_path"] == "libEGL_nvidia.so.0"
