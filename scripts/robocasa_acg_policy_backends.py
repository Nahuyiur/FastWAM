#!/usr/bin/env python3
"""Policy backend adapters for RoboCasa-ACG online evaluation."""

from __future__ import annotations

import atexit
import os
import pathlib
import sys
from typing import Any

import numpy as np

try:
    from openpi_client import image_tools
except Exception:  # pragma: no cover - FastWAM eval does not require openpi_client
    image_tools = None

try:
    from openpi_client import websocket_client_policy as _websocket_client_policy
except Exception:  # pragma: no cover - only needed for pi0.5 websocket backend
    _websocket_client_policy = None


def convert_to_uint8(image: np.ndarray) -> np.ndarray:
    if image_tools is not None:
        return image_tools.convert_to_uint8(image)
    image = np.asarray(image)
    if np.issubdtype(image.dtype, np.floating):
        image = np.clip(image, 0.0, 1.0) * 255.0
    return np.ascontiguousarray(np.clip(image, 0, 255).astype(np.uint8))


def resize_with_pad_fallback(image: np.ndarray, height: int, width: int) -> np.ndarray:
    from PIL import Image

    image = convert_to_uint8(image)
    src_h, src_w = image.shape[:2]
    scale = min(width / src_w, height / src_h)
    new_w = max(1, int(round(src_w * scale)))
    new_h = max(1, int(round(src_h * scale)))
    pil = Image.fromarray(image)
    resized = pil.resize((new_w, new_h), resample=Image.BILINEAR)
    canvas = Image.new("RGB", (width, height))
    left = (width - new_w) // 2
    top = (height - new_h) // 2
    canvas.paste(resized, (left, top))
    return np.asarray(canvas, dtype=np.uint8)


def preprocess_image(image: np.ndarray, resize_size: int) -> np.ndarray:
    image = np.ascontiguousarray(image)
    if image_tools is not None:
        resized = image_tools.resize_with_pad(image, resize_size, resize_size)
        return image_tools.convert_to_uint8(resized)
    return resize_with_pad_fallback(image, resize_size, resize_size)


class CameraFrameIntegrityError(RuntimeError):
    """Raised when a renderer returns a malformed or implausible RGB frame."""


def camera_frame_integrity_metrics(image: np.ndarray) -> dict[str, float | list[int]]:
    """Return conservative corruption diagnostics for a RoboCasa RGB frame."""

    raw = np.asarray(image)
    if raw.ndim != 3 or raw.shape[2] != 3:
        raise CameraFrameIntegrityError(
            f"expected HWC RGB frame, got shape={tuple(raw.shape)} dtype={raw.dtype}"
        )
    if raw.shape[0] < 32 or raw.shape[1] < 32:
        raise CameraFrameIntegrityError(f"RGB frame is unexpectedly small: {tuple(raw.shape)}")
    if np.issubdtype(raw.dtype, np.floating) and not np.isfinite(raw).all():
        raise CameraFrameIntegrityError("RGB frame contains NaN or Inf")

    value = convert_to_uint8(raw).astype(np.float32, copy=False)
    horizontal_delta = float(np.abs(np.diff(value, axis=1)).mean())
    vertical_delta = float(np.abs(np.diff(value, axis=0)).mean())
    mean_neighbor_delta = 0.5 * (horizontal_delta + vertical_delta)
    intensity_std = float(value.std())
    edge_to_std_ratio = mean_neighbor_delta / max(intensity_std, 1.0e-6)

    quantized = (value.astype(np.uint8) >> 4).astype(np.int32)
    bins = quantized[..., 0] * 256 + quantized[..., 1] * 16 + quantized[..., 2]
    histogram = np.bincount(bins.reshape(-1), minlength=4096)
    probability = histogram[histogram > 0].astype(np.float64)
    probability /= probability.sum()
    coarse_rgb_entropy = float(-(probability * np.log2(probability)).sum())
    return {
        "shape": [int(v) for v in value.shape],
        "intensity_std": intensity_std,
        "horizontal_neighbor_delta": horizontal_delta,
        "vertical_neighbor_delta": vertical_delta,
        "mean_neighbor_delta": mean_neighbor_delta,
        "edge_to_std_ratio": edge_to_std_ratio,
        "coarse_rgb_entropy": coarse_rgb_entropy,
    }


def validate_camera_frame(image: np.ndarray, camera_name: str) -> dict[str, float | list[int]]:
    """Fail closed on the stripe/noise/blank frames observed with broken OSMesa."""

    metrics = camera_frame_integrity_metrics(image)
    reasons: list[str] = []
    if float(metrics["mean_neighbor_delta"]) > 24.0:
        reasons.append("excessive adjacent-pixel variation")
    if (
        float(metrics["edge_to_std_ratio"]) > 0.55
        and float(metrics["mean_neighbor_delta"]) > 16.0
    ):
        reasons.append("edge energy is implausibly high relative to image variance")
    if (
        float(metrics["coarse_rgb_entropy"]) < 1.25
        and float(metrics["intensity_std"]) < 30.0
    ):
        reasons.append("frame is nearly blank or low-information")
    if reasons:
        raise CameraFrameIntegrityError(
            f"invalid RoboCasa RGB frame from {camera_name}: {', '.join(reasons)}; "
            f"metrics={metrics}"
        )
    return metrics


class WebsocketPolicyClient:
    def __init__(self, host: str, port: int):
        if _websocket_client_policy is None:
            raise ImportError("openpi_client is required for --policy-backend websocket")
        self._client = _websocket_client_policy.WebsocketClientPolicy(host, port)

    def infer(self, element: dict[str, Any]) -> dict[str, Any]:
        return self._client.infer(element)


class FastWAMPolicyClient:
    DEFAULT_PROMPT = "A video recorded from a robot's point of view executing the following instruction: {task}"

    def __init__(
        self,
        *,
        repo: str,
        task_config: str,
        checkpoint: str,
        norm_stats: str,
        text_cache: str,
        device: str,
        mixed_precision: str,
        num_video_frames: int,
        action_horizon: int,
        num_inference_steps: int,
        seed: int,
        rand_device: str,
    ):
        import torch
        from hydra import compose, initialize_config_dir
        from hydra.utils import instantiate
        from omegaconf import OmegaConf

        repo_path = pathlib.Path(repo).resolve()
        if not repo_path.exists():
            raise FileNotFoundError(f"FastWAM repo not found: {repo_path}")
        src_path = repo_path / "src"
        if str(src_path) not in sys.path:
            sys.path.insert(0, str(src_path))

        from fastwam.datasets.lerobot.utils.normalizer import LinearNormalizer, load_dataset_stats_from_json
        from fastwam.runtime import _mixed_precision_to_model_dtype, _normalize_mixed_precision
        from fastwam.utils.config_resolvers import register_default_resolvers

        register_default_resolvers()
        with initialize_config_dir(config_dir=str(repo_path / "configs"), version_base="1.3"):
            cfg = compose(config_name="train", overrides=[f"task={task_config}"])

        precision = _normalize_mixed_precision(mixed_precision or str(cfg.mixed_precision))
        model_dtype = _mixed_precision_to_model_dtype(precision)
        self.model = instantiate(cfg.model, model_dtype=model_dtype, device=device)
        self.model.load_checkpoint(str(checkpoint))
        self.model.eval()

        self.torch = torch
        self.device = self.model.device
        self.model_dtype = self.model.torch_dtype
        self.text_cache = pathlib.Path(text_cache)
        self.context_len = int(cfg.data.train.context_len)
        self.text_encoder_id = str(cfg.data.train.text_encoder_id)
        self.num_video_frames = int(num_video_frames)
        self.action_horizon = int(action_horizon)
        self.num_inference_steps = int(num_inference_steps)
        self.seed = int(seed)
        self.rand_device = str(rand_device)

        shape_meta = OmegaConf.to_container(cfg.data.train.shape_meta, resolve=True)
        processor_cfg = cfg.data.train.processor
        norm_exception_mode = (
            {}
            if processor_cfg.norm_exception_mode is None
            else OmegaConf.to_container(processor_cfg.norm_exception_mode, resolve=True)
        )
        stats = load_dataset_stats_from_json(str(norm_stats))
        self.normalizer = LinearNormalizer(
            shape_meta=shape_meta,
            use_stepwise_action_norm=bool(processor_cfg.use_stepwise_action_norm),
            default_mode=processor_cfg.norm_default_mode,
            exception_mode=norm_exception_mode,
            stats=stats,
        )

    def _prompt_for_cache(self, task_prompt: str | None) -> str:
        prompt = str(task_prompt or "")
        if prompt.startswith("A video recorded from a robot's point of view"):
            return prompt
        return self.DEFAULT_PROMPT.format(task=prompt)

    def _load_text_context(self, prompt: str):
        import hashlib

        hashed = hashlib.sha256(prompt.encode("utf-8")).hexdigest()
        path = self.text_cache / f"{hashed}.t5_len{self.context_len}.{self.text_encoder_id}.pt"
        if not path.exists():
            raise FileNotFoundError(f"Missing FastWAM text cache for prompt={prompt!r}: {path}")
        payload = self.torch.load(path, map_location="cpu")
        context = payload["context"].to(self.torch.float32)
        mask = payload["mask"].bool()
        if context.ndim != 2 or mask.ndim != 1:
            raise ValueError(f"Invalid text cache payload in {path}: {tuple(context.shape)}, {tuple(mask.shape)}")
        context = context.clone()
        context[~mask] = 0.0
        return context, self.torch.ones_like(mask, dtype=self.torch.bool)

    def _normalize_state(self, state: np.ndarray):
        state_t = self.torch.as_tensor(state, dtype=self.torch.float32).reshape(1, -1)
        return self.normalizer.normalizers["state"]["default"].forward(state_t)[0]

    def _denormalize_action(self, action):
        action_t = self.torch.as_tensor(action, dtype=self.torch.float32)
        return self.normalizer.normalizers["action"]["default"].backward(action_t).cpu().numpy()

    def _image_tensor(self, left_image: np.ndarray, wrist_image: np.ndarray):
        validate_camera_frame(left_image, "agentview_left")
        validate_camera_frame(wrist_image, "eye_in_hand")
        merged = np.concatenate([convert_to_uint8(left_image), convert_to_uint8(wrist_image)], axis=1)
        x = self.torch.from_numpy(np.asarray(merged, dtype=np.float32))
        x = x.to(device=self.device, dtype=self.model_dtype)
        x = x * (2.0 / 255.0) - 1.0
        return x.permute(2, 0, 1).unsqueeze(0).contiguous()

    def infer(self, element: dict[str, Any]) -> dict[str, Any]:
        context, context_mask = self._load_text_context(self._prompt_for_cache(element.get("prompt")))
        input_image = self._image_tensor(element["observation/base_image"], element["observation/wrist_image"])
        proprio = self._normalize_state(np.asarray(element["observation/state"], dtype=np.float32))
        out = self.model.infer_action(
            prompt=None,
            input_image=input_image,
            action_horizon=self.action_horizon,
            num_video_frames=self.num_video_frames,
            proprio=proprio,
            context=context,
            context_mask=context_mask,
            negative_prompt=None,
            text_cfg_scale=1.0,
            num_inference_steps=self.num_inference_steps,
            seed=self.seed,
            rand_device=self.rand_device,
            tiled=False,
        )
        return {"actions": self._denormalize_action(out["action"])}


class MegatronFastWAMPolicyClient:
    """Single-replica RoboCasa policy backed by a reshardable Megatron DCP."""

    DEFAULT_PROMPT = FastWAMPolicyClient.DEFAULT_PROMPT

    def __init__(
        self,
        *,
        repo: str,
        checkpoint: str,
        vae_checkpoint: str,
        norm_stats: str,
        text_cache: str,
        device: str,
        mixed_precision: str,
        action_dim: int,
        proprio_dim: int,
        action_horizon: int,
        num_video_frames: int,
        num_inference_steps: int,
        seed: int,
    ):
        import torch
        from dataclasses import replace

        os.environ.setdefault("GIT_PYTHON_REFRESH", "quiet")
        repo_path = pathlib.Path(repo).resolve()
        if str(repo_path) not in sys.path:
            sys.path.insert(0, str(repo_path))
        src_path = repo_path / "src"
        if str(src_path) not in sys.path:
            sys.path.insert(0, str(src_path))

        from fast_wam.checkpoint import load_megatron_dcp
        from fast_wam.config import FastWAMConfig
        from fast_wam.distributed import initialize, transformer_config
        from fast_wam.model import FastWAMModel
        from fast_wam.train.vae import WanVideoVAE38Encoder
        from fastwam.datasets.lerobot.utils.normalizer import (
            LinearNormalizer,
            load_dataset_stats_from_json,
        )

        if device not in {"cuda", "cuda:0"}:
            raise ValueError(
                "Megatron RoboCasa eval isolates one process per visible GPU; "
                f"expected cuda:0, got {device!r}"
            )
        if mixed_precision not in {"bf16", "fp32", "float32"}:
            raise ValueError(
                "Megatron RoboCasa eval supports mixed_precision in "
                f"{{'bf16', 'fp32', 'float32'}}, got {mixed_precision!r}"
            )
        os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
        os.environ.setdefault("MASTER_PORT", "29591")
        os.environ.setdefault("RANK", "0")
        os.environ.setdefault("LOCAL_RANK", "0")
        os.environ.setdefault("WORLD_SIZE", "1")
        self.torch = torch
        self._closed = False
        training_config = FastWAMConfig.from_megatron_checkpoint(checkpoint)
        mismatches = []
        if training_config.action.action_dim != int(action_dim):
            mismatches.append(
                f"action_dim checkpoint={training_config.action.action_dim} cli={int(action_dim)}"
            )
        if training_config.proprio_dim != int(proprio_dim):
            mismatches.append(
                f"proprio_dim checkpoint={training_config.proprio_dim} cli={int(proprio_dim)}"
            )
        if mismatches:
            raise ValueError(
                "Megatron eval contract does not match checkpoint: " + "; ".join(mismatches)
            )
        initialize(1)
        atexit.register(self.close)

        dtype = torch.bfloat16 if mixed_precision == "bf16" else torch.float32
        self.config = replace(
            training_config,
            action_horizon=int(action_horizon),
            num_video_frames=int(num_video_frames),
            num_inference_steps=int(num_inference_steps),
            inference_seed=int(seed),
        )
        self.runtime_contract = {
            "checkpoint_training_attention_backend": self.config.training_attention_backend,
            "checkpoint_training_kernel_mode": self.config.training_kernel_mode,
            "checkpoint_joint_action_video_attention": self.config.joint_action_video_attention,
            "checkpoint_action_dim": self.config.action.action_dim,
            "checkpoint_proprio_dim": self.config.proprio_dim,
            "eval_action_horizon": self.config.action_horizon,
            "eval_num_video_frames": self.config.num_video_frames,
            "eval_num_inference_steps": self.config.num_inference_steps,
            "eval_seed": self.config.inference_seed,
        }
        self.model = FastWAMModel(
            self.config,
            transformer_config(self.config, 1, dtype),
        ).to(device=torch.device("cuda", 0), dtype=dtype)
        load_megatron_dcp(self.model, checkpoint)
        self.model.eval()
        self.vae = WanVideoVAE38Encoder.from_pretrained(
            vae_checkpoint,
            device=torch.device("cuda", 0),
            dtype=dtype,
        )
        self.dtype = dtype
        self.text_cache = pathlib.Path(text_cache)
        self.context_len = self.config.context_len
        self.text_encoder_id = "wan22ti2v5b"
        self.seed = int(seed)
        self.action_dim = int(action_dim)
        self.action_horizon = int(action_horizon)

        shape_meta = {
            "images": [],
            "action": [{"key": "default", "raw_shape": int(action_dim), "shape": int(action_dim)}],
            "state": [{"key": "default", "raw_shape": int(proprio_dim), "shape": int(proprio_dim)}],
        }
        self.normalizer = LinearNormalizer(
            shape_meta=shape_meta,
            use_stepwise_action_norm=False,
            default_mode="min/max",
            exception_mode={},
            stats=load_dataset_stats_from_json(str(norm_stats)),
        )

    def _prompt_for_cache(self, task_prompt: str | None) -> str:
        prompt = str(task_prompt or "")
        if prompt.startswith("A video recorded from a robot's point of view"):
            return prompt
        return self.DEFAULT_PROMPT.format(task=prompt)

    def _load_text_context(self, prompt: str):
        import hashlib

        hashed = hashlib.sha256(prompt.encode("utf-8")).hexdigest()
        path = self.text_cache / f"{hashed}.t5_len{self.context_len}.{self.text_encoder_id}.pt"
        if not path.is_file():
            raise FileNotFoundError(
                f"Missing FastWAM text cache for prompt={prompt!r}: {path}"
            )
        payload = self.torch.load(path, map_location="cpu", weights_only=True)
        context = payload["context"]
        mask = payload["mask"]
        expected_context = (self.context_len, self.config.video.text_dim)
        if context.shape != expected_context or mask.shape != (self.context_len,):
            raise ValueError(
                f"Invalid text cache payload in {path}: "
                f"context={tuple(context.shape)} mask={tuple(mask.shape)}"
            )
        context = context.to(device="cuda:0", dtype=self.dtype).unsqueeze(0)
        mask = mask.to(device="cuda:0", dtype=self.torch.bool).unsqueeze(0)
        context = context.clone()
        context[~mask] = 0
        return context, self.torch.ones_like(mask, dtype=self.torch.bool)

    def _normalize_state(self, state: np.ndarray):
        value = self.torch.as_tensor(state, dtype=self.torch.float32).reshape(1, -1)
        return self.normalizer.normalizers["state"]["default"].forward(value).to(
            device="cuda:0", dtype=self.dtype
        )

    def _denormalize_action(self, action):
        value = self.torch.as_tensor(action, dtype=self.torch.float32)
        return self.normalizer.normalizers["action"]["default"].backward(value).cpu().numpy()

    def _encode_image(self, left_image: np.ndarray, wrist_image: np.ndarray):
        validate_camera_frame(left_image, "agentview_left")
        validate_camera_frame(wrist_image, "eye_in_hand")
        merged = np.concatenate(
            [convert_to_uint8(left_image), convert_to_uint8(wrist_image)],
            axis=1,
        )
        video = self.torch.from_numpy(np.asarray(merged, dtype=np.float32))
        video = video.to(device="cuda:0", dtype=self.dtype) * (2.0 / 255.0) - 1.0
        video = video.permute(2, 0, 1).unsqueeze(0).unsqueeze(2).contiguous()
        return self.vae.encode_normalized_video(video)

    def infer(self, element: dict[str, Any]) -> dict[str, Any]:
        context, context_mask = self._load_text_context(
            self._prompt_for_cache(element.get("prompt"))
        )
        latent = self._encode_image(
            element["observation/base_image"],
            element["observation/wrist_image"],
        )
        proprio = self._normalize_state(
            np.asarray(element["observation/state"], dtype=np.float32)
        )
        if self.config.joint_action_video_attention:
            actions = self.model.infer_action_encoded(
                latent,
                context,
                context_mask,
                proprio,
                seed=self.seed,
                num_inference_steps=self.config.num_inference_steps,
                num_video_frames=self.config.num_video_frames,
            )
        else:
            actions = self.model.infer_action_only_encoded(
                latent,
                context,
                context_mask,
                proprio,
                seed=self.seed,
                num_inference_steps=self.config.num_inference_steps,
            )
        actions = self._denormalize_action(actions)
        expected = (self.action_horizon, self.action_dim)
        if actions.shape != expected:
            raise ValueError(
                f"Megatron action shape mismatch: expected={expected} got={actions.shape}"
            )
        if not np.isfinite(actions).all():
            raise FloatingPointError("Megatron policy produced non-finite actions")
        return {"actions": actions}

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        dist = self.torch.distributed
        if dist.is_available() and dist.is_initialized():
            dist.destroy_process_group()
