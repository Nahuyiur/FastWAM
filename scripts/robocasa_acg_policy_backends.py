#!/usr/bin/env python3
"""Policy backend adapters for RoboCasa-ACG online evaluation."""

from __future__ import annotations

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
        stats = load_dataset_stats_from_json(str(norm_stats))
        self.normalizer = LinearNormalizer(
            shape_meta=shape_meta,
            use_stepwise_action_norm=bool(processor_cfg.use_stepwise_action_norm),
            default_mode=processor_cfg.norm_default_mode,
            exception_mode=OmegaConf.to_container(processor_cfg.norm_exception_mode, resolve=True),
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
