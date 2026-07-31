"""Normalization and leader-only preprocessing for Megatron Fast-WAM."""

from __future__ import annotations

from pathlib import Path

import torch
import torch.distributed as dist

from .components import WanFrozenComponents, prepare_camera_image
from .config import FastWAMConfig
from .mcore import _tp_info
from .model import FastWAMModel


class MinMaxStats:
    def __init__(self, state_min, state_max, action_min, action_max):
        self.state_min = state_min.float()
        self.state_max = state_max.float()
        self.action_min = action_min.float()
        self.action_max = action_max.float()

    @classmethod
    def from_pretrained(cls, checkpoint: str | Path) -> "MinMaxStats":
        from safetensors import safe_open

        path = Path(checkpoint) / "policy_preprocessor_step_3_normalizer_processor.safetensors"
        with safe_open(path, framework="pt", device="cpu") as handle:
            return cls(
                handle.get_tensor("observation.state.min"),
                handle.get_tensor("observation.state.max"),
                handle.get_tensor("action.min"),
                handle.get_tensor("action.max"),
            )

    @staticmethod
    def _normalize(value: torch.Tensor, minimum: torch.Tensor, maximum: torch.Tensor) -> torch.Tensor:
        return 2.0 * (value - minimum) / (maximum - minimum) - 1.0

    @staticmethod
    def _unnormalize(value: torch.Tensor, minimum: torch.Tensor, maximum: torch.Tensor) -> torch.Tensor:
        return (value + 1.0) * 0.5 * (maximum - minimum) + minimum

    def normalize_state(self, state: torch.Tensor) -> torch.Tensor:
        return self._normalize(state.float(), self.state_min, self.state_max)

    def unnormalize_action(self, action: torch.Tensor) -> torch.Tensor:
        result = self._unnormalize(action.float(), self.action_min, self.action_max)
        gripper = result[..., -1] * 2.0 - 1.0
        result[..., -1] = torch.sign(-gripper)
        return result


def _broadcast_from_tp_leader(tensor: torch.Tensor | None, shape, dtype, device) -> torch.Tensor:
    group, world, rank = _tp_info()
    if world == 1:
        if tensor is None:
            raise RuntimeError("TP1 leader tensor is missing")
        return tensor.to(device=device, dtype=dtype)
    if rank != 0:
        tensor = torch.empty(shape, dtype=dtype, device=device)
    else:
        tensor = tensor.to(device=device, dtype=dtype)
    dist.broadcast(tensor, src=dist.get_global_rank(group, 0), group=group)
    return tensor


class FastWAMPolicy:
    """One-environment policy per DP replica; frozen encoders exist on TP rank 0."""

    def __init__(
        self,
        model: FastWAMModel,
        cfg: FastWAMConfig,
        stats: MinMaxStats,
        components: WanFrozenComponents | None,
    ):
        self.model = model
        self.cfg = cfg
        self.stats = stats
        self.components = components

    @property
    def is_tp_leader(self) -> bool:
        return _tp_info()[2] == 0

    @torch.no_grad()
    def predict_action_chunk(
        self,
        images: dict[str, torch.Tensor] | None,
        state: torch.Tensor | None,
        task: str | None,
    ) -> torch.Tensor:
        device = self.model.proprio_encoder.weight.device
        dtype = self.model.proprio_encoder.weight.dtype
        latent = context = mask = proprio = None
        if self.is_tp_leader:
            if self.components is None or images is None or state is None or task is None:
                raise ValueError("TP leader requires components, images, state and task")
            image = prepare_camera_image(images, self.cfg.image_size)
            latent = self.components.encode_image(image)
            prompt = self.cfg.prompt_template.format(task=task)
            context, mask = self.components.encode_prompt(
                prompt, max_length=self.cfg.tokenizer_max_len
            )
            if state.ndim == 1:
                state = state.unsqueeze(0)
            proprio = self.stats.normalize_state(state)

        latent_h, latent_w = self.cfg.image_size[0] // 16, self.cfg.image_size[1] // 16
        latent = _broadcast_from_tp_leader(
            latent, (1, self.cfg.video.in_dim, 1, latent_h, latent_w), dtype, device
        )
        context = _broadcast_from_tp_leader(
            context, (1, self.cfg.tokenizer_max_len, self.cfg.video.text_dim), dtype, device
        )
        mask = _broadcast_from_tp_leader(
            mask, (1, self.cfg.tokenizer_max_len), torch.bool, device
        )
        proprio = _broadcast_from_tp_leader(
            proprio, (1, self.cfg.proprio_dim), dtype, device
        )
        normalized = self.model.infer_action_encoded(latent, context, mask, proprio)
        return self.stats.unnormalize_action(normalized)
