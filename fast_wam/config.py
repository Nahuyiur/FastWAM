"""Configuration for Megatron Fast-WAM inference and LIBERO training."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class VideoExpertConfig:
    patch_size: tuple[int, int, int] = (1, 2, 2)
    in_dim: int = 48
    hidden_dim: int = 3072
    ffn_dim: int = 14336
    freq_dim: int = 256
    text_dim: int = 4096
    out_dim: int = 48
    num_heads: int = 24
    attn_head_dim: int = 128
    num_layers: int = 30
    eps: float = 1.0e-6


@dataclass(frozen=True)
class ActionExpertConfig:
    action_dim: int = 7
    hidden_dim: int = 1024
    ffn_dim: int = 4096
    num_heads: int = 24
    attn_head_dim: int = 128
    num_layers: int = 30
    text_dim: int = 4096
    freq_dim: int = 256
    eps: float = 1.0e-6


@dataclass(frozen=True)
class FastWAMConfig:
    """The released LIBERO Fast-WAM architecture and optimization contract."""

    video: VideoExpertConfig = field(default_factory=VideoExpertConfig)
    action: ActionExpertConfig = field(default_factory=ActionExpertConfig)
    proprio_dim: int = 8
    action_horizon: int = 32
    n_action_steps: int = 10
    image_size: tuple[int, int] = (224, 448)
    context_len: int = 128
    tokenizer_max_len: int = 128
    prompt_template: str = (
        "A video recorded from a robot's point of view executing the following instruction: {task}"
    )
    num_inference_steps: int = 10
    inference_seed: int = 42
    sigma_shift: float = 5.0
    fp32_attention: bool = True
    training_attention_backend: str = "flex"
    training_kernel_mode: str = "reference"
    video_train_shift: float = 5.0
    action_train_shift: float = 5.0
    num_train_timesteps: int = 1000
    temporal_downsample_factor: int = 4
    spatial_downsample_factor: int = 16
    loss_lambda_video: float = 1.0
    loss_lambda_action: float = 1.0

    def __post_init__(self) -> None:
        if self.video.num_layers != self.action.num_layers:
            raise ValueError("Video and action experts must have the same number of layers.")
        if self.video.num_heads != self.action.num_heads:
            raise ValueError("Video and action experts must have the same number of attention heads.")
        if self.video.attn_head_dim != self.action.attn_head_dim:
            raise ValueError("Video and action experts must have the same attention head dimension.")
        if self.action.action_dim <= 0 or self.action_horizon <= 0:
            raise ValueError("Action dimensions and horizon must be positive.")
        if self.training_attention_backend not in {"sdpa", "flex", "structured_sdpa"}:
            raise ValueError(
                "training_attention_backend must be one of "
                "{'sdpa', 'flex', 'structured_sdpa'}, "
                f"got {self.training_attention_backend!r}."
            )
        if self.training_kernel_mode not in {"reference", "optimized"}:
            raise ValueError(
                "training_kernel_mode must be one of {'reference', 'optimized'}, "
                f"got {self.training_kernel_mode!r}."
            )
        if self.num_train_timesteps <= 0:
            raise ValueError("num_train_timesteps must be positive.")
        if self.temporal_downsample_factor <= 0:
            raise ValueError("temporal_downsample_factor must be positive.")
        if self.spatial_downsample_factor <= 0:
            raise ValueError("spatial_downsample_factor must be positive.")

    @classmethod
    def from_pretrained(cls, checkpoint: str | Path) -> "FastWAMConfig":
        """Read the architecture fields serialized by a LeRobot Fast-WAM checkpoint."""

        path = Path(checkpoint)
        config_path = path / "config.json" if path.is_dir() else path.with_name("config.json")
        if not config_path.is_file():
            raise FileNotFoundError(config_path)
        with config_path.open() as stream:
            raw: dict[str, Any] = json.load(stream)

        video_raw = dict(raw.get("video_dit_config") or {})
        action_raw = dict(raw.get("action_dit_config") or {})
        video = VideoExpertConfig(
            patch_size=tuple(video_raw.get("patch_size", (1, 2, 2))),
            in_dim=int(video_raw.get("in_dim", 48)),
            hidden_dim=int(video_raw.get("hidden_dim", 3072)),
            ffn_dim=int(video_raw.get("ffn_dim", 14336)),
            freq_dim=int(video_raw.get("freq_dim", 256)),
            text_dim=int(video_raw.get("text_dim", 4096)),
            out_dim=int(video_raw.get("out_dim", 48)),
            num_heads=int(video_raw.get("num_heads", 24)),
            attn_head_dim=int(video_raw.get("attn_head_dim", 128)),
            num_layers=int(video_raw.get("num_layers", 30)),
            eps=float(video_raw.get("eps", 1.0e-6)),
        )
        action = ActionExpertConfig(
            action_dim=int(action_raw.get("action_dim", raw.get("action_dim", 7))),
            hidden_dim=int(action_raw.get("hidden_dim", 1024)),
            ffn_dim=int(action_raw.get("ffn_dim", 4096)),
            num_heads=int(action_raw.get("num_heads", 24)),
            attn_head_dim=int(action_raw.get("attn_head_dim", 128)),
            num_layers=int(action_raw.get("num_layers", 30)),
            text_dim=int(action_raw.get("text_dim", 4096)),
            freq_dim=int(action_raw.get("freq_dim", 256)),
            eps=float(action_raw.get("eps", 1.0e-6)),
        )
        image_size = tuple(int(v) for v in raw.get("image_size", (224, 448)))
        return cls(
            video=video,
            action=action,
            proprio_dim=int(raw.get("proprio_dim", 8)),
            action_horizon=int(raw.get("action_horizon", 32)),
            n_action_steps=int(raw.get("n_action_steps", 10)),
            image_size=image_size,
            context_len=int(raw.get("context_len", 128)),
            tokenizer_max_len=int(raw.get("tokenizer_max_len", 128)),
            prompt_template=str(raw.get("prompt_template", cls.prompt_template)),
            num_inference_steps=int(raw.get("num_inference_steps", 10)),
            inference_seed=int(raw.get("inference_seed", 42)),
            sigma_shift=float(raw.get("sigma_shift") or 5.0),
            fp32_attention=bool(raw.get("fp32_attention", True)),
            training_attention_backend=str(raw.get("training_attention_backend", "flex")),
            training_kernel_mode=str(raw.get("training_kernel_mode", "reference")),
            video_train_shift=float(raw.get("video_train_shift", 5.0)),
            action_train_shift=float(raw.get("action_train_shift", 5.0)),
            num_train_timesteps=int(raw.get("num_train_timesteps", 1000)),
            temporal_downsample_factor=int(raw.get("temporal_downsample_factor", 4)),
            spatial_downsample_factor=int(raw.get("spatial_downsample_factor", 16)),
            loss_lambda_video=float(raw.get("loss_lambda_video", 1.0)),
            loss_lambda_action=float(raw.get("loss_lambda_action", 1.0)),
        )

    @classmethod
    def tiny(cls) -> "FastWAMConfig":
        """Small shape-compatible config used by CPU tests."""

        return cls(
            video=VideoExpertConfig(
                patch_size=(1, 2, 2),
                in_dim=4,
                hidden_dim=64,
                ffn_dim=128,
                freq_dim=32,
                text_dim=48,
                out_dim=4,
                num_heads=4,
                attn_head_dim=16,
                num_layers=2,
            ),
            action=ActionExpertConfig(
                action_dim=3,
                hidden_dim=32,
                ffn_dim=64,
                num_heads=4,
                attn_head_dim=16,
                num_layers=2,
                text_dim=48,
                freq_dim=32,
            ),
            proprio_dim=5,
            action_horizon=8,
            n_action_steps=4,
            image_size=(32, 64),
            context_len=12,
            tokenizer_max_len=12,
            num_inference_steps=3,
            training_attention_backend="sdpa",
        )
