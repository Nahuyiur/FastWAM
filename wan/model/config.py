"""Wan architecture presets and argument conversion."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple


@dataclass
class WanConfig:
    """Configuration for the DiffSynth-compatible Wan DiT.

    Field names mirror DiffSynth/official Wan configs where practical.
    """

    dim: int
    in_dim: int
    ffn_dim: int
    out_dim: int
    text_dim: int
    freq_dim: int
    eps: float
    patch_size: Tuple[int, int, int]
    num_heads: int
    num_layers: int
    has_image_input: bool = False
    has_image_pos_emb: bool = False
    has_ref_conv: bool = False
    require_vae_embedding: bool = True
    require_clip_embedding: bool = True
    seperated_timestep: bool = False
    fuse_vae_embedding_in_latents: bool = False


PRESETS = {
    # Official Wan2.1-T2V-1.3B config.json.
    "t2v-1.3b": WanConfig(
        dim=1536,
        in_dim=16,
        ffn_dim=8960,
        out_dim=16,
        text_dim=4096,
        freq_dim=256,
        eps=1e-6,
        patch_size=(1, 2, 2),
        num_heads=12,
        num_layers=30,
        has_image_input=False,
    ),
    # Official Wan2.1-T2V-14B dimensions. Kept as a preset for checkpoint load
    # and inference; practical training should use TP/FSDP work in a follow-up.
    "t2v-14b": WanConfig(
        dim=5120,
        in_dim=16,
        ffn_dim=13824,
        out_dim=16,
        text_dim=4096,
        freq_dim=256,
        eps=1e-6,
        patch_size=(1, 2, 2),
        num_heads=40,
        num_layers=40,
        has_image_input=False,
    ),
    # DiffSynth/official Wan2.2-TI2V-5B DiT. This model uses the Wan2.2 VAE
    # latent space (48 channels, spatial factor 16) and fuses the encoded
    # first frame into latents for image-to-video conditioning.
    "ti2v-5b": WanConfig(
        dim=3072,
        in_dim=48,
        ffn_dim=14336,
        out_dim=48,
        text_dim=4096,
        freq_dim=256,
        eps=1e-6,
        patch_size=(1, 2, 2),
        num_heads=24,
        num_layers=30,
        has_image_input=False,
        require_vae_embedding=False,
        require_clip_embedding=False,
        seperated_timestep=True,
        fuse_vae_embedding_in_latents=True,
    ),
    # Small but shape-compatible enough to test the full training/inference
    # logic quickly. Not checkpoint-compatible with official weights.
    "tiny": WanConfig(
        dim=128,
        in_dim=4,
        ffn_dim=512,
        out_dim=4,
        text_dim=64,
        freq_dim=64,
        eps=1e-6,
        patch_size=(1, 2, 2),
        num_heads=4,
        num_layers=4,
        has_image_input=False,
    ),
}


def _parse_patch_size(value: str, default: Tuple[int, int, int]) -> Tuple[int, int, int]:
    if not value:
        return default
    parts = tuple(int(x.strip()) for x in value.split(","))
    if len(parts) != 3:
        raise ValueError(f"--wan-patch-size expects three comma-separated ints, got {value!r}")
    return parts


def wan_config_from_args(args) -> WanConfig:
    """Build WanConfig from Megatron args."""
    base = PRESETS[getattr(args, "wan_preset", "t2v-1.3b")]
    cfg = WanConfig(**base.__dict__)

    overrides = {
        "dim": getattr(args, "wan_dim", 0),
        "in_dim": getattr(args, "wan_in_dim", 0),
        "out_dim": getattr(args, "wan_out_dim", 0),
        "text_dim": getattr(args, "wan_text_dim", 0),
        "ffn_dim": getattr(args, "wan_ffn_dim", 0),
        "freq_dim": getattr(args, "wan_freq_dim", 0),
        "num_heads": getattr(args, "wan_num_heads", 0),
        "num_layers": getattr(args, "wan_num_layers", 0),
    }
    for key, value in overrides.items():
        if value:
            setattr(cfg, key, value)
    cfg.patch_size = _parse_patch_size(getattr(args, "wan_patch_size", ""), cfg.patch_size)
    cfg.eps = getattr(args, "wan_eps", cfg.eps)
    if getattr(args, "wan_has_image_input", False):
        cfg.has_image_input = True
    if getattr(args, "wan_seperated_timestep", False):
        cfg.seperated_timestep = True
    if getattr(args, "wan_fuse_vae_embedding_in_latents", False):
        cfg.fuse_vae_embedding_in_latents = True
    return cfg
