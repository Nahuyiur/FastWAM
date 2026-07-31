#!/usr/bin/env python3
"""Decode Wan latent tensors to an MP4 with the official Wan VAE."""

from __future__ import annotations

import argparse
import gc
import os
import subprocess
import sys
from pathlib import Path

import torch

sys.path.insert(0, os.path.dirname(__file__))

from prepare_diffsynth_sample import (  # noqa: E402
    DEFAULT_DIFFSYNTH_ROOT,
    DEFAULT_SEARCH_ROOTS,
    _load_diffsynth_components,
    _load_state_dict,
    _path_is_file,
    _resolve_assets,
)


def _select_latents(obj, key: str | None):
    if key is not None:
        if key not in obj:
            raise KeyError(f"{key} not found in latent file; available keys={sorted(obj)}")
        return obj[key], key
    for candidate in ("pred_latents", "input_latents", "gt_latents", "latents"):
        value = obj.get(candidate)
        if value is not None:
            return value, candidate
    raise KeyError(f"No latent tensor found; available keys={sorted(obj)}")


def _to_video_uint8(decoded):
    video = decoded.detach().float().cpu().clamp(-1, 1)
    if video.ndim != 5:
        raise ValueError(f"Decoded video must be [B,C,T,H,W], got {tuple(video.shape)}")
    video = video[0]
    video = ((video + 1.0) * 127.5).round().clamp(0, 255).to(torch.uint8)
    return video.permute(1, 2, 3, 0).contiguous().numpy()


def _write_mp4(frames, output: Path, fps: float):
    output.parent.mkdir(parents=True, exist_ok=True)
    height, width = frames.shape[1], frames.shape[2]
    cmd = [
        os.environ.get("FFMPEG_BIN", "ffmpeg"),
        "-nostdin",
        "-hide_banner",
        "-loglevel",
        "error",
        "-y",
        "-f",
        "rawvideo",
        "-pix_fmt",
        "rgb24",
        "-s",
        f"{width}x{height}",
        "-r",
        str(fps),
        "-i",
        "-",
        "-an",
        "-c:v",
        "libx264",
        "-pix_fmt",
        "yuv420p",
        "-crf",
        "18",
        str(output),
    ]
    proc = subprocess.run(cmd, input=frames.tobytes(), check=True)
    return proc.returncode


@torch.no_grad()
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--latents", required=True, help=".pt file containing pred_latents/input_latents/gt_latents")
    parser.add_argument("--output", required=True, help="Output .mp4 path")
    parser.add_argument("--latent-key", default=None)
    parser.add_argument("--diffsynth-root", default=os.environ.get("DIFFSYNTH_ROOT", DEFAULT_DIFFSYNTH_ROOT))
    parser.add_argument("--vae-ckpt", default=None)
    parser.add_argument("--wan-version", default="auto", choices=["auto", "2.1", "2.2"])
    parser.add_argument("--search-root", action="append", default=list(DEFAULT_SEARCH_ROOTS))
    parser.add_argument("--auto-search", action="store_true")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--dtype", default="bf16", choices=["fp32", "fp16", "bf16"])
    parser.add_argument("--fps", type=float, default=8.0)
    parser.add_argument("--tiled", action="store_true")
    parser.add_argument("--tile-size", type=int, nargs=2, default=(30, 52))
    parser.add_argument("--tile-stride", type=int, nargs=2, default=(15, 26))
    args = parser.parse_args()

    class AssetArgs:
        search_root = args.search_root
        auto_search = args.auto_search
        vae_ckpt = args.vae_ckpt
        wan_version = "2.1" if args.wan_version == "auto" else args.wan_version
        text_encoder_ckpt = None
        tokenizer_path = None

    dtype = {"fp32": torch.float32, "fp16": torch.float16, "bf16": torch.bfloat16}[args.dtype]
    obj = torch.load(args.latents, map_location="cpu", weights_only=False)
    if not isinstance(obj, dict):
        raise TypeError(f"Expected {args.latents} to contain a dict, got {type(obj)}")
    latents, latent_key = _select_latents(obj, args.latent_key)
    if latents.ndim == 4:
        latents = latents.unsqueeze(0)
    if latents.ndim != 5:
        raise ValueError(f"{latent_key} must be [C,F,H,W] or [B,C,F,H,W], got {tuple(latents.shape)}")

    wan_version = args.wan_version
    if wan_version == "auto":
        wan_version = "2.2" if latents.shape[1] == 48 else "2.1"
    AssetArgs.wan_version = wan_version
    vae, _, _ = _resolve_assets(AssetArgs)
    if not vae or not _path_is_file(Path(vae)):
        raise FileNotFoundError(f"WAN_VAE_CKPT not found for Wan{wan_version}; resolved value={vae}")

    _, _, WanVideoVAE, WanVideoVAE38, vae_state_dict_converter = _load_diffsynth_components(args.diffsynth_root)
    VAEClass = WanVideoVAE38 if wan_version == "2.2" else WanVideoVAE
    vae_model = VAEClass().eval().requires_grad_(False).to(device=args.device, dtype=dtype)
    state = _load_state_dict(vae)
    vae_model.load_state_dict(vae_state_dict_converter(state), strict=True)
    del state
    gc.collect()

    decoded = vae_model.decode(
        latents.to(device=args.device, dtype=dtype),
        device=args.device,
        tiled=args.tiled,
        tile_size=tuple(args.tile_size),
        tile_stride=tuple(args.tile_stride),
    )
    frames = _to_video_uint8(decoded)
    output = Path(args.output)
    _write_mp4(frames, output, args.fps)
    print(
        f"decoded key={latent_key} wan_version={wan_version} "
        f"latents={tuple(latents.shape)} frames={frames.shape} fps={args.fps}"
    )
    print(f"saved {output}")


if __name__ == "__main__":
    main()
