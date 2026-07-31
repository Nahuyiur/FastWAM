#!/usr/bin/env python3
"""Prepare a real Wan VAE/UMT5 pre-encoded sample via DiffSynth.

This script is the production counterpart to `prepare_overfit_sample.py`.
It uses DiffSynth's Wan VAE and UMT5 text encoder to create the same `.pt`
schema consumed by `wan/pretrain.py`:

    input_latents: [C, F, H/vae_factor, W/vae_factor]
    context: [L, 4096]

It never downloads model assets implicitly. Pass explicit asset paths or set
the WAN_* env vars; otherwise the script reports a blocker and exits with 2.
"""

from __future__ import annotations

import argparse
import gc
import importlib.util
import os
import subprocess
import sys
import types
from pathlib import Path

import torch

DEFAULT_DIFFSYNTH_ROOT = "/aifs4su/mmcode/codeclm/DiffSynth-Studio"
DEFAULT_SEARCH_ROOTS = (
    "/aifs4su/mmcode/codeclm/DiffSynth-Studio/models",
    "/aifs4su/mmcode/codeclm/checkpoints/hf_models",
    "/aifs4su/mmcode/codeclm/checkpoints",
    "/aifs4su/mmcode/codeclm/.cache/huggingface/hub",
    "/root/.cache/huggingface/hub",
)


def _path_is_dir(path: Path) -> bool:
    try:
        return path.is_dir()
    except OSError:
        return False


def _path_is_file(path: Path) -> bool:
    try:
        return path.is_file()
    except OSError:
        return False


def _first_existing_file(roots: list[Path], names: tuple[str, ...]) -> str | None:
    for root in roots:
        if not _path_is_dir(root):
            continue
        for name in names:
            direct = root / name
            if _path_is_file(direct):
                return str(direct)
            try:
                for found in root.rglob(name):
                    if _path_is_file(found):
                        return str(found)
            except OSError:
                continue
    return None


def _first_tokenizer_dir(roots: list[Path]) -> str | None:
    def probably_umt5(path: Path) -> bool:
        lowered = str(path).lower()
        return "umt5" in lowered or "google" in lowered

    for root in roots:
        if not _path_is_dir(root):
            continue
        if probably_umt5(root) and (_path_is_file(root / "spiece.model") or _path_is_file(root / "tokenizer.json")):
            return str(root)
        try:
            for found in root.rglob("spiece.model"):
                if _path_is_file(found) and probably_umt5(found.parent):
                    return str(found.parent)
            for found in root.rglob("tokenizer.json"):
                if _path_is_file(found) and probably_umt5(found.parent):
                    return str(found.parent)
        except OSError:
            continue
    return None


def _resolve_assets(args):
    roots = [Path(p) for p in args.search_root]
    wan_version = getattr(args, "wan_version", "2.1")
    vae = args.vae_ckpt or os.environ.get("WAN_VAE_CKPT")
    text_encoder = args.text_encoder_ckpt or os.environ.get("WAN_TEXT_ENCODER_CKPT")
    tokenizer = args.tokenizer_path or os.environ.get("WAN_TOKENIZER_PATH")

    if vae is None and args.auto_search:
        if wan_version == "2.2":
            vae_names = ("Wan2.2_VAE.safetensors", "Wan2.2_VAE.pth")
        else:
            vae_names = ("Wan2.1_VAE.safetensors", "Wan2.1_VAE.pth")
        vae = _first_existing_file(roots, vae_names)
    if text_encoder is None and args.auto_search:
        text_encoder = _first_existing_file(
            roots,
            (
                "models_t5_umt5-xxl-enc-bf16.safetensors",
                "models_t5_umt5-xxl-enc-bf16.pth",
            ),
        )
    if tokenizer is None and args.auto_search:
        tokenizer = _first_tokenizer_dir(roots)
    return vae, text_encoder, tokenizer


def _blocker(message: str) -> int:
    print(f"BLOCKER: {message}", file=sys.stderr)
    return 2


def _read_video_frames(video: str, width: int, height: int, num_frames: int, fps: float):
    import numpy as np
    from PIL import Image

    cmd = [
        os.environ.get("FFMPEG_BIN", "ffmpeg"),
        "-nostdin",
        "-hide_banner",
        "-loglevel",
        "error",
        "-i",
        video,
        "-vf",
        f"fps={fps},scale={width}:{height}:force_original_aspect_ratio=increase,crop={width}:{height}",
        "-frames:v",
        str(num_frames),
        "-f",
        "rawvideo",
        "-pix_fmt",
        "rgb24",
        "-",
    ]
    raw = subprocess.check_output(cmd)
    expected = num_frames * height * width * 3
    arr = np.frombuffer(raw, dtype=np.uint8)
    if arr.size < expected:
        raise RuntimeError(f"ffmpeg returned {arr.size} bytes, expected at least {expected}")
    arr = arr[:expected].reshape(num_frames, height, width, 3)
    return [Image.fromarray(frame, mode="RGB") for frame in arr]


def _load_module(module_name: str, path: Path):
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot import {module_name} from {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _load_diffsynth_components(root: str):
    root_path = Path(root)
    if not root_path.is_dir():
        raise FileNotFoundError(f"DiffSynth root not found: {root}")
    try:
        import ftfy  # noqa: F401
    except ModuleNotFoundError:
        ftfy_stub = types.ModuleType("ftfy")
        ftfy_stub.fix_text = lambda text: text
        sys.modules["ftfy"] = ftfy_stub
    try:
        import regex  # noqa: F401
    except ModuleNotFoundError:
        import re

        sys.modules["regex"] = re
    models_dir = root_path / "diffsynth" / "models"
    utils_dir = root_path / "diffsynth" / "utils" / "state_dict_converters"
    text_mod = _load_module("diffsynth_wan_video_text_encoder_local", models_dir / "wan_video_text_encoder.py")
    vae_mod = _load_module("diffsynth_wan_video_vae_local", models_dir / "wan_video_vae.py")
    vae_converter_mod = _load_module("diffsynth_wan_video_vae_converter_local", utils_dir / "wan_video_vae.py")
    return (
        text_mod.HuggingfaceTokenizer,
        text_mod.WanTextEncoder,
        vae_mod.WanVideoVAE,
        vae_mod.WanVideoVAE38,
        vae_converter_mod.WanVideoVAEStateDictConverter,
    )


def _load_state_dict(path: str):
    path_obj = Path(path)
    if path_obj.suffix == ".safetensors":
        from safetensors.torch import load_file

        return load_file(str(path_obj), device="cpu")
    state = torch.load(path_obj, map_location="cpu", weights_only=False)
    if isinstance(state, dict):
        for key in ("state_dict", "model", "module"):
            if key in state and isinstance(state[key], dict):
                return state[key]
    return state


def _preprocess_video(frames, dtype, device):
    import numpy as np

    tensors = []
    for image in frames:
        frame = torch.tensor(np.array(image, dtype=np.float32), dtype=dtype, device=device)
        frame = frame * (2.0 / 255.0) - 1.0
        tensors.append(frame.permute(2, 0, 1))
    return torch.stack(tensors, dim=1).unsqueeze(0)


@torch.no_grad()
def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--video", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--prompt", default="a short real-world video clip")
    parser.add_argument("--diffsynth-root", default=os.environ.get("DIFFSYNTH_ROOT", DEFAULT_DIFFSYNTH_ROOT))
    parser.add_argument("--vae-ckpt", default=None)
    parser.add_argument("--text-encoder-ckpt", default=None)
    parser.add_argument("--tokenizer-path", default=None)
    parser.add_argument("--wan-version", default="2.1", choices=["2.1", "2.2"])
    parser.add_argument("--search-root", action="append", default=list(DEFAULT_SEARCH_ROOTS))
    parser.add_argument("--auto-search", action="store_true")
    parser.add_argument("--check-only", action="store_true")
    parser.add_argument("--height", type=int, default=128)
    parser.add_argument("--width", type=int, default=128)
    parser.add_argument("--num-frames", type=int, default=17)
    parser.add_argument("--fps", type=float, default=4.0)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--dtype", default="bf16", choices=["fp32", "fp16", "bf16"])
    parser.add_argument("--tiled", action="store_true")
    parser.add_argument("--tile-size", type=int, nargs=2, default=(30, 52))
    parser.add_argument("--tile-stride", type=int, nargs=2, default=(15, 26))
    parser.add_argument(
        "--fuse-first-frame",
        action="store_true",
        help="Store input_latents[:, :1] as first_frame_latents for Wan2.2 TI2V conditioning.",
    )
    args = parser.parse_args()

    vae, text_encoder, tokenizer = _resolve_assets(args)
    missing = []
    for label, value in (("WAN_VAE_CKPT", vae), ("WAN_TEXT_ENCODER_CKPT", text_encoder), ("WAN_TOKENIZER_PATH", tokenizer)):
        if not value:
            missing.append(label)
        elif label == "WAN_TOKENIZER_PATH":
            if not _path_is_dir(Path(value)):
                missing.append(f"{label} (not a directory: {value})")
        elif not _path_is_file(Path(value)):
            missing.append(f"{label} (not a file: {value})")
    if missing:
        roots = ", ".join(args.search_root)
        return _blocker(
            "missing official Wan preprocessing assets: "
            + ", ".join(missing)
            + f". Searched only when --auto-search is set. Search roots: {roots}"
        )

    if args.num_frames % 4 != 1:
        return _blocker("--num-frames must be 4k+1 for Wan VAE temporal stride")
    vae_factor = 16 if args.wan_version == "2.2" else 8
    spatial_divisor = 32 if args.wan_version == "2.2" else 16
    if args.height % spatial_divisor != 0 or args.width % spatial_divisor != 0:
        return _blocker(f"--height/--width must be divisible by {spatial_divisor} for Wan{args.wan_version}")

    print(f"DiffSynth root: {args.diffsynth_root}")
    print(f"VAE: {vae}")
    print(f"Text encoder: {text_encoder}")
    print(f"Tokenizer: {tokenizer}")
    if args.check_only:
        print("asset check passed")
        return 0

    dtype = {"fp32": torch.float32, "fp16": torch.float16, "bf16": torch.bfloat16}[args.dtype]
    HuggingfaceTokenizer, WanTextEncoder, WanVideoVAE, WanVideoVAE38, vae_state_dict_converter = _load_diffsynth_components(
        args.diffsynth_root
    )
    VAEClass = WanVideoVAE38 if args.wan_version == "2.2" else WanVideoVAE

    tokenizer_model = HuggingfaceTokenizer(name=tokenizer, seq_len=512, clean="whitespace")
    ids, mask = tokenizer_model(args.prompt, return_mask=True, add_special_tokens=True)
    ids = ids.to(args.device)
    mask = mask.to(args.device)
    text_model = WanTextEncoder().eval().requires_grad_(False).to(device=args.device, dtype=dtype)
    text_state = _load_state_dict(text_encoder)
    text_model.load_state_dict(text_state, strict=True)
    del text_state
    seq_lens = mask.gt(0).sum(dim=1).long()
    context = text_model(ids, mask).to(dtype=dtype, device="cpu")
    del text_model
    gc.collect()
    if args.device.startswith("cuda") and torch.cuda.is_available():
        torch.cuda.empty_cache()

    vae_model = VAEClass().eval().requires_grad_(False).to(device=args.device, dtype=dtype)
    vae_state = _load_state_dict(vae)
    vae_model.load_state_dict(vae_state_dict_converter(vae_state), strict=True)
    del vae_state
    gc.collect()

    frames = _read_video_frames(args.video, args.width, args.height, args.num_frames, args.fps)
    video = _preprocess_video(frames, dtype=dtype, device=args.device)
    input_latents = vae_model.encode(
        video,
        device=args.device,
        tiled=args.tiled,
        tile_size=tuple(args.tile_size),
        tile_stride=tuple(args.tile_stride),
    ).to(dtype=dtype, device="cpu")
    del vae_model
    if args.device.startswith("cuda") and torch.cuda.is_available():
        torch.cuda.empty_cache()

    for i, length in enumerate(seq_lens.tolist()):
        context[i, length:] = 0

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    sample = {
        "input_latents": input_latents[0].contiguous(),
        "context": context[0].contiguous(),
        "prompt": args.prompt,
        "video_path": str(Path(args.video).resolve()),
        "schema": "wan_overfit_preencoded_v1",
        "preprocess": {
            "source": "diffsynth",
            "diffsynth_root": str(Path(args.diffsynth_root).resolve()),
            "wan_version": args.wan_version,
            "vae": vae,
            "vae_factor": vae_factor,
            "text_encoder": text_encoder,
            "tokenizer": tokenizer,
            "height": args.height,
            "width": args.width,
            "num_frames": args.num_frames,
            "fps": args.fps,
        },
    }
    if args.fuse_first_frame:
        sample["first_frame_latents"] = input_latents[0, :, 0:1].contiguous()
        sample["fuse_vae_embedding_in_latents"] = True
    torch.save(sample, output)
    print(f"saved {output}")
    print(f"latents={tuple(input_latents[0].shape)} context={tuple(context[0].shape)} video={args.video}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
