"""Official/DiffSynth Wan checkpoint loading helpers."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Iterable, List

import torch
import torch.distributed as dist

from wan.model.config import PRESETS, WanConfig

try:
    from megatron.core import parallel_state
except Exception:  # pragma: no cover
    parallel_state = None


def _strip_prefix(state: Dict[str, torch.Tensor], prefixes: Iterable[str]):
    out = {}
    for key, value in state.items():
        new_key = key
        changed = True
        while changed:
            changed = False
            for prefix in prefixes:
                if new_key.startswith(prefix):
                    new_key = new_key[len(prefix) :]
                    changed = True
        out[new_key] = value
    return out


_DIFFUSERS_RENAME_TEMPLATE = {
    "blocks.0.attn1.norm_k.weight": "blocks.0.self_attn.norm_k.weight",
    "blocks.0.attn1.norm_q.weight": "blocks.0.self_attn.norm_q.weight",
    "blocks.0.attn1.to_k.bias": "blocks.0.self_attn.k.bias",
    "blocks.0.attn1.to_k.weight": "blocks.0.self_attn.k.weight",
    "blocks.0.attn1.to_out.0.bias": "blocks.0.self_attn.o.bias",
    "blocks.0.attn1.to_out.0.weight": "blocks.0.self_attn.o.weight",
    "blocks.0.attn1.to_q.bias": "blocks.0.self_attn.q.bias",
    "blocks.0.attn1.to_q.weight": "blocks.0.self_attn.q.weight",
    "blocks.0.attn1.to_v.bias": "blocks.0.self_attn.v.bias",
    "blocks.0.attn1.to_v.weight": "blocks.0.self_attn.v.weight",
    "blocks.0.attn2.norm_k.weight": "blocks.0.cross_attn.norm_k.weight",
    "blocks.0.attn2.norm_q.weight": "blocks.0.cross_attn.norm_q.weight",
    "blocks.0.attn2.to_k.bias": "blocks.0.cross_attn.k.bias",
    "blocks.0.attn2.to_k.weight": "blocks.0.cross_attn.k.weight",
    "blocks.0.attn2.to_out.0.bias": "blocks.0.cross_attn.o.bias",
    "blocks.0.attn2.to_out.0.weight": "blocks.0.cross_attn.o.weight",
    "blocks.0.attn2.to_q.bias": "blocks.0.cross_attn.q.bias",
    "blocks.0.attn2.to_q.weight": "blocks.0.cross_attn.q.weight",
    "blocks.0.attn2.to_v.bias": "blocks.0.cross_attn.v.bias",
    "blocks.0.attn2.to_v.weight": "blocks.0.cross_attn.v.weight",
    "blocks.0.attn2.add_k_proj.bias": "blocks.0.cross_attn.k_img.bias",
    "blocks.0.attn2.add_k_proj.weight": "blocks.0.cross_attn.k_img.weight",
    "blocks.0.attn2.add_v_proj.bias": "blocks.0.cross_attn.v_img.bias",
    "blocks.0.attn2.add_v_proj.weight": "blocks.0.cross_attn.v_img.weight",
    "blocks.0.attn2.norm_added_k.weight": "blocks.0.cross_attn.norm_k_img.weight",
    "blocks.0.ffn.net.0.proj.bias": "blocks.0.ffn.0.bias",
    "blocks.0.ffn.net.0.proj.weight": "blocks.0.ffn.0.weight",
    "blocks.0.ffn.net.2.bias": "blocks.0.ffn.2.bias",
    "blocks.0.ffn.net.2.weight": "blocks.0.ffn.2.weight",
    "blocks.0.norm2.bias": "blocks.0.norm3.bias",
    "blocks.0.norm2.weight": "blocks.0.norm3.weight",
    "blocks.0.scale_shift_table": "blocks.0.modulation",
    "condition_embedder.text_embedder.linear_1.bias": "text_embedding.0.bias",
    "condition_embedder.text_embedder.linear_1.weight": "text_embedding.0.weight",
    "condition_embedder.text_embedder.linear_2.bias": "text_embedding.2.bias",
    "condition_embedder.text_embedder.linear_2.weight": "text_embedding.2.weight",
    "condition_embedder.time_embedder.linear_1.bias": "time_embedding.0.bias",
    "condition_embedder.time_embedder.linear_1.weight": "time_embedding.0.weight",
    "condition_embedder.time_embedder.linear_2.bias": "time_embedding.2.bias",
    "condition_embedder.time_embedder.linear_2.weight": "time_embedding.2.weight",
    "condition_embedder.time_proj.bias": "time_projection.1.bias",
    "condition_embedder.time_proj.weight": "time_projection.1.weight",
    "condition_embedder.image_embedder.ff.net.0.proj.bias": "img_emb.proj.1.bias",
    "condition_embedder.image_embedder.ff.net.0.proj.weight": "img_emb.proj.1.weight",
    "condition_embedder.image_embedder.ff.net.2.bias": "img_emb.proj.3.bias",
    "condition_embedder.image_embedder.ff.net.2.weight": "img_emb.proj.3.weight",
    "condition_embedder.image_embedder.norm1.bias": "img_emb.proj.0.bias",
    "condition_embedder.image_embedder.norm1.weight": "img_emb.proj.0.weight",
    "condition_embedder.image_embedder.norm2.bias": "img_emb.proj.4.bias",
    "condition_embedder.image_embedder.norm2.weight": "img_emb.proj.4.weight",
    "patch_embedding.bias": "patch_embedding.bias",
    "patch_embedding.weight": "patch_embedding.weight",
    "scale_shift_table": "head.modulation",
    "proj_out.bias": "head.head.bias",
    "proj_out.weight": "head.head.weight",
}


def convert_diffusers_wan_state_dict(state: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    """Convert Diffusers-style Wan DiT keys into DiffSynth/WanModel keys.

    This mirrors DiffSynth's `WanVideoDiTFromDiffusers` converter. It is kept
    local so official Diffusers-format shards can be loaded without importing
    DiffSynth in the Megatron training process.
    """
    converted = {}
    for name, tensor in state.items():
        if name in _DIFFUSERS_RENAME_TEMPLATE:
            converted[_DIFFUSERS_RENAME_TEMPLATE[name]] = tensor
            continue

        parts = name.split(".")
        if len(parts) >= 3 and parts[0] == "blocks" and parts[1].isdigit():
            template_name = ".".join([parts[0], "0", *parts[2:]])
            target_template = _DIFFUSERS_RENAME_TEMPLATE.get(template_name)
            if target_template is not None:
                target_parts = target_template.split(".")
                converted[".".join([target_parts[0], parts[1], *target_parts[2:]])] = tensor
                continue
        converted[name] = tensor
    return converted


def _looks_like_diffusers_wan(state: Dict[str, torch.Tensor]) -> bool:
    markers = (
        "condition_embedder.",
        "proj_out.",
        "scale_shift_table",
    )
    for key in state:
        if any(key.startswith(marker) for marker in markers):
            return True
        if ".attn1." in key or ".attn2." in key:
            return True
    return False


def _maybe_convert_state(state: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    state = _strip_prefix(
        state,
        prefixes=(
            "pipe.dit.",
            "dit.",
            "model.",
            "module.",
            "diffusion_model.",
            "model.diffusion_model.",
            "transformer.",
        ),
    )
    if _looks_like_diffusers_wan(state):
        state = convert_diffusers_wan_state_dict(state)
    return state


def _is_probably_dit_file(path: Path) -> bool:
    name = path.name.lower()
    blocked = ("vae", "t5", "umt5", "clip", "text_encoder", "image_encoder", "tokenizer")
    if any(token in name for token in blocked):
        return False
    preferred = ("diffusion_pytorch_model", "wan", "dit", "model")
    return any(token in name for token in preferred)


def _checkpoint_files(path: Path) -> List[Path]:
    search_dirs = [path, path / "transformer", path / "dit", path / "model"]
    patterns = (
        "diffusion_pytorch_model*.safetensors",
        "*dit*.safetensors",
        "model*.safetensors",
        "*.safetensors",
        "*dit*.pth",
        "model*.pth",
        "*.pth",
        "*dit*.pt",
        "model*.pt",
        "*.pt",
        "*.bin",
    )
    for directory in search_dirs:
        if not directory.is_dir():
            continue
        for pattern in patterns:
            files = [file for file in sorted(directory.glob(pattern)) if _is_probably_dit_file(file)]
            if files:
                return files
    return []


def load_torch_or_safetensors(path: str | Path) -> Dict[str, torch.Tensor]:
    """Load one checkpoint file.

    Supports `.pt`, `.pth`, `.bin`, and `.safetensors`. Safetensors is optional
    and imported only when needed.
    """
    path = Path(path)
    if path.suffix == ".safetensors":
        try:
            from safetensors.torch import load_file
        except Exception as exc:
            raise RuntimeError(
                f"Loading {path} requires safetensors in the active Python env"
            ) from exc
        return load_file(str(path), device="cpu")

    obj = torch.load(path, map_location="cpu", weights_only=False)
    if isinstance(obj, dict):
        for key in ("state_dict", "model", "module"):
            if key in obj and isinstance(obj[key], dict):
                return obj[key]
    if not isinstance(obj, dict):
        raise TypeError(f"Unsupported checkpoint object in {path}: {type(obj)}")
    return obj


def load_checkpoint_dir(path: str | Path) -> Dict[str, torch.Tensor]:
    """Load all checkpoint shards from a directory or a single file."""
    path = Path(path)
    files = _checkpoint_files_or_file(path)
    state = {}
    for file in files:
        state.update(load_torch_or_safetensors(file))
    return _maybe_convert_state(state)


def _checkpoint_files_or_file(path: str | Path) -> List[Path]:
    path = Path(path)
    if path.is_file():
        return [path]
    if path.is_dir():
        files = _checkpoint_files(path)
        if files:
            return files
        raise FileNotFoundError(f"No Wan DiT checkpoint shards found in {path}")
    else:
        raise FileNotFoundError(path)


def _tp_rank_world():
    if parallel_state is None or not dist.is_available() or not dist.is_initialized():
        return 0, 1
    try:
        group = parallel_state.get_tensor_model_parallel_group()
        return group.rank(), group.size()
    except Exception:
        return 0, 1


def _wrapped_linear_key(key: str, target_state: Dict[str, torch.Tensor]) -> str | None:
    if key in target_state:
        return key
    for suffix in (".weight", ".bias"):
        if key.endswith(suffix):
            candidate = f"{key[:-len(suffix)]}.linear{suffix}"
            if candidate in target_state:
                return candidate
    return None


def _slice_for_target(key: str, full: torch.Tensor, target: torch.Tensor, rank: int, world: int) -> torch.Tensor:
    if tuple(full.shape) == tuple(target.shape):
        return full
    if world == 1:
        raise ValueError(f"{key}: checkpoint shape {tuple(full.shape)} != target shape {tuple(target.shape)}")
    if full.ndim == 2 and target.ndim == 2:
        if full.shape[0] == target.shape[0] * world and full.shape[1] == target.shape[1]:
            return full.chunk(world, dim=0)[rank].contiguous()
        if full.shape[0] == target.shape[0] and full.shape[1] == target.shape[1] * world:
            return full.chunk(world, dim=1)[rank].contiguous()
    if full.ndim == 1 and target.ndim == 1 and full.shape[0] == target.shape[0] * world:
        return full.chunk(world, dim=0)[rank].contiguous()
    raise ValueError(f"{key}: cannot shard checkpoint shape {tuple(full.shape)} to target shape {tuple(target.shape)}")


def adapt_official_state_to_model(
    state: Dict[str, torch.Tensor], target: torch.nn.Module
) -> tuple[Dict[str, torch.Tensor], List[str]]:
    """Rename wrapper keys and slice full official weights for the local TP rank."""
    target_state = target.state_dict()
    rank, world = _tp_rank_world()
    adapted: Dict[str, torch.Tensor] = {}
    unexpected: List[str] = []
    for key, tensor in state.items():
        target_key = _wrapped_linear_key(key, target_state)
        if target_key is None:
            continue
        try:
            adapted[target_key] = _slice_for_target(target_key, tensor, target_state[target_key], rank, world)
        except ValueError as exc:
            unexpected.append(f"{key} ({exc})")
    return adapted, unexpected


def load_official_wan_checkpoint(model: torch.nn.Module, path: str | Path, strict: bool = False):
    """Load official/DiffSynth Wan weights into `WanModel` or `WanFlowTrainingModel`.

    If `model` is the training wrapper, weights are loaded into `model.dit`.
    Returns `(missing, unexpected)` from `load_state_dict`.
    """
    target = getattr(model, "dit", model)
    target_state = target.state_dict()
    loaded_keys = set()
    unexpected = []
    for file in _checkpoint_files_or_file(path):
        state = _maybe_convert_state(load_torch_or_safetensors(file))
        state, shard_unexpected = adapt_official_state_to_model(state, target)
        unexpected.extend(shard_unexpected)
        if state:
            incompatible = target.load_state_dict(state, strict=False)
            unexpected.extend(incompatible.unexpected_keys)
            loaded_keys.update(state)
        del state
    missing = [
        key
        for key in target_state
        if key not in loaded_keys and not key.endswith("_extra_state")
    ]
    if strict and (missing or unexpected):
        raise RuntimeError(
            f"Strict Wan checkpoint load failed: missing={missing[:16]}, unexpected={unexpected[:16]}"
        )
    return missing, unexpected


def config_from_official_dir(path: str | Path, fallback_preset: str = "t2v-1.3b") -> WanConfig:
    """Load WanConfig from official `config.json` if present."""
    path = Path(path)
    config_path = path / "config.json" if path.is_dir() else path.with_name("config.json")
    if not config_path.is_file():
        return PRESETS[fallback_preset]
    with open(config_path) as f:
        raw = json.load(f)
    cfg = PRESETS[fallback_preset]
    data = cfg.__dict__.copy()
    for key in (
        "dim",
        "in_dim",
        "ffn_dim",
        "out_dim",
        "freq_dim",
        "num_heads",
        "num_layers",
        "eps",
        "has_image_input",
        "has_image_pos_emb",
        "has_ref_conv",
        "require_vae_embedding",
        "require_clip_embedding",
        "seperated_timestep",
        "fuse_vae_embedding_in_latents",
    ):
        if key in raw:
            data[key] = raw[key]
    data["text_dim"] = raw.get("text_dim", raw.get("text_len_dim", data["text_dim"]))
    if raw.get("model_type") == "ti2v" or data.get("in_dim") == 48:
        data.setdefault("require_vae_embedding", False)
        data.setdefault("require_clip_embedding", False)
        data["require_vae_embedding"] = False
        data["require_clip_embedding"] = False
        data["seperated_timestep"] = True
        data["fuse_vae_embedding_in_latents"] = True
    # Official config omits patch_size; Wan T2V uses (1, 2, 2).
    if "patch_size" in raw:
        data["patch_size"] = tuple(raw["patch_size"])
    return WanConfig(**data)
