"""Build the deterministic Wan2.2 -> ActionDiT Fast-WAM initialization DCP."""

from __future__ import annotations

import argparse
import gc
import json
import math
import random
from dataclasses import replace
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.distributed as dist
import torch.nn.functional as F

from ..checkpoint import save_megatron_dcp
from ..config import FastWAMConfig
from ..distributed import initialize, transformer_config
from ..model import ActionExpert, FastWAMModel
from ..mcore import _tp_info


def _checkpoint_files(path: str | Path) -> list[Path]:
    root = Path(path).expanduser().resolve()
    if root.is_file():
        return [root]
    files = sorted(root.glob("diffusion_pytorch_model-*.safetensors"))
    if not files:
        direct = root / "diffusion_pytorch_model.safetensors"
        if direct.is_file():
            files = [direct]
    if not files:
        raise FileNotFoundError(f"No Wan DiT safetensors found under {root}")
    return files


class _SourceIndex:
    def __init__(self, path: str | Path):
        try:
            from safetensors import safe_open
        except ImportError as exc:  # pragma: no cover
            raise RuntimeError("Fast-WAM initialization requires safetensors") from exc
        self.files = _checkpoint_files(path)
        self._locations: dict[str, Path] = {}
        for file in self.files:
            with safe_open(file, framework="pt", device="cpu") as handle:
                for key in handle.keys():
                    if key in self._locations:
                        raise ValueError(f"Duplicate checkpoint key {key}")
                    self._locations[key] = file

    def get(self, key: str) -> torch.Tensor:
        try:
            from safetensors import safe_open
        except ImportError as exc:  # pragma: no cover
            raise RuntimeError("Fast-WAM initialization requires safetensors") from exc
        file = self._locations.get(key)
        if file is None:
            raise KeyError(key)
        with safe_open(file, framework="pt", device="cpu") as handle:
            return handle.get_tensor(key)

    def shape(self, key: str) -> tuple[int, ...]:
        try:
            from safetensors import safe_open
        except ImportError as exc:  # pragma: no cover
            raise RuntimeError("Fast-WAM initialization requires safetensors") from exc
        file = self._locations.get(key)
        if file is None:
            raise KeyError(key)
        with safe_open(file, framework="pt", device="cpu") as handle:
            return tuple(handle.get_slice(key).get_shape())


def _source_name(target_name: str) -> str:
    return target_name.replace(".linear.weight", ".weight").replace(
        ".linear.bias", ".bias"
    )


def _linear_module(
    target_name: str,
    modules: dict[str, torch.nn.Module],
) -> torch.nn.Module | None:
    marker = ".linear."
    if marker not in target_name:
        return None
    module_name = target_name.split(marker, 1)[0]
    module = modules.get(module_name)
    if module is None or not hasattr(module, "parallel"):
        raise KeyError(f"Cannot resolve parallel wrapper for {target_name}")
    return module


def _full_target_shape(
    target_name: str,
    target: torch.Tensor,
    modules: dict[str, torch.nn.Module],
    tp_size: int,
) -> tuple[int, ...]:
    shape = list(target.shape)
    module = _linear_module(target_name, modules)
    if module is None or tp_size == 1:
        return tuple(shape)
    parallel = getattr(module, "parallel")
    if parallel == "column":
        shape[0] *= tp_size
    elif parallel == "row" and target_name.endswith(".weight"):
        shape[1] *= tp_size
    elif parallel not in {"row", "replicated"}:
        raise ValueError(f"{target_name}: unsupported parallel mode {parallel!r}")
    return tuple(shape)


def _local_target(
    target_name: str,
    full: torch.Tensor,
    target: torch.Tensor,
    modules: dict[str, torch.nn.Module],
    tp_rank: int,
    tp_size: int,
) -> torch.Tensor:
    if tuple(full.shape) == tuple(target.shape):
        return full
    module = _linear_module(target_name, modules)
    if module is None:
        raise ValueError(
            f"{target_name}: replicated parameter shape {tuple(target.shape)} "
            f"does not match full tensor {tuple(full.shape)}"
        )
    parallel = getattr(module, "parallel")
    if parallel == "column":
        local = full.chunk(tp_size, dim=0)[tp_rank]
    elif parallel == "row" and target_name.endswith(".weight"):
        local = full.chunk(tp_size, dim=1)[tp_rank]
    else:
        local = full
    if tuple(local.shape) != tuple(target.shape):
        raise ValueError(
            f"{target_name}: local shape {tuple(local.shape)} != target {tuple(target.shape)}"
        )
    return local.contiguous()


def _interpolate_last_dimension(tensor: torch.Tensor, size: int) -> torch.Tensor:
    if tensor.shape[-1] == size:
        return tensor
    flat = tensor.reshape(-1, 1, tensor.shape[-1]).float()
    flat = F.interpolate(flat, size=size, mode="linear", align_corners=True)
    return flat.reshape(*tensor.shape[:-1], size)


def resize_action_backbone_tensor(
    source: torch.Tensor,
    target_shape: tuple[int, ...],
    *,
    alpha_scaling: bool = True,
) -> torch.Tensor:
    """Exact current Fast-WAM `preprocess_action_dit_backbone.py` transform."""

    if tuple(source.shape) == target_shape:
        return source
    output = source.float()
    while output.ndim < len(target_shape):
        output = output.unsqueeze(0)
    while output.ndim > len(target_shape):
        if output.shape[0] != 1:
            raise ValueError(
                f"Cannot reduce source rank {tuple(source.shape)} to {target_shape}"
            )
        output = output.squeeze(0)
    for dimension, size in enumerate(target_shape):
        if output.shape[dimension] == size:
            continue
        permutation = [
            index for index in range(output.ndim) if index != dimension
        ] + [dimension]
        inverse = [0] * output.ndim
        for index, value in enumerate(permutation):
            inverse[value] = index
        permuted = output.permute(*permutation).contiguous()
        permuted = _interpolate_last_dimension(permuted, size)
        output = permuted.permute(*inverse).contiguous()
    if tuple(output.shape) != target_shape:
        raise ValueError(
            f"Resize {tuple(source.shape)} -> {target_shape} produced {tuple(output.shape)}"
        )
    if (
        alpha_scaling
        and source.ndim >= 2
        and source.shape[-1] != target_shape[-1]
    ):
        output.mul_(math.sqrt(float(source.shape[-1]) / float(target_shape[-1])))
    return output.to(source.dtype)


@torch.no_grad()
def initialize_from_wan(
    model: FastWAMModel,
    wan_checkpoint: str | Path,
) -> dict[str, int]:
    """Stream VideoDiT weights and synthesize the ActionDiT backbone."""

    source = _SourceIndex(wan_checkpoint)
    _, tp_size, tp_rank = _tp_info()
    counts = {
        "video_copied": 0,
        "action_copied": 0,
        "action_interpolated": 0,
        "action_random": 0,
    }

    def load_expert(expert: torch.nn.Module, *, synthesize: bool) -> None:
        parameters = dict(expert.named_parameters())
        modules = dict(expert.named_modules())
        missing: list[str] = []
        for target_name, target in parameters.items():
            if synthesize and target_name.startswith(("action_encoder.", "head.")):
                counts["action_random"] += 1
                continue
            source_name = _source_name(target_name)
            try:
                source_shape = source.shape(source_name)
            except KeyError:
                missing.append(source_name)
                continue
            full_shape = _full_target_shape(
                target_name,
                target,
                modules,
                tp_size,
            )
            value = source.get(source_name)
            if synthesize and source_shape != full_shape:
                value = resize_action_backbone_tensor(value, full_shape)
                counts["action_interpolated"] += 1
            elif source_shape != full_shape:
                raise ValueError(
                    f"{source_name}: Wan shape {source_shape} != VideoDiT shape {full_shape}"
                )
            elif synthesize:
                counts["action_copied"] += 1
            else:
                counts["video_copied"] += 1
            local = _local_target(
                target_name,
                value,
                target,
                modules,
                tp_rank,
                tp_size,
            )
            target.copy_(local.to(device=target.device, dtype=target.dtype))
            del value, local
        if missing:
            raise RuntimeError(
                f"Missing {len(missing)} Wan keys for {'ActionDiT' if synthesize else 'VideoDiT'}: "
                f"{missing[:16]}"
            )

    load_expert(model.video_expert, synthesize=False)
    load_expert(model.action_expert, synthesize=True)
    return counts


@torch.no_grad()
def initialize_random_io_modules(
    model: FastWAMModel,
    *,
    seed: int,
) -> int:
    """Use dense `nn.Linear` initialization for modules absent from Wan.

    Megatron's TP linears normally use Xavier initialization, while upstream
    ActionDiT uses PyTorch `nn.Linear.reset_parameters` (Kaiming-uniform).
    Constructing one dense ActionExpert also advances RNG through the
    interpolated backbone before creating its final head, preserving upstream
    module-construction order.  The dense temporary is released immediately.
    """

    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(seed)
        reference_action = ActionExpert(
            model.fast_wam_config.action,
            megatron_config=None,
        )
        reference_proprio = torch.nn.Linear(
            model.fast_wam_config.proprio_dim,
            model.fast_wam_config.video.text_dim,
        )

    _, tp_size, tp_rank = _tp_info()
    copied = 0
    for target_expert, reference_expert, prefixes in (
        (
            model.action_expert,
            reference_action,
            ("action_encoder.", "head."),
        ),
    ):
        targets = dict(target_expert.named_parameters())
        modules = dict(target_expert.named_modules())
        references = dict(reference_expert.named_parameters())
        for target_name, target in targets.items():
            if not target_name.startswith(prefixes):
                continue
            reference = references[target_name]
            local = _local_target(
                target_name,
                reference,
                target,
                modules,
                tp_rank,
                tp_size,
            )
            target.copy_(local.to(device=target.device, dtype=target.dtype))
            copied += 1

    for target_name, target in model.proprio_encoder.named_parameters():
        reference_name = target_name.removeprefix("linear.")
        reference = dict(reference_proprio.named_parameters())[reference_name]
        local = (
            reference.chunk(tp_size, dim=0)[tp_rank].contiguous()
            if tuple(reference.shape) != tuple(target.shape)
            else reference
        )
        if tuple(local.shape) != tuple(target.shape):
            raise ValueError(
                f"proprio_encoder.{target_name}: {tuple(local.shape)} != {tuple(target.shape)}"
            )
        target.copy_(local.to(device=target.device, dtype=target.dtype))
        copied += 1
    del reference_action, reference_proprio
    gc.collect()
    return copied


def _seed_before_model(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--wan-checkpoint", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--tp", type=int, default=1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--dtype", choices=("bfloat16", "float32"), default="bfloat16")
    parser.add_argument("--action-dim", type=int, default=7)
    parser.add_argument("--proprio-dim", type=int, default=8)
    args = parser.parse_args()
    dtype = torch.bfloat16 if args.dtype == "bfloat16" else torch.float32
    parallel = initialize(args.tp)
    _seed_before_model(args.seed)
    base_config = FastWAMConfig()
    config = replace(
        base_config,
        action=replace(base_config.action, action_dim=args.action_dim),
        proprio_dim=args.proprio_dim,
    )
    model = FastWAMModel(
        config,
        transformer_config(config, args.tp, dtype),
    ).to(device=torch.cuda.current_device(), dtype=dtype)
    counts = initialize_from_wan(model, args.wan_checkpoint)
    counts["official_random_io_tensors"] = initialize_random_io_modules(
        model,
        seed=args.seed,
    )
    output = Path(args.output).expanduser().resolve()
    save_megatron_dcp(model, output)
    if dist.get_rank() == 0:
        manifest = {
            "format": "megatron-dcp",
            "stage": "wan2.2-video-dit-plus-interpolated-action-dit",
            "wan_checkpoint": str(Path(args.wan_checkpoint).expanduser().resolve()),
            "seed_before_model_construction": args.seed,
            "random_io_initialization": (
                "dense ActionExpert then proprio nn.Linear under a seed-42 CPU RNG scope; "
                "copy action_encoder/head/proprio_encoder into TP shards"
            ),
            "dtype": args.dtype,
            "tp_at_save": args.tp,
            "action_dim": args.action_dim,
            "proprio_dim": args.proprio_dim,
            "action_backbone_policy": {
                "skip_prefixes": ["action_encoder.", "head."],
                "interpolation": "sequential_1d_linear_align_corners_true",
                "alpha_scaling": True,
            },
            "counts": counts,
        }
        (output / "fast_wam_initialization.json").write_text(
            json.dumps(manifest, indent=2),
            encoding="utf-8",
        )
        print(json.dumps(manifest, indent=2), flush=True)
    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
