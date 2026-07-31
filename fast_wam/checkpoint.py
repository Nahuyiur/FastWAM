"""Streaming LeRobot safetensors loading and Megatron DCP resharding."""

from __future__ import annotations

import gc
import json
from collections.abc import Iterator
from pathlib import Path

import torch
import torch.distributed as dist

from .mcore import _USE_MCORE

try:
    if not _USE_MCORE:
        raise ImportError
    from megatron.core import parallel_state
except ImportError:  # pragma: no cover - CPU test environments
    parallel_state = None


_TRAINING_DCP_COMMON_KEYS = frozenset(
    {
        "args",
        "checkpoint_version",
        "content_metadata",
        "iteration",
        "num_floating_point_operations_so_far",
        "optimizer",
        "opt_param_scheduler",
    }
)


def _unexpected_dcp_keys(keys) -> list[str]:
    return [key for key in keys if key not in _TRAINING_DCP_COMMON_KEYS]


def _tp_rank_world() -> tuple[int, int]:
    if parallel_state is None or not dist.is_available() or not dist.is_initialized():
        return 0, 1
    group = parallel_state.get_tensor_model_parallel_group()
    return group.rank(), group.size()


def _checkpoint_files(checkpoint: str | Path) -> list[Path]:
    path = Path(checkpoint)
    if path.is_file():
        return [path]
    direct = path / "model.safetensors"
    if direct.is_file():
        return [direct]
    index = path / "model.safetensors.index.json"
    if index.is_file():
        data = json.loads(index.read_text())
        return sorted({path / name for name in data["weight_map"].values()})
    files = sorted(path.glob("*.safetensors"))
    if not files:
        raise FileNotFoundError(f"No safetensors checkpoint found under {path}")
    return files


def _source_name(target_name: str) -> str:
    name = target_name.replace(".linear.weight", ".weight")
    name = name.replace(".linear.bias", ".bias")
    return f"model.{name}"


def _slice_for_rank(
    source: torch.Tensor,
    target: torch.Tensor,
    *,
    name: str,
    tp_rank: int,
    tp_world: int,
) -> torch.Tensor:
    if source.shape == target.shape:
        return source
    candidates = []
    if source.ndim == target.ndim:
        for dim in range(source.ndim):
            expected = list(source.shape)
            if source.shape[dim] % tp_world:
                continue
            expected[dim] //= tp_world
            if tuple(expected) == tuple(target.shape):
                candidates.append(dim)
    if len(candidates) != 1:
        raise ValueError(
            f"{name}: cannot uniquely shard {tuple(source.shape)} to {tuple(target.shape)} "
            f"for TP={tp_world}"
        )
    return source.chunk(tp_world, dim=candidates[0])[tp_rank].contiguous()


def _iter_safetensor_files(files: list[Path]) -> Iterator[tuple[Path, object]]:
    try:
        from safetensors import safe_open
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError("Fast-WAM checkpoint loading requires safetensors") from exc
    for file in files:
        with safe_open(file, framework="pt", device="cpu") as handle:
            yield file, handle


@torch.no_grad()
def load_lerobot_checkpoint(
    model: torch.nn.Module, checkpoint: str | Path, *, strict: bool = True
) -> tuple[list[str], list[str]]:
    """Stream a LeRobot Fast-WAM checkpoint directly into the local TP shard.

    Only one source tensor is materialized at a time, so the 12 GB checkpoint
    does not need a second full in-memory state dict.
    """

    files = _checkpoint_files(checkpoint)
    targets = dict(model.named_parameters())
    source_to_target = {_source_name(name): name for name in targets}
    loaded: set[str] = set()
    unexpected: list[str] = []
    tp_rank, tp_world = _tp_rank_world()

    for _, handle in _iter_safetensor_files(files):
        for source_name in handle.keys():
            target_name = source_to_target.get(source_name)
            if target_name is None:
                unexpected.append(source_name)
                continue
            target = targets[target_name]
            tensor = _slice_for_rank(
                handle.get_tensor(source_name),
                target,
                name=source_name,
                tp_rank=tp_rank,
                tp_world=tp_world,
            )
            target.copy_(tensor.to(device=target.device, dtype=target.dtype))
            loaded.add(target_name)

    missing = sorted(set(targets) - loaded)
    if strict and (missing or unexpected):
        raise RuntimeError(
            "Strict Fast-WAM load failed: "
            f"missing={missing[:16]} ({len(missing)} total), "
            f"unexpected={unexpected[:16]} ({len(unexpected)} total)"
        )
    return missing, unexpected


def _dcp_path(checkpoint: str | Path) -> Path:
    path = Path(checkpoint)
    tracker = path / "latest_checkpointed_iteration.txt"
    if tracker.is_file():
        return path / f"iter_{int(tracker.read_text().strip()):07d}"
    return path


def save_megatron_dcp(model: torch.nn.Module, checkpoint: str | Path) -> None:
    """Save current TP shards as a reshardable distributed checkpoint."""

    path = Path(checkpoint)
    path.mkdir(parents=True, exist_ok=True)
    if dist.is_available() and dist.is_initialized():
        from megatron.core import dist_checkpointing

        metadata = {
            "dp_cp_group": parallel_state.get_data_parallel_group(with_context_parallel=True)
        }
        dist_checkpointing.save(model.sharded_state_dict(metadata=metadata), str(path))
        return

    import torch.distributed.checkpoint as dcp

    state = {key: value for key, value in model.state_dict().items() if not key.endswith("_extra_state")}
    dcp.save(state, checkpoint_id=str(path), no_dist=True)


def load_megatron_dcp(model: torch.nn.Module, checkpoint: str | Path) -> None:
    """Load a DCP checkpoint, allowing a different TP/DP topology."""

    path = _dcp_path(checkpoint)
    if dist.is_available() and dist.is_initialized():
        from megatron.core import dist_checkpointing
        from megatron.core.dist_checkpointing.strategies.torch import TorchDistLoadShardedStrategy

        metadata = {
            "dp_cp_group": parallel_state.get_data_parallel_group(with_context_parallel=True)
        }
        template = model.sharded_state_dict(metadata=metadata)
        state = dist_checkpointing.load(
            template,
            str(path),
            TorchDistLoadShardedStrategy(),
            strict="assume_ok_unexpected",
        )
    else:
        import torch.distributed.checkpoint as dcp

        state = {
            key: value for key, value in model.state_dict().items() if not key.endswith("_extra_state")
        }
        dcp.load(state, checkpoint_id=str(path), no_dist=True)
    incompatible = model.load_state_dict(state, strict=False)
    missing = [key for key in incompatible.missing_keys if not key.endswith("_extra_state")]
    unexpected = _unexpected_dcp_keys(incompatible.unexpected_keys)
    # TorchDist may materialize a second full set of BF16 tensors while loading
    # a fresh DCP.  Do not leave those tensors (or allocator cache blocks) live
    # while the optimizer, frozen VAE, and first training activations are built.
    del state
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    if missing or unexpected:
        raise RuntimeError(
            f"Fast-WAM DCP load mismatch: missing={missing[:16]}, "
            f"unexpected={unexpected[:16]}"
        )
