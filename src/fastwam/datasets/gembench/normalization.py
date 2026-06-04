from __future__ import annotations

from pathlib import Path
from typing import Iterable

import numpy as np
import torch

from fastwam.datasets.lerobot.transforms.action_state_merger import ConcatLeftAlign
from fastwam.datasets.lerobot.utils.normalizer import (
    LinearNormalizer,
    load_dataset_stats_from_json,
    save_dataset_stats_to_json,
)


DEFAULT_SHAPE_META = {
    "images": [
        {"key": "front", "raw_shape": [3, 128, 128], "shape": [3, 224, 224]},
        {"key": "wrist", "raw_shape": [3, 128, 128], "shape": [3, 224, 224]},
        {"key": "left_shoulder", "raw_shape": [3, 128, 128], "shape": [3, 224, 224]},
    ],
    "action": [{"key": "default", "raw_shape": 8, "shape": 8}],
    "state": [{"key": "default", "raw_shape": 8, "shape": 8}],
}


def _empty_stats(dim: int) -> dict[str, torch.Tensor]:
    zeros = torch.zeros(dim, dtype=torch.float32)
    ones = torch.ones(dim, dtype=torch.float32)
    return {
        "global_min": zeros.clone(),
        "global_max": ones.clone(),
        "global_mean": zeros.clone(),
        "global_std": ones.clone(),
        "global_q01": zeros.clone(),
        "global_q99": ones.clone(),
    }


def fixed_dataset_stats(action_dim: int = 8, state_dim: int = 8) -> dict:
    return {
        "action": {"default": _empty_stats(int(action_dim))},
        "state": {"default": _empty_stats(int(state_dim))},
    }


def _stack_stats(values: list[np.ndarray], dim: int) -> dict[str, torch.Tensor]:
    if not values:
        return _empty_stats(dim)
    arr = np.concatenate([np.asarray(v, dtype=np.float32).reshape(-1, dim) for v in values], axis=0)
    return {
        "global_min": torch.from_numpy(arr.min(axis=0)),
        "global_max": torch.from_numpy(arr.max(axis=0)),
        "global_mean": torch.from_numpy(arr.mean(axis=0)),
        "global_std": torch.from_numpy(arr.std(axis=0) + 1e-8),
        "global_q01": torch.from_numpy(np.quantile(arr, 0.01, axis=0).astype(np.float32)),
        "global_q99": torch.from_numpy(np.quantile(arr, 0.99, axis=0).astype(np.float32)),
    }


def scanned_dataset_stats(
    samples: Iterable[tuple[np.ndarray, np.ndarray]],
    *,
    action_dim: int = 8,
    state_dim: int = 8,
) -> dict:
    actions: list[np.ndarray] = []
    states: list[np.ndarray] = []
    for action, state in samples:
        actions.append(np.asarray(action, dtype=np.float32))
        states.append(np.asarray(state, dtype=np.float32))
    return {
        "action": {"default": _stack_stats(actions, int(action_dim))},
        "state": {"default": _stack_stats(states, int(state_dim))},
    }


class GEMBenchProcessorShim:
    """Small compatibility shim for Wan22Trainer.evaluate()."""

    def __init__(
        self,
        stats: dict,
        *,
        action_dim: int = 8,
        proprio_dim: int = 8,
        norm_default_mode: str = "-2.0/2.0",
        shape_meta: dict | None = None,
    ):
        self.action_output_dim = int(action_dim)
        self.proprio_output_dim = int(proprio_dim)
        self.shape_meta = shape_meta or DEFAULT_SHAPE_META
        self.action_state_merger = ConcatLeftAlign()
        self.action_state_merger.set_shape_meta(self.shape_meta)
        self.normalizer = LinearNormalizer(
            shape_meta=self.shape_meta,
            use_stepwise_action_norm=False,
            default_mode=norm_default_mode,
            exception_mode=None,
            stats=stats,
        )

    def normalize(self, action: torch.Tensor, proprio: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        batch = {
            "action": {"default": action.to(torch.float32)},
            "state": {"default": proprio.to(torch.float32)},
        }
        batch = self.normalizer.forward(batch)
        batch = self.action_state_merger.forward(batch)
        return (
            batch["action"],
            batch["state"],
            batch["action_dim_is_pad"],
            batch["state_dim_is_pad"],
        )


def load_or_create_stats(
    stats_path: str | None,
    *,
    action_dim: int = 8,
    state_dim: int = 8,
    save_if_missing: bool = True,
) -> dict:
    if stats_path:
        path = Path(stats_path).expanduser()
        if path.exists():
            return load_dataset_stats_from_json(str(path))
    stats = fixed_dataset_stats(action_dim=action_dim, state_dim=state_dim)
    if stats_path and save_if_missing:
        path = Path(stats_path).expanduser()
        path.parent.mkdir(parents=True, exist_ok=True)
        save_dataset_stats_to_json(stats, str(path))
    return stats
