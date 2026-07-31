"""Recompute Fast-WAM's episode-aggregated LIBERO normalization statistics."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

import torch

from .data import _EpisodeCache, _load_episodes, official_dataset_dirs


def _sliding_window_with_replication(action: torch.Tensor, horizon: int) -> torch.Tensor:
    indices = torch.arange(action.shape[0])[:, None] + torch.arange(horizon)[None, :]
    return action[indices.clamp(max=action.shape[0] - 1)]


def compute_dataset_stats(root: str | Path) -> dict[str, Any]:
    """Match ``BaseLerobotDataset.get_dataset_stats`` for the official four suites."""

    episodes = _load_episodes(
        official_dataset_dirs(root),
        validate_official_release=True,
    )
    cache = _EpisodeCache(capacity=1)
    values: dict[str, dict[str, list[torch.Tensor]]] = {
        "state": defaultdict(list),
        "action": defaultdict(list),
    }

    for episode_number, episode in enumerate(episodes, start=1):
        table = cache.get(episode)
        tensors = {
            "state": table["state"].unsqueeze(1),
            "action": _sliding_window_with_replication(table["action"], 32),
        }
        for kind, tensor in tensors.items():
            values[kind]["min"].append(tensor.amin(dim=0))
            values[kind]["max"].append(tensor.amax(dim=0))
            values[kind]["mean"].append(tensor.mean(dim=0))
            values[kind]["var"].append(tensor.var(dim=0))
            values[kind]["q01"].append(torch.quantile(tensor, 0.01, dim=0))
            values[kind]["q99"].append(torch.quantile(tensor, 0.99, dim=0))
        if episode_number % 100 == 0 or episode_number == len(episodes):
            print(f"stats: {episode_number}/{len(episodes)} episodes", flush=True)

    def aggregate(kind: str) -> dict[str, torch.Tensor]:
        current = values[kind]
        stepwise_min = torch.stack(current["min"]).amin(dim=0)
        stepwise_max = torch.stack(current["max"]).amax(dim=0)
        stepwise_q01 = torch.stack(current["q01"]).amin(dim=0)
        stepwise_q99 = torch.stack(current["q99"]).amax(dim=0)
        means = torch.stack(current["mean"])
        variances = torch.stack(current["var"])
        stepwise_mean = means.mean(dim=0)
        stepwise_std = (
            variances + (means - stepwise_mean).square()
        ).mean(dim=0).sqrt()
        global_mean = means.mean(dim=(0, 1))
        global_std = (
            variances + (means - global_mean).square()
        ).mean(dim=(0, 1)).sqrt()
        return {
            "stepwise_min": stepwise_min,
            "stepwise_max": stepwise_max,
            "global_min": stepwise_min.amin(dim=0),
            "global_max": stepwise_max.amax(dim=0),
            "stepwise_q01": stepwise_q01,
            "stepwise_q99": stepwise_q99,
            "global_q01": stepwise_q01.amin(dim=0),
            "global_q99": stepwise_q99.amax(dim=0),
            "stepwise_mean": stepwise_mean,
            "stepwise_std": stepwise_std,
            "global_mean": global_mean,
            "global_std": global_std,
        }

    return {
        "state": {"default": aggregate("state")},
        "action": {"default": aggregate("action")},
        "num_episodes": len(episodes),
        "num_transition": sum(episode.length for episode in episodes),
    }


def _json_value(value: Any) -> Any:
    if isinstance(value, torch.Tensor):
        return value.cpu().tolist()
    if isinstance(value, dict):
        return {key: _json_value(item) for key, item in value.items()}
    return value


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    output = Path(args.output).expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    stats = compute_dataset_stats(args.root)
    temporary = output.with_name(f".{output.name}.tmp")
    temporary.write_text(
        json.dumps(_json_value(stats), indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    temporary.replace(output)
    print(f"wrote {output}")


if __name__ == "__main__":
    main()
