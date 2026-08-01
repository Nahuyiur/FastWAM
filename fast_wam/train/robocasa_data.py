"""RoboCasa ACG dataset bridge for the Megatron Fast-WAM overlay."""

from __future__ import annotations

import sys
import json
import math
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist
from hydra import compose, initialize_config_dir
from hydra.utils import instantiate

from .robocasa_webdataset import (
    RoboCasaIndexedSubset,
    RoboCasaWebDataset,
    load_source_indices,
)


def _instantiate_rank_ordered(config, **kwargs):
    """Avoid all ranks opening the same metadata files at exactly the same time."""

    if not dist.is_available() or not dist.is_initialized():
        return instantiate(config, **kwargs)
    value = None
    if dist.get_rank() == 0:
        value = instantiate(config, **kwargs)
    dist.barrier()
    if dist.get_rank() != 0:
        value = instantiate(config, **kwargs)
    dist.barrier()
    return value


class RoboCasaLatentCache:
    """Read immutable BF16 latent shards created by prepare_robocasa_latents."""

    def __init__(self, root: str | Path, expected_samples: int):
        self.root = Path(root).expanduser().resolve()
        manifest_path = self.root / "manifest.json"
        if not manifest_path.is_file():
            raise FileNotFoundError(f"Incomplete RoboCasa latent cache: {manifest_path}")
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if not manifest.get("complete"):
            raise ValueError(f"RoboCasa latent cache is not marked complete: {manifest_path}")
        if int(manifest["num_samples"]) != int(expected_samples):
            raise ValueError(
                f"RoboCasa latent cache sample mismatch: cache={manifest['num_samples']} "
                f"dataset={expected_samples}"
            )
        self.sample_shape = tuple(int(value) for value in manifest["sample_shape"])
        self.shards = list(manifest["shards"])
        self._mapped: dict[int, torch.Tensor] = {}

    def __getitem__(self, index: int) -> torch.Tensor:
        for shard in self.shards:
            start = int(shard["start"])
            count = int(shard["count"])
            if start <= index < start + count:
                shard_id = int(shard["id"])
                if shard_id not in self._mapped:
                    path = self.root / str(shard["file"])
                    expected_bytes = count * math.prod(self.sample_shape) * 2
                    if path.stat().st_size != expected_bytes:
                        raise ValueError(f"Bad latent shard size: {path}")
                    self._mapped[shard_id] = torch.from_file(
                        str(path),
                        shared=False,
                        size=count * math.prod(self.sample_shape),
                        dtype=torch.bfloat16,
                    ).view(count, *self.sample_shape)
                return self._mapped[shard_id][index - start]
        raise IndexError(index)


class RoboCasaLatentDataset(torch.utils.data.Dataset):
    """Attach cached VAE latents without decoding the source MP4 files."""

    def __init__(
        self,
        dataset,
        latent_cache: str | Path,
        source_indices: list[int] | None = None,
    ):
        self.dataset = dataset
        self.source_indices = tuple(
            range(len(dataset)) if source_indices is None else source_indices
        )
        self.latent_cache = RoboCasaLatentCache(
            latent_cache,
            len(self.source_indices),
        )
        self.episodes = dataset.episodes
        required = (
            "windows",
            "action_horizon",
            "shape_meta",
            "num_frames",
            "_load_episode_arrays",
            "_normalize_action_state",
            "_get_cached_text_context",
        )
        missing = [name for name in required if not hasattr(dataset, name)]
        if missing:
            raise TypeError(
                "RoboCasa latent cache requires the ACG dataset metadata contract; "
                f"missing={missing}"
            )

    def __len__(self):
        return len(self.source_indices)

    def __getitem__(self, index):
        logical_index = int(index)
        last_error: Exception | None = None
        for _ in range(int(getattr(self.dataset, "max_getitem_retry", 1))):
            try:
                source_index = self.source_indices[logical_index]
                sample = self._get_metadata(source_index)
                sample["idx"] = torch.tensor(logical_index, dtype=torch.long)
                sample["source_idx"] = torch.tensor(source_index, dtype=torch.long)
                sample["input_latents"] = self.latent_cache[logical_index]
                return sample
            except Exception as error:
                last_error = error
                logical_index = int(np.random.randint(len(self)))
        raise RuntimeError(
            "Failed to load cached RoboCasa sample after metadata retries."
        ) from last_error

    def _get_metadata(self, index: int) -> dict:
        """Mirror the baseline sample contract while deliberately skipping `_load_video`."""

        from fastwam.datasets.lerobot.robot_video_dataset import DEFAULT_PROMPT

        episode_pos, start = self.dataset.windows[index]
        episode = self.dataset.episodes[episode_pos]
        states, actions, _ = self.dataset._load_episode_arrays(episode)
        horizon = int(self.dataset.action_horizon)
        action_raw = torch.as_tensor(
            actions[start : start + horizon], dtype=torch.float32
        )
        proprio_raw = torch.as_tensor(
            states[start : start + horizon], dtype=torch.float32
        )
        action_width = int(self.dataset.shape_meta["action"][0]["raw_shape"])
        proprio_width = int(self.dataset.shape_meta["state"][0]["raw_shape"])
        if action_raw.shape != (horizon, action_width):
            raise ValueError(f"Bad cached action window shape: {tuple(action_raw.shape)}")
        if proprio_raw.shape != (horizon, proprio_width):
            raise ValueError(
                f"Bad cached proprio window shape: {tuple(proprio_raw.shape)}"
            )
        action, proprio, action_dim_is_pad, proprio_dim_is_pad = (
            self.dataset._normalize_action_state(action_raw, proprio_raw)
        )
        prompt = DEFAULT_PROMPT.format(task=episode.task_text)
        context, context_mask = self.dataset._get_cached_text_context(prompt)
        return {
            "action": action,
            "proprio": proprio,
            "prompt": prompt,
            "context": context,
            "context_mask": context_mask,
            "image_is_pad": torch.zeros(self.dataset.num_frames, dtype=torch.bool),
            "action_is_pad": torch.zeros(horizon, dtype=torch.bool),
            "proprio_is_pad": torch.zeros(horizon, dtype=torch.bool),
            "action_dim_is_pad": action_dim_is_pad,
            "proprio_dim_is_pad": proprio_dim_is_pad,
            "idx": torch.tensor(index, dtype=torch.long),
            "episode_index": torch.tensor(episode.episode_index, dtype=torch.long),
            "window_start": torch.tensor(start, dtype=torch.long),
        }


def build_robocasa_datasets(
    repo_root: str | Path,
    task_config: str,
    *,
    train_latent_cache: str | Path | None = None,
    valid_latent_cache: str | Path | None = None,
    train_webdataset: str | Path | None = None,
    valid_webdataset: str | Path | None = None,
    train_index_file: str | Path | None = None,
    valid_index_file: str | Path | None = None,
):
    """Instantiate the unchanged baseline train/validation dataset contracts."""

    root = Path(repo_root).expanduser().resolve()
    config_dir = root / "configs"
    src_dir = root / "src"
    if not config_dir.is_dir() or not src_dir.is_dir():
        raise FileNotFoundError(
            f"RoboCasa Fast-WAM repo must contain configs/ and src/: {root}"
        )
    if str(src_dir) not in sys.path:
        sys.path.insert(0, str(src_dir))

    from fastwam.utils.config_resolvers import register_default_resolvers

    register_default_resolvers()
    with initialize_config_dir(config_dir=str(config_dir), version_base="1.3"):
        cfg = compose(config_name="train", overrides=[f"task={task_config}"])

    train_stats = cfg.data.train.get("pretrained_norm_stats")
    val_stats = cfg.data.val.get("pretrained_norm_stats") or train_stats
    if train_webdataset and (train_latent_cache or train_index_file):
        raise ValueError(
            "train_webdataset is mutually exclusive with train_latent_cache/train_index_file"
        )
    if valid_webdataset and (valid_latent_cache or valid_index_file):
        raise ValueError(
            "valid_webdataset is mutually exclusive with valid_latent_cache/valid_index_file"
        )
    if train_webdataset:
        train_dataset = RoboCasaWebDataset(train_webdataset)
    else:
        train_dataset = _instantiate_rank_ordered(cfg.data.train)
        train_indices = load_source_indices(train_index_file, len(train_dataset))
        if train_latent_cache:
            train_dataset = RoboCasaLatentDataset(
                train_dataset,
                train_latent_cache,
                train_indices,
            )
        elif train_indices is not None:
            train_dataset = RoboCasaIndexedSubset(train_dataset, train_indices)
    if valid_webdataset:
        val_dataset = RoboCasaWebDataset(valid_webdataset)
    else:
        val_dataset = _instantiate_rank_ordered(
            cfg.data.val,
            pretrained_norm_stats=val_stats,
        )
        valid_indices = load_source_indices(valid_index_file, len(val_dataset))
        if valid_latent_cache:
            val_dataset = RoboCasaLatentDataset(
                val_dataset,
                valid_latent_cache,
                valid_indices,
            )
        elif valid_indices is not None:
            val_dataset = RoboCasaIndexedSubset(val_dataset, valid_indices)
    return train_dataset, val_dataset, cfg
