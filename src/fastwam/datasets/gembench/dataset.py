from __future__ import annotations

import hashlib
import logging
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Iterable, Sequence

import numpy as np
import torch
import torch.nn.functional as F
from omegaconf import DictConfig, OmegaConf

from fastwam.datasets.lerobot.robot_video_dataset import DEFAULT_PROMPT

from .instructions import (
    instruction_for_taskvar,
    load_instruction_map,
    resolve_taskvars,
)
from .lmdb_reader import LMDBEpisodeStore
from .normalization import GEMBenchProcessorShim, DEFAULT_SHAPE_META, load_or_create_stats, scanned_dataset_stats
from .vae_cache import GEMBenchVAELatentCache, build_expected_dataset_config

logger = logging.getLogger(__name__)

# Matches the official robot-3dlotus keystep generator default --cameras order.
RAW_CAMERA_ORDER = ("left_shoulder", "right_shoulder", "wrist", "front")


class GEMBenchKeystepsDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        root: str = "/mnt/yuhan/datasets/GEMBench",
        split: str = "train",
        subset: str = "keysteps_bbox",
        seed: str = "seed0",
        data_dir: str | None = None,
        taskvars: Sequence[str] | str | None = None,
        skip_missing_taskvars: bool = True,
        num_video_frames: int = 9,
        action_horizon: int = 8,
        video_size: Sequence[int] = (224, 672),
        camera_order: Sequence[str] = ("front", "wrist", "left_shoulder"),
        raw_camera_order: Sequence[str] = RAW_CAMERA_ORDER,
        val_set_proportion: float = 0.02,
        is_training_set: bool = True,
        split_seed: int = 42,
        max_episodes_per_taskvar: int | None = None,
        instruction_json_path: str | None = None,
        instruction_index: int = 0,
        text_embedding_cache_dir: str | None = None,
        context_len: int = 128,
        text_dim: int = 4096,
        text_encoder_id: str = "wan22ti2v5b",
        allow_missing_text_embeds: bool = False,
        pretrained_norm_stats: str | None = None,
        norm_default_mode: str = "-2.0/2.0",
        stats_scan_limit: int = 0,
        vae_latent_cache_dir: str | None = None,
        vae_latent_cache_encode_autocast: bool | None = None,
        processor: Any | None = None,
    ):
        self.root = Path(root).expanduser().resolve()
        self.split = str(split)
        self.subset = str(subset)
        self.seed = str(seed)
        self.data_dir = Path(data_dir).expanduser().resolve() if data_dir else self.root / f"{self.split}_dataset" / self.subset / self.seed
        self.skip_missing_taskvars = bool(skip_missing_taskvars)
        self.num_video_frames = int(num_video_frames)
        self.action_horizon = int(action_horizon)
        self.video_size = [int(video_size[0]), int(video_size[1])]
        self.camera_order = [str(c) for c in camera_order]
        self.raw_camera_order = [str(c) for c in raw_camera_order]
        self.val_set_proportion = float(val_set_proportion)
        self.is_training_set = bool(is_training_set)
        self.split_seed = int(split_seed)
        self.max_episodes_per_taskvar = None if max_episodes_per_taskvar is None else int(max_episodes_per_taskvar)
        self.instruction_index = int(instruction_index)
        self.text_embedding_cache_dir = None if text_embedding_cache_dir is None else str(text_embedding_cache_dir)
        self.context_len = int(context_len)
        self.text_dim = int(text_dim)
        self.text_encoder_id = str(text_encoder_id)
        self.allow_missing_text_embeds = bool(allow_missing_text_embeds)
        self.norm_default_mode = str(norm_default_mode)
        self.stats_scan_limit = int(stats_scan_limit)
        self.vae_latent_cache_dir = (
            None if vae_latent_cache_dir is None else Path(vae_latent_cache_dir).expanduser().resolve()
        )
        self.vae_latent_cache_encode_autocast = vae_latent_cache_encode_autocast

        if self.num_video_frames <= 1:
            raise ValueError("num_video_frames must be > 1")
        if self.num_video_frames % 4 != 1:
            raise ValueError(f"num_video_frames must satisfy T % 4 == 1, got {self.num_video_frames}")
        if self.action_horizon <= 0:
            raise ValueError("action_horizon must be positive")
        if self.video_size[1] % 16 != 0 or self.video_size[0] % 16 != 0:
            raise ValueError(f"video_size must be multiples of 16, got {self.video_size}")
        if self.video_size[1] % len(self.camera_order) != 0:
            raise ValueError(
                f"video width {self.video_size[1]} must be divisible by number of cameras {len(self.camera_order)}"
            )

        self.camera_indices = [self._camera_index(camera) for camera in self.camera_order]
        self.instruction_map = load_instruction_map(instruction_json_path)
        self.store = LMDBEpisodeStore(self.data_dir)
        self.taskvars = self._resolve_available_taskvars(taskvars)
        self.index = self._build_index()
        if not self.index:
            raise ValueError(f"No GEMBench episodes found under {self.data_dir}")
        self.vae_latent_cache = self._load_vae_latent_cache()

        self.shape_meta = self._shape_meta_from_processor(processor)
        stats = self._load_or_scan_stats(pretrained_norm_stats)
        self.processor = GEMBenchProcessorShim(
            stats,
            action_dim=8,
            proprio_dim=8,
            norm_default_mode=self.norm_default_mode,
            shape_meta=self.shape_meta,
        )
        # Wan22Trainer.evaluate() expects this path from the LeRobot dataset.
        self.lerobot_dataset = SimpleNamespace(processor=self.processor)
        logger.info(
            "GEMBench dataset ready: data_dir=%s split=%s is_training_set=%s taskvars=%d samples=%d vae_cache=%s",
            self.data_dir,
            self.split,
            self.is_training_set,
            len(self.taskvars),
            len(self.index),
            self.vae_latent_cache_dir is not None,
        )

    def __len__(self) -> int:
        return len(self.index)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        if idx < 0 or idx >= len(self.index):
            raise IndexError(idx)
        taskvar, episode_key = self.index[idx]
        episode = self.store.get(taskvar, episode_key)
        return self._episode_to_sample(taskvar, episode_key, episode)

    def close(self) -> None:
        self.store.close()

    def _camera_index(self, camera: str) -> int:
        if camera not in self.raw_camera_order:
            raise ValueError(f"Unknown GEMBench camera {camera!r}; raw order is {self.raw_camera_order}")
        return self.raw_camera_order.index(camera)

    def _resolve_available_taskvars(self, taskvars: Sequence[str] | str | None) -> list[str]:
        requested = resolve_taskvars(taskvars)
        if requested is None:
            return self.store.list_taskvars()

        out: list[str] = []
        missing: list[str] = []
        for taskvar in requested:
            if self.store.has_taskvar(taskvar):
                out.append(taskvar)
            else:
                missing.append(taskvar)
        if missing and not self.skip_missing_taskvars:
            raise FileNotFoundError(
                f"Missing GEMBench taskvars under {self.data_dir}: {missing[:10]}"
                + ("..." if len(missing) > 10 else "")
            )
        if missing:
            logger.warning("Skipping %d missing/incomplete GEMBench taskvars, first missing=%s", len(missing), missing[:5])
        return out

    def _build_index(self) -> list[tuple[str, bytes]]:
        rng = np.random.default_rng(self.split_seed)
        index: list[tuple[str, bytes]] = []
        for taskvar in self.taskvars:
            keys = self.store.list_episode_keys(taskvar)
            if self.val_set_proportion > 0:
                order = np.arange(len(keys))
                rng.shuffle(order)
                split_idx = int(len(order) * (1.0 - self.val_set_proportion))
                selected_idx = order[:split_idx] if self.is_training_set else order[split_idx:]
                keys = [keys[int(i)] for i in sorted(selected_idx.tolist())]
            if self.max_episodes_per_taskvar is not None:
                keys = keys[: self.max_episodes_per_taskvar]
            index.extend((taskvar, key) for key in keys)
        return index

    def _load_vae_latent_cache(self) -> GEMBenchVAELatentCache | None:
        if self.vae_latent_cache_dir is None:
            return None
        expected_config = build_expected_dataset_config(
            root=self.root,
            split=self.split,
            subset=self.subset,
            seed=self.seed,
            num_video_frames=self.num_video_frames,
            action_horizon=self.action_horizon,
            video_size=self.video_size,
            camera_order=self.camera_order,
        )
        return GEMBenchVAELatentCache(
            self.vae_latent_cache_dir,
            expected_dataset_config=expected_config,
            expected_index=self.index,
            expected_encode_autocast=self.vae_latent_cache_encode_autocast,
        )

    def _shape_meta_from_processor(self, processor_cfg: Any | None) -> dict:
        shape_meta = OmegaConf.to_container(DictConfig(DEFAULT_SHAPE_META), resolve=True)
        camera_width = self.video_size[1] // len(self.camera_order)
        for meta, camera in zip(shape_meta["images"], self.camera_order):
            meta["key"] = camera
            meta["shape"] = [3, self.video_size[0], camera_width]
        if processor_cfg is not None:
            plain = OmegaConf.to_container(processor_cfg, resolve=True) if isinstance(processor_cfg, DictConfig) else processor_cfg
            if isinstance(plain, dict) and plain.get("shape_meta") is not None:
                shape_meta = plain["shape_meta"]
        return shape_meta

    def _load_or_scan_stats(self, pretrained_norm_stats: str | None) -> dict:
        if self.stats_scan_limit <= 0:
            return load_or_create_stats(pretrained_norm_stats, action_dim=8, state_dim=8)

        stats_path = Path(pretrained_norm_stats).expanduser() if pretrained_norm_stats else None
        if stats_path is not None and stats_path.exists():
            return load_or_create_stats(str(stats_path), action_dim=8, state_dim=8)

        samples = []
        for taskvar, key in self.index[: self.stats_scan_limit]:
            episode = self.store.get(taskvar, key)
            action, proprio = self._episode_action_proprio(episode)
            samples.append((action, proprio))
        stats = scanned_dataset_stats(samples, action_dim=8, state_dim=8)
        if stats_path is not None:
            stats_path.parent.mkdir(parents=True, exist_ok=True)
            from fastwam.datasets.lerobot.utils.normalizer import save_dataset_stats_to_json

            save_dataset_stats_to_json(stats, str(stats_path))
        return stats

    def _episode_to_sample(self, taskvar: str, episode_key: bytes, episode: dict) -> dict[str, Any]:
        video = None if self.vae_latent_cache is not None else self._episode_video(episode)
        video_latents = None if self.vae_latent_cache is None else self.vae_latent_cache.get(taskvar, episode_key)
        action_raw, proprio_raw = self._episode_action_proprio(episode)
        action, proprio, action_dim_is_pad, proprio_dim_is_pad = self.processor.normalize(
            torch.as_tensor(action_raw, dtype=torch.float32),
            torch.as_tensor(proprio_raw, dtype=torch.float32),
        )
        instruction = instruction_for_taskvar(
            taskvar,
            instruction_map=self.instruction_map,
            instruction_index=self.instruction_index,
        )
        prompt = DEFAULT_PROMPT.format(task=instruction)
        context, context_mask = self._get_cached_text_context(prompt)
        context = context.clone()
        context[~context_mask.bool()] = 0.0
        context_mask = torch.ones_like(context_mask, dtype=torch.bool)
        sample = {
            "action": action,
            "proprio": proprio,
            "prompt": prompt,
            "context": context,
            "context_mask": context_mask,
            "image_is_pad": torch.zeros(self.num_video_frames, dtype=torch.bool),
            "action_is_pad": torch.zeros(self.action_horizon, dtype=torch.bool),
            "proprio_is_pad": torch.zeros(self.action_horizon, dtype=torch.bool),
            "action_dim_is_pad": action_dim_is_pad,
            "proprio_dim_is_pad": proprio_dim_is_pad,
            "taskvar": taskvar,
            "episode_key": episode_key.decode("ascii", errors="ignore"),
        }
        if video_latents is None:
            sample["video"] = video
        else:
            sample["video_latents"] = video_latents
        return sample

    def _episode_video(self, episode: dict) -> torch.Tensor:
        rgb = np.asarray(episode["rgb"])
        if rgb.ndim != 5 or rgb.shape[-1] != 3:
            raise ValueError(f"Expected GEMBench rgb [T,N,H,W,3], got {rgb.shape}")
        frame_idx = self._frame_indices(rgb.shape[0], self.num_video_frames)
        rgb = rgb[frame_idx][:, self.camera_indices]
        tensor = torch.from_numpy(np.ascontiguousarray(rgb)).permute(0, 1, 4, 2, 3).to(torch.float32) / 255.0
        t, n, c, h, w = tensor.shape
        camera_h = self.video_size[0]
        camera_w = self.video_size[1] // n
        tensor = tensor.reshape(t * n, c, h, w)
        tensor = F.interpolate(tensor, size=(camera_h, camera_w), mode="bilinear", align_corners=False)
        tensor = tensor.reshape(t, n, c, camera_h, camera_w)
        video = torch.cat([tensor[:, i] for i in range(n)], dim=-1)
        video = video * 2.0 - 1.0
        return video.permute(1, 0, 2, 3).contiguous()

    def _episode_action_proprio(self, episode: dict) -> tuple[np.ndarray, np.ndarray]:
        action = np.asarray(episode["action"], dtype=np.float32)
        if action.ndim != 2 or action.shape[1] != 8:
            raise ValueError(f"Expected GEMBench action [T,8], got {action.shape}")
        frame_idx = self._frame_indices(action.shape[0], self.num_video_frames)
        proprio_idx = frame_idx[:-1]
        next_idx = frame_idx[1:]
        if self.action_horizon != len(next_idx):
            positions = np.linspace(0, len(next_idx) - 1, self.action_horizon)
            take = np.rint(positions).astype(np.int64)
            next_idx = next_idx[take]
            proprio_idx = proprio_idx[take]
        return action[next_idx].astype(np.float32), action[proprio_idx].astype(np.float32)

    @staticmethod
    def _frame_indices(length: int, target_length: int) -> np.ndarray:
        if length <= 0:
            raise ValueError("Episode has no frames")
        if target_length <= 1:
            return np.zeros((target_length,), dtype=np.int64)
        return np.rint(np.linspace(0, length - 1, target_length)).astype(np.int64)

    def _get_cached_text_context(self, prompt: str) -> tuple[torch.Tensor, torch.Tensor]:
        if self.text_embedding_cache_dir is None:
            if self.allow_missing_text_embeds:
                return self._empty_text_context()
            raise ValueError("text_embedding_cache_dir is not set.")
        cache_dir = Path(self.text_embedding_cache_dir)
        hashed = hashlib.sha256(prompt.encode("utf-8")).hexdigest()
        cache_path = cache_dir / f"{hashed}.t5_len{self.context_len}.{self.text_encoder_id}.pt"
        if not cache_path.exists():
            if self.allow_missing_text_embeds:
                return self._empty_text_context()
            raise FileNotFoundError(
                f"Missing text embedding cache: {cache_path}. Run scripts/precompute_gembench_text_embeds.py first."
            )
        payload = torch.load(cache_path, map_location="cpu")
        context = payload["context"].to(torch.float32)
        mask = payload["mask"].bool()
        if context.ndim != 2 or mask.ndim != 1:
            raise ValueError(f"Invalid text cache payload shapes in {cache_path}: {tuple(context.shape)}, {tuple(mask.shape)}")
        return context, mask

    def _empty_text_context(self) -> tuple[torch.Tensor, torch.Tensor]:
        return (
            torch.zeros((self.context_len, self.text_dim), dtype=torch.float32),
            torch.zeros((self.context_len,), dtype=torch.bool),
        )
