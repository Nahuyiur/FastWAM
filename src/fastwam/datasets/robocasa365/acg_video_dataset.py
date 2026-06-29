import csv
import hashlib
import json
from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import torch
import torchvision.transforms.functional as transforms_F
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf

from fastwam.datasets.lerobot.lerobot.datasets.video_utils import decode_video_frames
from fastwam.datasets.lerobot.robot_video_dataset import DEFAULT_PROMPT
from fastwam.datasets.lerobot.utils.normalizer import load_dataset_stats_from_json
from fastwam.utils.logging_config import get_logger

logger = get_logger(__name__)


@dataclass(frozen=True)
class EpisodeRecord:
    repo_root: Path
    repo_name: str
    episode_index: int
    length: int
    task_text: str
    data_chunk_index: int
    data_file_index: int
    video_offsets: dict[str, float]
    video_chunk_indices: dict[str, int]
    video_file_indices: dict[str, int]


class RoboCasa365ACGVideoDataset(torch.utils.data.Dataset):
    """RoboCasa365 LeRobot v3 adapter for the RoboCasa-ACG manifest split.

    The public RoboCasa365 mirrors use LeRobot v3 chunked metadata:
    - metadata lives in parquet files, not v2 jsonl files;
    - each data parquet/mp4 file contains many episodes;
    - parquet timestamps reset per episode, while mp4 timestamps are continuous
      inside the chunk-level video file.

    This dataset returns the same training contract as RobotVideoDataset:
    `video` [C,T,H,W], normalized `action` [H,A], normalized `proprio` [H,S],
    cached `context/context_mask`, and padding masks.
    """

    def __init__(
        self,
        dataset_dirs: Sequence[str],
        shape_meta: dict[str, Any],
        episode_manifest_path: str,
        episode_manifest_split: str,
        num_frames: int = 9,
        action_horizon: int = 32,
        frame_offsets: Sequence[int] | None = None,
        video_size: Sequence[int] = (224, 448),
        camera_keys: Sequence[str] | None = None,
        processor: Any | None = None,
        pretrained_norm_stats: str | None = None,
        text_embedding_cache_dir: str | None = None,
        context_len: int = 128,
        text_encoder_id: str = "wan22ti2v5b",
        concat_multi_camera: str = "horizontal",
        window_stride: int = 1,
        max_episodes: int | None = None,
        max_windows_per_episode: int | None = None,
        max_samples: int | None = None,
        is_training_set: bool = False,
        allow_missing_text_embeds: bool = False,
        video_backend: str = "pyav",
        tolerance_s: float = 0.04,
        data_file_cache_size: int = 4,
        episode_cache_size: int = 512,
        max_getitem_retry: int = 5,
    ):
        if not dataset_dirs:
            raise ValueError("`dataset_dirs` must contain at least one RoboCasa365 repo.")
        if not episode_manifest_path:
            raise ValueError("`episode_manifest_path` is required.")
        if not episode_manifest_split:
            raise ValueError("`episode_manifest_split` is required.")
        if action_horizon <= 0:
            raise ValueError("`action_horizon` must be positive.")

        self.dataset_dirs = [Path(p).expanduser() for p in dataset_dirs]
        self.shape_meta = OmegaConf.to_container(shape_meta, resolve=True)
        self.episode_manifest_path = Path(episode_manifest_path).expanduser()
        self.episode_manifest_split = str(episode_manifest_split)
        self.num_frames = int(num_frames)
        self.action_horizon = int(action_horizon)
        self.frame_offsets = tuple(int(v) for v in (frame_offsets or range(self.num_frames)))
        self.video_size = (int(video_size[0]), int(video_size[1]))
        self.concat_multi_camera = str(concat_multi_camera)
        self.window_stride = int(window_stride)
        self.max_windows_per_episode = None if max_windows_per_episode is None else int(max_windows_per_episode)
        self.max_samples = None if max_samples is None else int(max_samples)
        self.is_training_set = bool(is_training_set)
        self.allow_missing_text_embeds = bool(allow_missing_text_embeds)
        self.video_backend = str(video_backend)
        self.tolerance_s = float(tolerance_s)
        self.data_file_cache_size = int(data_file_cache_size)
        self.episode_cache_size = int(episode_cache_size)
        self.max_getitem_retry = int(max_getitem_retry)
        self.context_len = int(context_len)
        self.text_encoder_id = str(text_encoder_id)
        self.text_embedding_cache_dir = None if text_embedding_cache_dir is None else Path(text_embedding_cache_dir)

        if self.window_stride <= 0:
            raise ValueError("`window_stride` must be positive.")
        if max(self.frame_offsets) != self.action_horizon:
            raise ValueError(
                "`frame_offsets` must end at action_horizon so video covers the full action window: "
                f"max_offset={max(self.frame_offsets)} action_horizon={self.action_horizon}"
            )
        if len(self.frame_offsets) != self.num_frames:
            raise ValueError(
                f"`num_frames` must equal len(frame_offsets), got {self.num_frames} and {len(self.frame_offsets)}"
            )

        if camera_keys is None:
            camera_keys = [meta["key"] for meta in self.shape_meta["images"]]
        self.camera_keys = [str(k) for k in camera_keys]
        self.video_keys = [
            key if key.startswith("observation.images.") else f"observation.images.{key}"
            for key in self.camera_keys
        ]
        if self.video_size[1] % len(self.video_keys) != 0 and self.concat_multi_camera == "horizontal":
            raise ValueError(f"video width {self.video_size[1]} is not divisible by cameras={len(self.video_keys)}")

        self.repo_name_to_root = {p.name: p for p in self.dataset_dirs}
        self.episodes = self._load_selected_episodes(max_episodes=max_episodes)
        self.windows = self._build_windows()
        if not self.windows:
            raise ValueError(
                f"No valid windows for split={self.episode_manifest_split!r}; "
                f"check action_horizon={self.action_horizon} and manifest={self.episode_manifest_path}"
            )

        self.processor = None
        if processor is not None:
            self.processor = instantiate(processor) if isinstance(processor, DictConfig) else processor
            if pretrained_norm_stats is None:
                raise ValueError("`pretrained_norm_stats` is required when `processor` is configured.")
            dataset_stats = load_dataset_stats_from_json(str(pretrained_norm_stats))
            self.processor.set_normalizer_from_stats(dataset_stats)
            if self.is_training_set:
                self.processor.train()
            else:
                self.processor.eval()

        self._data_file_cache: OrderedDict[Path, pd.DataFrame] = OrderedDict()
        self._episode_cache: OrderedDict[tuple[str, int], tuple[np.ndarray, np.ndarray, np.ndarray]] = OrderedDict()

        logger.info(
            "RoboCasa365ACGVideoDataset split=%s episodes=%d windows=%d cameras=%s action_horizon=%d frame_offsets=%s",
            self.episode_manifest_split,
            len(self.episodes),
            len(self.windows),
            self.video_keys,
            self.action_horizon,
            self.frame_offsets,
        )

    def __len__(self) -> int:
        return len(self.windows)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        sample_idx = int(idx)
        last_error: Exception | None = None
        for _ in range(self.max_getitem_retry):
            try:
                return self._get(sample_idx)
            except Exception as err:
                last_error = err
                logger.warning("RoboCasa sample %d failed (%s); retrying with random sample.", sample_idx, err)
                sample_idx = int(np.random.randint(len(self)))
        raise RuntimeError(f"Failed to load RoboCasa sample after {self.max_getitem_retry} retries.") from last_error

    def _load_selected_episodes(self, max_episodes: int | None) -> list[EpisodeRecord]:
        if not self.episode_manifest_path.exists():
            raise FileNotFoundError(f"Missing RoboCasa manifest: {self.episode_manifest_path}")

        manifest_rows: list[dict[str, str]] = []
        with self.episode_manifest_path.open(newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            required = {"split", "repo", "episode_index", "task_text"}
            missing = required - set(reader.fieldnames or [])
            if missing:
                raise KeyError(f"Manifest {self.episode_manifest_path} is missing columns: {sorted(missing)}")
            for row in reader:
                if row["split"] != self.episode_manifest_split:
                    continue
                if row["repo"] not in self.repo_name_to_root:
                    continue
                manifest_rows.append(row)

        if max_episodes is not None:
            manifest_rows = manifest_rows[: int(max_episodes)]
        if not manifest_rows:
            raise ValueError(
                f"Manifest selected no episodes for split={self.episode_manifest_split!r} "
                f"and repos={sorted(self.repo_name_to_root)}"
            )

        repo_episode_meta = {repo: self._load_episode_metadata(root) for repo, root in self.repo_name_to_root.items()}
        selected: list[EpisodeRecord] = []
        seen: set[tuple[str, int]] = set()
        for row in manifest_rows:
            repo_name = row["repo"]
            episode_index = int(row["episode_index"])
            key = (repo_name, episode_index)
            if key in seen:
                raise ValueError(f"Duplicate manifest episode: {key}")
            seen.add(key)

            meta_df = repo_episode_meta[repo_name]
            matches = meta_df.loc[meta_df["episode_index"] == episode_index]
            if len(matches) != 1:
                raise ValueError(f"Expected one episode metadata row for {key}, found {len(matches)}")
            meta = matches.iloc[0]
            video_offsets = {}
            video_chunk_indices = {}
            video_file_indices = {}
            for video_key in self.video_keys:
                prefix = f"videos/{video_key}"
                video_offsets[video_key] = float(meta[f"{prefix}/from_timestamp"])
                video_chunk_indices[video_key] = int(meta[f"{prefix}/chunk_index"])
                video_file_indices[video_key] = int(meta[f"{prefix}/file_index"])

            selected.append(
                EpisodeRecord(
                    repo_root=self.repo_name_to_root[repo_name],
                    repo_name=repo_name,
                    episode_index=episode_index,
                    length=int(meta["length"]),
                    task_text=str(row["task_text"]),
                    data_chunk_index=int(meta["data/chunk_index"]),
                    data_file_index=int(meta["data/file_index"]),
                    video_offsets=video_offsets,
                    video_chunk_indices=video_chunk_indices,
                    video_file_indices=video_file_indices,
                )
            )
        return selected

    @staticmethod
    def _load_episode_metadata(repo_root: Path) -> pd.DataFrame:
        info_path = repo_root / "meta" / "info.json"
        if not info_path.exists():
            raise FileNotFoundError(f"Missing RoboCasa info.json: {info_path}")
        info = json.loads(info_path.read_text(encoding="utf-8"))
        if str(info.get("codebase_version")) != "v3.0":
            raise ValueError(f"Expected LeRobot v3.0 metadata in {repo_root}, got {info.get('codebase_version')!r}")
        files = sorted((repo_root / "meta" / "episodes").glob("chunk-*/file-*.parquet"))
        if not files:
            raise FileNotFoundError(f"No episode metadata parquet files under {repo_root / 'meta' / 'episodes'}")
        return pd.concat([pd.read_parquet(path) for path in files], ignore_index=True)

    def _build_windows(self) -> list[tuple[int, int]]:
        windows: list[tuple[int, int]] = []
        for episode_pos, episode in enumerate(self.episodes):
            valid_count = max(0, episode.length - self.action_horizon)
            starts = list(range(0, valid_count, self.window_stride))
            if self.max_windows_per_episode is not None:
                starts = starts[: self.max_windows_per_episode]
            windows.extend((episode_pos, start) for start in starts)
            if self.max_samples is not None and len(windows) >= self.max_samples:
                return windows[: self.max_samples]
        return windows

    def _get(self, idx: int) -> dict[str, Any]:
        episode_pos, start = self.windows[idx]
        episode = self.episodes[episode_pos]
        states, actions, timestamps = self._load_episode_arrays(episode)

        action_raw = torch.as_tensor(actions[start : start + self.action_horizon], dtype=torch.float32)
        proprio_raw = torch.as_tensor(states[start : start + self.action_horizon], dtype=torch.float32)
        if action_raw.shape != (self.action_horizon, self.shape_meta["action"][0]["raw_shape"]):
            raise ValueError(f"Bad action window shape: {tuple(action_raw.shape)}")
        if proprio_raw.shape != (self.action_horizon, self.shape_meta["state"][0]["raw_shape"]):
            raise ValueError(f"Bad proprio window shape: {tuple(proprio_raw.shape)}")

        video_indices = [start + offset for offset in self.frame_offsets]
        video = self._load_video(episode, timestamps[video_indices])
        action, proprio, action_dim_is_pad, proprio_dim_is_pad = self._normalize_action_state(action_raw, proprio_raw)
        context, context_mask = self._get_cached_text_context(DEFAULT_PROMPT.format(task=episode.task_text))

        return {
            "video": video,
            "action": action,
            "proprio": proprio,
            "prompt": DEFAULT_PROMPT.format(task=episode.task_text),
            "context": context,
            "context_mask": context_mask,
            "image_is_pad": torch.zeros(self.num_frames, dtype=torch.bool),
            "action_is_pad": torch.zeros(self.action_horizon, dtype=torch.bool),
            "proprio_is_pad": torch.zeros(self.action_horizon, dtype=torch.bool),
            "action_dim_is_pad": action_dim_is_pad,
            "proprio_dim_is_pad": proprio_dim_is_pad,
            "idx": torch.tensor(idx, dtype=torch.long),
            "episode_index": torch.tensor(episode.episode_index, dtype=torch.long),
            "window_start": torch.tensor(start, dtype=torch.long),
        }

    def _load_episode_arrays(self, episode: EpisodeRecord) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        cache_key = (episode.repo_name, episode.episode_index)
        cached = self._episode_cache.get(cache_key)
        if cached is not None:
            self._episode_cache.move_to_end(cache_key)
            return cached

        file_path = self._data_file_path(episode)
        df = self._load_data_file(file_path)
        ep_df = df.loc[df["episode_index"] == episode.episode_index].sort_values("frame_index")
        if len(ep_df) != episode.length:
            raise ValueError(
                f"Episode length mismatch for {cache_key}: metadata={episode.length} parquet_rows={len(ep_df)}"
            )
        states = np.asarray(list(ep_df["observation.state"]), dtype=np.float32)
        actions = np.asarray(list(ep_df["action"]), dtype=np.float32)
        timestamps = ep_df["timestamp"].to_numpy(dtype=np.float32)
        result = (states, actions, timestamps)

        self._episode_cache[cache_key] = result
        self._episode_cache.move_to_end(cache_key)
        while len(self._episode_cache) > self.episode_cache_size:
            self._episode_cache.popitem(last=False)
        return result

    def _load_data_file(self, file_path: Path) -> pd.DataFrame:
        cached = self._data_file_cache.get(file_path)
        if cached is not None:
            self._data_file_cache.move_to_end(file_path)
            return cached
        table = pq.read_table(
            file_path,
            columns=["episode_index", "frame_index", "timestamp", "observation.state", "action"],
        )
        df = table.to_pandas()
        self._data_file_cache[file_path] = df
        self._data_file_cache.move_to_end(file_path)
        while len(self._data_file_cache) > self.data_file_cache_size:
            self._data_file_cache.popitem(last=False)
        return df

    def _data_file_path(self, episode: EpisodeRecord) -> Path:
        return (
            episode.repo_root
            / "data"
            / f"chunk-{episode.data_chunk_index:03d}"
            / f"file-{episode.data_file_index:03d}.parquet"
        )

    def _video_file_path(self, episode: EpisodeRecord, video_key: str) -> Path:
        return (
            episode.repo_root
            / "videos"
            / video_key
            / f"chunk-{episode.video_chunk_indices[video_key]:03d}"
            / f"file-{episode.video_file_indices[video_key]:03d}.mp4"
        )

    def _load_video(self, episode: EpisodeRecord, episode_timestamps: np.ndarray) -> torch.Tensor:
        per_camera: list[torch.Tensor] = []
        camera_h = self.video_size[0]
        camera_w = self.video_size[1] // len(self.video_keys) if self.concat_multi_camera == "horizontal" else self.video_size[1]
        if self.concat_multi_camera == "vertical":
            camera_h = self.video_size[0] // len(self.video_keys)

        for video_key in self.video_keys:
            query_timestamps = [float(episode.video_offsets[video_key] + ts) for ts in episode_timestamps]
            frames = decode_video_frames(
                self._video_file_path(episode, video_key),
                query_timestamps,
                tolerance_s=self.tolerance_s,
                backend=self.video_backend,
            )
            if frames.ndim != 4:
                raise ValueError(f"Decoded video must be [T,C,H,W], got {tuple(frames.shape)}")
            frames = transforms_F.resize(
                frames,
                size=[camera_h, camera_w],
                interpolation=transforms_F.InterpolationMode.BILINEAR,
                antialias=True,
            )
            per_camera.append(frames)

        if len(per_camera) == 1:
            video = per_camera[0]
        elif self.concat_multi_camera == "horizontal":
            video = torch.cat(per_camera, dim=-1)
        elif self.concat_multi_camera == "vertical":
            video = torch.cat(per_camera, dim=-2)
        else:
            raise ValueError(f"Unsupported concat_multi_camera={self.concat_multi_camera!r}")

        video = video * 2.0 - 1.0
        return video.permute(1, 0, 2, 3).contiguous()

    def _normalize_action_state(
        self,
        action_raw: torch.Tensor,
        proprio_raw: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        if self.processor is None:
            action_dim_is_pad = torch.zeros(action_raw.shape[-1], dtype=torch.bool)
            proprio_dim_is_pad = torch.zeros(proprio_raw.shape[-1], dtype=torch.bool)
            return action_raw, proprio_raw, action_dim_is_pad, proprio_dim_is_pad

        batch = {
            "action": {"default": action_raw},
            "state": {"default": proprio_raw},
        }
        batch = self.processor.action_state_transform(batch)
        batch = self.processor.normalizer.forward(batch)
        batch = self.processor.action_state_merger.forward(batch)
        return (
            batch["action"].to(torch.float32),
            batch["state"].to(torch.float32),
            batch["action_dim_is_pad"],
            batch["state_dim_is_pad"],
        )

    def _get_cached_text_context(self, prompt: str) -> tuple[torch.Tensor, torch.Tensor]:
        if self.text_embedding_cache_dir is None:
            if self.allow_missing_text_embeds:
                return self._empty_text_context()
            raise ValueError("`text_embedding_cache_dir` is not set.")
        hashed = hashlib.sha256(prompt.encode("utf-8")).hexdigest()
        cache_path = self.text_embedding_cache_dir / f"{hashed}.t5_len{self.context_len}.{self.text_encoder_id}.pt"
        if not cache_path.exists():
            if self.allow_missing_text_embeds:
                return self._empty_text_context()
            raise FileNotFoundError(f"Missing RoboCasa text cache: {cache_path}")
        payload = torch.load(cache_path, map_location="cpu")
        context = payload["context"].to(torch.float32)
        mask = payload["mask"].bool()
        if context.ndim != 2 or mask.ndim != 1:
            raise ValueError(f"Invalid text cache payload in {cache_path}: {tuple(context.shape)}, {tuple(mask.shape)}")
        if context.shape[0] != self.context_len or mask.shape[0] != self.context_len:
            raise ValueError(f"Text cache context_len mismatch in {cache_path}")
        context = context.clone()
        context[~mask] = 0.0
        return context, torch.ones_like(mask, dtype=torch.bool)

    def _empty_text_context(self) -> tuple[torch.Tensor, torch.Tensor]:
        return (
            torch.zeros((self.context_len, 4096), dtype=torch.float32),
            torch.zeros((self.context_len,), dtype=torch.bool),
        )
