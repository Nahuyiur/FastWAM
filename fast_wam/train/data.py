"""Official Fast-WAM LIBERO-v2.1 data semantics without a LeRobot dependency.

The implementation intentionally follows the current Fast-WAM repository:

* query 33 consecutive 20 Hz observations and 32 actions;
* clamp queries at episode boundaries and retain explicit padding masks;
* zero padded delta-action dimensions 0..5 before normalization;
* resize both cameras to 224x224, concatenate horizontally, then select
  frames 0,4,...,32 and normalize pixels to [-1,1];
* use the released global min/max statistics and cached UMT5 context.

Unlike the upstream dataset, decoding errors are fatal.  Randomly replacing a
bad sample would make Megatron resume and parity tests nondeterministic.
"""

from __future__ import annotations

import bisect
import hashlib
import json
import math
from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Sequence

import torch
from torch.utils.data import Dataset, default_collate

from ..config import FastWAMConfig


LIBERO_SUITE_DIRS = (
    "libero_spatial_no_noops_lerobot",
    "libero_object_no_noops_lerobot",
    "libero_goal_no_noops_lerobot",
    "libero_10_no_noops_lerobot",
)
CAMERA_KEYS = (
    "observation.images.image",
    "observation.images.wrist_image",
)
EXPECTED_EPISODES = 1712
EXPECTED_FRAMES = 277713
LATENT_CACHE_VERSION = 1
LATENT_SHAPE = (48, 3, 14, 28)
LATENT_DTYPE = torch.bfloat16


@dataclass(frozen=True)
class _Episode:
    root: Path
    index: int
    length: int
    task: str
    fps: int
    chunk_size: int
    data_template: str
    video_template: str

    @property
    def chunk(self) -> int:
        return self.index // self.chunk_size

    @property
    def data_path(self) -> Path:
        return self.root / self.data_template.format(
            episode_chunk=self.chunk,
            episode_index=self.index,
        )

    def video_path(self, camera: str) -> Path:
        return self.root / self.video_template.format(
            episode_chunk=self.chunk,
            episode_index=self.index,
            video_key=camera,
        )


class _EpisodeCache:
    """Small worker-local parquet cache."""

    def __init__(self, capacity: int):
        self.capacity = max(int(capacity), 0)
        self._items: OrderedDict[Path, dict[str, torch.Tensor]] = OrderedDict()
        self._frozen = False

    def freeze(self) -> None:
        self._frozen = True

    def get(self, episode: _Episode) -> dict[str, torch.Tensor]:
        path = episode.data_path
        cached = self._items.get(path)
        if cached is not None:
            if not self._frozen:
                self._items.move_to_end(path)
            return cached
        if self._frozen:
            raise KeyError(f"Frozen episode cache is missing {path}")

        try:
            import pyarrow.parquet as pq
        except ImportError as exc:  # pragma: no cover - environment dependency
            raise RuntimeError("Fast-WAM LIBERO loading requires pyarrow") from exc
        if not path.is_file():
            raise FileNotFoundError(path)
        table = pq.read_table(
            path,
            columns=["observation.state", "action", "timestamp"],
        )
        result = {
            "state": torch.tensor(
                table.column("observation.state").to_pylist(),
                dtype=torch.float32,
            ),
            "action": torch.tensor(
                table.column("action").to_pylist(),
                dtype=torch.float32,
            ),
            "timestamp": torch.tensor(
                table.column("timestamp").to_pylist(),
                dtype=torch.float32,
            ),
        }
        lengths = {value.shape[0] for value in result.values()}
        if lengths != {episode.length}:
            raise ValueError(
                f"{path}: metadata length={episode.length}, parquet lengths={sorted(lengths)}"
            )
        if self.capacity:
            self._items[path] = result
            self._items.move_to_end(path)
            while len(self._items) > self.capacity:
                self._items.popitem(last=False)
        return result


def _jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        raise FileNotFoundError(path)
    rows = []
    with path.open(encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON at {path}:{line_number}") from exc
    return rows


def _load_episodes(
    dataset_dirs: Sequence[str | Path],
    *,
    validate_official_release: bool,
) -> list[_Episode]:
    episodes: list[_Episode] = []
    total_frames = 0
    for root_value in dataset_dirs:
        root = Path(root_value).expanduser().resolve()
        info_path = root / "meta" / "info.json"
        if not info_path.is_file():
            raise FileNotFoundError(info_path)
        info = json.loads(info_path.read_text(encoding="utf-8"))
        if info.get("codebase_version") != "v2.1":
            raise ValueError(f"{root}: expected LeRobot v2.1, got {info.get('codebase_version')!r}")
        if int(info["fps"]) != 20:
            raise ValueError(f"{root}: Fast-WAM LIBERO requires 20 fps, got {info['fps']}")
        features = info["features"]
        if tuple(features["observation.state"]["shape"]) != (8,):
            raise ValueError(f"{root}: observation.state must have shape [8]")
        if tuple(features["action"]["shape"]) != (7,):
            raise ValueError(f"{root}: action must have shape [7]")
        for camera in CAMERA_KEYS:
            if camera not in features or features[camera].get("dtype") != "video":
                raise ValueError(f"{root}: missing video feature {camera}")

        episode_rows = _jsonl(root / "meta" / "episodes.jsonl")
        if len(episode_rows) != int(info["total_episodes"]):
            raise ValueError(
                f"{root}: episode count mismatch {len(episode_rows)} != {info['total_episodes']}"
            )
        for expected_index, row in enumerate(episode_rows):
            index = int(row["episode_index"])
            if index != expected_index:
                raise ValueError(
                    f"{root}: episodes must be contiguous, expected {expected_index}, got {index}"
                )
            tasks = row.get("tasks") or []
            if len(tasks) != 1:
                raise ValueError(f"{root}: episode {index} must contain exactly one task")
            length = int(row["length"])
            if length <= 0:
                raise ValueError(f"{root}: episode {index} has invalid length {length}")
            episodes.append(
                _Episode(
                    root=root,
                    index=index,
                    length=length,
                    task=str(tasks[0]),
                    fps=int(info["fps"]),
                    chunk_size=int(info["chunks_size"]),
                    data_template=str(info["data_path"]),
                    video_template=str(info["video_path"]),
                )
            )
            total_frames += length
        if sum(int(row["length"]) for row in episode_rows) != int(info["total_frames"]):
            raise ValueError(f"{root}: total_frames disagrees with episodes.jsonl")

    if validate_official_release:
        names = tuple(Path(path).name for path in dataset_dirs)
        if names != LIBERO_SUITE_DIRS:
            raise ValueError(
                "Official Fast-WAM suite order must be "
                f"{LIBERO_SUITE_DIRS}, got {names}"
            )
        if len(episodes) != EXPECTED_EPISODES or total_frames != EXPECTED_FRAMES:
            raise ValueError(
                "Official release cardinality mismatch: "
                f"episodes={len(episodes)} (expected {EXPECTED_EPISODES}), "
                f"frames={total_frames} (expected {EXPECTED_FRAMES})"
            )
    return episodes


def official_dataset_dirs(root: str | Path) -> tuple[Path, ...]:
    root = Path(root).expanduser().resolve()
    return tuple(root / name for name in LIBERO_SUITE_DIRS)


def dataset_fingerprint(
    dataset_dirs: Sequence[str | Path],
    stats_path: str | Path,
) -> str:
    """Fingerprint official metadata plus every payload's relative name/size."""

    digest = hashlib.sha256()
    for root_value in dataset_dirs:
        root = Path(root_value).expanduser().resolve()
        digest.update(root.name.encode("utf-8"))
        for relative in ("meta/info.json", "meta/episodes.jsonl"):
            path = root / relative
            digest.update(relative.encode("utf-8"))
            digest.update(path.read_bytes())
        info = json.loads((root / "meta/info.json").read_text(encoding="utf-8"))
        for row in _jsonl(root / "meta/episodes.jsonl"):
            episode_index = int(row["episode_index"])
            chunk = episode_index // int(info["chunks_size"])
            paths = [
                root
                / str(info["data_path"]).format(
                    episode_chunk=chunk,
                    episode_index=episode_index,
                )
            ]
            paths.extend(
                root
                / str(info["video_path"]).format(
                    episode_chunk=chunk,
                    episode_index=episode_index,
                    video_key=camera,
                )
                for camera in CAMERA_KEYS
            )
            for path in paths:
                stat = path.stat()
                digest.update(str(path.relative_to(root)).encode("utf-8"))
                digest.update(int(stat.st_size).to_bytes(8, "little"))
    stats = Path(stats_path).expanduser().resolve()
    digest.update(stats.read_bytes())
    return digest.hexdigest()


def latent_preprocessing_fingerprint(config: FastWAMConfig) -> str:
    """Version the exact pixels-to-standardized-latents contract."""

    payload = {
        "cache_version": LATENT_CACHE_VERSION,
        "camera_keys": CAMERA_KEYS,
        "observation_horizon": config.action_horizon + 1,
        "video_indices": list(
            range(
                0,
                config.action_horizon + 1,
                config.temporal_downsample_factor,
            )
        ),
        "camera_resize": [224, 224],
        "camera_interpolation": "bilinear_antialias",
        "concat": "width",
        "output_size": list(config.image_size),
        "output_resize": "bicubic_antialias_center_crop",
        "pixel_normalization": "(x-0.5)/0.5",
        "vae": "WanVideoVAE38Encoder.encode_normalized_video",
        "latent_shape": LATENT_SHAPE,
        "latent_dtype": "bfloat16",
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


class LatentCache:
    """Read immutable BF16 latent shards through memory-mapped tensors."""

    def __init__(
        self,
        root: str | Path,
        *,
        expected_samples: int,
        dataset_digest: str,
        preprocessing_digest: str,
    ):
        self.root = Path(root).expanduser().resolve()
        manifest_path = self.root / "manifest.json"
        if not manifest_path.is_file():
            raise FileNotFoundError(manifest_path)
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        expected = {
            "version": LATENT_CACHE_VERSION,
            "complete": True,
            "num_samples": expected_samples,
            "dtype": "bfloat16",
            "sample_shape": list(LATENT_SHAPE),
            "dataset_fingerprint": dataset_digest,
            "preprocessing_fingerprint": preprocessing_digest,
        }
        mismatches = {
            key: (manifest.get(key), value)
            for key, value in expected.items()
            if manifest.get(key) != value
        }
        if mismatches:
            raise ValueError(
                f"{manifest_path}: latent-cache contract mismatch: {mismatches}"
            )
        self.samples_per_shard = int(manifest["samples_per_shard"])
        self.shards = list(manifest["shards"])
        self._maps: dict[int, torch.Tensor] = {}
        sample_elements = math.prod(LATENT_SHAPE)
        for shard_id, shard in enumerate(self.shards):
            if int(shard["start"]) != shard_id * self.samples_per_shard:
                raise ValueError(f"{manifest_path}: non-contiguous shard {shard_id}")
            path = self.root / str(shard["file"])
            expected_bytes = int(shard["count"]) * sample_elements * 2
            if path.stat().st_size != expected_bytes:
                raise ValueError(
                    f"{path}: size={path.stat().st_size}, expected={expected_bytes}"
                )

    def __getitem__(self, index: int) -> torch.Tensor:
        shard_id = index // self.samples_per_shard
        shard = self.shards[shard_id]
        mapped = self._maps.get(shard_id)
        if mapped is None:
            count = int(shard["count"])
            mapped = torch.from_file(
                str(self.root / str(shard["file"])),
                shared=False,
                size=count * math.prod(LATENT_SHAPE),
                dtype=LATENT_DTYPE,
            ).view(count, *LATENT_SHAPE)
            self._maps[shard_id] = mapped
        return mapped[index - int(shard["start"])]


def _normalization_from_stats(
    stats_path: str | Path,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    path = Path(stats_path).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(path)
    stats = json.loads(path.read_text(encoding="utf-8"))
    if int(stats.get("num_episodes", -1)) != EXPECTED_EPISODES:
        raise ValueError(
            f"{path}: expected num_episodes={EXPECTED_EPISODES}, "
            f"got {stats.get('num_episodes')}"
        )
    if int(stats.get("num_transition", -1)) != EXPECTED_FRAMES:
        raise ValueError(
            f"{path}: expected num_transition={EXPECTED_FRAMES}, "
            f"got {stats.get('num_transition')}"
        )

    def scale_offset(kind: str) -> tuple[torch.Tensor, torch.Tensor]:
        field = stats[kind]["default"]
        low = torch.tensor(field["global_min"], dtype=torch.float32)
        high = torch.tensor(field["global_max"], dtype=torch.float32)
        value_range = high - low
        ignore = value_range < 1.0e-4
        value_range = value_range.clone()
        value_range[ignore] = 2.0
        scale = 2.0 / value_range
        offset = -1.0 - scale * low
        offset[ignore] = -low[ignore]
        return scale, offset

    action_scale, action_offset = scale_offset("action")
    state_scale, state_offset = scale_offset("state")
    if action_scale.shape != (7,) or state_scale.shape != (8,):
        raise ValueError(f"{path}: invalid action/state normalization dimensions")
    return action_scale, action_offset, state_scale, state_offset


def _decode_video_frames(
    video_path: Path,
    timestamps: list[float],
    *,
    tolerance_s: float = 1.0e-4,
) -> torch.Tensor:
    """Copy the LeRobot-v2.1 PyAV VideoReader timestamp-selection contract."""

    try:
        import torchvision
    except ImportError as exc:  # pragma: no cover - environment dependency
        raise RuntimeError("Fast-WAM LIBERO video loading requires torchvision") from exc
    if not video_path.is_file():
        raise FileNotFoundError(video_path)
    torchvision.set_video_backend("pyav")
    reader = torchvision.io.VideoReader(str(video_path), "video")
    reader.seek(min(timestamps), keyframes_only=True)
    loaded_frames: list[torch.Tensor] = []
    loaded_timestamps: list[float] = []
    last_timestamp = max(timestamps)
    try:
        for frame in reader:
            current = float(frame["pts"])
            loaded_frames.append(frame["data"])
            loaded_timestamps.append(current)
            if current >= last_timestamp:
                break
    finally:
        container = getattr(reader, "container", None)
        close = getattr(container, "close", None)
        if callable(close):
            close()
        reader = None
    if not loaded_frames:
        raise RuntimeError(f"No video frames decoded from {video_path}")

    query = torch.tensor(timestamps, dtype=torch.float32)
    loaded = torch.tensor(loaded_timestamps, dtype=torch.float32)
    distance = torch.cdist(query[:, None], loaded[:, None], p=1)
    minimum, indices = distance.min(dim=1)
    if not bool((minimum < tolerance_s).all()):
        raise RuntimeError(
            f"{video_path}: timestamp tolerance violation; "
            f"queries={query.tolist()}, loaded={loaded.tolist()}, min={minimum.tolist()}"
        )
    return torch.stack([loaded_frames[index] for index in indices]).float().div_(255.0)


def _prepare_video(
    episode: _Episode,
    timestamps: list[float],
    *,
    video_indices: Sequence[int],
    image_size: tuple[int, int],
) -> torch.Tensor:
    video = prepare_episode_video(
        episode,
        timestamps,
        image_size=image_size,
    )
    return video[list(video_indices)].permute(1, 0, 2, 3).contiguous()


def prepare_episode_video(
    episode: _Episode,
    timestamps: list[float],
    *,
    image_size: tuple[int, int],
) -> torch.Tensor:
    """Decode and preprocess every frame of one episode exactly once.

    The returned layout is ``[time, channel, height, width]``.  Keeping frame
    selection outside this function lets the latent-cache builder reuse one
    video decode for every overlapping 33-observation training window.
    """

    try:
        from torchvision.transforms import InterpolationMode
        from torchvision.transforms import functional as transform
    except ImportError as exc:  # pragma: no cover - environment dependency
        raise RuntimeError("Fast-WAM image preprocessing requires torchvision") from exc

    cameras = []
    for camera in CAMERA_KEYS:
        frames = _decode_video_frames(episode.video_path(camera), timestamps)
        # Match BaseLerobotDataset._get_image followed by ToTensor exactly.
        frames = (frames * 255.0).to(torch.uint8).to(torch.float32).div_(255.0)
        frames = transform.resize(
            frames,
            [224, 224],
            interpolation=InterpolationMode.BILINEAR,
            antialias=True,
        )
        cameras.append(frames)
    video = torch.cat(cameras, dim=-1)

    target_h, target_w = image_size
    source_h, source_w = video.shape[-2:]
    ratio = max(target_w / source_w, target_h / source_h)
    resized = (
        int(ratio * source_h + 0.5),
        int(ratio * source_w + 0.5),
    )
    video = transform.resize(
        video,
        resized,
        interpolation=InterpolationMode.BICUBIC,
        antialias=True,
    )
    video = transform.center_crop(video, [target_h, target_w])
    video = transform.normalize(video, mean=[0.5], std=[0.5])
    return video.contiguous()


class LiberoTrainingDataset(Dataset):
    """The four-suite official Fast-WAM LIBERO training dataset."""

    def __init__(
        self,
        *,
        root: str | Path | None = None,
        dataset_dirs: Sequence[str | Path] | None = None,
        stats_path: str | Path,
        text_cache_dir: str | Path,
        config: FastWAMConfig | None = None,
        parquet_cache_size: int = 4,
        validate_official_release: bool = True,
        latent_cache: str | Path | None = None,
    ):
        if (root is None) == (dataset_dirs is None):
            raise ValueError("Provide exactly one of root or dataset_dirs")
        if root is not None:
            dataset_dirs = official_dataset_dirs(root)
        assert dataset_dirs is not None
        dataset_dirs = tuple(Path(path).expanduser().resolve() for path in dataset_dirs)
        self.config = config or FastWAMConfig()
        if self.config.action_horizon != 32:
            raise ValueError("Official LIBERO recipe requires action_horizon=32")
        if self.config.temporal_downsample_factor != 4:
            raise ValueError("Official LIBERO recipe requires temporal_downsample_factor=4")
        if self.config.spatial_downsample_factor != 16:
            raise ValueError("Official Wan2.2 recipe requires spatial_downsample_factor=16")
        self.observation_horizon = self.config.action_horizon + 1
        self.video_indices = tuple(
            range(0, self.observation_horizon, self.config.temporal_downsample_factor)
        )
        if self.video_indices != tuple(range(0, 33, 4)):
            raise ValueError(f"Unexpected video indices {self.video_indices}")

        self.episodes = _load_episodes(
            dataset_dirs,
            validate_official_release=validate_official_release,
        )
        self._episode_ends: list[int] = []
        total = 0
        for episode in self.episodes:
            total += episode.length
            self._episode_ends.append(total)
        (
            self.action_scale,
            self.action_offset,
            self.state_scale,
            self.state_offset,
        ) = _normalization_from_stats(stats_path)
        self.text_cache_dir = Path(text_cache_dir).expanduser().resolve()
        self.context_len = self.config.context_len
        self._contexts: dict[str, tuple[torch.Tensor, torch.Tensor]] = {}
        for task in sorted({episode.task for episode in self.episodes}):
            prompt = self.config.prompt_template.format(task=task)
            self._contexts[prompt] = self._load_context(prompt)
        self.latent_cache = None
        if latent_cache is not None:
            self.latent_cache = LatentCache(
                latent_cache,
                expected_samples=len(self),
                dataset_digest=dataset_fingerprint(dataset_dirs, stats_path),
                preprocessing_digest=latent_preprocessing_fingerprint(self.config),
            )
        self._cache = _EpisodeCache(
            len(self.episodes)
            if self.latent_cache is not None
            else parquet_cache_size
        )
        if self.latent_cache is not None:
            # State/action/timestamp total only about 18 MB. Preloading them
            # before DataLoader workers fork avoids random parquet parsing on
            # every cached-latent training batch, while the read-only tensors
            # remain shared through copy-on-write.
            for episode in self.episodes:
                self._cache.get(episode)
            self._cache.freeze()

    def __len__(self) -> int:
        return self._episode_ends[-1]

    def _locate(self, index: int) -> tuple[_Episode, int]:
        if index < 0:
            index += len(self)
        if index < 0 or index >= len(self):
            raise IndexError(index)
        episode_position = bisect.bisect_right(self._episode_ends, index)
        episode_start = 0 if episode_position == 0 else self._episode_ends[episode_position - 1]
        return self.episodes[episode_position], index - episode_start

    def _load_context(self, prompt: str) -> tuple[torch.Tensor, torch.Tensor]:
        digest = hashlib.sha256(prompt.encode("utf-8")).hexdigest()
        path = self.text_cache_dir / (
            f"{digest}.t5_len{self.context_len}.wan22ti2v5b.pt"
        )
        if not path.is_file():
            raise FileNotFoundError(
                f"Missing text cache {path}; run python -m fast_wam.train.prepare_text first"
            )
        payload = torch.load(path, map_location="cpu", weights_only=True)
        context = payload["context"].clone()
        source_mask = payload["mask"].to(torch.bool)
        if context.ndim != 2 or tuple(context.shape) != (self.context_len, self.config.video.text_dim):
            raise ValueError(f"{path}: invalid context shape {tuple(context.shape)}")
        if source_mask.shape != (self.context_len,):
            raise ValueError(f"{path}: invalid mask shape {tuple(source_mask.shape)}")
        context[~source_mask] = 0
        # Fast-WAM deliberately exposes cached zero-padding to cross attention.
        return context, torch.ones_like(source_mask)

    def _context(self, prompt: str) -> tuple[torch.Tensor, torch.Tensor]:
        return self._contexts[prompt]

    def __getitem__(self, index: int) -> dict[str, Any]:
        episode, frame = self._locate(index)
        try:
            table = self._cache.get(episode)
            observation_indices = torch.arange(
                frame,
                frame + self.observation_horizon,
                dtype=torch.long,
            )
            action_indices = observation_indices[:-1]
            observation_pad = observation_indices >= episode.length
            action_pad = action_indices >= episode.length
            observation_indices.clamp_(max=episode.length - 1)
            action_indices.clamp_(max=episode.length - 1)

            state = table["state"][observation_indices].clone()
            action = table["action"][action_indices].clone()
            # Official FastWAMProcessor.delta_action_dim_mask behavior.
            action[action_pad, :6] = 0.0
            state = torch.clamp(
                state * self.state_scale + self.state_offset,
                -5.0,
                5.0,
            )
            action = torch.clamp(
                action * self.action_scale + self.action_offset,
                -5.0,
                5.0,
            )
            if self.latent_cache is None:
                timestamps = table["timestamp"][observation_indices].tolist()
                video = _prepare_video(
                    episode,
                    timestamps,
                    video_indices=self.video_indices,
                    image_size=self.config.image_size,
                )
                input_latents = None
            else:
                video = None
                input_latents = self.latent_cache[index]
            prompt = self.config.prompt_template.format(task=episode.task)
            context, context_mask = self._context(prompt)
        except Exception as exc:
            raise RuntimeError(
                f"Fast-WAM sample decode failed at global index {index}, "
                f"dataset={episode.root}, episode={episode.index}, frame={frame}"
            ) from exc

        result = {
            "action": action,
            "proprio": state[:-1],
            "prompt": prompt,
            "context": context,
            "context_mask": context_mask,
            "image_is_pad": observation_pad[list(self.video_indices)],
            "action_is_pad": action_pad,
            # Upstream keeps the original 33-step state padding mask even
            # though ``proprio`` itself is sliced to 32 steps.
            "proprio_is_pad": observation_pad,
            "idx": int(index),
        }
        if input_latents is None:
            result["video"] = video
        else:
            result["input_latents"] = input_latents
        return result


def libero_collate(batch: Iterable[dict[str, Any]]) -> dict[str, Any]:
    """Strict fixed-shape collate used by the Megatron data-loader patch."""

    items = list(batch)
    if not items:
        raise ValueError("Cannot collate an empty Fast-WAM batch")
    return default_collate(items)
