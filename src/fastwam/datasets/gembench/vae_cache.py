from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np
import torch


CACHE_VERSION = "gembench_vae_latents_v2"
DEFAULT_GEMBENCH_VAE_CACHE_DIR = (
    "/mnt/yuhan/datasets/GEMBench/fastwam_cache/vae_latents/"
    "keysteps_bbox_seed0_3cam224x672_t9_v1"
)


def episode_key_text(episode_key: bytes | str) -> str:
    if isinstance(episode_key, bytes):
        return episode_key.decode("ascii", errors="ignore")
    return str(episode_key)


def cache_key(taskvar: str, episode_key: bytes | str) -> str:
    return f"{taskvar}\t{episode_key_text(episode_key)}"


def expected_latent_shape(
    *,
    num_video_frames: int,
    video_size: Sequence[int],
    z_dim: int = 48,
    temporal_downsample_factor: int = 4,
    upsampling_factor: int = 16,
) -> tuple[int, int, int, int]:
    height, width = int(video_size[0]), int(video_size[1])
    if (int(num_video_frames) - 1) % int(temporal_downsample_factor) != 0:
        raise ValueError(
            "num_video_frames must satisfy (T - 1) % temporal_downsample_factor == 0: "
            f"T={num_video_frames}, factor={temporal_downsample_factor}"
        )
    if height % int(upsampling_factor) != 0 or width % int(upsampling_factor) != 0:
        raise ValueError(
            "video_size must be divisible by VAE upsampling_factor: "
            f"video_size={list(video_size)}, factor={upsampling_factor}"
        )
    latent_t = (int(num_video_frames) - 1) // int(temporal_downsample_factor) + 1
    return (int(z_dim), latent_t, height // int(upsampling_factor), width // int(upsampling_factor))


def build_expected_dataset_config(
    *,
    root: str | Path,
    split: str,
    subset: str,
    seed: str,
    num_video_frames: int,
    action_horizon: int,
    video_size: Sequence[int],
    camera_order: Sequence[str],
) -> dict[str, Any]:
    return {
        "root": str(Path(root).expanduser().resolve()),
        "split": str(split),
        "subset": str(subset),
        "seed": str(seed),
        "num_video_frames": int(num_video_frames),
        "action_horizon": int(action_horizon),
        "video_size": [int(video_size[0]), int(video_size[1])],
        "camera_order": [str(camera) for camera in camera_order],
    }


def write_json_atomic(path: str | Path, payload: dict[str, Any]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_name(f".{path.name}.tmp")
    with tmp_path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")
    tmp_path.replace(path)


def write_index_jsonl_atomic(path: str | Path, rows: Iterable[dict[str, Any]]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_name(f".{path.name}.tmp")
    with tmp_path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True) + "\n")
    tmp_path.replace(path)


class GEMBenchVAELatentCache:
    def __init__(
        self,
        cache_dir: str | Path,
        *,
        expected_dataset_config: dict[str, Any] | None = None,
        expected_index: Sequence[tuple[str, bytes]] | None = None,
        expected_vae_hash: str | None = None,
        expected_encode_autocast: bool | None = None,
    ):
        self.cache_dir = Path(cache_dir).expanduser().resolve()
        self.manifest_path = self.cache_dir / "manifest.json"
        self.index_path = self.cache_dir / "index.jsonl"
        self.latents_path = self.cache_dir / "video_latents.float32.npy"
        self.completed_path = self.cache_dir / "completed_rows.bool.npy"

        self.manifest = self._load_manifest()
        self._validate_manifest(expected_dataset_config, expected_vae_hash, expected_encode_autocast)
        self.rows = self._load_rows()
        self.row_by_key = {cache_key(row["taskvar"], row["episode_key"]): int(row["row_id"]) for row in self.rows}
        if len(self.row_by_key) != len(self.rows):
            raise ValueError(f"VAE cache index contains duplicate taskvar/episode_key rows: {self.index_path}")
        self.latents = np.load(self.latents_path, mmap_mode="r")
        self._validate_arrays()
        if expected_index is not None:
            self._validate_index_coverage(expected_index)

    def _load_manifest(self) -> dict[str, Any]:
        if not self.manifest_path.exists():
            raise FileNotFoundError(f"Missing VAE latent cache manifest: {self.manifest_path}")
        with self.manifest_path.open("r", encoding="utf-8") as handle:
            manifest = json.load(handle)
        if manifest.get("cache_version") != CACHE_VERSION:
            raise ValueError(
                f"Unsupported VAE latent cache version in {self.manifest_path}: "
                f"{manifest.get('cache_version')!r} != {CACHE_VERSION!r}"
            )
        if not bool(manifest.get("complete", False)):
            raise ValueError(f"VAE latent cache is not marked complete: {self.manifest_path}")
        return manifest

    def _validate_manifest(
        self,
        expected_dataset_config: dict[str, Any] | None,
        expected_vae_hash: str | None,
        expected_encode_autocast: bool | None,
    ) -> None:
        if expected_dataset_config is not None:
            dataset_cfg = self.manifest.get("dataset", {})
            for key, expected in expected_dataset_config.items():
                actual = dataset_cfg.get(key)
                if actual != expected:
                    raise ValueError(
                        "VAE latent cache dataset config mismatch: "
                        f"key={key!r} actual={actual!r} expected={expected!r} cache={self.cache_dir}"
                    )
        if expected_vae_hash is not None:
            actual_hash = self.manifest.get("vae", {}).get("hash")
            if actual_hash != expected_vae_hash:
                raise ValueError(
                    f"VAE latent cache hash mismatch: actual={actual_hash!r} expected={expected_vae_hash!r}"
                )
        if expected_encode_autocast is not None:
            actual_autocast = self.manifest.get("vae", {}).get("encode_autocast")
            if actual_autocast != expected_encode_autocast:
                raise ValueError(
                    "VAE latent cache autocast mode mismatch: "
                    f"actual={actual_autocast!r} expected={expected_encode_autocast!r} cache={self.cache_dir}"
                )

    def _load_rows(self) -> list[dict[str, Any]]:
        if not self.index_path.exists():
            raise FileNotFoundError(f"Missing VAE latent cache index: {self.index_path}")
        rows: list[dict[str, Any]] = []
        with self.index_path.open("r", encoding="utf-8") as handle:
            for line_no, line in enumerate(handle, start=1):
                line = line.strip()
                if not line:
                    continue
                row = json.loads(line)
                for field in ("row_id", "taskvar", "episode_key"):
                    if field not in row:
                        raise ValueError(f"Malformed VAE cache index line {line_no}: missing {field!r}")
                rows.append(row)
        rows.sort(key=lambda row: int(row["row_id"]))
        for expected_id, row in enumerate(rows):
            row_id = int(row["row_id"])
            if row_id != expected_id:
                raise ValueError(f"VAE cache index row_id must be contiguous: got {row_id}, expected {expected_id}")
        return rows

    def _validate_arrays(self) -> None:
        num_rows = int(self.manifest.get("num_rows", -1))
        latent_shape = tuple(int(x) for x in self.manifest.get("latent_shape", []))
        if len(latent_shape) != 4:
            raise ValueError(f"Invalid latent_shape in {self.manifest_path}: {latent_shape}")
        expected_shape = (num_rows, *latent_shape)
        if tuple(self.latents.shape) != expected_shape:
            raise ValueError(f"VAE latent array shape mismatch: {tuple(self.latents.shape)} vs {expected_shape}")
        if self.latents.dtype != np.float32:
            raise ValueError(f"VAE latent array dtype must be float32, got {self.latents.dtype}")
        if len(self.rows) != num_rows:
            raise ValueError(f"VAE cache index length mismatch: {len(self.rows)} vs manifest num_rows={num_rows}")
        if self.completed_path.exists():
            completed = np.load(self.completed_path, mmap_mode="r")
            if tuple(completed.shape) != (num_rows,) or completed.dtype != np.bool_:
                raise ValueError(f"Invalid completed_rows array in {self.completed_path}: {completed.shape}, {completed.dtype}")
            if not bool(np.all(completed)):
                raise ValueError(f"VAE latent cache has incomplete rows: {self.completed_path}")

    def _validate_index_coverage(self, expected_index: Sequence[tuple[str, bytes]]) -> None:
        missing = []
        for taskvar, episode_key in expected_index:
            key = cache_key(taskvar, episode_key)
            if key not in self.row_by_key:
                missing.append(key)
                if len(missing) >= 8:
                    break
        if missing:
            raise KeyError(f"VAE latent cache is missing dataset rows, first missing={missing}")

    def __len__(self) -> int:
        return len(self.rows)

    @property
    def latent_shape(self) -> tuple[int, int, int, int]:
        return tuple(int(x) for x in self.manifest["latent_shape"])

    def row_id(self, taskvar: str, episode_key: bytes | str) -> int:
        key = cache_key(taskvar, episode_key)
        try:
            return self.row_by_key[key]
        except KeyError as exc:
            raise KeyError(f"Missing VAE latent cache row for taskvar={taskvar!r} episode={episode_key_text(episode_key)!r}") from exc

    def get(self, taskvar: str, episode_key: bytes | str) -> torch.Tensor:
        row_id = self.row_id(taskvar, episode_key)
        latent = np.array(self.latents[row_id], dtype=np.float32, copy=True)
        return torch.from_numpy(latent)
