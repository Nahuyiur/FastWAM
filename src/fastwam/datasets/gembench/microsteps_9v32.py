from __future__ import annotations

import hashlib
import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Sequence

import numpy as np
import torch
import torch.nn.functional as F
from omegaconf import DictConfig, OmegaConf

from .instructions import instruction_for_taskvar, load_instruction_map
from .lmdb_reader import LMDBEpisodeStore
from .normalization import DEFAULT_SHAPE_META, GEMBenchProcessorShim, load_or_create_stats, scanned_dataset_stats
from .policy_local_frame import PolicyLocalFrameConfig, action_world_to_local, compute_policy_local_frame


SCHEMA_VERSION = "gembench_microsteps_9v32_v1"
VAE_LATENT_CACHE_VERSION = "gembench_microsteps_9v32_vae_latents_v1"
DEFAULT_PROMPT = "A video recorded from a robot's point of view executing the following instruction: {task}"
DEFAULT_FRAME_OFFSETS = (0, 4, 8, 12, 16, 20, 24, 28, 32)
DEFAULT_CAMERA_ORDER = ("front", "wrist", "left_shoulder")
DEFAULT_CACHE_CAMERA_ORDER = DEFAULT_CAMERA_ORDER
OFFICIAL_GEMBENCH_CAMERA_ORDER = ("left_shoulder", "right_shoulder", "wrist", "front")


def parse_taskvar(taskvar: str) -> tuple[str, int]:
    if "+" not in taskvar:
        raise ValueError(f"Expected taskvar `task+variation`, got {taskvar!r}")
    task, variation = taskvar.rsplit("+", 1)
    return task, int(variation)


def make_taskvar(task: str, variation: int) -> str:
    return f"{task}+{int(variation)}"


def safe_token(value: Any) -> str:
    text = str(value)
    return "".join(ch if ch.isalnum() or ch in "._-" else "_" for ch in text).strip("._") or "item"


def cache_episode_path(cache_dir: str | Path, *, seed: str, taskvar: str, episode_key: str) -> Path:
    return Path(cache_dir).expanduser() / str(seed) / safe_token(taskvar) / f"{safe_token(episode_key)}.npz"


def load_manifest(path: str | Path) -> dict[str, Any]:
    payload = json.loads(Path(path).expanduser().read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"GEMBench microsteps 9v32 manifest must be a JSON object: {path}")
    return payload


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).expanduser().open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def manifest_demo_rows(payload: dict[str, Any]) -> list[dict[str, Any]]:
    rows = payload.get("demos")
    if not isinstance(rows, list):
        raise ValueError("GEMBench microsteps 9v32 manifest is missing `demos` list.")
    out: list[dict[str, Any]] = []
    for row in rows:
        if not isinstance(row, dict):
            raise ValueError(f"Manifest demo row must be object, got {type(row).__name__}")
        out.append(row)
    return out


def _npz_scalar_str(payload: Any, key: str) -> str:
    value = np.asarray(payload[key])
    return str(value.item() if value.shape == () else value.tolist())


def _shape_meta_from_processor(processor_cfg: Any | None, *, camera_order: Sequence[str], video_size: Sequence[int]) -> dict:
    shape_meta = OmegaConf.to_container(DictConfig(DEFAULT_SHAPE_META), resolve=True)
    if processor_cfg is not None:
        plain = OmegaConf.to_container(processor_cfg, resolve=True) if isinstance(processor_cfg, DictConfig) else processor_cfg
        if isinstance(plain, dict) and plain.get("shape_meta") is not None:
            shape_meta = plain["shape_meta"]
            return shape_meta
    camera_width = int(video_size[1]) // len(camera_order)
    template = dict(shape_meta["images"][0]) if shape_meta.get("images") else {"raw_shape": [3, int(video_size[0]), camera_width]}
    images = []
    for camera in camera_order:
        meta = dict(template)
        meta["key"] = str(camera)
        meta["shape"] = [3, int(video_size[0]), camera_width]
        images.append(meta)
    shape_meta["images"] = images
    return shape_meta


def build_vae_cache_dataset_config(
    *,
    manifest_path: str | Path,
    manifest_sha256: str,
    rgb_cache_dir: str | Path,
    seed: str,
    frame_offsets: Sequence[int],
    action_horizon: int,
    window_stride: int,
    video_size: Sequence[int],
    camera_order: Sequence[str],
    cache_camera_order: Sequence[str],
) -> dict[str, Any]:
    return {
        "manifest_path": str(Path(manifest_path).expanduser().resolve()),
        "manifest_sha256": str(manifest_sha256),
        "rgb_cache_dir": str(Path(rgb_cache_dir).expanduser().resolve()),
        "seed": str(seed),
        "source_schema_version": SCHEMA_VERSION,
        "frame_offsets": [int(v) for v in frame_offsets],
        "num_video_frames": len(tuple(frame_offsets)),
        "action_horizon": int(action_horizon),
        "window_stride": int(window_stride),
        "video_size": [int(video_size[0]), int(video_size[1])],
        "camera_order": [str(v) for v in camera_order],
        "cache_camera_order": [str(v) for v in cache_camera_order],
    }


def window_cache_key(taskvar: str, episode_key: str, window_start: int) -> str:
    return f"{taskvar}\t{episode_key}\t{int(window_start)}"


class GEMBenchMicrosteps9V32VAELatentCache:
    def __init__(
        self,
        cache_dir: str | Path,
        *,
        expected_dataset_config: dict[str, Any] | None = None,
        expected_index: Sequence[tuple[int, int, dict[str, Any]]] | None = None,
    ):
        self.cache_dir = Path(cache_dir).expanduser().resolve()
        self.manifest_path = self.cache_dir / "manifest.json"
        self.index_path = self.cache_dir / "index.jsonl"
        self.latents_path = self.cache_dir / "video_latents.float32.npy"
        self.completed_path = self.cache_dir / "completed_windows.bool.npy"

        self.manifest = self._load_manifest()
        self._validate_manifest(expected_dataset_config)
        self.rows = self._load_rows()
        self.row_by_key = {
            window_cache_key(row["taskvar"], row["episode_key"], int(row["window_start"])): int(row["row_id"])
            for row in self.rows
        }
        if len(self.row_by_key) != len(self.rows):
            raise ValueError(f"VAE latent cache index contains duplicate rows: {self.index_path}")
        self.latents = np.load(self.latents_path, mmap_mode="r")
        self.completed = np.load(self.completed_path, mmap_mode="r")
        self._validate_arrays()
        if expected_index is not None:
            self._validate_index_coverage(expected_index)

    def _load_manifest(self) -> dict[str, Any]:
        if not self.manifest_path.exists():
            raise FileNotFoundError(f"Missing GEMBench 9v32 VAE cache manifest: {self.manifest_path}")
        payload = json.loads(self.manifest_path.read_text(encoding="utf-8"))
        if payload.get("cache_version") != VAE_LATENT_CACHE_VERSION:
            raise ValueError(
                f"Unsupported GEMBench 9v32 VAE cache version: {payload.get('cache_version')!r} "
                f"!= {VAE_LATENT_CACHE_VERSION!r}"
            )
        if not bool(payload.get("complete", False)):
            raise ValueError(f"GEMBench 9v32 VAE cache is not marked complete: {self.manifest_path}")
        return payload

    def _validate_manifest(self, expected_dataset_config: dict[str, Any] | None) -> None:
        if expected_dataset_config is None:
            return
        actual = self.manifest.get("dataset", {})
        for key, expected in expected_dataset_config.items():
            value = actual.get(key)
            if value != expected:
                raise ValueError(
                    "GEMBench 9v32 VAE cache dataset config mismatch: "
                    f"key={key!r} actual={value!r} expected={expected!r} cache={self.cache_dir}"
                )

    def _load_rows(self) -> list[dict[str, Any]]:
        if not self.index_path.exists():
            raise FileNotFoundError(f"Missing GEMBench 9v32 VAE cache index: {self.index_path}")
        rows: list[dict[str, Any]] = []
        with self.index_path.open("r", encoding="utf-8") as handle:
            for line_no, line in enumerate(handle, start=1):
                line = line.strip()
                if not line:
                    continue
                row = json.loads(line)
                for key in ("row_id", "taskvar", "episode_key", "window_start"):
                    if key not in row:
                        raise ValueError(f"Malformed VAE cache index line {line_no}: missing {key!r}")
                rows.append(row)
        rows.sort(key=lambda row: int(row["row_id"]))
        for expected_id, row in enumerate(rows):
            row_id = int(row["row_id"])
            if row_id != expected_id:
                raise ValueError(f"VAE cache index row_id mismatch: got {row_id} expected {expected_id}")
        return rows

    def _validate_arrays(self) -> None:
        expected_rows = int(self.manifest["num_windows"])
        expected_shape = tuple(int(v) for v in self.manifest["latent_shape"])
        if tuple(self.latents.shape) != (expected_rows, *expected_shape):
            raise ValueError(
                f"VAE latent array shape mismatch: actual={tuple(self.latents.shape)} "
                f"expected={(expected_rows, *expected_shape)}"
            )
        if tuple(self.completed.shape) != (expected_rows,):
            raise ValueError(
                f"VAE completion array shape mismatch: actual={tuple(self.completed.shape)} expected={(expected_rows,)}"
            )
        if not bool(np.all(self.completed)):
            missing = np.where(~np.asarray(self.completed))[0][:10].tolist()
            raise ValueError(f"GEMBench 9v32 VAE cache has incomplete rows, first_missing={missing}")

    def _validate_index_coverage(self, expected_index: Sequence[tuple[int, int, dict[str, Any]]]) -> None:
        if len(expected_index) != len(self.rows):
            raise ValueError(f"VAE cache row count mismatch: cache={len(self.rows)} expected={len(expected_index)}")
        for row_id, (row_idx, window_start, demo_row) in enumerate(expected_index):
            expected_key = window_cache_key(str(demo_row["taskvar"]), str(demo_row["episode_key"]), int(window_start))
            actual_row = self.rows[row_id]
            actual_key = window_cache_key(
                str(actual_row["taskvar"]),
                str(actual_row["episode_key"]),
                int(actual_row["window_start"]),
            )
            if actual_key != expected_key:
                raise ValueError(
                    f"VAE cache index mismatch at row_id={row_id}: actual={actual_key!r} expected={expected_key!r}"
                )

    def get(self, row: dict[str, Any], window_start: int) -> torch.Tensor:
        key = window_cache_key(str(row["taskvar"]), str(row["episode_key"]), int(window_start))
        row_id = self.row_by_key[key]
        return torch.from_numpy(np.asarray(self.latents[row_id]).copy())


class GEMBenchMicrosteps9V32Dataset(torch.utils.data.Dataset):
    """FastWAM-style GEMBench dataset backed by rendered dense microstep cache.

    Each item is a dense 32-action window with 9 visual anchors at offsets
    `[0,4,...,32]`. Simulator rendering is intentionally kept out of this class:
    missing cache files are a hard error so training cannot silently fall back to
    the legacy sparse `keysteps_bbox` 9/8 path.
    """

    def __init__(
        self,
        manifest_path: str,
        rgb_cache_dir: str,
        *,
        split: str = "train",
        seed: str = "seed0",
        frame_offsets: Sequence[int] = DEFAULT_FRAME_OFFSETS,
        action_horizon: int = 32,
        video_size: Sequence[int] = (224, 672),
        camera_order: Sequence[str] = DEFAULT_CAMERA_ORDER,
        cache_camera_order: Sequence[str] = DEFAULT_CACHE_CAMERA_ORDER,
        window_stride: int = 1,
        max_windows_per_demo: int | None = None,
        taskvars: Sequence[str] | str | None = None,
        instruction_json_path: str | None = None,
        instruction_index: int = 0,
        text_embedding_cache_dir: str | None = None,
        context_len: int = 128,
        text_dim: int = 4096,
        text_encoder_id: str = "umt5_xxl",
        cache_text_embeddings: bool = True,
        cache_gripper_arrays: bool = True,
        allow_missing_text_embeds: bool = False,
        pretrained_norm_stats: str | None = None,
        norm_default_mode: str = "-2.0/2.0",
        stats_scan_limit: int = 0,
        allow_partial_cache: bool = False,
        vae_latent_cache_dir: str | None = None,
        processor: Any | None = None,
    ):
        self.manifest_path = Path(manifest_path).expanduser().resolve()
        self.rgb_cache_dir = Path(rgb_cache_dir).expanduser().resolve()
        self.split = str(split)
        self.seed = str(seed)
        self.frame_offsets = tuple(int(v) for v in frame_offsets)
        self.action_horizon = int(action_horizon)
        self.video_size = [int(video_size[0]), int(video_size[1])]
        self.camera_order = [str(c) for c in camera_order]
        self.cache_camera_order = [str(c) for c in cache_camera_order]
        self.window_stride = int(window_stride)
        self.max_windows_per_demo = None if max_windows_per_demo is None else int(max_windows_per_demo)
        self.instruction_index = int(instruction_index)
        self.text_embedding_cache_dir = None if text_embedding_cache_dir is None else str(text_embedding_cache_dir)
        self.context_len = int(context_len)
        self.text_dim = int(text_dim)
        self.text_encoder_id = str(text_encoder_id)
        self.cache_text_embeddings = bool(cache_text_embeddings)
        self._text_context_cache: dict[str, tuple[torch.Tensor, torch.Tensor]] = {}
        self.cache_gripper_arrays = bool(cache_gripper_arrays)
        self._gripper_cache: dict[str, np.ndarray] = {}
        self.allow_missing_text_embeds = bool(allow_missing_text_embeds)
        self.norm_default_mode = str(norm_default_mode)
        self.stats_scan_limit = int(stats_scan_limit)
        self.allow_partial_cache = bool(allow_partial_cache)
        self.vae_latent_cache_dir = None if vae_latent_cache_dir in (None, "", "null") else str(vae_latent_cache_dir)

        if tuple(self.frame_offsets) != DEFAULT_FRAME_OFFSETS:
            raise ValueError(f"GEMBench 9v32 requires frame_offsets={DEFAULT_FRAME_OFFSETS}, got {self.frame_offsets}")
        if self.action_horizon != 32:
            raise ValueError(f"GEMBench 9v32 requires action_horizon=32, got {self.action_horizon}")
        if len(self.frame_offsets) != 9 or max(self.frame_offsets) != self.action_horizon:
            raise ValueError("GEMBench 9v32 requires 9 visual anchors ending at the 32nd action step.")
        if self.window_stride <= 0:
            raise ValueError("window_stride must be positive.")
        if self.video_size[1] % len(self.camera_order) != 0:
            raise ValueError(f"video width {self.video_size[1]} must be divisible by cameras={len(self.camera_order)}")

        missing_cameras = [cam for cam in self.camera_order if cam not in self.cache_camera_order]
        if missing_cameras:
            raise ValueError(f"Requested cameras not present in cache_camera_order: {missing_cameras}")
        self.camera_indices = [self.cache_camera_order.index(cam) for cam in self.camera_order]

        self.manifest = load_manifest(self.manifest_path)
        if self.manifest.get("schema_version") != SCHEMA_VERSION:
            raise ValueError(
                f"Expected manifest schema_version={SCHEMA_VERSION!r}, got {self.manifest.get('schema_version')!r}"
            )
        requested_taskvars = self._resolve_taskvars(taskvars)
        self.demo_rows = self._select_demo_rows(requested_taskvars)
        self.index = self._build_window_index()
        if not self.index:
            raise ValueError(f"No cached GEMBench 9v32 windows found from manifest={self.manifest_path}")
        self.vae_latent_cache = self._load_vae_latent_cache()

        self.instruction_map = load_instruction_map(instruction_json_path)
        self.shape_meta = _shape_meta_from_processor(processor, camera_order=self.camera_order, video_size=self.video_size)
        stats = self._load_or_scan_stats(pretrained_norm_stats)
        self.processor = GEMBenchProcessorShim(
            stats,
            action_dim=8,
            proprio_dim=8,
            norm_default_mode=self.norm_default_mode,
            shape_meta=self.shape_meta,
        )
        self.lerobot_dataset = SimpleNamespace(processor=self.processor)

    def __len__(self) -> int:
        return len(self.index)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        row_idx, start = self.index[idx]
        row = self.demo_rows[row_idx]
        if self.vae_latent_cache is not None:
            cache_path = self._cache_path(row)
            video_latents = self.vae_latent_cache.get(row, int(start))
            gripper = self._load_gripper(row, cache_path)
            action_raw = gripper[int(start) + 1 : int(start) + self.action_horizon + 1]
            proprio_raw = gripper[int(start) : int(start) + self.action_horizon]
            sample = self._make_sample_from_arrays(
                row=row,
                start=int(start),
                action_raw=action_raw,
                proprio_raw=proprio_raw,
                video=None,
            )
            sample["video_latents"] = video_latents
            return sample

        return self._load_rgb_window_sample(row_idx=row_idx, start=int(start))

    def sample_autoreg_sequence(
        self,
        *,
        num_chunks: int,
        stride: int = 32,
        seed: int | None = None,
        index: int | None = None,
    ) -> dict[str, Any]:
        """Return contiguous 9V32 windows for open-loop WAM diagnostics.

        This helper intentionally does not mutate `self.index`, so training
        sampling and VAE-cache provenance stay unchanged even when validation is
        configured with `max_windows_per_demo=1`.
        """
        num_chunks = int(num_chunks)
        stride = int(stride)
        if num_chunks <= 0:
            raise ValueError(f"num_chunks must be positive, got {num_chunks}")
        if stride <= 0:
            raise ValueError(f"stride must be positive, got {stride}")

        candidates = self._autoreg_anchor_candidates(num_chunks=num_chunks, stride=stride)
        if not candidates:
            raise ValueError(
                "No GEMBench 9V32 demo supports autoregressive rollout: "
                f"num_chunks={num_chunks} stride={stride} action_horizon={self.action_horizon} "
                f"manifest={self.manifest_path}"
            )
        if index is None:
            rng = np.random.default_rng(seed)
            candidate_idx = int(rng.integers(0, len(candidates)))
        else:
            candidate_idx = int(index) % len(candidates)

        row_idx, base_start = candidates[candidate_idx]
        row = self.demo_rows[row_idx]
        starts = [int(base_start) + chunk_idx * stride for chunk_idx in range(num_chunks)]
        samples = [self._load_rgb_window_sample(row_idx=row_idx, start=start) for start in starts]
        videos = torch.stack([sample["video"] for sample in samples], dim=0)
        gt_video_sequence = torch.cat(
            [videos[0, :, 0:1], *[videos[chunk_idx, :, 1:] for chunk_idx in range(num_chunks)]],
            dim=1,
        ).contiguous()
        return {
            "samples": samples,
            "video": videos,
            "gt_video_sequence": gt_video_sequence,
            "action": torch.stack([sample["action"] for sample in samples], dim=0),
            "proprio": torch.stack([sample["proprio"] for sample in samples], dim=0),
            "prompt": samples[0]["prompt"],
            "context": samples[0]["context"],
            "context_mask": samples[0]["context_mask"],
            "taskvar": str(row["taskvar"]),
            "episode_key": str(row["episode_key"]),
            "row_idx": int(row_idx),
            "base_start": int(base_start),
            "window_starts": starts,
            "num_chunks": num_chunks,
            "chunk_stride": stride,
            "candidate_count": len(candidates),
        }

    def _load_rgb_window_sample(self, *, row_idx: int, start: int) -> dict[str, Any]:
        row = self.demo_rows[int(row_idx)]
        length = int(row["length"])
        if int(start) < 0 or int(start) + self.action_horizon >= length:
            raise IndexError(
                f"Invalid GEMBench 9V32 window start={start} length={length} action_horizon={self.action_horizon}"
            )
        cache_path = self._cache_path(row)
        payload = np.load(cache_path, allow_pickle=False)
        try:
            self._validate_cache_payload(row, payload, cache_path)
            rgb = np.asarray(payload["rgb"])
            gripper = np.asarray(payload["gripper"], dtype=np.float32)
            frame_idx = np.asarray([int(start) + offset for offset in self.frame_offsets], dtype=np.int64)
            video = self._video_tensor(rgb[frame_idx][:, self.camera_indices])
            action_raw = gripper[int(start) + 1 : int(start) + self.action_horizon + 1]
            proprio_raw = gripper[int(start) : int(start) + self.action_horizon]
            return self._make_sample_from_arrays(
                row=row,
                start=int(start),
                action_raw=action_raw,
                proprio_raw=proprio_raw,
                video=video,
            )
        finally:
            payload.close()

    def _make_sample_from_arrays(
        self,
        *,
        row: dict[str, Any],
        start: int,
        action_raw: np.ndarray,
        proprio_raw: np.ndarray,
        video: torch.Tensor | None,
    ) -> dict[str, Any]:
        action, proprio, action_dim_is_pad, proprio_dim_is_pad = self.processor.normalize(
            torch.as_tensor(action_raw, dtype=torch.float32),
            torch.as_tensor(proprio_raw, dtype=torch.float32),
        )
        instruction = instruction_for_taskvar(
            str(row["taskvar"]),
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
            "image_is_pad": torch.zeros(len(self.frame_offsets), dtype=torch.bool),
            "action_is_pad": torch.zeros(self.action_horizon, dtype=torch.bool),
            "proprio_is_pad": torch.zeros(self.action_horizon, dtype=torch.bool),
            "action_dim_is_pad": action_dim_is_pad,
            "proprio_dim_is_pad": proprio_dim_is_pad,
            "taskvar": str(row["taskvar"]),
            "episode_key": str(row["episode_key"]),
            "window_start": int(start),
        }
        if video is not None:
            sample["video"] = video
        return sample

    def _autoreg_anchor_candidates(self, *, num_chunks: int, stride: int) -> list[tuple[int, int]]:
        required_span = (int(num_chunks) - 1) * int(stride) + self.action_horizon
        out: list[tuple[int, int]] = []
        for row_idx, row in enumerate(self.demo_rows):
            length = int(row["length"])
            count = max(0, length - required_span)
            out.extend((row_idx, start) for start in range(0, count, self.window_stride))
        return out

    def _resolve_taskvars(self, taskvars: Sequence[str] | str | None) -> set[str] | None:
        if taskvars is None:
            return None
        if isinstance(taskvars, str):
            values = [item.strip() for item in taskvars.split(",") if item.strip()]
        else:
            values = [str(item).strip() for item in taskvars if str(item).strip()]
        return set(values)

    def _select_demo_rows(self, requested_taskvars: set[str] | None) -> list[dict[str, Any]]:
        rows = []
        missing_cache: list[str] = []
        for row in manifest_demo_rows(self.manifest):
            if requested_taskvars is not None and str(row.get("taskvar")) not in requested_taskvars:
                continue
            if str(row.get("seed", self.seed)) != self.seed:
                continue
            cache_path = self._cache_path(row)
            if not cache_path.is_file():
                missing_cache.append(str(cache_path))
                continue
            rows.append(row)
        if missing_cache and not self.allow_partial_cache:
            preview = ", ".join(missing_cache[:5])
            more = "" if len(missing_cache) <= 5 else f" ... (+{len(missing_cache) - 5} more)"
            raise FileNotFoundError(
                "Missing GEMBench 9v32 dense RGB cache files. "
                "Render cache first or use a smaller manifest for smoke tests: "
                f"{preview}{more}"
            )
        return rows

    def _build_window_index(self) -> list[tuple[int, int]]:
        out: list[tuple[int, int]] = []
        for row_idx, row in enumerate(self.demo_rows):
            length = int(row["length"])
            count = max(0, length - self.action_horizon)
            starts = list(range(0, count, self.window_stride))
            if self.max_windows_per_demo is not None:
                starts = starts[: self.max_windows_per_demo]
            out.extend((row_idx, start) for start in starts)
        return out

    def _load_gripper(self, row: dict[str, Any], cache_path: Path) -> np.ndarray:
        key = str(cache_path)
        if self.cache_gripper_arrays and key in self._gripper_cache:
            return self._gripper_cache[key]
        payload = np.load(cache_path, allow_pickle=False)
        try:
            self._validate_cache_payload(row, payload, cache_path)
            gripper = np.asarray(payload["gripper"], dtype=np.float32)
            if self.cache_gripper_arrays:
                gripper = np.ascontiguousarray(gripper)
                self._gripper_cache[key] = gripper
            return gripper
        finally:
            payload.close()

    def _window_index_with_rows(self) -> list[tuple[int, int, dict[str, Any]]]:
        return [(row_idx, start, self.demo_rows[row_idx]) for row_idx, start in self.index]

    def _load_vae_latent_cache(self) -> GEMBenchMicrosteps9V32VAELatentCache | None:
        if self.vae_latent_cache_dir is None:
            return None
        manifest_sha = sha256_file(self.manifest_path)
        expected_config = build_vae_cache_dataset_config(
            manifest_path=self.manifest_path,
            manifest_sha256=manifest_sha,
            rgb_cache_dir=self.rgb_cache_dir,
            seed=self.seed,
            frame_offsets=self.frame_offsets,
            action_horizon=self.action_horizon,
            window_stride=self.window_stride,
            video_size=self.video_size,
            camera_order=self.camera_order,
            cache_camera_order=self.cache_camera_order,
        )
        return GEMBenchMicrosteps9V32VAELatentCache(
            self.vae_latent_cache_dir,
            expected_dataset_config=expected_config,
            expected_index=self._window_index_with_rows(),
        )

    def _cache_path(self, row: dict[str, Any]) -> Path:
        if row.get("cache_path"):
            path = Path(str(row["cache_path"])).expanduser()
            return path if path.is_absolute() else self.rgb_cache_dir / path
        return cache_episode_path(
            self.rgb_cache_dir,
            seed=str(row.get("seed", self.seed)),
            taskvar=str(row["taskvar"]),
            episode_key=str(row["episode_key"]),
        )

    def _validate_cache_payload(self, row: dict[str, Any], payload: Any, cache_path: Path) -> None:
        required = {"rgb", "gripper", "schema_version", "taskvar", "episode_key", "seed", "camera_order", "image_size"}
        missing = required.difference(payload.files)
        if missing:
            raise ValueError(f"Missing {sorted(missing)} in GEMBench 9v32 cache: {cache_path}")
        rgb = payload["rgb"]
        gripper = payload["gripper"]
        if _npz_scalar_str(payload, "schema_version") != SCHEMA_VERSION:
            raise ValueError(f"Cache schema_version mismatch in {cache_path}: {payload['schema_version']!r}")
        for key in ("taskvar", "episode_key", "seed"):
            expected = str(row.get(key, self.seed if key == "seed" else ""))
            actual = _npz_scalar_str(payload, key)
            if actual != expected:
                raise ValueError(f"Cache {key} mismatch in {cache_path}: payload={actual!r} expected={expected!r}")
        cache_order = tuple(str(value) for value in np.asarray(payload["camera_order"]).tolist())
        image_size = tuple(int(value) for value in np.asarray(payload["image_size"]).reshape(-1).tolist())
        length = int(row["length"])
        if rgb.ndim != 5 or rgb.shape[0] != length or rgb.shape[-1] != 3:
            raise ValueError(f"Expected rgb [L,C,H,W,3] length={length} in {cache_path}, got {rgb.shape}")
        if image_size != tuple(int(v) for v in rgb.shape[2:4]):
            raise ValueError(f"Cache image_size mismatch in {cache_path}: payload={image_size} rgb={tuple(rgb.shape[2:4])}")
        if int(rgb.shape[1]) != len(self.cache_camera_order):
            raise ValueError(
                f"Expected cache camera dimension={len(self.cache_camera_order)} in {cache_path}, got {rgb.shape[1]}"
            )
        if cache_order != tuple(self.cache_camera_order):
            raise ValueError(
                f"Cache camera_order mismatch in {cache_path}: payload={cache_order} expected={tuple(self.cache_camera_order)}"
            )
        if gripper.shape != (length, 8):
            raise ValueError(f"Expected gripper [{length},8] in {cache_path}, got {gripper.shape}")

    def _video_tensor(self, rgb: np.ndarray) -> torch.Tensor:
        if rgb.ndim != 5 or rgb.shape[-1] != 3:
            raise ValueError(f"Expected selected rgb [T,N,H,W,3], got {rgb.shape}")
        tensor = torch.from_numpy(np.ascontiguousarray(rgb)).permute(0, 1, 4, 2, 3).to(torch.float32) / 255.0
        t, n, c, h, w = tensor.shape
        camera_h = self.video_size[0]
        camera_w = self.video_size[1] // n
        if h != camera_h or w != camera_w:
            tensor = tensor.reshape(t * n, c, h, w)
            tensor = F.interpolate(tensor, size=(camera_h, camera_w), mode="bilinear", align_corners=False)
            tensor = tensor.reshape(t, n, c, camera_h, camera_w)
        video = torch.cat([tensor[:, i] for i in range(n)], dim=-1)
        video = video * 2.0 - 1.0
        return video.permute(1, 0, 2, 3).contiguous()

    def _load_or_scan_stats(self, pretrained_norm_stats: str | None) -> dict:
        if self.stats_scan_limit <= 0:
            return load_or_create_stats(pretrained_norm_stats, action_dim=8, state_dim=8)
        stats_path = Path(pretrained_norm_stats).expanduser() if pretrained_norm_stats else None
        if stats_path is not None and stats_path.exists():
            return load_or_create_stats(str(stats_path), action_dim=8, state_dim=8)
        samples = []
        for row_idx, start in self.index[: self.stats_scan_limit]:
            row = self.demo_rows[row_idx]
            payload = np.load(self._cache_path(row), allow_pickle=False)
            try:
                gripper = np.asarray(payload["gripper"], dtype=np.float32)
                action = gripper[int(start) + 1 : int(start) + self.action_horizon + 1]
                proprio = gripper[int(start) : int(start) + self.action_horizon]
                samples.append((action, proprio))
            finally:
                payload.close()
        stats = scanned_dataset_stats(samples, action_dim=8, state_dim=8)
        if stats_path is not None:
            from fastwam.datasets.lerobot.utils.normalizer import save_dataset_stats_to_json

            stats_path.parent.mkdir(parents=True, exist_ok=True)
            save_dataset_stats_to_json(stats, str(stats_path))
        return stats

    def _get_cached_text_context(self, prompt: str) -> tuple[torch.Tensor, torch.Tensor]:
        if self.text_embedding_cache_dir is None:
            if self.allow_missing_text_embeds:
                return self._empty_text_context()
            raise ValueError("text_embedding_cache_dir is not set.")
        cache_dir = Path(self.text_embedding_cache_dir)
        hashed = hashlib.sha256(prompt.encode("utf-8")).hexdigest()
        if self.cache_text_embeddings and hashed in self._text_context_cache:
            return self._text_context_cache[hashed]
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
        if self.cache_text_embeddings:
            self._text_context_cache[hashed] = (context, mask)
        return context, mask

    def _empty_text_context(self) -> tuple[torch.Tensor, torch.Tensor]:
        return (
            torch.zeros((self.context_len, self.text_dim), dtype=torch.float32),
            torch.zeros((self.context_len,), dtype=torch.bool),
        )


class GEMBenchKeyStepPolicy9V32Dataset(GEMBenchMicrosteps9V32Dataset):
    """Official-style key-step policy samples with FastWAM 9V32 video aux.

    The inherited `action` field remains the dense 32-step WAM auxiliary action
    window. The policy target is exposed separately as `policy_action`, a single
    normalized 8D action pointing from the current key-step observation to the
    next key-step gripper target. This keeps the policy contract explicit while
    preserving the existing 9V32 video/cache path.
    """

    def __init__(
        self,
        manifest_path: str,
        rgb_cache_dir: str,
        *,
        keysteps_dir: str | None = None,
        key_frameids_path: str | None = None,
        policy_max_index_demos: int | None = None,
        policy_include_final_key: bool = False,
        policy_min_key_delta: int = 1,
        policy_target_frame: str = "world",
        policy_pcd_data_dir: str | None = None,
        policy_local_xyz_shift: str = "center",
        policy_local_xyz_norm: bool = False,
        policy_local_rm_table: bool = True,
        policy_local_rm_robot: str = "none",
        policy_local_num_points: int = 4096,
        policy_local_sample_seed: int = 0,
        policy_local_train_voxel_size: float = 0.0,
        policy_local_require_open3d: bool = False,
        robot_3dlotus_root: str | None = None,
        **kwargs: Any,
    ):
        vae_latent_cache_dir = kwargs.pop("vae_latent_cache_dir", None)
        super().__init__(
            manifest_path=manifest_path,
            rgb_cache_dir=rgb_cache_dir,
            vae_latent_cache_dir=None,
            **kwargs,
        )
        self.policy_include_final_key = bool(policy_include_final_key)
        self.policy_min_key_delta = int(policy_min_key_delta)
        if self.policy_min_key_delta <= 0:
            raise ValueError(f"policy_min_key_delta must be positive, got {self.policy_min_key_delta}")
        self.policy_target_frame = str(policy_target_frame)
        if self.policy_target_frame not in ("world", "official_pcd_local"):
            raise ValueError(
                "policy_target_frame must be 'world' or 'official_pcd_local', "
                f"got {self.policy_target_frame!r}"
            )
        self.policy_local_frame_config = PolicyLocalFrameConfig(
            enabled=(self.policy_target_frame == "official_pcd_local"),
            xyz_shift=str(policy_local_xyz_shift),
            xyz_norm=bool(policy_local_xyz_norm),
            rm_table=bool(policy_local_rm_table),
            rm_robot=str(policy_local_rm_robot),
            num_points=int(policy_local_num_points),
            sample_seed=int(policy_local_sample_seed),
            voxel_size=float(policy_local_train_voxel_size),
            require_open3d=bool(policy_local_require_open3d),
        )
        self.policy_pcd_data_dir = None if policy_pcd_data_dir in (None, "", "null") else str(policy_pcd_data_dir)
        if self.policy_target_frame == "official_pcd_local" and self.policy_pcd_data_dir is None:
            raise ValueError("policy_target_frame='official_pcd_local' requires policy_pcd_data_dir.")
        self.robot_3dlotus_root = None if robot_3dlotus_root in (None, "", "null") else str(robot_3dlotus_root)
        resolved_keysteps_dir = keysteps_dir or self.manifest.get("keysteps_dir")
        self.keysteps_dir = None if resolved_keysteps_dir in (None, "", "null") else str(resolved_keysteps_dir)
        self.key_frameids_path = None if key_frameids_path in (None, "", "null") else str(key_frameids_path)
        self._key_frameids_by_demo = self._load_key_frameids_sidecar()
        self.policy_max_index_demos = None if policy_max_index_demos is None else int(policy_max_index_demos)
        self._keysteps_store: LMDBEpisodeStore | None = None
        self._policy_pcd_store: LMDBEpisodeStore | None = None
        self._policy_local_frame_cache: dict[tuple[str, str, int], dict[str, Any]] = {}

        self.index = self._build_key_transition_index()
        if not self.index:
            raise ValueError(f"No GEMBench key-step policy transitions found from manifest={self.manifest_path}")

        self.vae_latent_cache_dir = None if vae_latent_cache_dir in (None, "", "null") else str(vae_latent_cache_dir)
        self.vae_latent_cache = self._load_vae_latent_cache()
        if self.vae_latent_cache is not None:
            self._validate_policy_vae_cache_coverage()

    def __getitem__(self, idx: int) -> dict[str, Any]:
        row_idx, current_key_idx, next_key_idx, key_position = self.index[int(idx)]
        row = self.demo_rows[row_idx]
        start = int(current_key_idx)
        cache_path = self._cache_path(row)
        gripper = self._load_gripper(row, cache_path)
        action_raw = gripper[start + 1 : start + self.action_horizon + 1]
        proprio_raw = gripper[start : start + self.action_horizon]
        policy_action_raw = gripper[int(next_key_idx) : int(next_key_idx) + 1]
        policy_action_world_raw = policy_action_raw.copy()
        policy_proprio_raw = gripper[start : start + 1]

        if self.vae_latent_cache is not None:
            video_latents = self.vae_latent_cache.get(row, start)
            sample = self._make_sample_from_arrays(
                row=row,
                start=start,
                action_raw=action_raw,
                proprio_raw=proprio_raw,
                video=None,
            )
            sample["video_latents"] = video_latents
        else:
            sample = self._load_rgb_window_sample(row_idx=row_idx, start=start)

        policy_local_frame = None
        if self.policy_target_frame == "official_pcd_local":
            policy_local_frame = self._policy_local_frame(row, int(current_key_idx))
            policy_action_raw = action_world_to_local(policy_action_raw, policy_local_frame)

        policy_action, _, policy_action_dim_is_pad, _ = self.processor.normalize(
            torch.as_tensor(policy_action_raw, dtype=torch.float32),
            torch.as_tensor(policy_proprio_raw, dtype=torch.float32),
        )
        sample["policy_action"] = policy_action
        sample["policy_action_is_pad"] = torch.zeros(1, dtype=torch.bool)
        sample["policy_action_dim_is_pad"] = policy_action_dim_is_pad
        sample["policy_action_raw"] = torch.as_tensor(policy_action_raw, dtype=torch.float32)
        sample["policy_action_world_raw"] = torch.as_tensor(policy_action_world_raw, dtype=torch.float32)
        sample["policy_target_frame"] = self.policy_target_frame
        if policy_local_frame is not None:
            sample["policy_local_centroid"] = torch.as_tensor(policy_local_frame["centroid"], dtype=torch.float32)
            sample["policy_local_radius"] = torch.as_tensor([float(policy_local_frame["radius"])], dtype=torch.float32)
            if "pcd_current_key_idx" in policy_local_frame:
                sample["policy_pcd_current_key_idx"] = int(policy_local_frame["pcd_current_key_idx"])
            if "pcd_next_key_idx" in policy_local_frame:
                sample["policy_pcd_next_key_idx"] = int(policy_local_frame["pcd_next_key_idx"])
            if "pcd_next_action_world" in policy_local_frame:
                sample["policy_pcd_next_action_world"] = torch.as_tensor(
                    policy_local_frame["pcd_next_action_world"],
                    dtype=torch.float32,
                )
        sample["policy_current_key_idx"] = int(current_key_idx)
        sample["policy_next_key_idx"] = int(next_key_idx)
        sample["policy_key_position"] = int(key_position)
        sample["policy_action_horizon"] = 1
        sample["wam_aux_action_horizon"] = int(self.action_horizon)
        sample["policy_target_type"] = "next_key_step"
        return sample

    def _policy_local_frame(self, row: dict[str, Any], current_key_idx: int) -> dict[str, Any]:
        taskvar = str(row["taskvar"])
        episode_key = str(row["episode_key"])
        cache_key = (taskvar, episode_key, int(current_key_idx))
        cached = self._policy_local_frame_cache.get(cache_key)
        if cached is not None:
            return cached
        if self._policy_pcd_store is None:
            if self.policy_pcd_data_dir is None:
                raise ValueError("policy_pcd_data_dir is required for official_pcd_local policy targets.")
            self._policy_pcd_store = LMDBEpisodeStore(self.policy_pcd_data_dir)
        episode = self._policy_pcd_store.get(taskvar, episode_key)
        pcd_key_frameids = [int(v) for v in episode.get("key_frameids", [])]
        if pcd_key_frameids:
            try:
                pcd_step = pcd_key_frameids.index(int(current_key_idx))
            except ValueError as exc:
                raise ValueError(
                    f"Current key frame {current_key_idx} is not present in PCD key_frameids for "
                    f"{taskvar}/{episode_key}: {pcd_key_frameids}"
                ) from exc
        else:
            pcd_step = int(current_key_idx)
        xyz_seq = episode["xyz"]
        rgb_seq = episode.get("rgb")
        if not isinstance(xyz_seq, list):
            xyz_seq = np.asarray(xyz_seq).tolist()
        if rgb_seq is not None and not isinstance(rgb_seq, list):
            rgb_seq = np.asarray(rgb_seq).tolist()
        action = np.asarray(episode["action"], dtype=np.float32)
        if pcd_step >= len(xyz_seq) or pcd_step >= action.shape[0]:
            raise ValueError(
                f"PCD step out of range for {taskvar}/{episode_key}: step={pcd_step}, "
                f"xyz_steps={len(xyz_seq)}, action_steps={action.shape[0]}"
            )
        arm_links_info = None
        if self.policy_local_frame_config.rm_robot != "none":
            bbox_info = episode.get("bbox_info")
            pose_info = episode.get("pose_info")
            if bbox_info is None or pose_info is None:
                raise ValueError(f"PCD episode {taskvar}/{episode_key} lacks bbox_info/pose_info for robot filtering.")
            arm_links_info = (
                {key: np.asarray(value)[pcd_step] for key, value in bbox_info.items()},
                {key: np.asarray(value)[pcd_step] for key, value in pose_info.items()},
            )
        frame = compute_policy_local_frame(
            xyz=np.asarray(xyz_seq[pcd_step]),
            rgb=None if rgb_seq is None else np.asarray(rgb_seq[pcd_step]),
            ee_pose=action[pcd_step],
            arm_links_info=arm_links_info,
            config=self.policy_local_frame_config,
            sample_seed=int(self.policy_local_frame_config.sample_seed) + int(row.get("row_id", 0)) * 1009 + int(pcd_step),
            robot_3dlotus_root=self.robot_3dlotus_root,
        )
        frame["pcd_current_key_idx"] = int(pcd_key_frameids[pcd_step]) if pcd_key_frameids else int(current_key_idx)
        frame["pcd_next_key_idx"] = (
            int(pcd_key_frameids[pcd_step + 1])
            if pcd_key_frameids and pcd_step + 1 < len(pcd_key_frameids)
            else None
        )
        frame["pcd_next_action_world"] = (
            action[pcd_step + 1].astype(np.float32)
            if pcd_step + 1 < action.shape[0]
            else action[pcd_step].astype(np.float32)
        )
        self._policy_local_frame_cache[cache_key] = frame
        return frame

    def _build_key_transition_index(self) -> list[tuple[int, int, int, int]]:
        out: list[tuple[int, int, int, int]] = []
        demo_rows = self.demo_rows
        if self.policy_max_index_demos is not None:
            demo_rows = demo_rows[: self.policy_max_index_demos]
        for row_idx, row in enumerate(demo_rows):
            length = int(row["length"])
            key_frameids = self._normalized_key_frameids(row, length=length)
            if len(key_frameids) < 2:
                continue
            max_start = length - self.action_horizon - 1
            row_items: list[tuple[int, int, int, int]] = []
            for key_pos, current_key_idx in enumerate(key_frameids[:-1]):
                next_key_idx = key_frameids[key_pos + 1]
                if int(next_key_idx) - int(current_key_idx) < self.policy_min_key_delta:
                    continue
                if int(current_key_idx) > max_start:
                    continue
                row_items.append((row_idx, int(current_key_idx), int(next_key_idx), int(key_pos)))
            if self.policy_include_final_key:
                final_key = int(key_frameids[-1])
                if final_key <= max_start:
                    row_items.append((row_idx, final_key, final_key, len(key_frameids) - 1))
            if self.max_windows_per_demo is not None:
                row_items = row_items[: int(self.max_windows_per_demo)]
            out.extend(row_items)
        return out

    def _normalized_key_frameids(self, row: dict[str, Any], *, length: int) -> list[int]:
        raw = row.get("key_frameids")
        if raw:
            key_frameids = sorted({int(v) for v in raw if 0 <= int(v) < int(length)})
        elif self._key_frameids_by_demo:
            demo_key = self._sidecar_demo_key(row)
            raw = self._key_frameids_by_demo.get(demo_key)
            if raw is None:
                return []
            key_frameids = sorted({int(v) for v in raw if 0 <= int(v) < int(length)})
        else:
            if self.keysteps_dir is None:
                return []
            if self._keysteps_store is None:
                self._keysteps_store = LMDBEpisodeStore(self.keysteps_dir)
            key_frameids = sorted(
                {
                    int(v)
                    for v in self._keysteps_store.key_frameids(str(row["taskvar"]), str(row["episode_key"]))
                    if 0 <= int(v) < int(length)
                }
            )
        if not key_frameids:
            return []
        if key_frameids[0] != 0:
            key_frameids.insert(0, 0)
        return key_frameids

    def _load_key_frameids_sidecar(self) -> dict[str, list[int]]:
        if self.key_frameids_path is None:
            return {}
        path = Path(self.key_frameids_path).expanduser().resolve()
        if not path.is_file():
            raise FileNotFoundError(f"GEMBench key-frame sidecar not found: {path}")
        payload = json.loads(path.read_text(encoding="utf-8"))
        entries = payload.get("entries")
        if not isinstance(entries, dict):
            raise ValueError(f"Invalid GEMBench key-frame sidecar entries in {path}")
        out: dict[str, list[int]] = {}
        for key, value in entries.items():
            if not isinstance(value, list):
                raise ValueError(f"Invalid key_frameids for {key!r} in {path}: {type(value)}")
            out[str(key)] = [int(item) for item in value]
        return out

    @staticmethod
    def _sidecar_demo_key(row: dict[str, Any]) -> str:
        return f"{row['taskvar']}/{row['episode_key']}"

    def _load_vae_latent_cache(self) -> GEMBenchMicrosteps9V32VAELatentCache | None:
        if self.vae_latent_cache_dir is None:
            return None
        manifest_sha = sha256_file(self.manifest_path)
        expected_config = build_vae_cache_dataset_config(
            manifest_path=self.manifest_path,
            manifest_sha256=manifest_sha,
            rgb_cache_dir=self.rgb_cache_dir,
            seed=self.seed,
            frame_offsets=self.frame_offsets,
            action_horizon=self.action_horizon,
            window_stride=self.window_stride,
            video_size=self.video_size,
            camera_order=self.camera_order,
            cache_camera_order=self.cache_camera_order,
        )
        return GEMBenchMicrosteps9V32VAELatentCache(
            self.vae_latent_cache_dir,
            expected_dataset_config=expected_config,
            expected_index=None,
        )

    def _validate_policy_vae_cache_coverage(self) -> None:
        assert self.vae_latent_cache is not None
        missing: list[str] = []
        for row_idx, current_key_idx, _, _ in self.index:
            row = self.demo_rows[row_idx]
            key = window_cache_key(str(row["taskvar"]), str(row["episode_key"]), int(current_key_idx))
            if key not in self.vae_latent_cache.row_by_key:
                missing.append(key)
                if len(missing) >= 5:
                    break
        if missing:
            raise ValueError(
                "GEMBench key-step policy samples are not covered by the 9V32 VAE latent cache: "
                f"{missing}"
            )
