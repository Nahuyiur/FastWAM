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
from .normalization import DEFAULT_SHAPE_META, GEMBenchProcessorShim, load_or_create_stats, scanned_dataset_stats


SCHEMA_VERSION = "gembench_microsteps_9v32_v1"
DEFAULT_PROMPT = "A video recorded from a robot's point of view executing the following instruction: {task}"
DEFAULT_FRAME_OFFSETS = (0, 4, 8, 12, 16, 20, 24, 28, 32)
DEFAULT_CAMERA_ORDER = ("front", "wrist", "left_shoulder")
DEFAULT_CACHE_CAMERA_ORDER = DEFAULT_CAMERA_ORDER


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
    camera_width = int(video_size[1]) // len(camera_order)
    for meta, camera in zip(shape_meta["images"], camera_order):
        meta["key"] = str(camera)
        meta["shape"] = [3, int(video_size[0]), camera_width]
    if processor_cfg is not None:
        plain = OmegaConf.to_container(processor_cfg, resolve=True) if isinstance(processor_cfg, DictConfig) else processor_cfg
        if isinstance(plain, dict) and plain.get("shape_meta") is not None:
            shape_meta = plain["shape_meta"]
    return shape_meta


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
        allow_missing_text_embeds: bool = False,
        pretrained_norm_stats: str | None = None,
        norm_default_mode: str = "-2.0/2.0",
        stats_scan_limit: int = 0,
        allow_partial_cache: bool = False,
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
        self.allow_missing_text_embeds = bool(allow_missing_text_embeds)
        self.norm_default_mode = str(norm_default_mode)
        self.stats_scan_limit = int(stats_scan_limit)
        self.allow_partial_cache = bool(allow_partial_cache)

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
        finally:
            payload.close()

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

        return {
            "video": video,
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
