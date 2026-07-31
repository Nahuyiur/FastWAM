"""Datasets for Wan FlowMatch training.

Training consumes pre-encoded tensors:

* `input_latents`: Wan VAE latents, shape `[C, F, H, W]`
* `context`: UMT5 prompt embeddings, shape `[L, text_dim]`

This keeps the Megatron training path focused on the trainable DiT. Separate
scripts can prepare these tensors from real videos/prompts via DiffSynth or any
equivalent official Wan preprocessing stack.
"""

from __future__ import annotations

import json
from collections import OrderedDict
from pathlib import Path
from typing import Any, Dict, List

import torch
from torch.utils.data import Dataset

try:
    from megatron.training import print_rank_0
except Exception:  # pragma: no cover
    def print_rank_0(msg):
        print(msg)


def _load_tensor_from_obj(obj: Any, path: str, key: str | None = None) -> torch.Tensor:
    if isinstance(obj, dict):
        if key is not None:
            return obj[key]
        for candidate in ("input_latents", "latents", "latent", "context", "tensor"):
            if candidate in obj:
                return obj[candidate]
        raise KeyError(f"No tensor key found in {path}")
    return obj


def _load_tensor(path: str, key: str | None = None) -> torch.Tensor:
    path_obj = Path(path)
    suffix = path_obj.suffix
    if suffix == ".pt" or suffix == ".pth":
        obj = torch.load(path_obj, map_location="cpu", weights_only=False)
        return _load_tensor_from_obj(obj, path, key)
    if suffix == ".npy":
        import numpy as np

        return torch.from_numpy(np.load(path_obj))
    if suffix == ".safetensors":
        try:
            from safetensors.torch import load_file
        except Exception as exc:
            raise RuntimeError(f"Reading {path} requires safetensors") from exc
        tensors = load_file(str(path_obj), device="cpu")
        if key is not None:
            return tensors[key]
        for candidate in ("input_latents", "latents", "latent", "context", "tensor"):
            if candidate in tensors:
                return tensors[candidate]
        raise KeyError(f"No known tensor key in {path}; keys={list(tensors)[:8]}")
    raise ValueError(f"Unsupported tensor path: {path}")


class _ShardCache:
    """Small per-worker LRU cache for pre-extracted Wan shards."""

    def __init__(self, capacity: int = 2):
        self.capacity = max(0, capacity)
        self._cache: OrderedDict[tuple[str, str], Any] = OrderedDict()

    def get(self, path: str, key: str, index: int) -> torch.Tensor:
        suffix = Path(path).suffix
        if suffix == ".safetensors":
            return self._get_safetensors_slice(path, key, index)
        if suffix in (".pt", ".pth"):
            return self._get_torch_shard(path, key, index)
        raise ValueError(f"Unsupported shard path: {path}")

    def _remember(self, cache_key: tuple[str, str], value: Any) -> Any:
        if self.capacity == 0:
            return value
        self._cache[cache_key] = value
        self._cache.move_to_end(cache_key)
        while len(self._cache) > self.capacity:
            _, evicted = self._cache.popitem(last=False)
            close = getattr(evicted, "close", None)
            if callable(close):
                close()
        return value

    def _get_torch_shard(self, path: str, key: str, index: int) -> torch.Tensor:
        cache_key = ("torch", path)
        obj = self._cache.get(cache_key)
        if obj is None:
            obj = self._remember(cache_key, torch.load(path, map_location="cpu", weights_only=False))
        else:
            self._cache.move_to_end(cache_key)
        tensor = _load_tensor_from_obj(obj, path, key)
        return tensor[index]

    def _get_safetensors_slice(self, path: str, key: str, index: int) -> torch.Tensor:
        try:
            from safetensors import safe_open
        except Exception as exc:
            raise RuntimeError(f"Reading safetensors shard {path} requires safetensors") from exc

        cache_key = ("safetensors", path)
        handle = self._cache.get(cache_key)
        if handle is None:
            handle = self._remember(cache_key, safe_open(path, framework="pt", device="cpu"))
        else:
            self._cache.move_to_end(cache_key)
        return handle.get_slice(key)[index]


def _normalize_sample(obj: Dict[str, Any]) -> Dict[str, Any]:
    latents = obj.get("input_latents", obj.get("latents"))
    context = obj.get("context")
    first_frame_latents = obj.get("first_frame_latents")
    if latents is None:
        raise KeyError("Wan sample requires `input_latents` or `latents`")
    if context is None:
        raise KeyError("Wan sample requires `context`")

    if latents.ndim == 5 and latents.shape[0] == 1:
        latents = latents[0]
    if context.ndim == 3 and context.shape[0] == 1:
        context = context[0]
    sample = {
        "input_latents": latents.contiguous().float(),
        "context": context.contiguous().float(),
        "prompt": obj.get("prompt", ""),
        "video_path": obj.get("video_path", ""),
        "fuse_vae_embedding_in_latents": bool(obj.get("fuse_vae_embedding_in_latents", False)),
    }
    if first_frame_latents is not None:
        if first_frame_latents.ndim == 5 and first_frame_latents.shape[0] == 1:
            first_frame_latents = first_frame_latents[0]
        if first_frame_latents.ndim == 3:
            first_frame_latents = first_frame_latents.unsqueeze(1)
        expected = (sample["input_latents"].shape[0], 1, sample["input_latents"].shape[2], sample["input_latents"].shape[3])
        if tuple(first_frame_latents.shape) != expected:
            raise ValueError(
                f"first_frame_latents shape {tuple(first_frame_latents.shape)} must match {expected}"
            )
        sample["first_frame_latents"] = first_frame_latents.contiguous().float()
        sample["fuse_vae_embedding_in_latents"] = True
    return sample


class WanOverfitDataset(Dataset):
    """Repeat one pre-encoded sample for overfit verification."""

    def __init__(self, sample_path: str, num_samples: int = 10000):
        obj = torch.load(sample_path, map_location="cpu", weights_only=False)
        self.sample = _normalize_sample(obj)
        self.num_samples = num_samples
        print_rank_0(
            "WanOverfitDataset: "
            f"latents={tuple(self.sample['input_latents'].shape)}, "
            f"context={tuple(self.sample['context'].shape)}, "
            f"source={self.sample.get('video_path', '')}"
        )

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        return self.sample


class WanJsonlDataset(Dataset):
    """JSONL dataset of pre-encoded Wan training tensors.

    Each line can contain per-sample tensor paths:

    ```json
    {"latents": "/path/x.pt", "context": "/path/c.pt"}
    ```

    a single packed sample file:

    ```json
    {"sample_path": "/path/sample.pt"}
    ```

    or a packed shard row, which is the production path:

    ```json
    {"shard_path": "/path/shard_000001.safetensors", "index": 42}
    ```

    Shards store batched tensors under `input_latents` and `context` by default.
    This avoids one open/stat/deserialize per sample on Lustre.
    """

    def __init__(self, jsonl_path: str, num_samples: int | None = None, shard_cache_size: int = 2):
        self.path = jsonl_path
        self.offsets: List[int] = []
        with open(jsonl_path, "rb") as f:
            while True:
                offset = f.tell()
                line = f.readline()
                if not line:
                    break
                if line.strip():
                    self.offsets.append(offset)
        self.num_samples = num_samples or len(self.offsets)
        self.shard_cache_size = shard_cache_size
        self._fh = None
        self._shard_cache = None
        print_rank_0(f"WanJsonlDataset: indexed {len(self.offsets)} rows from {jsonl_path}")

    def __getstate__(self):
        state = self.__dict__.copy()
        state["_fh"] = None
        state["_shard_cache"] = None
        return state

    def __del__(self):
        if self._fh is not None:
            self._fh.close()

    def __len__(self):
        return self.num_samples

    def _file(self):
        if self._fh is None:
            self._fh = open(self.path, "r")
        return self._fh

    def _cache(self):
        if self._shard_cache is None:
            self._shard_cache = _ShardCache(self.shard_cache_size)
        return self._shard_cache

    def __getitem__(self, idx):
        idx = idx % len(self.offsets)
        fh = self._file()
        fh.seek(self.offsets[idx])
        row = json.loads(fh.readline())

        sample_path = row.get("sample_path")
        if isinstance(sample_path, str):
            obj = torch.load(sample_path, map_location="cpu", weights_only=False)
            sample = dict(obj)
            sample.setdefault("prompt", row.get("prompt", sample.get("prompt", "")))
            sample.setdefault("video_path", row.get("video_path", sample.get("video_path", "")))
            return _normalize_sample(sample)

        shard_path = row.get("shard_path", row.get("shard"))
        if isinstance(shard_path, str):
            shard_index = int(row.get("index", row.get("shard_index", 0)))
            latents_key = row.get("latents_key", "input_latents")
            context_key = row.get("context_key", "context")
            cache = self._cache()
            sample = {
                "input_latents": cache.get(shard_path, latents_key, shard_index),
                "context": cache.get(shard_path, context_key, shard_index),
                "prompt": row.get("prompt", ""),
                "video_path": row.get("video_path", ""),
            }
            first_frame_key = row.get("first_frame_latents_key")
            if first_frame_key:
                sample["first_frame_latents"] = cache.get(shard_path, first_frame_key, shard_index)
            if row.get("fuse_vae_embedding_in_latents", False):
                sample["fuse_vae_embedding_in_latents"] = True
            return _normalize_sample(sample)

        latents_path = row.get("latents_path", row.get("input_latents_path", row.get("latents")))
        context_path = row.get("context_path", row.get("context"))
        first_frame_path = row.get("first_frame_latents_path")
        latents_key = row.get("latents_key", "input_latents")
        context_key = row.get("context_key", "context")
        first_frame_key = row.get("first_frame_latents_key", "first_frame_latents")
        if not isinstance(latents_path, str) or not isinstance(context_path, str):
            raise ValueError("WanJsonlDataset expects latents/context paths, not inline tensors")

        sample = {
            "input_latents": _load_tensor(latents_path, key=latents_key),
            "context": _load_tensor(context_path, key=context_key),
            "prompt": row.get("prompt", ""),
            "video_path": row.get("video_path", ""),
        }
        if isinstance(first_frame_path, str):
            sample["first_frame_latents"] = _load_tensor(first_frame_path, key=first_frame_key)
        if row.get("fuse_vae_embedding_in_latents", False):
            sample["fuse_vae_embedding_in_latents"] = True
        return _normalize_sample(sample)


def wan_collate(batch):
    """Pad context length and stack fixed-shape latents."""
    batch = [b for b in batch if b is not None]
    if not batch:
        return None

    latent_shapes = {tuple(b["input_latents"].shape) for b in batch}
    if len(latent_shapes) != 1:
        raise ValueError(f"Wan batch has mismatched latent shapes: {sorted(latent_shapes)}")
    latents = torch.stack([b["input_latents"] for b in batch], dim=0)

    max_len = max(b["context"].shape[0] for b in batch)
    text_dim = batch[0]["context"].shape[-1]
    context = torch.zeros(len(batch), max_len, text_dim, dtype=batch[0]["context"].dtype)
    context_mask = torch.zeros(len(batch), max_len, dtype=torch.bool)
    for i, item in enumerate(batch):
        length = item["context"].shape[0]
        context[i, :length] = item["context"]
        context_mask[i, :length] = True

    result = {
        "input_latents": latents,
        "context": context,
        "context_mask": context_mask,
        "prompt": [b.get("prompt", "") for b in batch],
        "video_path": [b.get("video_path", "") for b in batch],
    }
    first_flags = ["first_frame_latents" in b for b in batch]
    if any(first_flags) and not all(first_flags):
        raise ValueError("Wan batch mixes samples with and without first_frame_latents")
    fuse_flags = [bool(b.get("fuse_vae_embedding_in_latents", False)) for b in batch]
    if any(fuse_flags) and not all(fuse_flags):
        raise ValueError("Wan batch mixes fuse_vae_embedding_in_latents modes")
    result["fuse_vae_embedding_in_latents"] = torch.tensor(all(fuse_flags), dtype=torch.bool)
    if all(first_flags):
        first_shapes = {tuple(b["first_frame_latents"].shape) for b in batch}
        if len(first_shapes) != 1:
            raise ValueError(f"Wan batch has mismatched first-frame shapes: {sorted(first_shapes)}")
        result["first_frame_latents"] = torch.stack([b["first_frame_latents"] for b in batch], dim=0)
    return result
