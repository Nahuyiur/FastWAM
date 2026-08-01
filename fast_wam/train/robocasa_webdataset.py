"""Indexed WebDataset shards for RoboCasa Fast-WAM training.

The files are ordinary uncompressed POSIX tar shards with WebDataset-style
``<sample-key>.<extension>`` members.  A sidecar JSONL index makes the dataset
map-style, so Megatron keeps its exact sampler and resume semantics instead of
switching to an approximate streaming iterator.
"""

from __future__ import annotations

import io
import json
import os
from pathlib import Path

import numpy as np
import torch


FORMAT_VERSION = "fast_wam_robocasa_webdataset_v1"


def _read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def load_source_indices(path: str | Path | None, dataset_size: int) -> list[int] | None:
    """Load and validate a logical-index to source-index mapping."""

    if path in (None, ""):
        return None
    source = Path(path).expanduser().resolve()
    payload = json.loads(source.read_text(encoding="utf-8"))
    values = payload["source_indices"] if isinstance(payload, dict) else payload
    indices = [int(value) for value in values]
    if not indices:
        raise ValueError(f"RoboCasa index file is empty: {source}")
    if len(set(indices)) != len(indices):
        raise ValueError(f"RoboCasa index file contains duplicates: {source}")
    bad = [value for value in indices if not 0 <= value < int(dataset_size)]
    if bad:
        raise ValueError(f"RoboCasa index file is out of range: {bad[:8]}")
    return indices


class RoboCasaIndexedSubset(torch.utils.data.Dataset):
    """Map logical benchmark indices onto a fixed set of source windows."""

    def __init__(self, dataset, source_indices: list[int]):
        self.dataset = dataset
        self.source_indices = tuple(int(value) for value in source_indices)
        self.episodes = dataset.episodes

    def __len__(self) -> int:
        return len(self.source_indices)

    def __getitem__(self, index: int) -> dict:
        logical_index = int(index)
        source_index = self.source_indices[logical_index]
        sample = dict(self.dataset[source_index])
        returned_index = int(sample["idx"].item())
        if returned_index != source_index:
            raise RuntimeError(
                "RoboCasa source dataset substituted a failed indexed sample: "
                f"requested={source_index} returned={returned_index}"
            )
        sample["idx"] = torch.tensor(logical_index, dtype=torch.long)
        sample["source_idx"] = torch.tensor(source_index, dtype=torch.long)
        return sample


class RoboCasaWebDataset(torch.utils.data.Dataset):
    """Read indexed, uncompressed WebDataset tar shards with ``os.pread``."""

    def __init__(self, root: str | Path):
        self.root = Path(root).expanduser().resolve()
        manifest_path = self.root / "manifest.json"
        if not manifest_path.is_file():
            raise FileNotFoundError(f"Incomplete RoboCasa WebDataset: {manifest_path}")
        self.manifest = _read_json(manifest_path)
        if self.manifest.get("format") != FORMAT_VERSION:
            raise ValueError(f"Unsupported RoboCasa WebDataset format: {manifest_path}")
        if not self.manifest.get("complete"):
            raise ValueError(f"RoboCasa WebDataset is not marked complete: {manifest_path}")
        self.mode = str(self.manifest["mode"])
        if self.mode not in {"online", "offline"}:
            raise ValueError(f"Unsupported RoboCasa WebDataset mode: {self.mode}")
        index_path = self.root / str(self.manifest["index_file"])
        self.entries = [
            json.loads(line)
            for line in index_path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        if len(self.entries) != int(self.manifest["num_samples"]):
            raise ValueError(
                f"RoboCasa WebDataset index mismatch: index={len(self.entries)} "
                f"manifest={self.manifest['num_samples']}"
            )
        for logical_index, entry in enumerate(self.entries):
            if int(entry["logical_index"]) != logical_index:
                raise ValueError(f"Non-contiguous WebDataset index at row {logical_index}")
        self.contexts = dict(self.manifest["contexts"])
        self._fds: dict[str, int] = {}
        self._context_cache: dict[str, tuple[torch.Tensor, torch.Tensor, str]] = {}

    def __getstate__(self):
        state = dict(self.__dict__)
        state["_fds"] = {}
        state["_context_cache"] = {}
        return state

    def __del__(self):
        for descriptor in getattr(self, "_fds", {}).values():
            try:
                os.close(descriptor)
            except OSError:
                pass

    def __len__(self) -> int:
        return len(self.entries)

    def _read_member(self, entry: dict, name: str) -> bytes:
        member = entry["members"][name]
        shard = str(entry["shard"])
        descriptor = self._fds.get(shard)
        if descriptor is None:
            descriptor = os.open(self.root / shard, os.O_RDONLY)
            self._fds[shard] = descriptor
        value = os.pread(descriptor, int(member["size"]), int(member["offset"]))
        if len(value) != int(member["size"]):
            raise IOError(f"Short WebDataset read: {shard}:{name}")
        return value

    @staticmethod
    def _load_npy(value: bytes) -> np.ndarray:
        return np.array(np.load(io.BytesIO(value), allow_pickle=False), copy=True)

    @staticmethod
    def _load_npz(value: bytes) -> dict[str, np.ndarray]:
        with np.load(io.BytesIO(value), allow_pickle=False) as archive:
            return {name: np.array(archive[name], copy=True) for name in archive.files}

    def _load_context(self, context_id: str) -> tuple[torch.Tensor, torch.Tensor, str]:
        cached = self._context_cache.get(context_id)
        if cached is not None:
            return cached
        record = self.contexts[context_id]
        arrays = self._load_npz((self.root / str(record["file"])).read_bytes())
        cached = (
            torch.from_numpy(arrays["context"]),
            torch.from_numpy(arrays["context_mask"]),
            str(record["prompt"]),
        )
        self._context_cache[context_id] = cached
        return cached

    def __getitem__(self, index: int) -> dict:
        logical_index = int(index)
        entry = self.entries[logical_index]
        metadata = self._load_npz(self._read_member(entry, "metadata"))
        context, context_mask, prompt = self._load_context(str(entry["context_id"]))
        sample = {
            name: torch.from_numpy(value)
            for name, value in metadata.items()
        }
        sample.update(
            {
                "prompt": prompt,
                "context": context,
                "context_mask": context_mask,
                "idx": torch.tensor(logical_index, dtype=torch.long),
                "source_idx": torch.tensor(int(entry["source_index"]), dtype=torch.long),
            }
        )
        if self.mode == "online":
            video = self._load_npy(self._read_member(entry, "video"))
            sample["video"] = torch.from_numpy(video)
        else:
            bits = self._load_npy(self._read_member(entry, "latent"))
            sample["input_latents"] = torch.from_numpy(bits).view(torch.bfloat16)
        return sample
