"""Build resumable indexed WebDataset shards for RoboCasa Fast-WAM."""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import os
import shutil
import tarfile
from pathlib import Path

import numpy as np
import torch

from .robocasa_data import RoboCasaLatentDataset, build_robocasa_datasets
from .robocasa_webdataset import (
    FORMAT_VERSION,
    RoboCasaIndexedSubset,
    load_source_indices,
)


def _json_atomic(path: Path, payload: object) -> None:
    temporary = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, path)


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(16 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _sha256_json(payload: object) -> str:
    return _sha256_bytes(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    )


def _npy_bytes(value: torch.Tensor | np.ndarray) -> bytes:
    array = value.detach().cpu().contiguous().numpy() if torch.is_tensor(value) else value
    stream = io.BytesIO()
    np.save(stream, array, allow_pickle=False)
    return stream.getvalue()


def _npz_bytes(values: dict[str, torch.Tensor | np.ndarray]) -> bytes:
    arrays = {
        name: value.detach().cpu().contiguous().numpy()
        if torch.is_tensor(value)
        else value
        for name, value in values.items()
    }
    stream = io.BytesIO()
    np.savez(stream, **arrays)
    return stream.getvalue()


def _add_bytes(archive: tarfile.TarFile, name: str, value: bytes) -> None:
    info = tarfile.TarInfo(name=name)
    info.size = len(value)
    info.mtime = 0
    info.mode = 0o644
    info.uid = 0
    info.gid = 0
    info.uname = ""
    info.gname = ""
    archive.addfile(info, io.BytesIO(value))


def _select_indices(
    dataset_size: int,
    *,
    index_file: str | None,
    max_samples: int | None,
    selection: str,
    seed: int,
) -> list[int]:
    external = load_source_indices(index_file, dataset_size)
    if external is not None:
        return external
    count = dataset_size if max_samples is None else min(dataset_size, int(max_samples))
    if count <= 0:
        raise ValueError("max_samples must be positive")
    if selection == "prefix":
        return list(range(count))
    if selection == "spread":
        return np.linspace(0, dataset_size - 1, count, dtype=np.int64).tolist()
    if selection == "random":
        generator = torch.Generator(device="cpu").manual_seed(int(seed))
        return torch.randperm(dataset_size, generator=generator)[:count].tolist()
    raise ValueError(f"Unknown sample selection: {selection}")


def _context_id(sample: dict) -> str:
    digest = hashlib.sha256()
    digest.update(str(sample["prompt"]).encode("utf-8"))
    for name in ("context", "context_mask"):
        value = sample[name].detach().cpu().contiguous()
        digest.update(value.view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()[:24]


def _write_context(root: Path, sample: dict, context_id: str) -> None:
    directory = root / "contexts"
    directory.mkdir(parents=True, exist_ok=True)
    array_path = directory / f"{context_id}.npz"
    record_path = directory / f"{context_id}.json"
    if not array_path.is_file():
        temporary = array_path.with_name(f".{array_path.name}.tmp.{os.getpid()}")
        temporary.write_bytes(
            _npz_bytes(
                {
                    "context": sample["context"],
                    "context_mask": sample["context_mask"],
                }
            )
        )
        os.replace(temporary, array_path)
    record = {
        "file": str(array_path.relative_to(root)),
        "prompt": str(sample["prompt"]),
    }
    if record_path.is_file():
        if json.loads(record_path.read_text(encoding="utf-8")) != record:
            raise ValueError(f"Context hash collision: {context_id}")
    else:
        _json_atomic(record_path, record)


def _sample_members(sample: dict, mode: str, logical_index: int) -> dict[str, bytes]:
    excluded = {
        "video",
        "input_latents",
        "prompt",
        "context",
        "context_mask",
        "idx",
        "source_idx",
    }
    metadata = {
        name: value
        for name, value in sample.items()
        if name not in excluded and torch.is_tensor(value)
    }
    key = f"{logical_index:09d}"
    members = {f"{key}.metadata.npz": _npz_bytes(metadata)}
    if mode == "online":
        members[f"{key}.video.npy"] = _npy_bytes(sample["video"].float())
    else:
        latent = sample["input_latents"].detach().cpu().to(torch.bfloat16).contiguous()
        members[f"{key}.latent.npy"] = _npy_bytes(latent.view(torch.uint16))
    return members


def _scan_shard(root: Path, shard_name: str) -> list[dict]:
    groups: dict[str, dict] = {}
    with tarfile.open(root / shard_name, mode="r:") as archive:
        for member in archive:
            if not member.isfile():
                continue
            key, suffix = member.name.split(".", 1)
            record = groups.setdefault(key, {"members": {}})
            if suffix == "json":
                stream = archive.extractfile(member)
                if stream is None:
                    raise IOError(f"Cannot read {shard_name}:{member.name}")
                record.update(json.loads(stream.read()))
            elif suffix == "metadata.npz":
                record["members"]["metadata"] = {
                    "offset": member.offset_data,
                    "size": member.size,
                }
            elif suffix == "video.npy":
                record["members"]["video"] = {
                    "offset": member.offset_data,
                    "size": member.size,
                }
            elif suffix == "latent.npy":
                record["members"]["latent"] = {
                    "offset": member.offset_data,
                    "size": member.size,
                }
    entries = []
    for key in sorted(groups):
        record = groups[key]
        logical_index = int(key)
        required = {"metadata", "video" if record["mode"] == "online" else "latent"}
        if not required.issubset(record["members"]):
            raise ValueError(f"Incomplete WebDataset sample {shard_name}:{key}")
        entries.append(
            {
                "logical_index": logical_index,
                "source_index": int(record["source_index"]),
                "context_id": str(record["context_id"]),
                "shard": shard_name,
                "members": record["members"],
            }
        )
    return entries


def _write_shard(
    root: Path,
    shard_name: str,
    dataset,
    start: int,
    stop: int,
    mode: str,
) -> None:
    final_path = root / shard_name
    temporary = final_path.with_name(f".{final_path.name}.tmp.{os.getpid()}")
    with tarfile.open(temporary, mode="w", format=tarfile.USTAR_FORMAT) as archive:
        for logical_index in range(start, stop):
            sample = dataset[logical_index]
            context_id = _context_id(sample)
            _write_context(root, sample, context_id)
            source_index = int(sample.get("source_idx", sample["idx"]).item())
            key = f"{logical_index:09d}"
            descriptor = json.dumps(
                {
                    "logical_index": logical_index,
                    "source_index": source_index,
                    "context_id": context_id,
                    "mode": mode,
                },
                sort_keys=True,
            ).encode("utf-8")
            _add_bytes(archive, f"{key}.json", descriptor)
            for name, value in _sample_members(sample, mode, logical_index).items():
                _add_bytes(archive, name, value)
    os.replace(temporary, final_path)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", required=True)
    parser.add_argument("--task-config", default="robocasa_acg_v1_fastwam_8gpu")
    parser.add_argument("--split", choices=("train", "valid"), default="train")
    parser.add_argument("--mode", choices=("online", "offline"), required=True)
    parser.add_argument("--latent-cache", default=None)
    parser.add_argument("--output", required=True)
    parser.add_argument("--source-index-file", default=None)
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--selection", choices=("prefix", "spread", "random"), default="random")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--samples-per-shard", type=int, default=64)
    parser.add_argument("--allow-low-disk", action="store_true")
    args = parser.parse_args()

    if args.mode == "offline" and not args.latent_cache:
        parser.error("--latent-cache is required for offline WebDataset")
    if args.mode == "online" and args.latent_cache:
        parser.error("--latent-cache is only valid for offline WebDataset")

    train_dataset, valid_dataset, cfg = build_robocasa_datasets(
        args.repo_root,
        args.task_config,
    )
    source_dataset = train_dataset if args.split == "train" else valid_dataset
    source_indices = _select_indices(
        len(source_dataset),
        index_file=args.source_index_file,
        max_samples=args.max_samples,
        selection=args.selection,
        seed=args.seed,
    )
    output = Path(args.output).expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    index_payload = {
        "dataset_size": len(source_dataset),
        "selection": args.selection if args.source_index_file is None else "external",
        "seed": args.seed,
        "source_indices": source_indices,
    }
    source_index_path = output / "source_indices.json"
    if source_index_path.is_file():
        if json.loads(source_index_path.read_text(encoding="utf-8")) != index_payload:
            raise ValueError(f"Refusing to replace a different index: {source_index_path}")
    else:
        _json_atomic(source_index_path, index_payload)

    if args.mode == "online":
        dataset = RoboCasaIndexedSubset(source_dataset, source_indices)
    else:
        dataset = RoboCasaLatentDataset(source_dataset, args.latent_cache, source_indices)
    episode_manifest = Path(
        str(cfg.data.train.episode_manifest_path)
    ).expanduser().resolve()
    latent_manifest = (
        Path(args.latent_cache).expanduser().resolve() / "manifest.json"
        if args.latent_cache
        else None
    )
    contract = {
        "format": FORMAT_VERSION,
        "complete": False,
        "mode": args.mode,
        "task_config": args.task_config,
        "split": args.split,
        "source_dataset_size": len(source_dataset),
        "num_samples": len(dataset),
        "source_indices_sha256": _sha256_json(source_indices),
        "episode_manifest": str(episode_manifest),
        "episode_manifest_sha256": _sha256_file(episode_manifest),
        "latent_manifest": str(latent_manifest) if latent_manifest else None,
        "latent_manifest_sha256": (
            _sha256_file(latent_manifest) if latent_manifest else None
        ),
        "samples_per_shard": int(args.samples_per_shard),
        "camera_keys": list(cfg.data.train.camera_keys),
        "frame_offsets": list(cfg.data.train.frame_offsets),
        "video_size": list(cfg.data.train.video_size),
        "online_video_dtype": "float32" if args.mode == "online" else None,
        "offline_latent_dtype": "bfloat16" if args.mode == "offline" else None,
    }
    build_path = output / "build.json"
    if build_path.is_file():
        if json.loads(build_path.read_text(encoding="utf-8")) != contract:
            raise ValueError(f"Refusing to resume a different WebDataset: {build_path}")
    else:
        _json_atomic(build_path, contract)

    probe = dataset[0]
    probe_bytes = sum(len(value) for value in _sample_members(probe, args.mode, 0).values())
    estimated_bytes = int(probe_bytes * len(dataset) * 1.02)
    free_bytes = shutil.disk_usage(output).free
    if estimated_bytes > int(free_bytes * 0.9) and not args.allow_low_disk:
        raise RuntimeError(
            f"Estimated WebDataset size {estimated_bytes / 2**30:.1f} GiB exceeds "
            f"90% of free space {free_bytes / 2**30:.1f} GiB; use --allow-low-disk to override"
        )

    shard_names = []
    for shard_id, start in enumerate(range(0, len(dataset), args.samples_per_shard)):
        stop = min(start + args.samples_per_shard, len(dataset))
        shard_name = f"shard-{shard_id:05d}.tar"
        shard_names.append(shard_name)
        shard_path = output / shard_name
        if shard_path.is_file():
            entries = _scan_shard(output, shard_name)
            if len(entries) == stop - start:
                continue
            raise ValueError(f"Incomplete existing WebDataset shard: {shard_path}")
        _write_shard(output, shard_name, dataset, start, stop, args.mode)
        print(f"completed {shard_name} samples={stop - start}", flush=True)

    entries = []
    for shard_name in shard_names:
        entries.extend(_scan_shard(output, shard_name))
    entries.sort(key=lambda value: int(value["logical_index"]))
    if len(entries) != len(dataset):
        raise RuntimeError(f"WebDataset index is incomplete: {len(entries)} != {len(dataset)}")
    index_file = "index.jsonl"
    index_temporary = output / f".{index_file}.tmp.{os.getpid()}"
    index_temporary.write_text(
        "".join(json.dumps(value, sort_keys=True) + "\n" for value in entries),
        encoding="utf-8",
    )
    os.replace(index_temporary, output / index_file)
    contexts = {
        path.stem: json.loads(path.read_text(encoding="utf-8"))
        for path in sorted((output / "contexts").glob("*.json"))
    }
    total_bytes = sum((output / name).stat().st_size for name in shard_names)
    manifest = {
        **contract,
        "complete": True,
        "index_file": index_file,
        "source_index_file": source_index_path.name,
        "contexts": contexts,
        "shards": [
            {"file": name, "size_bytes": (output / name).stat().st_size}
            for name in shard_names
        ],
        "total_shard_bytes": total_bytes,
        "estimated_full_dataset_bytes": round(
            total_bytes * len(source_dataset) / len(dataset)
        ),
    }
    _json_atomic(output / "manifest.json", manifest)
    print(
        f"completed RoboCasa WebDataset mode={args.mode} samples={len(dataset)} "
        f"size_gib={total_bytes / 2**30:.3f} output={output}",
        flush=True,
    )


if __name__ == "__main__":
    main()
