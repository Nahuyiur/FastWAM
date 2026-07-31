"""Build a resumable, immutable BF16 Wan-VAE latent cache for LIBERO."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path

import torch
import torch.distributed as dist

from ..config import FastWAMConfig
from .data import (
    LATENT_CACHE_VERSION,
    LATENT_SHAPE,
    LiberoTrainingDataset,
    dataset_fingerprint,
    latent_preprocessing_fingerprint,
    official_dataset_dirs,
    prepare_episode_video,
)
from .vae import WanVideoVAE38Encoder


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(16 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _write_json_atomic(path: Path, payload: dict) -> None:
    temporary = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def _distributed() -> tuple[int, int, int]:
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    if world_size > 1:
        torch.cuda.set_device(local_rank)
        dist.init_process_group(backend="nccl")
        return dist.get_rank(), dist.get_world_size(), local_rank
    if not torch.cuda.is_available():
        raise RuntimeError("Latent preparation requires one accelerator")
    torch.cuda.set_device(local_rank)
    return 0, 1, local_rank


def _contract(
    args,
    config: FastWAMConfig,
    vae_sha256: str,
    *,
    num_samples: int,
) -> dict:
    directories = official_dataset_dirs(args.dataset_root)
    return {
        "version": LATENT_CACHE_VERSION,
        "num_samples": num_samples,
        "dtype": "bfloat16",
        "sample_shape": list(LATENT_SHAPE),
        "samples_per_shard": args.samples_per_shard,
        "dataset_fingerprint": dataset_fingerprint(
            directories,
            args.stats_path,
        ),
        "preprocessing_fingerprint": latent_preprocessing_fingerprint(config),
        "vae_checkpoint": Path(args.vae_checkpoint).name,
        "vae_sha256": vae_sha256,
    }


def _shard_metadata(
    output: Path,
    *,
    num_samples: int,
    samples_per_shard: int,
) -> list[dict]:
    shards = []
    for shard_id, start in enumerate(
        range(0, num_samples, samples_per_shard)
    ):
        count = min(samples_per_shard, num_samples - start)
        shards.append(
            {
                "id": shard_id,
                "file": f"latents-{shard_id:05d}.bf16",
                "start": start,
                "count": count,
                "size_bytes": count * math.prod(LATENT_SHAPE) * 2,
            }
        )
    return shards


def _write_shard(path: Path, latents: list[torch.Tensor]) -> None:
    value = torch.cat(latents, dim=0).to(dtype=torch.bfloat16).contiguous()
    temporary = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    with temporary.open("wb") as stream:
        stream.write(value.view(torch.uint8).numpy().tobytes())
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", required=True)
    parser.add_argument("--stats-path", required=True)
    parser.add_argument("--text-cache", required=True)
    parser.add_argument("--vae-checkpoint", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--parquet-cache-size", type=int, default=32)
    parser.add_argument("--samples-per-shard", type=int, default=1024)
    args = parser.parse_args()
    if args.batch_size <= 0 or args.samples_per_shard <= 0:
        parser.error("batch size and samples per shard must be positive")

    rank, world_size, local_rank = _distributed()
    output = Path(args.output).expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    config = FastWAMConfig()
    dataset = LiberoTrainingDataset(
        root=args.dataset_root,
        stats_path=args.stats_path,
        text_cache_dir=args.text_cache,
        config=config,
        parquet_cache_size=args.parquet_cache_size,
        validate_official_release=True,
    )
    vae_digest = _sha256(Path(args.vae_checkpoint)) if rank == 0 else None
    if world_size > 1:
        values = [vae_digest]
        dist.broadcast_object_list(values, src=0)
        vae_digest = values[0]
    assert isinstance(vae_digest, str)
    contract = _contract(
        args,
        config,
        vae_digest,
        num_samples=len(dataset),
    )
    build_path = output / "build.json"
    if rank == 0:
        if build_path.is_file():
            previous = json.loads(build_path.read_text(encoding="utf-8"))
            if previous != contract:
                raise ValueError(
                    f"{build_path}: refusing to resume a different cache contract"
                )
        else:
            _write_json_atomic(build_path, contract)
    if world_size > 1:
        dist.barrier()

    shards = _shard_metadata(
        output,
        num_samples=len(dataset),
        samples_per_shard=args.samples_per_shard,
    )
    assigned = []
    for shard in shards[rank::world_size]:
        path = output / shard["file"]
        if path.is_file() and path.stat().st_size == shard["size_bytes"]:
            continue
        assigned.append(shard)

    encoder = (
        WanVideoVAE38Encoder.from_pretrained(
            args.vae_checkpoint,
            device=torch.device("cuda", local_rank),
            dtype=torch.bfloat16,
        )
        if assigned
        else None
    )
    video_offsets = torch.tensor(dataset.video_indices, dtype=torch.long)
    for shard in assigned:
        assert encoder is not None
        shard_start = int(shard["start"])
        shard_end = shard_start + int(shard["count"])
        cursor = shard_start
        buffered: list[torch.Tensor] = []
        while cursor < shard_end:
            episode, frame = dataset._locate(cursor)
            segment_count = min(shard_end - cursor, episode.length - frame)
            table = dataset._cache.get(episode)
            episode_video = prepare_episode_video(
                episode,
                table["timestamp"].tolist(),
                image_size=config.image_size,
            )
            for batch_start in range(0, segment_count, args.batch_size):
                batch_count = min(args.batch_size, segment_count - batch_start)
                frames = (
                    torch.arange(
                        frame + batch_start,
                        frame + batch_start + batch_count,
                        dtype=torch.long,
                    )[:, None]
                    + video_offsets[None, :]
                )
                frames.clamp_(max=episode.length - 1)
                video = (
                    episode_video[frames]
                    .permute(0, 2, 1, 3, 4)
                    .contiguous()
                )
                buffered.append(
                    encoder.encode_normalized_video(
                        video.cuda(non_blocking=True)
                    ).cpu()
                )
            cursor += segment_count
        if sum(value.shape[0] for value in buffered) != int(shard["count"]):
            raise RuntimeError(
                f"rank {rank}: shard {shard['id']} produced the wrong sample count"
            )
        _write_shard(output / str(shard["file"]), buffered)
        print(f"[rank {rank}] completed shard {shard['id']}", flush=True)

    if world_size > 1:
        dist.barrier()
    if rank == 0:
        missing = [
            shard["file"]
            for shard in shards
            if not (output / shard["file"]).is_file()
            or (output / shard["file"]).stat().st_size != shard["size_bytes"]
        ]
        if missing:
            raise RuntimeError(f"latent cache incomplete; missing/bad shards: {missing[:16]}")
        _write_json_atomic(
            output / "manifest.json",
            {
                **contract,
                "complete": True,
                "shards": shards,
            },
        )
        print(f"Completed latent cache: {output}", flush=True)
    if world_size > 1:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
