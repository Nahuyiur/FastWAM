"""Build a resumable BF16 Wan-VAE latent cache for RoboCasa ACG windows."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path

import torch
import torch.distributed as dist
from torch.utils.data import DataLoader, Subset

from .robocasa_data import build_robocasa_datasets
from .vae import WanVideoVAE38Encoder


LATENT_SHAPE = (48, 3, 14, 28)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(16 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _write_json_atomic(path: Path, payload: dict) -> None:
    temporary = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, path)


def _initialize_distributed() -> tuple[int, int, int]:
    world = int(os.environ.get("WORLD_SIZE", "1"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    torch.cuda.set_device(local_rank)
    if world > 1:
        dist.init_process_group(backend="nccl")
        return dist.get_rank(), dist.get_world_size(), local_rank
    return 0, 1, local_rank


def _write_shard(path: Path, batches: list[torch.Tensor]) -> None:
    value = torch.cat(batches, dim=0).to(torch.bfloat16).contiguous()
    temporary = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    with temporary.open("wb") as stream:
        stream.write(value.view(torch.uint8).numpy().tobytes())
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", required=True)
    parser.add_argument("--task-config", default="robocasa_acg_v1_fastwam_8gpu")
    parser.add_argument("--split", choices=("train", "valid"), default="train")
    parser.add_argument("--vae-checkpoint", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--samples-per-shard", type=int, default=1024)
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--source-index-file", default=None)
    args = parser.parse_args()

    rank, world, local_rank = _initialize_distributed()
    train_dataset, valid_dataset, cfg = build_robocasa_datasets(
        args.repo_root,
        args.task_config,
        train_index_file=args.source_index_file if args.split == "train" else None,
        valid_index_file=args.source_index_file if args.split == "valid" else None,
    )
    dataset = train_dataset if args.split == "train" else valid_dataset
    num_samples = len(dataset)
    if args.max_samples is not None:
        num_samples = min(num_samples, int(args.max_samples))
    output = Path(args.output).expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    vae_path = Path(args.vae_checkpoint).expanduser().resolve()
    manifest_path = Path(str(cfg.data.train.episode_manifest_path)).expanduser().resolve()
    digests = [
        _sha256(manifest_path) if rank == 0 else None,
        _sha256(vae_path) if rank == 0 else None,
    ]
    if world > 1:
        dist.broadcast_object_list(digests, src=0)
    contract = {
        "version": 1,
        "complete": False,
        "task_config": args.task_config,
        "split": args.split,
        "num_samples": num_samples,
        "sample_shape": list(LATENT_SHAPE),
        "dtype": "bfloat16",
        "encoding_batch_size": args.batch_size,
        "samples_per_shard": args.samples_per_shard,
        "episode_manifest": str(manifest_path),
        "episode_manifest_sha256": digests[0],
        "vae_checkpoint": str(vae_path),
        "vae_sha256": digests[1],
        "camera_keys": list(cfg.data.train.camera_keys),
        "frame_offsets": list(cfg.data.train.frame_offsets),
        "video_size": list(cfg.data.train.video_size),
        "source_index_file": (
            str(Path(args.source_index_file).expanduser().resolve())
            if args.source_index_file
            else None
        ),
        "source_index_sha256": (
            _sha256(Path(args.source_index_file).expanduser().resolve())
            if args.source_index_file
            else None
        ),
    }
    build_path = output / "build.json"
    if rank == 0:
        if build_path.is_file():
            previous = json.loads(build_path.read_text())
            if previous != contract:
                raise ValueError(f"Refusing to resume a different cache contract: {build_path}")
        else:
            _write_json_atomic(build_path, contract)
    if world > 1:
        dist.barrier()

    shards = []
    for shard_id, start in enumerate(range(0, num_samples, args.samples_per_shard)):
        count = min(args.samples_per_shard, num_samples - start)
        shards.append(
            {
                "id": shard_id,
                "file": f"latents-{shard_id:05d}.bf16",
                "start": start,
                "count": count,
                "size_bytes": count * math.prod(LATENT_SHAPE) * 2,
            }
        )

    assigned = []
    for shard in shards[rank::world]:
        path = output / shard["file"]
        if path.is_file() and path.stat().st_size == shard["size_bytes"]:
            continue
        assigned.append(shard)
    encoder = (
        WanVideoVAE38Encoder.from_pretrained(
            vae_path,
            device=torch.device("cuda", local_rank),
            dtype=torch.bfloat16,
        )
        if assigned
        else None
    )
    for shard in assigned:
        assert encoder is not None
        start = int(shard["start"])
        stop = start + int(shard["count"])
        loader = DataLoader(
            Subset(dataset, range(start, stop)),
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            pin_memory=True,
            persistent_workers=args.num_workers > 0,
        )
        encoded = []
        for batch in loader:
            encoded.append(
                encoder.encode_normalized_video(
                    batch["video"].cuda(non_blocking=True)
                ).cpu()
            )
        if sum(value.shape[0] for value in encoded) != int(shard["count"]):
            raise RuntimeError(f"rank {rank}: bad sample count for shard {shard['id']}")
        _write_shard(output / shard["file"], encoded)
        print(f"[rank {rank}] completed shard {shard['id']}", flush=True)

    if world > 1:
        dist.barrier()
    if rank == 0:
        missing = [
            shard["file"]
            for shard in shards
            if not (output / shard["file"]).is_file()
            or (output / shard["file"]).stat().st_size != shard["size_bytes"]
        ]
        if missing:
            raise RuntimeError(f"RoboCasa latent cache incomplete: {missing[:16]}")
        _write_json_atomic(
            output / "manifest.json",
            {**contract, "complete": True, "shards": shards},
        )
        print(f"Completed RoboCasa latent cache: {output}", flush=True)
    if world > 1:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
