#!/usr/bin/env python
from __future__ import annotations

import argparse
import os
import shutil
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT / "src"))

from fastwam.datasets.gembench.dataset import GEMBenchKeystepsDataset
from fastwam.datasets.gembench.vae_cache import (
    CACHE_VERSION,
    DEFAULT_GEMBENCH_VAE_CACHE_DIR,
    build_expected_dataset_config,
    expected_latent_shape,
    write_index_jsonl_atomic,
    write_json_atomic,
)
from fastwam.models.wan22.helpers.io import hash_model_file
from fastwam.models.wan22.helpers.loader import load_wan22_ti2v_5b_vae


def _dtype(name: str) -> torch.dtype:
    key = str(name).strip().lower()
    if key in {"bf16", "bfloat16"}:
        return torch.bfloat16
    if key in {"fp16", "float16", "half"}:
        return torch.float16
    if key in {"fp32", "float32"}:
        return torch.float32
    raise ValueError(f"Unsupported dtype: {name}")


def _nearest_existing_parent(path: Path) -> Path:
    path = path.expanduser()
    probe = path if path.exists() else path.parent
    while not probe.exists():
        if probe == probe.parent:
            break
        probe = probe.parent
    return probe


def _check_free_space(cache_dir: Path, min_free_gb: float) -> None:
    parent = _nearest_existing_parent(cache_dir)
    free_gb = shutil.disk_usage(parent).free / (1024 ** 3)
    if free_gb < float(min_free_gb):
        raise RuntimeError(
            f"Not enough free space for VAE latent cache under {parent}: "
            f"free={free_gb:.2f}GB required>={min_free_gb:.2f}GB"
        )



def _autocast_context(device: str, dtype: torch.dtype, enabled: bool):
    if not enabled:
        from contextlib import nullcontext
        return nullcontext()
    if not str(device).startswith("cuda"):
        from contextlib import nullcontext
        return nullcontext()
    if dtype not in (torch.bfloat16, torch.float16):
        from contextlib import nullcontext
        return nullcontext()
    return torch.autocast(device_type="cuda", dtype=dtype)

def _manifest_payload(args, dataset: GEMBenchKeystepsDataset, vae, vae_path: str, vae_hash: str, latent_shape: tuple[int, int, int, int], *, complete: bool) -> dict:
    return {
        "cache_version": CACHE_VERSION,
        "complete": bool(complete),
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "dataset": build_expected_dataset_config(
            root=dataset.root,
            split=dataset.split,
            subset=dataset.subset,
            seed=dataset.seed,
            num_video_frames=dataset.num_video_frames,
            action_horizon=dataset.action_horizon,
            video_size=dataset.video_size,
            camera_order=dataset.camera_order,
        ),
        "num_rows": len(dataset.index),
        "latent_shape": list(latent_shape),
        "latents_dtype": "float32",
        "torch_dtype_used_for_encode": str(_dtype(args.torch_dtype)).replace("torch.", ""),
        "vae": {
            "model_id": args.model_id,
            "tokenizer_model_id": args.tokenizer_model_id,
            "redirect_common_files": bool(args.redirect_common_files),
            "path": str(vae_path),
            "hash": str(vae_hash),
            "z_dim": int(vae.model.z_dim),
            "temporal_downsample_factor": int(vae.temporal_downsample_factor),
            "upsampling_factor": int(vae.upsampling_factor),
            "encode_autocast": bool(args.encode_autocast),
            "autocast_dtype": str(_dtype(args.torch_dtype)).replace("torch.", "") if bool(args.encode_autocast) else None,
        },
    }


def _index_rows(dataset: GEMBenchKeystepsDataset) -> list[dict]:
    rows = []
    for row_id, (taskvar, episode_key) in enumerate(dataset.index):
        rows.append(
            {
                "row_id": row_id,
                "taskvar": str(taskvar),
                "episode_key": episode_key.decode("ascii", errors="ignore"),
            }
        )
    return rows


def _manifest_matches(path: Path, expected: dict) -> bool:
    if not path.exists():
        return False
    import json

    with path.open("r", encoding="utf-8") as handle:
        actual = json.load(handle)
    keys = ["cache_version", "dataset", "num_rows", "latent_shape", "latents_dtype", "vae"]
    for key in keys:
        if actual.get(key) != expected.get(key):
            return False
    return True


def _prepare_arrays(cache_dir: Path, manifest: dict, *, resume: bool):
    latents_path = cache_dir / "video_latents.float32.npy"
    completed_path = cache_dir / "completed_rows.bool.npy"
    shape = (int(manifest["num_rows"]), *[int(x) for x in manifest["latent_shape"]])
    if resume and latents_path.exists() and completed_path.exists():
        latents = np.lib.format.open_memmap(latents_path, mode="r+", dtype=np.float32, shape=shape)
        completed = np.lib.format.open_memmap(completed_path, mode="r+", dtype=np.bool_, shape=(shape[0],))
    else:
        latents = np.lib.format.open_memmap(latents_path, mode="w+", dtype=np.float32, shape=shape)
        completed = np.lib.format.open_memmap(completed_path, mode="w+", dtype=np.bool_, shape=(shape[0],))
        completed[:] = False
        completed.flush()
    return latents, completed


def main() -> None:
    parser = argparse.ArgumentParser(description="Precompute Wan VAE latents for GEMBench keystep training.")
    parser.add_argument("--root", default=os.environ.get("GEMBENCH_ROOT", "/mnt/yuhan/datasets/GEMBench"))
    parser.add_argument("--cache-dir", default=os.environ.get("GEMBENCH_VAE_CACHE_DIR", DEFAULT_GEMBENCH_VAE_CACHE_DIR))
    parser.add_argument("--split", default="train")
    parser.add_argument("--subset", default="keysteps_bbox")
    parser.add_argument("--seed", default="seed0")
    parser.add_argument("--num-video-frames", type=int, default=9)
    parser.add_argument("--action-horizon", type=int, default=8)
    parser.add_argument("--video-size", type=int, nargs=2, default=(224, 672))
    parser.add_argument("--camera-order", nargs="+", default=("front", "wrist", "left_shoulder"))
    parser.add_argument("--taskvars", default=None)
    parser.add_argument("--limit", type=int, default=None, help="Only encode the first N rows; for smoke generation only.")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--torch-dtype", default="bf16")
    parser.set_defaults(encode_autocast=True)
    parser.add_argument("--encode-autocast", dest="encode_autocast", action="store_true")
    parser.add_argument("--no-encode-autocast", dest="encode_autocast", action="store_false")
    parser.add_argument("--model-id", default="Wan-AI/Wan2.2-TI2V-5B")
    parser.add_argument("--tokenizer-model-id", default="Wan-AI/Wan2.1-T2V-1.3B")
    parser.add_argument("--redirect-common-files", action="store_true")
    parser.add_argument("--no-resume", action="store_true")
    parser.add_argument("--min-free-gb", type=float, default=5.0)
    parser.add_argument("--log-every", type=int, default=25)
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--shard-id", type=int, default=0)
    args = parser.parse_args()
    if int(args.num_shards) < 1:
        raise ValueError("--num-shards must be >= 1")
    if int(args.shard_id) < 0 or int(args.shard_id) >= int(args.num_shards):
        raise ValueError("--shard-id must be in [0, num_shards)")

    cache_dir = Path(args.cache_dir).expanduser().resolve()
    _check_free_space(cache_dir, args.min_free_gb)
    cache_dir.mkdir(parents=True, exist_ok=True)

    dataset = GEMBenchKeystepsDataset(
        root=args.root,
        split=args.split,
        subset=args.subset,
        seed=args.seed,
        taskvars=args.taskvars,
        num_video_frames=args.num_video_frames,
        action_horizon=args.action_horizon,
        video_size=args.video_size,
        camera_order=args.camera_order,
        val_set_proportion=0.0,
        is_training_set=True,
        text_embedding_cache_dir=None,
        allow_missing_text_embeds=True,
        pretrained_norm_stats=None,
        stats_scan_limit=0,
    )
    if args.limit is not None:
        dataset.index = dataset.index[: max(int(args.limit), 0)]
    print(
        f"[vae-cache] rows={len(dataset.index)} cache_dir={cache_dir} "
        f"shard={int(args.shard_id)}/{int(args.num_shards)} encode_autocast={bool(args.encode_autocast)}"
    )

    vae, vae_path = load_wan22_ti2v_5b_vae(
        device=args.device,
        torch_dtype=_dtype(args.torch_dtype),
        model_id=args.model_id,
        tokenizer_model_id=args.tokenizer_model_id,
        redirect_common_files=bool(args.redirect_common_files),
    )
    vae.eval()
    vae_hash = hash_model_file(vae_path)
    latent_shape = expected_latent_shape(
        num_video_frames=dataset.num_video_frames,
        video_size=dataset.video_size,
        z_dim=int(vae.model.z_dim),
        temporal_downsample_factor=int(vae.temporal_downsample_factor),
        upsampling_factor=int(vae.upsampling_factor),
    )

    manifest_incomplete = _manifest_payload(args, dataset, vae, vae_path, vae_hash, latent_shape, complete=False)
    manifest_complete = dict(manifest_incomplete)
    manifest_complete["complete"] = True
    manifest_path = cache_dir / "manifest.json"
    manifest_exists = manifest_path.exists()
    manifest_matches = _manifest_matches(manifest_path, manifest_incomplete)
    if manifest_exists and not manifest_matches and not args.no_resume:
        raise RuntimeError(
            f"Existing VAE cache manifest does not match requested config: {manifest_path}. "
            "Use --no-resume only when you intentionally want to rebuild this cache directory."
        )
    resume = not args.no_resume and manifest_matches
    if not resume:
        write_index_jsonl_atomic(cache_dir / "index.jsonl", _index_rows(dataset))
        write_json_atomic(manifest_path, manifest_incomplete)
    else:
        print("[vae-cache] resuming existing cache")

    latents, completed = _prepare_arrays(cache_dir, manifest_incomplete, resume=resume)
    start = time.time()
    encoded = 0
    dtype = _dtype(args.torch_dtype)
    with torch.no_grad(), _autocast_context(args.device, dtype, bool(args.encode_autocast)):
        for row_id, (taskvar, episode_key) in enumerate(dataset.index):
            if row_id % int(args.num_shards) != int(args.shard_id):
                continue
            if bool(completed[row_id]):
                continue
            episode = dataset.store.get(taskvar, episode_key)
            video = dataset._episode_video(episode).unsqueeze(0).to(device=args.device, dtype=dtype)
            latent = vae.encode(video, device=args.device, tiled=False)
            expected = (1, *latent_shape)
            if tuple(latent.shape) != expected:
                raise RuntimeError(f"Unexpected latent shape for row={row_id}: {tuple(latent.shape)} vs {expected}")
            latents[row_id] = latent[0].detach().to(device="cpu", dtype=torch.float32).numpy()
            completed[row_id] = True
            encoded += 1
            if encoded % max(int(args.log_every), 1) == 0:
                latents.flush()
                completed.flush()
                elapsed = max(time.time() - start, 1e-6)
                print(f"[vae-cache] encoded={encoded} row={row_id + 1}/{len(dataset.index)} rate={encoded / elapsed:.3f} rows/s")
    latents.flush()
    completed.flush()
    if not bool(np.all(completed)):
        missing_count = int((~np.asarray(completed)).sum())
        missing = np.where(~np.asarray(completed))[0][:10].tolist()
        print(f"[vae-cache] shard complete but cache still incomplete missing_count={missing_count} first_missing={missing}")
        if int(args.num_shards) <= 1:
            raise RuntimeError(f"Cache still has incomplete rows: {missing}")
        return
    write_json_atomic(manifest_path, manifest_complete)
    print(f"[vae-cache] complete rows={len(dataset.index)} latents={cache_dir / 'video_latents.float32.npy'}")


if __name__ == "__main__":
    main()
