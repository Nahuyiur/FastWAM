#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import time
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT / "src"))

from fastwam.datasets.gembench.microsteps_9v32 import (  # noqa: E402
    DEFAULT_CACHE_CAMERA_ORDER,
    DEFAULT_CAMERA_ORDER,
    DEFAULT_FRAME_OFFSETS,
    SCHEMA_VERSION,
    VAE_LATENT_CACHE_VERSION,
    GEMBenchMicrosteps9V32Dataset,
    build_vae_cache_dataset_config,
    sha256_file,
    window_cache_key,
)
from fastwam.datasets.gembench.vae_cache import expected_latent_shape, write_index_jsonl_atomic, write_json_atomic  # noqa: E402
from fastwam.models.wan22.helpers.io import hash_model_file  # noqa: E402
from fastwam.models.wan22.helpers.loader import load_wan22_ti2v_5b_vae  # noqa: E402


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
    probe = path if path.exists() else path.parent
    while not probe.exists() and probe != probe.parent:
        probe = probe.parent
    return probe


def _check_free_space(cache_dir: Path, min_free_gb: float) -> None:
    parent = _nearest_existing_parent(cache_dir)
    free_gb = shutil.disk_usage(parent).free / (1024**3)
    if free_gb < float(min_free_gb):
        raise RuntimeError(
            f"Not enough free space for GEMBench 9v32 VAE cache under {parent}: "
            f"free={free_gb:.2f}GB required>={min_free_gb:.2f}GB"
        )


def _autocast_context(device: str, dtype: torch.dtype, enabled: bool):
    if not enabled or not str(device).startswith("cuda") or dtype not in (torch.bfloat16, torch.float16):
        from contextlib import nullcontext

        return nullcontext()
    return torch.autocast(device_type="cuda", dtype=dtype)


def _dataset(args) -> GEMBenchMicrosteps9V32Dataset:
    return GEMBenchMicrosteps9V32Dataset(
        manifest_path=args.manifest,
        rgb_cache_dir=args.rgb_cache_dir,
        split="train",
        seed=args.seed,
        frame_offsets=args.frame_offsets,
        action_horizon=args.action_horizon,
        video_size=args.video_size,
        camera_order=args.camera_order,
        cache_camera_order=args.cache_camera_order,
        window_stride=args.window_stride,
        max_windows_per_demo=args.max_windows_per_demo,
        taskvars=args.taskvars,
        text_embedding_cache_dir=None,
        allow_missing_text_embeds=True,
        pretrained_norm_stats=args.pretrained_norm_stats,
        stats_scan_limit=0,
        vae_latent_cache_dir=None,
    )


def _index_rows(dataset: GEMBenchMicrosteps9V32Dataset) -> list[dict]:
    rows = []
    for row_id, (row_idx, start) in enumerate(dataset.index):
        row = dataset.demo_rows[row_idx]
        rows.append(
            {
                "row_id": row_id,
                "taskvar": str(row["taskvar"]),
                "episode_key": str(row["episode_key"]),
                "window_start": int(start),
                "key": window_cache_key(str(row["taskvar"]), str(row["episode_key"]), int(start)),
            }
        )
    return rows


def _manifest_payload(
    args,
    dataset: GEMBenchMicrosteps9V32Dataset,
    *,
    latent_shape: tuple[int, int, int, int],
    vae_path: str,
    vae_hash: str,
    complete: bool,
) -> dict:
    manifest_sha = sha256_file(args.manifest)
    return {
        "cache_version": VAE_LATENT_CACHE_VERSION,
        "complete": bool(complete),
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "source_schema_version": SCHEMA_VERSION,
        "dataset": build_vae_cache_dataset_config(
            manifest_path=args.manifest,
            manifest_sha256=manifest_sha,
            rgb_cache_dir=args.rgb_cache_dir,
            seed=args.seed,
            frame_offsets=args.frame_offsets,
            action_horizon=args.action_horizon,
            window_stride=args.window_stride,
            video_size=args.video_size,
            camera_order=args.camera_order,
            cache_camera_order=args.cache_camera_order,
        ),
        "num_demos": len(dataset.demo_rows),
        "num_windows": len(dataset.index),
        "latent_shape": [int(v) for v in latent_shape],
        "latents_dtype": "float32",
        "vae": {
            "model_id": args.model_id,
            "tokenizer_model_id": args.tokenizer_model_id,
            "redirect_common_files": bool(args.redirect_common_files),
            "path": str(vae_path),
            "hash": str(vae_hash),
            "z_dim": int(latent_shape[0]),
            "temporal_downsample_factor": 4,
            "upsampling_factor": 16,
            "encode_autocast": bool(args.encode_autocast),
            "autocast_dtype": str(_dtype(args.torch_dtype)).replace("torch.", "") if bool(args.encode_autocast) else None,
        },
    }


def _manifest_matches(path: Path, expected: dict) -> bool:
    if not path.exists():
        return False
    actual = json.loads(path.read_text(encoding="utf-8"))
    for key in ("cache_version", "dataset", "num_demos", "num_windows", "latent_shape", "latents_dtype", "vae"):
        if actual.get(key) != expected.get(key):
            return False
    return True


def _prepare_arrays(cache_dir: Path, manifest: dict, *, resume: bool):
    latents_path = cache_dir / "video_latents.float32.npy"
    completed_path = cache_dir / "completed_windows.bool.npy"
    shape = (int(manifest["num_windows"]), *[int(v) for v in manifest["latent_shape"]])
    if resume and latents_path.exists() and completed_path.exists():
        latents = np.lib.format.open_memmap(latents_path, mode="r+", dtype=np.float32, shape=shape)
        completed = np.lib.format.open_memmap(completed_path, mode="r+", dtype=np.bool_, shape=(shape[0],))
    else:
        latents = np.lib.format.open_memmap(latents_path, mode="w+", dtype=np.float32, shape=shape)
        completed = np.lib.format.open_memmap(completed_path, mode="w+", dtype=np.bool_, shape=(shape[0],))
        completed[:] = False
        completed.flush()
    return latents, completed


def _write_complete_if_ready(cache_dir: Path, manifest_complete: dict, completed: np.ndarray) -> bool:
    completed_view = np.asarray(completed)
    if not bool(np.all(completed_view)):
        return False
    write_json_atomic(cache_dir / "manifest.json", manifest_complete)
    return True


def main() -> None:
    parser = argparse.ArgumentParser(description="Precompute window-level Wan VAE latents for GEMBench microsteps 9V32.")
    parser.add_argument("--manifest", default=os.environ.get("GEMBENCH_9V32_MANIFEST", "/mnt/yuhan/datasets/GEMBench/fastwam_cache/microsteps_9v32_manifest.json"))
    parser.add_argument("--rgb-cache-dir", default=os.environ.get("GEMBENCH_9V32_RGB_CACHE_DIR", "/mnt/yuhan/datasets/GEMBench/fastwam_cache/microsteps_9v32_rgb"))
    parser.add_argument("--cache-dir", default=os.environ.get("GEMBENCH_9V32_VAE_CACHE_DIR", "/mnt/yuhan/datasets/GEMBench/fastwam_cache/vae_latents/microsteps_9v32_seed0_3cam224x672_t9_a32_v1"))
    parser.add_argument("--seed", default="seed0")
    parser.add_argument("--frame-offsets", type=int, nargs="+", default=list(DEFAULT_FRAME_OFFSETS))
    parser.add_argument("--action-horizon", type=int, default=32)
    parser.add_argument("--video-size", type=int, nargs=2, default=(224, 672))
    parser.add_argument("--camera-order", nargs="+", default=list(DEFAULT_CAMERA_ORDER))
    parser.add_argument("--cache-camera-order", nargs="+", default=list(DEFAULT_CACHE_CAMERA_ORDER))
    parser.add_argument("--window-stride", type=int, default=1)
    parser.add_argument("--max-windows-per-demo", type=int, default=None)
    parser.add_argument("--taskvars", default=None)
    parser.add_argument("--limit-windows", type=int, default=None, help="Keep only the first N window rows; smoke generation only.")
    parser.add_argument("--pretrained-norm-stats", default=None)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--torch-dtype", default="bf16")
    parser.set_defaults(encode_autocast=True)
    parser.add_argument("--encode-autocast", dest="encode_autocast", action="store_true")
    parser.add_argument("--no-encode-autocast", dest="encode_autocast", action="store_false")
    parser.add_argument("--model-id", default="Wan-AI/Wan2.2-TI2V-5B")
    parser.add_argument("--tokenizer-model-id", default="Wan-AI/Wan2.1-T2V-1.3B")
    parser.add_argument("--redirect-common-files", action="store_true")
    parser.add_argument("--no-resume", action="store_true")
    parser.add_argument("--init-only", action="store_true")
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--shard-id", type=int, default=0)
    parser.add_argument("--min-free-gb", type=float, default=20.0)
    parser.add_argument("--log-every", type=int, default=100)
    args = parser.parse_args()

    if int(args.num_shards) < 1:
        raise ValueError("--num-shards must be >= 1")
    if int(args.shard_id) < 0 or int(args.shard_id) >= int(args.num_shards):
        raise ValueError("--shard-id must be in [0, num_shards)")

    cache_dir = Path(args.cache_dir).expanduser().resolve()
    _check_free_space(cache_dir, args.min_free_gb)
    cache_dir.mkdir(parents=True, exist_ok=True)

    dataset = _dataset(args)
    if args.limit_windows is not None:
        dataset.index = dataset.index[: max(int(args.limit_windows), 0)]
    print(
        f"[9v32-vae-cache] demos={len(dataset.demo_rows)} windows={len(dataset.index)} "
        f"cache_dir={cache_dir} shard={int(args.shard_id)}/{int(args.num_shards)}"
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
        num_video_frames=len(args.frame_offsets),
        video_size=args.video_size,
        z_dim=int(vae.model.z_dim),
        temporal_downsample_factor=int(vae.temporal_downsample_factor),
        upsampling_factor=int(vae.upsampling_factor),
    )

    manifest_incomplete = _manifest_payload(
        args,
        dataset,
        latent_shape=latent_shape,
        vae_path=vae_path,
        vae_hash=vae_hash,
        complete=False,
    )
    manifest_complete = dict(manifest_incomplete)
    manifest_complete["complete"] = True
    manifest_path = cache_dir / "manifest.json"
    manifest_matches = _manifest_matches(manifest_path, manifest_incomplete)
    if manifest_path.exists() and not manifest_matches and not args.no_resume:
        raise RuntimeError(
            f"Existing GEMBench 9v32 VAE cache manifest does not match requested config: {manifest_path}. "
            "Use --no-resume only when intentionally rebuilding this cache directory."
        )
    resume = not args.no_resume and manifest_matches
    if not resume:
        write_index_jsonl_atomic(cache_dir / "index.jsonl", _index_rows(dataset))
        write_json_atomic(manifest_path, manifest_incomplete)
    else:
        print("[9v32-vae-cache] resuming existing cache")

    latents, completed = _prepare_arrays(cache_dir, manifest_incomplete, resume=resume)
    if args.init_only:
        print(f"[9v32-vae-cache] init complete cache_dir={cache_dir}")
        return

    by_demo: dict[int, list[tuple[int, int]]] = defaultdict(list)
    for window_id, (row_idx, start) in enumerate(dataset.index):
        if window_id % int(args.num_shards) != int(args.shard_id):
            continue
        by_demo[int(row_idx)].append((int(window_id), int(start)))

    dtype = _dtype(args.torch_dtype)
    encoded = 0
    start_time = time.time()
    with torch.no_grad(), _autocast_context(args.device, dtype, bool(args.encode_autocast)):
        for row_idx in sorted(by_demo):
            row = dataset.demo_rows[row_idx]
            cache_path = dataset._cache_path(row)
            payload = np.load(cache_path, allow_pickle=False)
            try:
                dataset._validate_cache_payload(row, payload, cache_path)
                rgb = np.asarray(payload["rgb"])
                for window_id, window_start in by_demo[row_idx]:
                    if bool(completed[window_id]):
                        continue
                    frame_idx = np.asarray([int(window_start) + offset for offset in dataset.frame_offsets], dtype=np.int64)
                    video = dataset._video_tensor(rgb[frame_idx][:, dataset.camera_indices])
                    video = video.unsqueeze(0).to(device=args.device, dtype=dtype)
                    latent = vae.encode(video, device=args.device, tiled=False)
                    expected = (1, *latent_shape)
                    if tuple(latent.shape) != expected:
                        raise RuntimeError(f"Unexpected latent shape for window_id={window_id}: {tuple(latent.shape)} vs {expected}")
                    latents[window_id] = latent[0].detach().to(device="cpu", dtype=torch.float32).numpy()
                    completed[window_id] = True
                    encoded += 1
                    if encoded % max(int(args.log_every), 1) == 0:
                        latents.flush()
                        completed.flush()
                        elapsed = max(time.time() - start_time, 1e-6)
                        print(
                            f"[9v32-vae-cache] encoded={encoded} last_window={window_id + 1}/{len(dataset.index)} "
                            f"rate={encoded / elapsed:.3f} windows/s"
                        )
            finally:
                payload.close()

    latents.flush()
    completed.flush()
    if _write_complete_if_ready(cache_dir, manifest_complete, completed):
        print(f"[9v32-vae-cache] complete windows={len(dataset.index)} latents={cache_dir / 'video_latents.float32.npy'}")
    else:
        missing_count = int((~np.asarray(completed)).sum())
        missing = np.where(~np.asarray(completed))[0][:10].tolist()
        print(f"[9v32-vae-cache] shard complete; cache incomplete missing_count={missing_count} first_missing={missing}")


if __name__ == "__main__":
    main()
