#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT / "src"))

from fastwam.datasets.gembench.dataset import GEMBenchKeystepsDataset
from fastwam.datasets.gembench.vae_cache import (
    DEFAULT_GEMBENCH_VAE_CACHE_DIR,
    build_expected_dataset_config,
    cache_key,
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

def _write_json(path: str | Path, payload: dict[str, Any]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _write_text(path: str | Path, text: str) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def _balanced_indices(dataset: GEMBenchKeystepsDataset, limit: int) -> list[int]:
    if limit <= 0:
        return []
    out = []
    seen = set()
    for idx, (taskvar, _) in enumerate(dataset.index):
        if taskvar in seen:
            continue
        out.append(idx)
        seen.add(taskvar)
        if len(out) >= limit:
            return out
    used = set(out)
    for idx in range(len(dataset)):
        if idx in used:
            continue
        out.append(idx)
        if len(out) >= limit:
            return out
    return out


def _cache_status(cache_dir: str | Path) -> dict[str, Any]:
    cache_dir = Path(cache_dir).expanduser().resolve()
    manifest_path = cache_dir / "manifest.json"
    index_path = cache_dir / "index.jsonl"
    latents_path = cache_dir / "video_latents.float32.npy"
    completed_path = cache_dir / "completed_rows.bool.npy"
    status: dict[str, Any] = {
        "cache_dir": str(cache_dir),
        "manifest_path": str(manifest_path),
        "manifest_exists": manifest_path.exists(),
        "index_exists": index_path.exists(),
        "latents_exists": latents_path.exists(),
        "completed_rows_exists": completed_path.exists(),
        "manifest_complete": False,
        "manifest_num_rows": None,
        "index_rows": None,
        "completed_rows": None,
        "completed_total": None,
        "latents_shape": None,
        "latents_dtype": None,
        "errors": [],
    }
    if manifest_path.exists():
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            status["manifest"] = manifest
            status["manifest_complete"] = bool(manifest.get("complete", False))
            status["manifest_num_rows"] = manifest.get("num_rows")
        except Exception as exc:
            status["errors"].append(f"manifest_read_failed: {exc}")
    if index_path.exists():
        try:
            rows = 0
            row_keys: set[str] = set()
            with index_path.open("r", encoding="utf-8") as handle:
                for line in handle:
                    line = line.strip()
                    if not line:
                        continue
                    row = json.loads(line)
                    rows += 1
                    row_keys.add(cache_key(str(row["taskvar"]), str(row["episode_key"])))
            status["index_rows"] = rows
            status["_index_keys"] = row_keys
        except Exception as exc:
            status["errors"].append(f"index_read_failed: {exc}")
    if completed_path.exists():
        try:
            completed = np.load(completed_path, mmap_mode="r")
            status["completed_rows"] = int(completed.sum())
            status["completed_total"] = int(completed.shape[0])
            status["completed_dtype"] = str(completed.dtype)
        except Exception as exc:
            status["errors"].append(f"completed_rows_read_failed: {exc}")
    if latents_path.exists():
        try:
            latents = np.load(latents_path, mmap_mode="r")
            status["latents_shape"] = [int(x) for x in latents.shape]
            status["latents_dtype"] = str(latents.dtype)
        except Exception as exc:
            status["errors"].append(f"latents_read_failed: {exc}")
    return status


def _assert_tensor_close(key: str, raw: torch.Tensor, cached: torch.Tensor, *, atol: float) -> float:
    if tuple(raw.shape) != tuple(cached.shape):
        raise AssertionError(f"{key} shape mismatch: {tuple(raw.shape)} vs {tuple(cached.shape)}")
    if raw.dtype == torch.bool or raw.dtype in (torch.int8, torch.int16, torch.int32, torch.int64, torch.long):
        if not torch.equal(raw, cached):
            raise AssertionError(f"{key} tensor mismatch")
        return 0.0
    max_abs = (raw.float() - cached.float()).abs().max().item() if raw.numel() else 0.0
    if max_abs > atol:
        raise AssertionError(f"{key} max_abs_diff={max_abs:.6g} > {atol}")
    return float(max_abs)


def _compare_samples(raw: dict, cached: dict, *, atol: float, use_relation: bool) -> dict[str, float]:
    if "video" not in raw:
        raise AssertionError("raw sample must contain RGB video")
    if "video_latents" not in cached:
        raise AssertionError("cached sample must contain video_latents")
    if "video" in cached:
        raise AssertionError("cached sample must not contain RGB video")
    scalar_keys = ("prompt", "taskvar", "episode_key")
    for key in scalar_keys:
        if raw[key] != cached[key]:
            raise AssertionError(f"{key} mismatch: {raw[key]!r} vs {cached[key]!r}")
    tensor_keys = [
        "action",
        "proprio",
        "context",
        "context_mask",
        "image_is_pad",
        "action_is_pad",
        "proprio_is_pad",
        "action_dim_is_pad",
        "proprio_dim_is_pad",
    ]
    if use_relation:
        tensor_keys.extend(
            key
            for key in raw.keys()
            if key.startswith("relation_") and isinstance(raw[key], torch.Tensor)
        )
    diffs: dict[str, float] = {}
    for key in sorted(set(tensor_keys)):
        if key not in raw or key not in cached:
            raise AssertionError(f"{key} missing from raw/cached sample")
        diffs[key] = _assert_tensor_close(key, raw[key], cached[key], atol=atol)
    return diffs


def _dataset_kwargs(args, *, vae_cache: bool, is_training_set: bool) -> dict[str, Any]:
    kwargs = dict(
        root=args.root,
        split="train",
        subset="keysteps_bbox",
        seed="seed0",
        taskvars=args.taskvars,
        skip_missing_taskvars=True,
        num_video_frames=9,
        action_horizon=8,
        video_size=(224, 672),
        camera_order=("front", "wrist", "left_shoulder"),
        val_set_proportion=0.02,
        is_training_set=is_training_set,
        split_seed=42,
        max_episodes_per_taskvar=args.max_episodes_per_taskvar,
        text_embedding_cache_dir=args.cache_dir_text,
        pretrained_norm_stats=args.pretrained_norm_stats,
        norm_default_mode=args.norm_default_mode,
        stats_scan_limit=0,
    )
    if vae_cache:
        kwargs["vae_latent_cache_dir"] = args.vae_cache_dir
        kwargs["vae_latent_cache_encode_autocast"] = bool(args.encode_autocast)
    return kwargs


def _expected_dataset_config(dataset: GEMBenchKeystepsDataset) -> dict[str, Any]:
    return build_expected_dataset_config(
        root=dataset.root,
        split=dataset.split,
        subset=dataset.subset,
        seed=dataset.seed,
        num_video_frames=dataset.num_video_frames,
        action_horizon=dataset.action_horizon,
        video_size=dataset.video_size,
        camera_order=dataset.camera_order,
    )


def _check_cache_files(
    cache_status: dict[str, Any],
    expected_index: list[tuple[str, bytes]],
    expected_config: dict[str, Any],
) -> dict[str, Any]:
    manifest = cache_status.get("manifest") or {}
    manifest_rows = cache_status.get("manifest_num_rows")
    index_rows = cache_status.get("index_rows")
    completed_total = cache_status.get("completed_total")
    completed_rows = cache_status.get("completed_rows")
    latents_shape = cache_status.get("latents_shape") or []
    index_keys = cache_status.get("_index_keys") or set()
    missing = []
    for taskvar, episode_key in expected_index:
        key = cache_key(taskvar, episode_key)
        if key not in index_keys:
            missing.append(key)
            if len(missing) >= 8:
                break
    dataset_cfg = manifest.get("dataset") or {}
    checks = {
        "manifest_exists": bool(cache_status.get("manifest_exists")),
        "index_exists": bool(cache_status.get("index_exists")),
        "latents_exists": bool(cache_status.get("latents_exists")),
        "completed_rows_exists": bool(cache_status.get("completed_rows_exists")),
        "manifest_complete": bool(cache_status.get("manifest_complete")),
        "manifest_num_rows_matches_index": manifest_rows is not None and manifest_rows == index_rows,
        "completed_rows_shape_matches_manifest": manifest_rows is not None and completed_total == manifest_rows,
        "all_rows_completed": manifest_rows is not None and completed_rows == manifest_rows,
        "latents_first_dim_matches_manifest": bool(latents_shape) and manifest_rows is not None and int(latents_shape[0]) == int(manifest_rows),
        "latents_dtype_float32": cache_status.get("latents_dtype") == "float32",
        "dataset_config_matches": all(dataset_cfg.get(key) == value for key, value in expected_config.items()),
        "expected_subset_index_covered": not missing,
        "no_cache_file_read_errors": not cache_status.get("errors"),
    }
    return {
        "checks": checks,
        "missing_expected_index_rows": missing,
        "expected_index_rows": len(expected_index),
        "expected_dataset_config": expected_config,
        "cache_dataset_config": dataset_cfg,
    }


def _markdown(payload: dict[str, Any]) -> str:
    lines = [
        "# GEMBench VAE Cache Contract",
        "",
        f"Status: `{payload['status']}`",
        "",
        "| check | passed |",
        "|---|---|",
    ]
    for key, value in sorted((payload.get("checks") or {}).items()):
        lines.append(f"| {key} | `{bool(value)}` |")
    lines.extend(
        [
            "",
            "## Cache",
            "",
            "```json",
            json.dumps(payload.get("cache", {}), indent=2, sort_keys=True),
            "```",
        ]
    )
    if payload.get("samples"):
        lines.extend(["", "## Samples", ""])
        for sample in payload["samples"]:
            lines.append(
                f"- idx={sample['idx']} taskvar={sample['taskvar']} episode={sample['episode_key']} "
                f"max_tensor_abs_diff={sample.get('max_tensor_abs_diff')} latent_max_abs_diff={sample.get('latent_max_abs_diff')}"
            )
    if payload.get("error"):
        lines.extend(["", "## Error", "", "```text", str(payload["error"]), "```"])
    lines.append("")
    return "\n".join(lines)


def build_payload(args: argparse.Namespace) -> dict[str, Any]:
    raw_train = GEMBenchKeystepsDataset(**_dataset_kwargs(args, vae_cache=False, is_training_set=True))
    raw_val = GEMBenchKeystepsDataset(**_dataset_kwargs(args, vae_cache=False, is_training_set=False))
    expected_config = _expected_dataset_config(raw_train)
    cache_status = _cache_status(args.vae_cache_dir)
    cache_checks = _check_cache_files(
        cache_status,
        expected_index=raw_train.index + raw_val.index,
        expected_config=expected_config,
    )
    checks = dict(cache_checks["checks"])
    payload: dict[str, Any] = {
        "status": "failed",
        "checks": checks,
        "splits": {
            "train_len": len(raw_train),
            "val_len": len(raw_val),
            "train_taskvars": len(raw_train.taskvars),
            "val_taskvars": len(raw_val.taskvars),
        },
        "cache": {key: value for key, value in cache_status.items() if key != "_index_keys"},
        "cache_contract": cache_checks,
        "samples": [],
        "taskvars": args.taskvars,
        "skip_latent_encode": bool(args.skip_latent_encode),
        "encode_autocast": bool(args.encode_autocast),
    }
    complete_enough = (
        checks["manifest_complete"]
        and checks["all_rows_completed"]
        and checks["expected_subset_index_covered"]
        and checks["dataset_config_matches"]
        and checks["latents_dtype_float32"]
        and checks["latents_first_dim_matches_manifest"]
        and checks["no_cache_file_read_errors"]
    )
    if not complete_enough:
        payload["status"] = "incomplete" if not checks["all_rows_completed"] or not checks["manifest_complete"] else "failed"
        return payload

    cached_train = GEMBenchKeystepsDataset(**_dataset_kwargs(args, vae_cache=True, is_training_set=True))
    cached_val = GEMBenchKeystepsDataset(**_dataset_kwargs(args, vae_cache=True, is_training_set=False))
    checks["train_len_matches"] = len(raw_train) == len(cached_train)
    checks["val_len_matches"] = len(raw_val) == len(cached_val)
    checks["train_index_matches"] = raw_train.index == cached_train.index
    checks["val_index_matches"] = raw_val.index == cached_val.index
    if not checks["train_len_matches"]:
        raise AssertionError(f"train length mismatch: raw={len(raw_train)} cached={len(cached_train)}")
    if not checks["val_len_matches"]:
        raise AssertionError(f"val length mismatch: raw={len(raw_val)} cached={len(cached_val)}")
    if not checks["train_index_matches"]:
        raise AssertionError("cached train split index differs from raw train split index")
    if not checks["val_index_matches"]:
        raise AssertionError("cached val split index differs from raw val split index")
    print(
        f"split_ok train={len(raw_train)} val={len(raw_val)} "
        f"cache_rows={len(cached_train.vae_latent_cache)}"
    )

    indices = _balanced_indices(raw_train, min(args.samples, len(raw_train)))
    vae = None
    dtype = _dtype(args.torch_dtype)
    if not args.skip_latent_encode:
        vae, vae_path = load_wan22_ti2v_5b_vae(
            device=args.device,
            torch_dtype=dtype,
            model_id=args.model_id,
            tokenizer_model_id=args.tokenizer_model_id,
            redirect_common_files=bool(args.redirect_common_files),
        )
        vae.eval()
        vae_hash = hash_model_file(vae_path)
        payload["loaded_vae"] = {"path": str(vae_path), "hash": vae_hash}
        manifest_hash = cached_train.vae_latent_cache.manifest.get("vae", {}).get("hash")
        checks["vae_hash_matches_manifest"] = manifest_hash == vae_hash
        if manifest_hash != vae_hash:
            raise AssertionError(f"VAE hash mismatch: manifest={manifest_hash} loaded={vae_hash}")
    else:
        checks["vae_hash_matches_manifest"] = True

    for idx in indices:
        raw = raw_train[idx]
        cached = cached_train[idx]
        diffs = _compare_samples(raw, cached, atol=args.atol, use_relation=False)
        sample_payload: dict[str, Any] = {
            "idx": idx,
            "taskvar": raw["taskvar"],
            "episode_key": raw["episode_key"],
            "tensor_max_abs_diff_by_key": diffs,
            "max_tensor_abs_diff": max(diffs.values()) if diffs else 0.0,
            "latent_max_abs_diff": None,
        }
        if vae is not None:
            with torch.no_grad(), _autocast_context(args.device, dtype, bool(args.encode_autocast)):
                encoded = vae.encode(raw["video"].unsqueeze(0).to(device=args.device, dtype=dtype), device=args.device, tiled=False)
            cached_latent = cached["video_latents"].unsqueeze(0).to(dtype=dtype).float().cpu()
            encoded_cpu = encoded.to(dtype=dtype).float().cpu()
            max_abs = (encoded_cpu - cached_latent).abs().max().item()
            sample_payload["latent_max_abs_diff"] = float(max_abs)
            if max_abs > args.latent_atol:
                raise AssertionError(
                    f"latent parity failed idx={idx} taskvar={raw['taskvar']} episode={raw['episode_key']} "
                    f"max_abs={max_abs:.6g} > {args.latent_atol}"
                )
            print(f"latent_ok idx={idx} taskvar={raw['taskvar']} episode={raw['episode_key']} max_abs={max_abs:.3g}")
        else:
            print(f"sample_ok idx={idx} taskvar={raw['taskvar']} episode={raw['episode_key']}")
        payload["samples"].append(sample_payload)
    checks["sample_count_matches_request"] = len(payload["samples"]) == len(indices)
    checks["sample_tensor_parity_passed"] = True
    checks["latent_parity_passed"] = bool(args.skip_latent_encode) or all(
        (sample.get("latent_max_abs_diff") or 0.0) <= args.latent_atol for sample in payload["samples"]
    )
    payload["status"] = "passed" if all(checks.values()) else "failed"
    return payload


def add_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--root", default=os.environ.get("GEMBENCH_ROOT", "/mnt/yuhan/datasets/GEMBench"))
    parser.add_argument("--vae-cache-dir", default=os.environ.get("GEMBENCH_VAE_CACHE_DIR", DEFAULT_GEMBENCH_VAE_CACHE_DIR))
    parser.add_argument("--cache-dir-text", default="data/text_embeds_cache/gembench_keysteps_bbox")
    parser.add_argument("--pretrained-norm-stats", default="data/gembench_keysteps_bbox_dataset_stats.json")
    parser.add_argument("--norm-default-mode", default="z-score")
    parser.add_argument("--taskvars", default=None)
    parser.add_argument("--max-episodes-per-taskvar", type=int, default=None)
    parser.add_argument("--samples", type=int, default=8)
    parser.add_argument("--atol", type=float, default=1.0e-5)
    parser.add_argument("--latent-atol", type=float, default=1.0e-3)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--torch-dtype", default="bf16")
    parser.set_defaults(encode_autocast=True)
    parser.add_argument("--encode-autocast", dest="encode_autocast", action="store_true")
    parser.add_argument("--no-encode-autocast", dest="encode_autocast", action="store_false")
    parser.add_argument("--skip-latent-encode", action="store_true")
    parser.add_argument("--model-id", default="Wan-AI/Wan2.2-TI2V-5B")
    parser.add_argument("--tokenizer-model-id", default="Wan-AI/Wan2.1-T2V-1.3B")
    parser.add_argument("--redirect-common-files", action="store_true")
    parser.add_argument("--json-output")
    parser.add_argument("--markdown-output")
    parser.add_argument(
        "--allow-incomplete-cache",
        action="store_true",
        help="Return success for structured progress reporting when the cache is still being precomputed.",
    )


def main() -> int:
    parser = argparse.ArgumentParser(description="Verify GEMBench VAE latent cache semantic parity.")
    add_args(parser)
    args = parser.parse_args()
    try:
        payload = build_payload(args)
    except Exception as exc:
        payload = {
            "status": "failed",
            "checks": {},
            "error": f"{type(exc).__name__}: {exc}",
        }
    if args.json_output:
        _write_json(args.json_output, payload)
    if args.markdown_output:
        _write_text(args.markdown_output, _markdown(payload))
    print(f"status={payload['status']}")
    if payload["status"] == "passed":
        print("vae_cache_contract_ok")
        return 0
    if payload["status"] == "incomplete" and args.allow_incomplete_cache:
        return 0
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
