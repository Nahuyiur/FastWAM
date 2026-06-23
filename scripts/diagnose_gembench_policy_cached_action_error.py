#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
from hydra.utils import instantiate
from omegaconf import OmegaConf

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from fastwam.evaluation.gembench_official.common import git_provenance, utc_now, write_json
from fastwam.evaluation.gembench_official.policy import GEMBenchOfficialActioner


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=True) + "\n")


def _action_list(value: Any) -> list[float]:
    return [float(v) for v in np.asarray(value, dtype=np.float64).reshape(-1).tolist()]


def _normalize_quat(quat: np.ndarray) -> np.ndarray:
    q = np.asarray(quat, dtype=np.float64).reshape(4)
    norm = float(np.linalg.norm(q))
    if not np.isfinite(norm) or norm <= 1.0e-8:
        return np.asarray([0.0, 0.0, 0.0, 1.0], dtype=np.float64)
    return q / norm


def _quat_angle_deg(pred: np.ndarray, target: np.ndarray) -> float:
    qp = _normalize_quat(pred)
    qt = _normalize_quat(target)
    dot = float(abs(np.dot(qp, qt)))
    dot = min(max(dot, 0.0), 1.0)
    return float(2.0 * math.acos(dot) * 180.0 / math.pi)


def _gripper_bin(value: float) -> int:
    return 1 if float(value) > 0.5 else 0


def _postprocess_chunk(actioner: GEMBenchOfficialActioner, denorm: np.ndarray) -> np.ndarray:
    return np.stack([actioner._postprocess_action(row) for row in np.asarray(denorm, dtype=np.float32)], axis=0)


def _prediction_seed(actioner: GEMBenchOfficialActioner, *, taskvar: str, episode_key: str, step_id: int) -> int | None:
    episode_id = int(str(episode_key).replace("episode", ""))
    return actioner._prediction_seed(taskvar=taskvar, episode_id=episode_id, step_id=int(step_id))


def _denormalize(actioner: GEMBenchOfficialActioner, action: torch.Tensor) -> np.ndarray:
    return actioner._denormalize_action_chunk(action)


def _metric_row(
    *,
    variant: str,
    sample_index: int,
    sample: dict[str, Any],
    normalized_action: torch.Tensor,
    denormalized_action: np.ndarray,
    executed_action: np.ndarray,
    elapsed_s: float,
) -> dict[str, Any]:
    gt_world = np.asarray(sample["policy_action_world_raw"], dtype=np.float64).reshape(1, 8)
    gt_model = np.asarray(sample["policy_action_raw"], dtype=np.float64).reshape(1, 8)
    pred_denorm = np.asarray(denormalized_action, dtype=np.float64).reshape(-1, 8)
    pred_exec = np.asarray(executed_action, dtype=np.float64).reshape(-1, 8)
    pred_norm = normalized_action.detach().to(dtype=torch.float32, device="cpu").numpy().reshape(-1, 8)
    first_denorm = pred_denorm[0]
    first_exec = pred_exec[0]
    first_norm = pred_norm[0]
    xyz_abs = np.abs(first_denorm[:3] - gt_world[0, :3])
    exec_xyz_abs = np.abs(first_exec[:3] - gt_world[0, :3])
    return {
        "variant": variant,
        "sample_index": int(sample_index),
        "taskvar": str(sample["taskvar"]),
        "episode_key": str(sample["episode_key"]),
        "window_start": int(sample["window_start"]),
        "policy_current_key_idx": int(sample["policy_current_key_idx"]),
        "policy_next_key_idx": int(sample["policy_next_key_idx"]),
        "policy_key_position": int(sample["policy_key_position"]),
        "policy_target_frame": str(sample["policy_target_frame"]),
        "policy_action_horizon": int(sample["policy_action_horizon"]),
        "wam_aux_action_horizon": int(sample["wam_aux_action_horizon"]),
        "gt_world_action": _action_list(gt_world[0]),
        "gt_model_action": _action_list(gt_model[0]),
        "pred_normalized_action": _action_list(first_norm),
        "pred_denormalized_action": _action_list(first_denorm),
        "pred_executed_action": _action_list(first_exec),
        "xyz_error": float(np.linalg.norm(first_denorm[:3] - gt_world[0, :3])),
        "xyz_abs_error": [float(v) for v in xyz_abs.tolist()],
        "executed_xyz_error": float(np.linalg.norm(first_exec[:3] - gt_world[0, :3])),
        "executed_xyz_abs_error": [float(v) for v in exec_xyz_abs.tolist()],
        "quat_angle_deg": _quat_angle_deg(first_denorm[3:7], gt_world[0, 3:7]),
        "pred_gripper_bin": _gripper_bin(first_denorm[7]),
        "gt_gripper_bin": _gripper_bin(gt_world[0, 7]),
        "gripper_mismatch": bool(_gripper_bin(first_denorm[7]) != _gripper_bin(gt_world[0, 7])),
        "chunk_horizon": int(pred_denorm.shape[0]),
        "elapsed_s": float(elapsed_s),
    }


def _summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    by_variant: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        by_variant.setdefault(str(row["variant"]), []).append(row)
    out: dict[str, Any] = {}
    for variant, variant_rows in sorted(by_variant.items()):
        xyz = np.asarray([float(row["xyz_error"]) for row in variant_rows], dtype=np.float64)
        exec_xyz = np.asarray([float(row["executed_xyz_error"]) for row in variant_rows], dtype=np.float64)
        quat = np.asarray([float(row["quat_angle_deg"]) for row in variant_rows], dtype=np.float64)
        abs_xyz = np.asarray([row["xyz_abs_error"] for row in variant_rows], dtype=np.float64)
        signed = np.asarray(
            [
                np.asarray(row["pred_denormalized_action"][:3], dtype=np.float64)
                - np.asarray(row["gt_world_action"][:3], dtype=np.float64)
                for row in variant_rows
            ],
            dtype=np.float64,
        )
        out[variant] = {
            "num_rows": int(len(variant_rows)),
            "mean_xyz_error": float(xyz.mean()) if xyz.size else None,
            "median_xyz_error": float(np.median(xyz)) if xyz.size else None,
            "p90_xyz_error": float(np.percentile(xyz, 90.0)) if xyz.size else None,
            "max_xyz_error": float(xyz.max()) if xyz.size else None,
            "mean_executed_xyz_error": float(exec_xyz.mean()) if exec_xyz.size else None,
            "median_executed_xyz_error": float(np.median(exec_xyz)) if exec_xyz.size else None,
            "mean_quat_angle_deg": float(quat.mean()) if quat.size else None,
            "median_quat_angle_deg": float(np.median(quat)) if quat.size else None,
            "gripper_mismatch_rate": float(
                sum(bool(row["gripper_mismatch"]) for row in variant_rows) / max(len(variant_rows), 1)
            ),
            "signed_xyz_error_mean": [float(v) for v in signed.mean(axis=0).tolist()] if signed.size else None,
            "signed_xyz_error_median": [float(v) for v in np.median(signed, axis=0).tolist()] if signed.size else None,
            "abs_xyz_error_mean": [float(v) for v in abs_xyz.mean(axis=0).tolist()] if abs_xyz.size else None,
            "abs_xyz_error_median": [float(v) for v in np.median(abs_xyz, axis=0).tolist()] if abs_xyz.size else None,
            "worst_rows": sorted(
                [
                    {
                        "sample_index": int(row["sample_index"]),
                        "taskvar": row["taskvar"],
                        "episode_key": row["episode_key"],
                        "xyz_error": float(row["xyz_error"]),
                        "xyz_abs_error": row["xyz_abs_error"],
                    }
                    for row in variant_rows
                ],
                key=lambda item: item["xyz_error"],
                reverse=True,
            )[:10],
        }
    return out


def _sample_infer_action(
    *,
    actioner: GEMBenchOfficialActioner,
    sample: dict[str, Any],
    variant: str,
    seed: int | None,
) -> torch.Tensor:
    input_image = sample["video"][:, 0].unsqueeze(0)
    proprio = sample["proprio"][0]
    kwargs: dict[str, Any] = {
        "input_image": input_image,
        "action_horizon": int(sample["policy_action_horizon"]),
        "proprio": proprio,
        "negative_prompt": "",
        "text_cfg_scale": 1.0,
        "num_inference_steps": int(actioner.num_inference_steps),
        "sigma_shift": None,
        "seed": seed,
        "rand_device": actioner.rand_device,
        "tiled": actioner.tiled,
    }
    if variant == "cached_context_infer_action":
        kwargs["prompt"] = None
        kwargs["context"] = sample["context"]
        kwargs["context_mask"] = sample["context_mask"]
    elif variant == "prompt_infer_action":
        kwargs["prompt"] = sample["prompt"]
    else:
        raise ValueError(f"Unsupported infer_action variant: {variant}")
    with torch.no_grad():
        pred = actioner.model.infer_action(**kwargs)
    return pred["action"]


def _sample_infer_joint(
    *,
    actioner: GEMBenchOfficialActioner,
    sample: dict[str, Any],
    seed: int | None,
) -> torch.Tensor:
    input_image = sample["video"][:, 0].unsqueeze(0)
    proprio = sample["proprio"][0]
    with torch.no_grad():
        pred = actioner.model.infer(
            prompt=None,
            input_image=input_image,
            num_frames=int(sample["video"].shape[1]),
            action=sample["action"],
            action_horizon=int(sample["policy_action_horizon"]),
            proprio=proprio,
            context=sample["context"],
            context_mask=sample["context_mask"],
            negative_prompt="",
            text_cfg_scale=1.0,
            action_cfg_scale=1.0,
            num_inference_steps=int(actioner.num_inference_steps),
            sigma_shift=None,
            seed=seed,
            rand_device=actioner.rand_device,
            tiled=actioner.tiled,
        )
    return pred["action"]


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Diagnose FastWAM key-step policy action error on cached GEMBench dataset inputs, "
            "separating dataset/model inference from live simulator observation effects."
        )
    )
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--dataset-key", default="val", choices=("train", "val"))
    parser.add_argument("--max-samples", type=int, default=50)
    parser.add_argument("--sample-offset", type=int, default=0)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--mixed-precision", default="bf16", choices=("no", "fp16", "bf16"))
    parser.add_argument("--num-inference-steps", type=int, default=10)
    parser.add_argument("--rand-device", default="cpu")
    parser.add_argument("--model-seed", type=int, default=123)
    parser.add_argument("--tiled", action="store_true")
    parser.add_argument(
        "--variants",
        default="cached_context_infer_action,prompt_infer_action",
        help="Comma-separated variants: cached_context_infer_action,prompt_infer_action,cached_context_infer_joint",
    )
    parser.add_argument(
        "--rgb-cache-dir-override",
        default=None,
        help="Temporarily override data.{train,val}.rgb_cache_dir from the run config for A/B diagnostics.",
    )
    parser.add_argument(
        "--vae-latent-cache-dir-override",
        default=None,
        help=(
            "Temporarily override data.{train,val}.vae_latent_cache_dir. "
            "Use 'none' to disable stale VAE cache references."
        ),
    )
    parser.add_argument(
        "--allow-partial-cache-override",
        action="store_true",
        help="Set dataset allow_partial_cache=true for diagnostics against a small rendered cache subset.",
    )
    parser.add_argument("--output-root", default=None)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    run_dir = Path(args.run_dir).expanduser().resolve()
    checkpoint = Path(args.checkpoint).expanduser().resolve()
    cfg = OmegaConf.load(run_dir / "config.yaml")
    dataset_cfg = cfg.data.get(str(args.dataset_key))
    if dataset_cfg is None:
        raise ValueError(f"config.yaml has no data.{args.dataset_key}")
    if args.rgb_cache_dir_override:
        dataset_cfg.rgb_cache_dir = str(Path(args.rgb_cache_dir_override).expanduser().resolve())
    if args.vae_latent_cache_dir_override:
        value = str(args.vae_latent_cache_dir_override)
        dataset_cfg.vae_latent_cache_dir = None if value.lower() in {"none", "null", ""} else str(
            Path(value).expanduser().resolve()
        )
    if args.allow_partial_cache_override:
        dataset_cfg.allow_partial_cache = True
    dataset = instantiate(dataset_cfg)
    end = min(len(dataset), int(args.sample_offset) + int(args.max_samples))
    sample_indices = list(range(int(args.sample_offset), end))
    variants = [item.strip() for item in str(args.variants).split(",") if item.strip()]

    output_root = Path(
        args.output_root
        or run_dir
        / "diagnostics"
        / f"cached_action_error_{args.dataset_key}_{time.strftime('%Y%m%d_%H%M%S')}"
    ).expanduser()
    output_root.mkdir(parents=True, exist_ok=True)
    manifest = {
        "eval_type": "gembench_policy_cached_input_action_diagnostic",
        "official_full_score": False,
        "write_official_preds": False,
        "generated_at": utc_now(),
        "run_dir": str(run_dir),
        "checkpoint": str(checkpoint),
        "dataset_key": str(args.dataset_key),
        "dataset_class": type(dataset).__name__,
        "dataset_len": int(len(dataset)),
        "sample_indices": sample_indices,
        "variants": variants,
        "num_inference_steps": int(args.num_inference_steps),
        "model_seed": int(args.model_seed),
        "overrides": {
            "rgb_cache_dir": str(Path(args.rgb_cache_dir_override).expanduser().resolve())
            if args.rgb_cache_dir_override
            else None,
            "vae_latent_cache_dir": None
            if str(args.vae_latent_cache_dir_override).lower() in {"none", "null", ""}
            else (
                str(Path(args.vae_latent_cache_dir_override).expanduser().resolve())
                if args.vae_latent_cache_dir_override
                else None
            ),
            "allow_partial_cache": bool(args.allow_partial_cache_override),
        },
        "note": (
            "This diagnostic uses cached dataset RGB/proprio/context instead of RLBench live observations. "
            "It is not an official GEMBench score."
        ),
        "git": git_provenance(),
    }
    write_json(output_root / "diagnostic_manifest.json", manifest)
    if args.dry_run:
        print(json.dumps(manifest, ensure_ascii=True, indent=2), flush=True)
        return 0

    actioner = GEMBenchOfficialActioner.from_run_dir(
        run_dir=run_dir,
        checkpoint=checkpoint,
        device=str(args.device),
        mixed_precision=str(args.mixed_precision),
        num_inference_steps=int(args.num_inference_steps),
        relation_mode="none",
        rand_device=str(args.rand_device),
        model_seed=int(args.model_seed),
        tiled=bool(args.tiled),
        chunk_action_horizon=1,
        min_chunk_action_horizon=None,
    )

    rows: list[dict[str, Any]] = []
    for sample_index in sample_indices:
        sample = dataset[int(sample_index)]
        if "video" not in sample:
            raise ValueError(
                f"Sample {sample_index} has no video. Use a dataset split without VAE-only samples for cached-input diagnostics."
            )
        seed = _prediction_seed(
            actioner,
            taskvar=str(sample["taskvar"]),
            episode_key=str(sample["episode_key"]),
            step_id=int(sample["policy_key_position"]),
        )
        for variant in variants:
            start = time.time()
            if variant == "cached_context_infer_joint":
                pred_action = _sample_infer_joint(actioner=actioner, sample=sample, seed=seed)
            else:
                pred_action = _sample_infer_action(actioner=actioner, sample=sample, variant=variant, seed=seed)
            denorm = _denormalize(actioner, pred_action)
            executed = _postprocess_chunk(actioner, denorm)
            row = _metric_row(
                variant=variant,
                sample_index=int(sample_index),
                sample=sample,
                normalized_action=pred_action,
                denormalized_action=denorm,
                executed_action=executed,
                elapsed_s=time.time() - start,
            )
            rows.append(row)
            print(
                "[cached-action-diagnosis] "
                f"variant={variant} sample={sample_index} taskvar={row['taskvar']} "
                f"episode={row['episode_key']} xyz={row['xyz_error']:.6f} "
                f"quat={row['quat_angle_deg']:.3f} grip_mismatch={row['gripper_mismatch']}",
                flush=True,
            )

    _write_jsonl(output_root / "cached_action_diagnostics.jsonl", rows)
    summary = {
        **manifest,
        "output_root": str(output_root),
        "num_rows": int(len(rows)),
        "summary_by_variant": _summarize(rows),
    }
    write_json(output_root / "summary.json", summary)
    print(json.dumps(summary, ensure_ascii=True, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
