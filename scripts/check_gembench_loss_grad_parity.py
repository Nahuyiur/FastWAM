#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import os
import random
import sys
from contextlib import nullcontext
from pathlib import Path
from typing import Any

import numpy as np
import torch
from hydra import compose, initialize_config_dir
from hydra.core.global_hydra import GlobalHydra
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf
from torch.utils.data import DataLoader

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT / "src"))

from fastwam.runtime import _mixed_precision_to_model_dtype, _normalize_mixed_precision
from fastwam.utils.config_resolvers import register_default_resolvers


def _write_json(path: str | Path, payload: dict[str, Any]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _write_text(path: str | Path, text: str) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def _seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)



def _wrap_online_vae_rng_for_parity(model) -> None:
    original_encode = model._encode_video_latents

    def wrapped_encode(*args, **kwargs):
        cpu_state = torch.random.get_rng_state()
        cuda_states = torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
        out = original_encode(*args, **kwargs)
        torch.random.set_rng_state(cpu_state)
        if cuda_states is not None:
            torch.cuda.set_rng_state_all(cuda_states)
        return out

    model._encode_video_latents = wrapped_encode

def _autocast_context(device: torch.device, dtype: torch.dtype):
    if device.type != "cuda":
        return nullcontext()
    if dtype == torch.bfloat16:
        return torch.autocast(device_type="cuda", dtype=torch.bfloat16)
    if dtype == torch.float16:
        return torch.autocast(device_type="cuda", dtype=torch.float16)
    return nullcontext()


def _compose_cfg(task: str, overrides: list[str]) -> DictConfig:
    cfg = compose(config_name="train", overrides=[f"task={task}", *overrides])
    OmegaConf.resolve(cfg)
    return cfg


def _tensor_max_abs(a: torch.Tensor, b: torch.Tensor) -> float:
    if tuple(a.shape) != tuple(b.shape):
        raise AssertionError(f"shape mismatch: {tuple(a.shape)} vs {tuple(b.shape)}")
    if a.dtype == torch.bool or b.dtype == torch.bool or not torch.is_floating_point(a):
        if not torch.equal(a, b):
            raise AssertionError("non-floating tensor mismatch")
        return 0.0
    return float((a.float() - b.float()).abs().max().item()) if a.numel() else 0.0


def _assert_batch_contract(rgb_batch: dict[str, Any], cached_batch: dict[str, Any], atol: float) -> dict[str, float]:
    if "video" not in rgb_batch:
        raise AssertionError("RGB batch must contain `video`.")
    if "video_latents" not in cached_batch:
        raise AssertionError("Cached batch must contain `video_latents`.")
    if "video" in cached_batch:
        raise AssertionError("Cached batch must not contain RGB `video`.")
    if "video_latents" in rgb_batch:
        raise AssertionError("RGB batch must not contain `video_latents`.")

    for key in ("taskvar", "episode_key", "prompt"):
        if list(rgb_batch[key]) != list(cached_batch[key]):
            raise AssertionError(f"{key} mismatch: {rgb_batch[key]!r} vs {cached_batch[key]!r}")

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
    diffs: dict[str, float] = {}
    for key in tensor_keys:
        if key not in rgb_batch or key not in cached_batch:
            raise AssertionError(f"{key} missing from one batch")
        diff = _tensor_max_abs(rgb_batch[key], cached_batch[key])
        diffs[key] = diff
        if diff > atol:
            raise AssertionError(f"{key} max_abs_diff={diff:.6g} > {atol}")
    return diffs


def _next_batch(dataset, batch_size: int, skip_batches: int) -> dict[str, Any]:
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=False)
    iterator = iter(loader)
    batch = None
    for _ in range(skip_batches + 1):
        batch = next(iterator)
    assert batch is not None
    return batch


def _loss_payload(loss: torch.Tensor, loss_dict: dict[str, float]) -> dict[str, float]:
    out = {"loss_total": float(loss.detach().float().cpu().item())}
    for key, value in loss_dict.items():
        out[str(key)] = float(value)
    return out


def _assert_metric_close(name: str, a: float, b: float, *, atol: float, rtol: float) -> dict[str, float]:
    abs_diff = abs(float(a) - float(b))
    denom = max(abs(float(a)), abs(float(b)), 1.0)
    rel_diff = abs_diff / denom
    if abs_diff > atol and rel_diff > rtol:
        raise AssertionError(
            f"{name} mismatch: rgb={a:.8g} cached={b:.8g} abs={abs_diff:.6g} rel={rel_diff:.6g} "
            f"tolerance atol={atol} rtol={rtol}"
        )
    return {"rgb": float(a), "cached": float(b), "abs_diff": abs_diff, "rel_diff": rel_diff}


def _collect_grad_stats(model: torch.nn.Module, *, max_params: int, max_elements_per_param: int) -> tuple[float, dict[str, torch.Tensor]]:
    norm_sq = torch.zeros((), dtype=torch.float64)
    selected: dict[str, torch.Tensor] = {}
    for name, param in model.named_parameters():
        if param.grad is None:
            continue
        grad = param.grad.detach().float().cpu().flatten()
        norm_sq += grad.double().pow(2).sum()
        if len(selected) < max_params:
            if grad.numel() > max_elements_per_param:
                stride = max(1, grad.numel() // max_elements_per_param)
                grad = grad[::stride][:max_elements_per_param].contiguous()
            selected[name] = grad.clone()
    return float(norm_sq.sqrt().item()), selected


def _grad_cosine(a: torch.Tensor, b: torch.Tensor) -> float:
    if a.numel() != b.numel():
        raise AssertionError(f"grad sample shape mismatch: {a.numel()} vs {b.numel()}")
    denom = float(a.norm().item() * b.norm().item())
    if denom == 0.0:
        return 1.0 if float(a.abs().max().item()) == 0.0 and float(b.abs().max().item()) == 0.0 else 0.0
    return float(torch.dot(a, b).item() / denom)


def _run_loss(model, batch: dict[str, Any], *, seed: int, device: torch.device, dtype: torch.dtype, backward: bool):
    _seed_everything(seed)
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(device)
    model.zero_grad(set_to_none=True)
    ctx = nullcontext() if backward else torch.no_grad()
    with ctx:
        with _autocast_context(device, dtype):
            loss, loss_dict = model.training_loss(batch)
    if backward:
        loss.backward()
    if torch.cuda.is_available():
        torch.cuda.synchronize(device)
    return loss, loss_dict


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Check FastWAM GEMBench RGB path vs VAE-cache loss/grad parity.")
    parser.add_argument("--rgb-task", default="gembench_keysteps_bbox_3cam224_1e-4")
    parser.add_argument("--cached-task", default="gembench_keysteps_bbox_3cam224_vaecache_1e-4")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--skip-batches", type=int, default=0)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--mixed-precision", default=None, help="Override cfg mixed_precision for model dtype/autocast.")
    parser.add_argument("--dataset-atol", type=float, default=1e-6)
    parser.add_argument("--loss-atol", type=float, default=1e-3)
    parser.add_argument("--loss-rtol", type=float, default=1e-3)
    parser.add_argument("--grad-norm-rtol", type=float, default=1e-2)
    parser.add_argument("--grad-cosine-min", type=float, default=0.999)
    parser.add_argument("--backward", action="store_true", help="Also compare backward grad norm and sampled grad cosine.")
    parser.add_argument(
        "--no-preserve-rng-around-online-vae",
        action="store_true",
        help="Do not restore RNG state around online VAE encode during parity checks.",
    )
    parser.add_argument("--grad-param-samples", type=int, default=8)
    parser.add_argument("--grad-max-elements-per-param", type=int, default=200000)
    parser.add_argument("--json-output", default="runs/gembench_verification/loss_grad_parity.json")
    parser.add_argument("--markdown-output", default="runs/gembench_verification/loss_grad_parity.md")
    parser.add_argument("overrides", nargs="*", help="Additional Hydra overrides applied to both configs.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    os.chdir(PROJECT_ROOT)
    register_default_resolvers()
    GlobalHydra.instance().clear()

    common_overrides = [
        "wandb.enabled=false",
        f"batch_size={args.batch_size}",
        "num_workers=0",
        *args.overrides,
    ]
    with initialize_config_dir(config_dir=str(PROJECT_ROOT / "configs"), version_base="1.3"):
        rgb_cfg = _compose_cfg(args.rgb_task, common_overrides)
        cached_cfg = _compose_cfg(args.cached_task, common_overrides)

    precision = args.mixed_precision or str(rgb_cfg.mixed_precision)
    dtype = _mixed_precision_to_model_dtype(_normalize_mixed_precision(precision))
    device = torch.device(args.device)

    rgb_ds = instantiate(rgb_cfg.data.train)
    cached_ds = instantiate(cached_cfg.data.train)
    if len(rgb_ds) != len(cached_ds):
        raise AssertionError(f"dataset length mismatch: {len(rgb_ds)} vs {len(cached_ds)}")
    if list(rgb_ds.index) != list(cached_ds.index):
        raise AssertionError("dataset index mismatch between RGB and cached tasks")

    rgb_batch = _next_batch(rgb_ds, args.batch_size, args.skip_batches)
    cached_batch = _next_batch(cached_ds, args.batch_size, args.skip_batches)
    dataset_diffs = _assert_batch_contract(rgb_batch, cached_batch, atol=args.dataset_atol)

    model = instantiate(rgb_cfg.model, model_dtype=dtype, device=str(device))
    model.train()
    for param in model.vae.parameters():
        param.requires_grad_(False)
    if model.text_encoder is not None:
        for param in model.text_encoder.parameters():
            param.requires_grad_(False)
    if not args.no_preserve_rng_around_online_vae:
        _wrap_online_vae_rng_for_parity(model)

    loss_rgb, loss_dict_rgb = _run_loss(model, rgb_batch, seed=args.seed, device=device, dtype=dtype, backward=args.backward)
    rgb_loss_metrics = _loss_payload(loss_rgb, loss_dict_rgb)
    grad_rgb_norm = None
    grad_rgb_samples = None
    if args.backward:
        grad_rgb_norm, grad_rgb_samples = _collect_grad_stats(
            model,
            max_params=args.grad_param_samples,
            max_elements_per_param=args.grad_max_elements_per_param,
        )

    loss_cached, loss_dict_cached = _run_loss(model, cached_batch, seed=args.seed, device=device, dtype=dtype, backward=args.backward)
    cached_loss_metrics = _loss_payload(loss_cached, loss_dict_cached)
    grad_cached_norm = None
    grad_cached_samples = None
    if args.backward:
        grad_cached_norm, grad_cached_samples = _collect_grad_stats(
            model,
            max_params=args.grad_param_samples,
            max_elements_per_param=args.grad_max_elements_per_param,
        )

    loss_comparison = {}
    for key in sorted(set(rgb_loss_metrics) | set(cached_loss_metrics)):
        if key not in rgb_loss_metrics or key not in cached_loss_metrics:
            raise AssertionError(f"loss metric {key} missing from one path")
        loss_comparison[key] = _assert_metric_close(
            key,
            rgb_loss_metrics[key],
            cached_loss_metrics[key],
            atol=args.loss_atol,
            rtol=args.loss_rtol,
        )

    grad_payload = None
    if args.backward:
        assert grad_rgb_norm is not None and grad_cached_norm is not None
        grad_norm_cmp = _assert_metric_close(
            "grad_norm",
            grad_rgb_norm,
            grad_cached_norm,
            atol=0.0,
            rtol=args.grad_norm_rtol,
        )
        assert grad_rgb_samples is not None and grad_cached_samples is not None
        cosine_payload = {}
        for name, rgb_grad in grad_rgb_samples.items():
            if name not in grad_cached_samples:
                raise AssertionError(f"sampled grad {name} missing from cached backward")
            cosine = _grad_cosine(rgb_grad, grad_cached_samples[name])
            if cosine < args.grad_cosine_min:
                raise AssertionError(f"grad cosine for {name} is {cosine:.8f} < {args.grad_cosine_min}")
            cosine_payload[name] = cosine
        grad_payload = {
            "grad_norm": grad_norm_cmp,
            "sampled_grad_cosine": cosine_payload,
        }

    payload: dict[str, Any] = {
        "status": "passed",
        "rgb_task": args.rgb_task,
        "cached_task": args.cached_task,
        "batch_size": args.batch_size,
        "skip_batches": args.skip_batches,
        "seed": args.seed,
        "device": str(device),
        "mixed_precision": precision,
        "dataset_size": len(rgb_ds),
        "batch_meta": {
            "taskvar": list(rgb_batch["taskvar"]),
            "episode_key": list(rgb_batch["episode_key"]),
        },
        "dataset_diffs": dataset_diffs,
        "loss": loss_comparison,
        "backward_checked": bool(args.backward),
        "preserve_rng_around_online_vae": not args.no_preserve_rng_around_online_vae,
        "grad": grad_payload,
    }
    _write_json(args.json_output, payload)

    lines = [
        "# GEMBench FastWAM Loss/Grad Parity",
        "",
        f"status: {payload['status']}",
        f"rgb_task: `{args.rgb_task}`",
        f"cached_task: `{args.cached_task}`",
        f"batch_size: {args.batch_size}",
        f"seed: {args.seed}",
        "",
        "## Loss",
    ]
    for key, cmp_payload in loss_comparison.items():
        lines.append(
            f"- {key}: rgb={cmp_payload['rgb']:.8g}, cached={cmp_payload['cached']:.8g}, "
            f"abs={cmp_payload['abs_diff']:.6g}, rel={cmp_payload['rel_diff']:.6g}"
        )
    if grad_payload is not None:
        lines.extend(["", "## Grad"])
        gn = grad_payload["grad_norm"]
        lines.append(f"- grad_norm: rgb={gn['rgb']:.8g}, cached={gn['cached']:.8g}, rel={gn['rel_diff']:.6g}")
        for name, cosine in grad_payload["sampled_grad_cosine"].items():
            lines.append(f"- cosine `{name}`: {cosine:.8f}")
    _write_text(args.markdown_output, "\n".join(lines) + "\n")

    print("loss_grad_parity_ok")
    for key, cmp_payload in loss_comparison.items():
        print(
            f"{key}: rgb={cmp_payload['rgb']:.8g} cached={cmp_payload['cached']:.8g} "
            f"abs={cmp_payload['abs_diff']:.6g} rel={cmp_payload['rel_diff']:.6g}"
        )
    if grad_payload is not None:
        print(f"grad_norm_rel={grad_payload['grad_norm']['rel_diff']:.6g}")


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        payload = {"status": "failed", "error": str(exc)}
        try:
            args = parse_args()
            _write_json(args.json_output, payload)
        except Exception:
            pass
        raise
