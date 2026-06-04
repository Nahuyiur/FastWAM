#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import os
import random
import sys
from contextlib import contextmanager, nullcontext
from pathlib import Path
from typing import Any

import numpy as np
import torch
from hydra import compose, initialize_config_dir
from hydra.core.global_hydra import GlobalHydra
from hydra.utils import instantiate
from omegaconf import OmegaConf
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


def _autocast_context(device: torch.device, dtype: torch.dtype):
    if device.type != "cuda":
        return nullcontext()
    if dtype == torch.bfloat16:
        return torch.autocast(device_type="cuda", dtype=torch.bfloat16)
    if dtype == torch.float16:
        return torch.autocast(device_type="cuda", dtype=torch.float16)
    return nullcontext()


def _compose_cfg(task: str, overrides: list[str]):
    cfg = compose(config_name="train", overrides=[f"task={task}", *overrides])
    OmegaConf.resolve(cfg)
    return cfg


def _next_batch(dataset, batch_size: int, skip_batches: int) -> dict[str, Any]:
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=False)
    iterator = iter(loader)
    batch = None
    for _ in range(skip_batches + 1):
        batch = next(iterator)
    assert batch is not None
    return batch


def _slice_batch(batch: Any, start: int, end: int, total_batch: int) -> Any:
    if torch.is_tensor(batch):
        if batch.ndim > 0 and batch.shape[0] == total_batch:
            return batch[start:end]
        return batch
    if isinstance(batch, dict):
        return {key: _slice_batch(value, start, end, total_batch) for key, value in batch.items()}
    if isinstance(batch, tuple):
        return tuple(_slice_batch(value, start, end, total_batch) for value in batch)
    if isinstance(batch, list):
        if len(batch) == total_batch:
            return batch[start:end]
        return [_slice_batch(value, start, end, total_batch) for value in batch]
    return batch


def _loss_payload(loss: torch.Tensor, loss_dict: dict[str, float]) -> dict[str, float]:
    out = {"loss_total": float(loss.detach().float().cpu().item())}
    for key, value in loss_dict.items():
        out[str(key)] = float(value)
    return out


def _weighted_add(dst: dict[str, float], src: dict[str, float], weight: float) -> None:
    for key, value in src.items():
        dst[key] = float(dst.get(key, 0.0) + float(value) * weight)


def _assert_metric_close(name: str, a: float, b: float, *, atol: float, rtol: float) -> dict[str, float]:
    abs_diff = abs(float(a) - float(b))
    denom = max(abs(float(a)), abs(float(b)), 1.0)
    rel_diff = abs_diff / denom
    if abs_diff > atol and rel_diff > rtol:
        raise AssertionError(
            f"{name} mismatch: full={a:.8g} accum={b:.8g} abs={abs_diff:.6g} rel={rel_diff:.6g} "
            f"tolerance atol={atol} rtol={rtol}"
        )
    return {"full": float(a), "accum": float(b), "abs_diff": abs_diff, "rel_diff": rel_diff}


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


class _FixedTrainingNoise:
    def __init__(
        self,
        *,
        video_noise: torch.Tensor,
        action_noise: torch.Tensor,
        video_timestep: torch.Tensor,
        action_timestep: torch.Tensor,
    ) -> None:
        self.video_noise = video_noise
        self.action_noise = action_noise
        self.video_timestep = video_timestep
        self.action_timestep = action_timestep
        self.reset()

    def reset(self) -> None:
        self.video_noise_offset = 0
        self.action_noise_offset = 0
        self.video_timestep_offset = 0
        self.action_timestep_offset = 0

    @property
    def total_batch(self) -> int:
        return int(self.video_noise.shape[0])

    def randn_like(self, reference: torch.Tensor, *args, **kwargs) -> torch.Tensor:
        if reference.ndim == self.video_noise.ndim:
            source = self.video_noise
            offset = self.video_noise_offset
            batch = int(reference.shape[0])
            self.video_noise_offset += batch
        elif reference.ndim == self.action_noise.ndim:
            source = self.action_noise
            offset = self.action_noise_offset
            batch = int(reference.shape[0])
            self.action_noise_offset += batch
        else:
            raise AssertionError(f"unexpected randn_like reference shape: {tuple(reference.shape)}")
        out = source[offset : offset + batch]
        if out.shape[0] != batch:
            raise AssertionError("fixed noise exhausted")
        return out.to(device=reference.device, dtype=reference.dtype)

    def sample_video_timestep(self, batch_size: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        offset = self.video_timestep_offset
        self.video_timestep_offset += int(batch_size)
        out = self.video_timestep[offset : offset + int(batch_size)]
        if out.shape[0] != int(batch_size):
            raise AssertionError("fixed video timesteps exhausted")
        return out.to(device=device, dtype=dtype)

    def sample_action_timestep(self, batch_size: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        offset = self.action_timestep_offset
        self.action_timestep_offset += int(batch_size)
        out = self.action_timestep[offset : offset + int(batch_size)]
        if out.shape[0] != int(batch_size):
            raise AssertionError("fixed action timesteps exhausted")
        return out.to(device=device, dtype=dtype)

    def assert_consumed(self) -> None:
        expected = self.total_batch
        observed = {
            "video_noise": self.video_noise_offset,
            "action_noise": self.action_noise_offset,
            "video_timestep": self.video_timestep_offset,
            "action_timestep": self.action_timestep_offset,
        }
        bad = {key: value for key, value in observed.items() if value != expected}
        if bad:
            raise AssertionError(f"fixed training noise not fully consumed: expected={expected}, observed={observed}")


@contextmanager
def _patched_training_noise(model, provider: _FixedTrainingNoise):
    original_randn_like = torch.randn_like
    original_video_sample = model.train_video_scheduler.sample_training_t
    original_action_sample = model.train_action_scheduler.sample_training_t
    torch.randn_like = provider.randn_like
    model.train_video_scheduler.sample_training_t = provider.sample_video_timestep
    model.train_action_scheduler.sample_training_t = provider.sample_action_timestep
    try:
        yield
    finally:
        torch.randn_like = original_randn_like
        model.train_video_scheduler.sample_training_t = original_video_sample
        model.train_action_scheduler.sample_training_t = original_action_sample


def _freeze_non_trainable_components(model) -> None:
    if hasattr(model, "vae") and model.vae is not None:
        for param in model.vae.parameters():
            param.requires_grad_(False)
    if getattr(model, "text_encoder", None) is not None:
        for param in model.text_encoder.parameters():
            param.requires_grad_(False)


def _make_fixed_training_noise(model, batch: dict[str, Any], *, seed: int, device: torch.device, dtype: torch.dtype):
    _seed_everything(seed)
    video_shape = tuple(batch["video_latents"].shape if "video_latents" in batch else batch["video"].shape)
    action_shape = tuple(batch["action"].shape)
    video_noise = torch.randn(video_shape, device=device, dtype=dtype)
    action_noise = torch.randn(action_shape, device=device, dtype=dtype)
    video_timestep = model.train_video_scheduler.sample_training_t(video_shape[0], device=device, dtype=dtype)
    action_timestep = model.train_action_scheduler.sample_training_t(action_shape[0], device=device, dtype=dtype)
    return _FixedTrainingNoise(
        video_noise=video_noise,
        action_noise=action_noise,
        video_timestep=video_timestep,
        action_timestep=action_timestep,
    )


def _run_full(model, batch, provider, *, device: torch.device, dtype: torch.dtype):
    model.zero_grad(set_to_none=True)
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(device)
    provider.reset()
    with _patched_training_noise(model, provider):
        with _autocast_context(device, dtype):
            loss, loss_dict = model.training_loss(batch)
        loss.backward()
        provider.assert_consumed()
    if torch.cuda.is_available():
        torch.cuda.synchronize(device)
    return _loss_payload(loss, loss_dict)


def _run_accum(model, batch, provider, *, micro_batch_size: int, device: torch.device, dtype: torch.dtype):
    total_batch = int(batch["action"].shape[0])
    model.zero_grad(set_to_none=True)
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(device)
    provider.reset()
    weighted_loss: dict[str, float] = {}
    with _patched_training_noise(model, provider):
        for start in range(0, total_batch, micro_batch_size):
            end = min(total_batch, start + micro_batch_size)
            micro = _slice_batch(batch, start, end, total_batch)
            weight = float(end - start) / float(total_batch)
            with _autocast_context(device, dtype):
                loss, loss_dict = model.training_loss(micro)
            _weighted_add(weighted_loss, _loss_payload(loss, loss_dict), weight)
            (loss * weight).backward()
        provider.assert_consumed()
    if torch.cuda.is_available():
        torch.cuda.synchronize(device)
    return weighted_loss


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Check optimizer-step semantic parity for GEMBench batch regrouping.")
    parser.add_argument("--task", default="gembench_keysteps_bbox_3cam224_vaecache_b4a1_1e-4")
    parser.add_argument("--effective-batch-size", type=int, default=4)
    parser.add_argument("--micro-batch-size", type=int, default=2)
    parser.add_argument("--skip-batches", type=int, default=0)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--mixed-precision", default=None)
    parser.add_argument("--loss-atol", type=float, default=1e-3)
    parser.add_argument("--loss-rtol", type=float, default=1e-3)
    parser.add_argument("--grad-norm-rtol", type=float, default=1e-2)
    parser.add_argument("--grad-cosine-min", type=float, default=0.999)
    parser.add_argument("--grad-param-samples", type=int, default=8)
    parser.add_argument("--grad-max-elements-per-param", type=int, default=200000)
    parser.add_argument("--json-output", default="runs/gembench_verification/accumulation_parity.json")
    parser.add_argument("--markdown-output", default="runs/gembench_verification/accumulation_parity.md")
    parser.add_argument("overrides", nargs="*", help="Additional Hydra overrides.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.effective_batch_size <= 0 or args.micro_batch_size <= 0:
        raise ValueError("batch sizes must be positive")
    if args.effective_batch_size % args.micro_batch_size != 0:
        raise ValueError("effective batch size must be divisible by micro batch size for this parity check")

    os.chdir(PROJECT_ROOT)
    register_default_resolvers()
    GlobalHydra.instance().clear()
    # The parity verdict should not depend on random initialization of any
    # parameters that are not covered by the loaded checkpoints.
    _seed_everything(args.seed)

    overrides = [
        "wandb.enabled=false",
        f"batch_size={args.effective_batch_size}",
        "num_workers=0",
        *args.overrides,
    ]
    with initialize_config_dir(config_dir=str(PROJECT_ROOT / "configs"), version_base="1.3"):
        cfg = _compose_cfg(args.task, overrides)

    precision = args.mixed_precision or str(cfg.mixed_precision)
    dtype = _mixed_precision_to_model_dtype(_normalize_mixed_precision(precision))
    device = torch.device(args.device)

    dataset = instantiate(cfg.data.train)
    batch = _next_batch(dataset, args.effective_batch_size, args.skip_batches)
    if "video_latents" not in batch:
        raise AssertionError("accumulation parity should run on the VAE-cache path and requires `video_latents`")

    model = instantiate(cfg.model, model_dtype=dtype, device=str(device))
    model.train()
    _freeze_non_trainable_components(model)

    provider = _make_fixed_training_noise(model, batch, seed=args.seed, device=device, dtype=dtype)

    full_loss = _run_full(model, batch, provider, device=device, dtype=dtype)
    full_grad_norm, full_grad_samples = _collect_grad_stats(
        model,
        max_params=args.grad_param_samples,
        max_elements_per_param=args.grad_max_elements_per_param,
    )

    accum_loss = _run_accum(model, batch, provider, micro_batch_size=args.micro_batch_size, device=device, dtype=dtype)
    accum_grad_norm, accum_grad_samples = _collect_grad_stats(
        model,
        max_params=args.grad_param_samples,
        max_elements_per_param=args.grad_max_elements_per_param,
    )

    loss_comparison = {}
    for key in sorted(set(full_loss) | set(accum_loss)):
        if key not in full_loss or key not in accum_loss:
            raise AssertionError(f"loss metric {key} missing from one path")
        loss_comparison[key] = _assert_metric_close(
            key,
            full_loss[key],
            accum_loss[key],
            atol=args.loss_atol,
            rtol=args.loss_rtol,
        )

    grad_norm_cmp = _assert_metric_close(
        "grad_norm",
        full_grad_norm,
        accum_grad_norm,
        atol=0.0,
        rtol=args.grad_norm_rtol,
    )
    cosine_payload = {}
    for name, full_grad in full_grad_samples.items():
        if name not in accum_grad_samples:
            raise AssertionError(f"sampled grad {name} missing from accumulation backward")
        cosine = _grad_cosine(full_grad, accum_grad_samples[name])
        if cosine < args.grad_cosine_min:
            raise AssertionError(f"grad cosine for {name} is {cosine:.8f} < {args.grad_cosine_min}")
        cosine_payload[name] = cosine

    payload: dict[str, Any] = {
        "status": "passed",
        "task": args.task,
        "effective_batch_size": args.effective_batch_size,
        "micro_batch_size": args.micro_batch_size,
        "grad_accumulation_steps": args.effective_batch_size // args.micro_batch_size,
        "skip_batches": args.skip_batches,
        "seed": args.seed,
        "device": str(device),
        "mixed_precision": precision,
        "dataset_size": len(dataset),
        "batch_meta": {
            "taskvar": list(batch["taskvar"]),
            "episode_key": list(batch["episode_key"]),
        },
        "seed_before_model_init": True,
        "fixed_training_noise": True,
        "loss": loss_comparison,
        "grad": {
            "grad_norm": grad_norm_cmp,
            "sampled_grad_cosine": cosine_payload,
        },
    }
    _write_json(args.json_output, payload)

    lines = [
        "# GEMBench FastWAM Accumulation Parity",
        "",
        f"status: {payload['status']}",
        f"task: `{args.task}`",
        f"effective_batch_size: {args.effective_batch_size}",
        f"micro_batch_size: {args.micro_batch_size}",
        f"seed: {args.seed}",
        "",
        "## Loss",
    ]
    for key, cmp_payload in loss_comparison.items():
        lines.append(
            f"- {key}: full={cmp_payload['full']:.8g}, accum={cmp_payload['accum']:.8g}, "
            f"abs={cmp_payload['abs_diff']:.6g}, rel={cmp_payload['rel_diff']:.6g}"
        )
    lines.extend(["", "## Grad"])
    gn = payload["grad"]["grad_norm"]
    lines.append(f"- grad_norm: full={gn['full']:.8g}, accum={gn['accum']:.8g}, rel={gn['rel_diff']:.6g}")
    for name, cosine in cosine_payload.items():
        lines.append(f"- cosine `{name}`: {cosine:.8f}")
    _write_text(args.markdown_output, "\n".join(lines) + "\n")

    print("accumulation_parity_ok")
    for key, cmp_payload in loss_comparison.items():
        print(
            f"{key}: full={cmp_payload['full']:.8g} accum={cmp_payload['accum']:.8g} "
            f"abs={cmp_payload['abs_diff']:.6g} rel={cmp_payload['rel_diff']:.6g}"
        )
    print(f"grad_norm_rel={grad_norm_cmp['rel_diff']:.6g}")


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
