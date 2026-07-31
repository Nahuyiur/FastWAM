#!/usr/bin/env python3
"""Benchmark DiffSynth Wan2.2-TI2V-5B train-step throughput on preencoded data.

This is intentionally an apples-to-apples baseline for the Megatron matrix:
it uses DiffSynth's original WanModel and model_fn_wan_video, but consumes the
same preencoded latent/context sample so VAE and UMT5 preprocessing do not
dominate the measured DiT optimizer step.
"""

from __future__ import annotations

import argparse
import csv
import importlib
import sys
import time
import types
from pathlib import Path

import torch
import torch.nn.functional as F
from accelerate import Accelerator
from accelerate.utils import DistributedDataParallelKwargs
from einops import rearrange

TI2V_5B_KWARGS = {
    "has_image_input": False,
    "patch_size": (1, 2, 2),
    "in_dim": 48,
    "dim": 3072,
    "ffn_dim": 14336,
    "freq_dim": 256,
    "text_dim": 4096,
    "out_dim": 48,
    "num_heads": 24,
    "num_layers": 30,
    "eps": 1e-6,
    "seperated_timestep": True,
    "require_clip_embedding": False,
    "require_vae_embedding": False,
    "fuse_vae_embedding_in_latents": True,
}


def install_diffsynth_namespace(diffsynth_root: Path):
    """Import DiffSynth model modules without executing diffsynth/__init__.py."""
    package_root = diffsynth_root / "diffsynth"
    namespaces = {
        "diffsynth": package_root,
        "diffsynth.core": package_root / "core",
        "diffsynth.models": package_root / "models",
    }
    for name, path in namespaces.items():
        module = sys.modules.get(name)
        if module is None:
            module = types.ModuleType(name)
            sys.modules[name] = module
        module.__path__ = [str(path)]


def load_diffsynth_wan_symbols(diffsynth_root: Path):
    install_diffsynth_namespace(diffsynth_root)
    wan_mod = importlib.import_module("diffsynth.models.wan_video_dit")
    grad_mod = importlib.import_module("diffsynth.core.gradient")
    return (
        wan_mod.WanModel,
        wan_mod.sinusoidal_embedding_1d,
        grad_mod.gradient_checkpoint_forward,
    )


def model_fn_wan_video_minimal(
    dit,
    latents: torch.Tensor,
    timestep: torch.Tensor,
    context: torch.Tensor,
    sinusoidal_embedding_1d,
    gradient_checkpoint_forward,
    use_gradient_checkpointing: bool,
    fuse_vae_embedding_in_latents: bool,
):
    if dit.seperated_timestep and fuse_vae_embedding_in_latents:
        timestep = torch.concat(
            [
                torch.zeros(
                    (1, latents.shape[3] * latents.shape[4] // 4),
                    dtype=latents.dtype,
                    device=latents.device,
                ),
                torch.ones(
                    (latents.shape[2] - 1, latents.shape[3] * latents.shape[4] // 4),
                    dtype=latents.dtype,
                    device=latents.device,
                )
                * timestep,
            ]
        ).flatten()
        t = dit.time_embedding(sinusoidal_embedding_1d(dit.freq_dim, timestep).unsqueeze(0))
        t_mod = dit.time_projection(t).unflatten(2, (6, dit.dim))
    else:
        t = dit.time_embedding(sinusoidal_embedding_1d(dit.freq_dim, timestep))
        t_mod = dit.time_projection(t).unflatten(1, (6, dit.dim))

    context = dit.text_embedding(context)
    x = dit.patchify(latents)
    f, h, w = x.shape[2:]
    x = rearrange(x, "b c f h w -> b (f h w) c").contiguous()

    freqs = torch.cat(
        [
            dit.freqs[0][:f].view(f, 1, 1, -1).expand(f, h, w, -1),
            dit.freqs[1][:h].view(1, h, 1, -1).expand(f, h, w, -1),
            dit.freqs[2][:w].view(1, 1, w, -1).expand(f, h, w, -1),
        ],
        dim=-1,
    ).reshape(f * h * w, 1, -1).to(x.device)

    for block in dit.blocks:
        x = gradient_checkpoint_forward(
            block,
            use_gradient_checkpointing,
            False,
            x,
            context,
            t_mod,
            freqs,
        )

    x = dit.head(x, t)
    return dit.unpatchify(x, (f, h, w))


class DiffSynthWanTrainStep(torch.nn.Module):
    def __init__(self, diffsynth_root: Path, use_gradient_checkpointing: bool):
        super().__init__()
        (
            WanModel,
            self.sinusoidal_embedding_1d,
            self.gradient_checkpoint_forward,
        ) = load_diffsynth_wan_symbols(diffsynth_root)
        self.dit = WanModel(**TI2V_5B_KWARGS)
        self.use_gradient_checkpointing = use_gradient_checkpointing

    def forward(self, input_latents, first_frame_latents, context, timestep, noise):
        latents = (1.0 - 0.5) * input_latents + 0.5 * noise
        latents = latents.clone()
        latents[:, :, 0:1] = first_frame_latents
        target = noise - input_latents
        noise_pred = model_fn_wan_video_minimal(
            dit=self.dit,
            latents=latents,
            timestep=timestep,
            context=context,
            sinusoidal_embedding_1d=self.sinusoidal_embedding_1d,
            gradient_checkpoint_forward=self.gradient_checkpoint_forward,
            fuse_vae_embedding_in_latents=True,
            use_gradient_checkpointing=self.use_gradient_checkpointing,
        )
        return F.mse_loss(noise_pred[:, :, 1:].float(), target[:, :, 1:].float())


def load_sample(path: Path):
    obj = torch.load(path, map_location="cpu", weights_only=False)
    input_latents = obj.get("input_latents", obj.get("latents"))
    if input_latents.ndim == 4:
        input_latents = input_latents.unsqueeze(0)
    first_frame_latents = obj["first_frame_latents"]
    if first_frame_latents.ndim == 4:
        first_frame_latents = first_frame_latents.unsqueeze(0)
    context = obj["context"]
    if context.ndim == 2:
        context = context.unsqueeze(0)
    return input_latents, first_frame_latents, context


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--diffsynth-root", required=True, type=Path)
    parser.add_argument("--sample", required=True, type=Path)
    parser.add_argument("--steps", type=int, default=8)
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--learning-rate", type=float, default=1e-5)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--no-gradient-checkpointing", action="store_true")
    args = parser.parse_args()

    accelerator = Accelerator(
        kwargs_handlers=[DistributedDataParallelKwargs(find_unused_parameters=False)],
    )
    device = accelerator.device

    old_dtype = torch.get_default_dtype()
    torch.set_default_dtype(torch.bfloat16)
    try:
        model = DiffSynthWanTrainStep(
            diffsynth_root=args.diffsynth_root,
            use_gradient_checkpointing=not args.no_gradient_checkpointing,
        )
    finally:
        torch.set_default_dtype(old_dtype)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    model, optimizer = accelerator.prepare(model, optimizer)

    input_latents, first_frame_latents, context = load_sample(args.sample)
    input_latents = input_latents.to(device=device, dtype=torch.bfloat16)
    first_frame_latents = first_frame_latents.to(device=device, dtype=torch.bfloat16)
    context = context.to(device=device, dtype=torch.bfloat16)
    timestep = torch.tensor([500.0], device=device, dtype=torch.bfloat16)

    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats(device)

    rows = []
    for step in range(1, args.steps + 1):
        accelerator.wait_for_everyone()
        if torch.cuda.is_available():
            torch.cuda.synchronize(device)
        start = time.perf_counter()

        noise = torch.randn_like(input_latents)
        loss = model(input_latents, first_frame_latents, context, timestep, noise)
        accelerator.backward(loss)
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)

        accelerator.wait_for_everyone()
        if torch.cuda.is_available():
            torch.cuda.synchronize(device)
        elapsed_ms = (time.perf_counter() - start) * 1000.0
        loss_value = float(loss.detach().float().item())
        rows.append((step, elapsed_ms, loss_value))
        if accelerator.is_main_process:
            print(
                f"diffsynth iteration {step:4d}/{args.steps:4d} | "
                f"elapsed time per iteration (ms): {elapsed_ms:.1f} | "
                f"mse loss: {loss_value:.6E}",
                flush=True,
            )

    stable = rows[args.warmup :] if len(rows) > args.warmup else rows
    avg_ms = sum(row[1] for row in stable) / len(stable)
    max_allocated_gb = 0.0
    max_reserved_gb = 0.0
    if torch.cuda.is_available():
        max_allocated_gb = torch.cuda.max_memory_allocated(device) / (1024**3)
        max_reserved_gb = torch.cuda.max_memory_reserved(device) / (1024**3)
    stats = torch.tensor(
        [accelerator.process_index, max_allocated_gb, max_reserved_gb],
        device=device,
        dtype=torch.float32,
    )
    gathered = accelerator.gather(stats).detach().cpu().view(-1, 3)

    if accelerator.is_main_process:
        output = args.output
        if output is not None:
            output.parent.mkdir(parents=True, exist_ok=True)
            with output.open("w", newline="") as handle:
                writer = csv.DictWriter(
                    handle,
                    fieldnames=["rank", "max_allocated_gb", "max_reserved_gb"],
                    delimiter="\t",
                )
                writer.writeheader()
                for rank, allocated, reserved in gathered.tolist():
                    writer.writerow(
                        {
                            "rank": int(rank),
                            "max_allocated_gb": allocated,
                            "max_reserved_gb": reserved,
                        }
                    )
        print(f"diffsynth avg_ms_excl_warmup={avg_ms:.3f}")
        print(f"diffsynth max_allocated_gb={gathered[:, 1].max().item():.3f}")
        print(f"diffsynth max_reserved_gb={gathered[:, 2].max().item():.3f}")


if __name__ == "__main__":
    main()
