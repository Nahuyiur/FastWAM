#!/usr/bin/env python3
"""Wan latent-space inference and overfit evaluation.

This script can load either official/DiffSynth Wan weights or a Megatron DCP
checkpoint produced by `wan/pretrain.py`. It runs the official Wan FlowMatch
Euler scheduler in latent space and stores the predicted latents. Video decode
is intentionally separate because it requires the official Wan VAE assets.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import torch
import torch.distributed as dist
import torch.nn.functional as F

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from wan.model.checkpoint import load_official_wan_checkpoint
from wan.model.config import PRESETS
from wan.model.scheduler import WanFlowMatchScheduler
from wan.model.wan_dit import WanFlowTrainingModel, WanModel

try:
    from megatron.core import parallel_state
    from megatron.core.tensor_parallel.random import model_parallel_cuda_manual_seed
    from megatron.core.transformer.transformer_config import TransformerConfig
except Exception:  # pragma: no cover - CPU-only syntax checks may not have full Megatron deps.
    parallel_state = None
    model_parallel_cuda_manual_seed = None
    TransformerConfig = None


def _resolve_dcp_path(ckpt_dir):
    ckpt_dir = Path(ckpt_dir)
    tracker = ckpt_dir / "latest_checkpointed_iteration.txt"
    if tracker.is_file():
        iteration = int(tracker.read_text().strip())
        return ckpt_dir / f"iter_{iteration:07d}"
    return ckpt_dir


def _load_dcp(model, ckpt_dir, distributed=False):
    import torch.distributed.checkpoint as dcp

    ckpt_path = _resolve_dcp_path(ckpt_dir)
    if distributed:
        from megatron.core import dist_checkpointing
        from megatron.core.dist_checkpointing.strategies.torch import TorchDistLoadShardedStrategy

        sharded_state_dict = model.sharded_state_dict(
            metadata={"dp_cp_group": parallel_state.get_data_parallel_group(with_context_parallel=True)}
        )
        state_dict = dist_checkpointing.load(
            sharded_state_dict,
            str(ckpt_path),
            TorchDistLoadShardedStrategy(),
            strict="assume_ok_unexpected",
        )
    else:
        state_dict = {
            key: value for key, value in model.state_dict().items() if not key.endswith("_extra_state")
        }
        dcp.load(state_dict, checkpoint_id=str(ckpt_path), no_dist=True)
    incompatible = model.load_state_dict(state_dict, strict=False)
    if _is_rank0():
        print(f"loaded DCP {ckpt_path}")
        missing = [key for key in incompatible.missing_keys if not key.endswith("_extra_state")]
        unexpected = list(incompatible.unexpected_keys)
        print(f"DCP load_state_dict: missing={len(missing)} unexpected={len(unexpected)}")
        if missing[:8]:
            print(f"first missing keys: {missing[:8]}")
        if unexpected[:8]:
            print(f"first unexpected keys: {unexpected[:8]}")


def _is_rank0():
    return not dist.is_available() or not dist.is_initialized() or dist.get_rank() == 0


def _init_model_parallel(tp_size: int, cp_size: int, force_distributed: bool = False):
    if tp_size <= 1 and cp_size <= 1 and not force_distributed:
        return False
    if parallel_state is None or TransformerConfig is None:
        raise RuntimeError("Distributed inference requires Megatron imports")
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    torch.cuda.set_device(local_rank)
    if not dist.is_initialized():
        dist.init_process_group(backend="nccl")
    world_size = dist.get_world_size()
    expected = tp_size * cp_size
    if world_size != expected:
        raise ValueError(f"Distributed inference requires world_size == tp_size * cp_size, got {world_size=} {tp_size=} {cp_size=}")
    parallel_state.initialize_model_parallel(
        tensor_model_parallel_size=tp_size,
        pipeline_model_parallel_size=1,
        context_parallel_size=cp_size,
    )
    model_parallel_cuda_manual_seed(1234)
    if _is_rank0():
        print(f"initialized distributed inference: world_size={world_size} tensor_model_parallel_size={tp_size} context_parallel_size={cp_size}")
    return True


def _minimal_transformer_config(
    cfg,
    tp_size: int,
    cp_size: int,
    dtype: torch.dtype,
    attention_backend: str,
    local_qkv: bool,
):
    if TransformerConfig is None:
        return None
    config = TransformerConfig(
        num_layers=1,
        hidden_size=cfg.dim,
        num_attention_heads=cfg.num_heads,
        ffn_hidden_size=cfg.ffn_dim,
        tensor_model_parallel_size=tp_size,
        pipeline_model_parallel_size=1,
        context_parallel_size=cp_size,
        sequence_parallel=False,
        gradient_accumulation_fusion=False,
        params_dtype=dtype,
        bf16=dtype == torch.bfloat16,
        fp16=dtype == torch.float16,
    )
    config.wan_attention_backend = attention_backend
    config.wan_local_qkv = bool(local_qkv)
    return config


def _tp_broadcast_tensor(tensor: torch.Tensor) -> torch.Tensor:
    if parallel_state is None or not dist.is_available() or not dist.is_initialized():
        return tensor
    try:
        group = parallel_state.get_tensor_model_parallel_group()
    except Exception:
        return tensor
    if group is not None and group.size() > 1:
        src = dist.get_global_rank(group, 0)
        dist.broadcast(tensor, src=src, group=group)
    return tensor


@torch.no_grad()
def sample_latents(dit, context, shape, steps, sigma_shift, seed, device, dtype, first_frame_latents=None):
    scheduler = WanFlowMatchScheduler()
    scheduler.set_timesteps(steps, shift=sigma_shift, training=False)
    gen = torch.Generator(device=device)
    gen.manual_seed(seed)
    latents = torch.randn(shape, generator=gen, device=device, dtype=dtype)
    _tp_broadcast_tensor(latents)
    context = context.to(device=device, dtype=dtype)
    if first_frame_latents is not None:
        first_frame_latents = first_frame_latents.to(device=device, dtype=dtype)
        if first_frame_latents.ndim == 4:
            first_frame_latents = first_frame_latents.unsqueeze(0)
        expected = (shape[0], shape[1], 1, shape[3], shape[4])
        if tuple(first_frame_latents.shape) != expected:
            raise ValueError(f"first_frame_latents shape {tuple(first_frame_latents.shape)} must match {expected}")
        latents[:, :, 0:1] = first_frame_latents
    for i, timestep in enumerate(scheduler.timesteps):
        timestep = timestep.unsqueeze(0).to(device=device, dtype=dtype)
        pred = dit(
            latents,
            timestep=timestep,
            context=context,
            fuse_vae_embedding_in_latents=first_frame_latents is not None,
        )
        latents = scheduler.step(pred, timestep, latents)
        if first_frame_latents is not None:
            latents[:, :, 0:1] = first_frame_latents
        if _is_rank0() and (i + 1) % max(1, steps // 4) == 0:
            print(f"step {i + 1}/{steps}: pred_norm={pred.float().norm().item():.4f}")
    return latents


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sample", required=True, help="Pre-encoded .pt sample with context and optional GT latents")
    parser.add_argument("--output", required=True)
    parser.add_argument("--preset", default="tiny", choices=sorted(PRESETS))
    parser.add_argument("--official-ckpt", default=None)
    parser.add_argument("--dcp-ckpt", default=None)
    parser.add_argument("--steps", type=int, default=50)
    parser.add_argument("--sigma-shift", type=float, default=5.0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--dtype", default="bf16", choices=["fp32", "fp16", "bf16"])
    parser.add_argument(
        "--tensor-model-parallel-size",
        type=int,
        default=1,
        help="Run inference with Megatron TP. Launch with torchrun and set this to the DCP TP size.",
    )
    parser.add_argument(
        "--distributed-dcp-load",
        action="store_true",
        help="Use Megatron distributed checkpointing even for TP=1, enabling DCP reshard loads.",
    )
    parser.add_argument(
        "--context-parallel-size",
        type=int,
        default=1,
        help="Run inference with Megatron context parallelism. Launch with torchrun using TP*CP ranks.",
    )
    parser.add_argument(
        "--wan-attention-backend",
        default="te",
        choices=["te", "sdpa"],
        help="Wan attention implementation for distributed inference.",
    )
    parser.add_argument(
        "--wan-local-qkv",
        action="store_true",
        help="Use TP-local QKV activations with TP-aware full-hidden Q/K RMSNorm for distributed inference.",
    )
    args = parser.parse_args()
    if args.wan_attention_backend == "te":
        os.environ.setdefault("NVTE_FLASH_ATTN", "1")
        os.environ.setdefault("NVTE_FUSED_ATTN", "0")
        os.environ.setdefault("NVTE_UNFUSED_ATTN", "0")

    dtype = {"fp32": torch.float32, "fp16": torch.float16, "bf16": torch.bfloat16}[args.dtype]
    distributed = _init_model_parallel(
        args.tensor_model_parallel_size,
        args.context_parallel_size,
        force_distributed=args.distributed_dcp_load,
    )
    device = torch.device(args.device)
    sample = torch.load(args.sample, map_location="cpu", weights_only=False)
    gt = sample.get("input_latents", sample.get("latents"))
    context = sample["context"]
    first_frame_latents = sample.get("first_frame_latents")
    if gt.ndim == 4:
        shape = (1, *gt.shape)
    elif gt.ndim == 5:
        shape = tuple(gt.shape)
    else:
        raise ValueError(f"Unsupported GT latent shape: {tuple(gt.shape)}")
    if context.ndim == 2:
        context = context.unsqueeze(0)

    cfg = PRESETS[args.preset]
    megatron_config = _minimal_transformer_config(
        cfg,
        args.tensor_model_parallel_size,
        args.context_parallel_size,
        dtype,
        args.wan_attention_backend,
        args.wan_local_qkv,
    ) if distributed else None
    if args.dcp_ckpt:
        wrapper = WanFlowTrainingModel(cfg, megatron_config=megatron_config)
        wrapper.to(device=device, dtype=dtype)
        _load_dcp(wrapper, args.dcp_ckpt, distributed=distributed)
        dit = wrapper.dit
    else:
        if distributed:
            wrapper = WanFlowTrainingModel(cfg, megatron_config=megatron_config)
            dit = wrapper.dit
        else:
            dit = WanModel(cfg)
        dit.to(dtype=dtype)
        if args.official_ckpt:
            missing, unexpected = load_official_wan_checkpoint(dit, args.official_ckpt, strict=False)
            if _is_rank0():
                print(f"loaded official ckpt: missing={len(missing)} unexpected={len(unexpected)}")
    dit.eval().to(device=device, dtype=dtype)
    if _is_rank0():
        print(
            f"wan_attention_backend={args.wan_attention_backend if distributed else 'sdpa'} "
            f"wan_local_qkv={args.wan_local_qkv if distributed else False}"
        )

    pred = sample_latents(
        dit=dit,
        context=context,
        shape=shape,
        steps=args.steps,
        sigma_shift=args.sigma_shift,
        seed=args.seed,
        device=device,
        dtype=dtype,
        first_frame_latents=first_frame_latents,
    ).cpu()

    if _is_rank0():
        result = {
            "pred_latents": pred,
            "gt_latents": gt,
            "prompt": sample.get("prompt", ""),
            "video_path": sample.get("video_path", ""),
            "steps": args.steps,
            "sigma_shift": args.sigma_shift,
            "tensor_model_parallel_size": args.tensor_model_parallel_size,
        }
        if gt is not None and tuple(pred.shape[1:]) == tuple(gt.shape):
            mse = F.mse_loss(pred[0].float(), gt.float()).item()
            result["latent_mse"] = mse
            print(f"latent_mse={mse:.8f}")
            if first_frame_latents is not None and gt.shape[1] > 1:
                mse_tail = F.mse_loss(pred[0, :, 1:].float(), gt[:, 1:].float()).item()
                result["latent_mse_without_first_frame"] = mse_tail
                print(f"latent_mse_without_first_frame={mse_tail:.8f}")
        if first_frame_latents is not None:
            result["first_frame_latents"] = first_frame_latents
            result["fuse_vae_embedding_in_latents"] = True
        output = Path(args.output)
        output.parent.mkdir(parents=True, exist_ok=True)
        torch.save(result, output)
        print(f"saved {output}")

    if distributed:
        dist.barrier()
        parallel_state.destroy_model_parallel()
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
