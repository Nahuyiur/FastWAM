"""Compare Wan full-gather QKV against TP-local QKV in one TP group."""

from __future__ import annotations

import argparse
import os

import torch
import torch.distributed as dist

from megatron.core import parallel_state
from megatron.core.tensor_parallel.random import model_parallel_cuda_manual_seed
from megatron.core.transformer.transformer_config import TransformerConfig
from wan.model.config import PRESETS
from wan.model.wan_dit import WanModel


def _init(tp_size: int):
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    torch.cuda.set_device(local_rank)
    dist.init_process_group("nccl", rank=rank, world_size=world_size)
    parallel_state.initialize_model_parallel(
        tensor_model_parallel_size=tp_size,
        pipeline_model_parallel_size=1,
        context_parallel_size=1,
    )
    model_parallel_cuda_manual_seed(1234)


def _config(cfg, dtype: torch.dtype, local_qkv: bool):
    config = TransformerConfig(
        num_layers=cfg.num_layers,
        hidden_size=cfg.dim,
        num_attention_heads=cfg.num_heads,
        ffn_hidden_size=cfg.ffn_dim,
        tensor_model_parallel_size=parallel_state.get_tensor_model_parallel_world_size(),
        pipeline_model_parallel_size=1,
        context_parallel_size=1,
        sequence_parallel=False,
        gradient_accumulation_fusion=False,
        params_dtype=dtype,
        bf16=dtype == torch.bfloat16,
        fp16=dtype == torch.float16,
    )
    config.wan_attention_backend = "te"
    config.wan_local_qkv = local_qkv
    return config


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--preset", default="tiny", choices=sorted(PRESETS))
    parser.add_argument("--tp-size", type=int, default=2)
    parser.add_argument("--dtype", default="bf16", choices=["fp32", "bf16"])
    parser.add_argument("--frames", type=int, default=5)
    parser.add_argument("--height", type=int, default=16)
    parser.add_argument("--width", type=int, default=16)
    parser.add_argument("--context-len", type=int, default=8)
    args = parser.parse_args()

    _init(args.tp_size)
    rank = dist.get_rank()
    dtype = torch.bfloat16 if args.dtype == "bf16" else torch.float32
    device = torch.device("cuda", torch.cuda.current_device())
    cfg = PRESETS[args.preset]

    full = WanModel(cfg, megatron_config=_config(cfg, dtype, local_qkv=False)).to(device=device, dtype=dtype).eval()
    local = WanModel(cfg, megatron_config=_config(cfg, dtype, local_qkv=True)).to(device=device, dtype=dtype).eval()
    local.load_state_dict(full.state_dict(), strict=True)

    gen = torch.Generator(device=device)
    gen.manual_seed(20260520)
    x = torch.randn(1, cfg.in_dim, args.frames, args.height, args.width, generator=gen, device=device, dtype=dtype)
    context = torch.randn(1, args.context_len, cfg.text_dim, generator=gen, device=device, dtype=dtype)
    timestep = torch.tensor([500.0], device=device, dtype=dtype)

    with torch.no_grad():
        y_full = full(x, timestep=timestep, context=context)
        y_local = local(x, timestep=timestep, context=context)
    diff = (y_full.float() - y_local.float()).abs()
    max_abs = diff.max()
    mse = diff.pow(2).mean()
    dist.all_reduce(max_abs, op=dist.ReduceOp.MAX)
    dist.all_reduce(mse, op=dist.ReduceOp.MAX)
    if rank == 0:
        print(f"local_qkv_parity preset={args.preset} dtype={args.dtype} tp={args.tp_size}")
        print(f"max_abs={max_abs.item():.8e}")
        print(f"mse={mse.item():.8e}")

    parallel_state.destroy_model_parallel()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
