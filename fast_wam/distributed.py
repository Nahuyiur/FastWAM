"""Megatron TP/DP initialization for Fast-WAM inference."""

from __future__ import annotations

import os
from dataclasses import dataclass

import torch
import torch.distributed as dist

from .config import FastWAMConfig


@dataclass(frozen=True)
class ParallelInfo:
    tp_size: int
    dp_size: int
    tp_rank: int
    dp_rank: int
    global_rank: int


def initialize(tp_size: int) -> ParallelInfo:
    if not torch.cuda.is_available():
        raise RuntimeError("Megatron Fast-WAM distributed inference requires a CUDA/PPU device")
    from megatron.core import parallel_state
    from megatron.core.tensor_parallel.random import model_parallel_cuda_manual_seed

    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    torch.cuda.set_device(local_rank)
    if not dist.is_initialized():
        dist.init_process_group(backend="nccl")
    world = dist.get_world_size()
    if world % tp_size:
        raise ValueError(f"world_size={world} must be divisible by tp_size={tp_size}")
    if not parallel_state.model_parallel_is_initialized():
        parallel_state.initialize_model_parallel(
            tensor_model_parallel_size=tp_size,
            pipeline_model_parallel_size=1,
            context_parallel_size=1,
        )
    model_parallel_cuda_manual_seed(42)
    return ParallelInfo(
        tp_size=tp_size,
        dp_size=world // tp_size,
        tp_rank=parallel_state.get_tensor_model_parallel_rank(),
        dp_rank=parallel_state.get_data_parallel_rank(with_context_parallel=False),
        global_rank=dist.get_rank(),
    )


def transformer_config(cfg: FastWAMConfig, tp_size: int, dtype: torch.dtype):
    from megatron.core.transformer.transformer_config import TransformerConfig

    return TransformerConfig(
        num_layers=cfg.video.num_layers,
        hidden_size=cfg.video.hidden_dim,
        num_attention_heads=cfg.video.num_heads,
        ffn_hidden_size=cfg.video.ffn_dim,
        tensor_model_parallel_size=tp_size,
        pipeline_model_parallel_size=1,
        context_parallel_size=1,
        sequence_parallel=False,
        gradient_accumulation_fusion=False,
        params_dtype=dtype,
        bf16=dtype == torch.bfloat16,
        fp16=dtype == torch.float16,
    )
