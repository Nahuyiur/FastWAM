"""DiffSynth-compatible Wan DiT.

This file ports the T2V core of `diffsynth.models.wan_video_dit.WanModel`
without importing DiffSynth. Names of modules and parameters are intentionally
kept compatible with official/DiffSynth Wan checkpoints:

  patch_embedding, text_embedding, time_embedding, time_projection,
  blocks.N.self_attn.{q,k,v,o}, blocks.N.cross_attn..., blocks.N.ffn,
  blocks.N.modulation, head.*

The wrapper `WanFlowTrainingModel` adds DiffSynth's FlowMatch SFT loss for use
inside Megatron's pretrain loop.
"""

from __future__ import annotations

import copy
import math
import os
from typing import Optional, Tuple

import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F

from wan.model.config import WanConfig
from wan.model.scheduler import WanFlowMatchScheduler

os.environ.setdefault("NVTE_FLASH_ATTN", "1")
os.environ.setdefault("NVTE_FUSED_ATTN", "0")
os.environ.setdefault("NVTE_UNFUSED_ATTN", "0")

try:
    from megatron.core import parallel_state
    from megatron.core.tensor_parallel.layers import ColumnParallelLinear, RowParallelLinear
    from megatron.core.tensor_parallel.mappings import (
        gather_from_sequence_parallel_region,
        scatter_to_sequence_parallel_region,
        scatter_to_tensor_model_parallel_region,
    )
    from megatron.core.transformer.module import MegatronModule
    from megatron.core.transformer.utils import (
        ensure_metadata_has_dp_cp_group,
        make_sharded_tensors_for_checkpoint,
        sharded_state_dict_default,
    )
except Exception:  # pragma: no cover - host-only syntax checks may not have Megatron imports.
    parallel_state = None
    gather_from_sequence_parallel_region = None
    scatter_to_sequence_parallel_region = None
    scatter_to_tensor_model_parallel_region = None
    ColumnParallelLinear = None
    RowParallelLinear = None
    MegatronModule = nn.Module
    ensure_metadata_has_dp_cp_group = None
    make_sharded_tensors_for_checkpoint = None
    sharded_state_dict_default = None

try:
    from megatron.core.extensions.transformer_engine import TEDotProductAttention
    from megatron.core.transformer.enums import AttnMaskType
except Exception:  # pragma: no cover - host-only syntax checks may not have Transformer Engine.
    TEDotProductAttention = None
    AttnMaskType = None


class _MinimalMegatronConfig:
    fp8 = None
    fp4 = None
    use_kitchen = False
    use_cpu_initialization = True
    perform_initialization = True
    params_dtype = torch.float32
    sequence_parallel = False
    gradient_accumulation_fusion = False
    defer_embedding_wgrad_compute = False
    wgrad_deferral_limit = 0
    expert_model_parallel_size = 1
    _cpu_offloading_context = None


def _init_method(weight):
    nn.init.xavier_uniform_(weight)


def _tp_info():
    if parallel_state is None or not dist.is_available() or not dist.is_initialized():
        return None, 1, 0
    try:
        group = parallel_state.get_tensor_model_parallel_group()
        return group, group.size(), group.rank()
    except Exception:
        return None, 1, 0


def _pp_info():
    if parallel_state is None or not dist.is_available() or not dist.is_initialized():
        return None, 1, 0
    try:
        group = parallel_state.get_pipeline_model_parallel_group()
        return group, group.size(), group.rank()
    except Exception:
        return None, 1, 0


def _cp_info():
    if parallel_state is None or not dist.is_available() or not dist.is_initialized():
        return None, 1, 0
    try:
        group = parallel_state.get_context_parallel_group()
        return group, group.size(), group.rank()
    except Exception:
        return None, 1, 0


def _tp_broadcast(tensor: torch.Tensor) -> torch.Tensor:
    group, world_size, _ = _tp_info()
    if group is not None and world_size > 1:
        src = dist.get_global_rank(group, 0)
        dist.broadcast(tensor, src=src, group=group)
    return tensor


def _cp_broadcast(tensor: torch.Tensor) -> torch.Tensor:
    group, world_size, _ = _cp_info()
    if group is not None and world_size > 1:
        src = dist.get_global_rank(group, 0)
        dist.broadcast(tensor, src=src, group=group)
    return tensor


def _model_parallel_broadcast(tensor: torch.Tensor) -> torch.Tensor:
    _tp_broadcast(tensor)
    _cp_broadcast(tensor)
    return tensor


def _dp_rank_without_context_parallel() -> int:
    if parallel_state is None or not dist.is_available() or not dist.is_initialized():
        return 0
    try:
        return parallel_state.get_data_parallel_rank(with_context_parallel=False)
    except Exception:
        return 0


def _all_gather_sequence(
    tensor: torch.Tensor,
    group,
    *,
    tensor_parallel_output_grad: bool = True,
) -> torch.Tensor:
    if group is None or group.size() == 1:
        return tensor
    if gather_from_sequence_parallel_region is not None:
        tensor_sbh = tensor.transpose(0, 1).contiguous()
        gathered = gather_from_sequence_parallel_region(
            tensor_sbh,
            tensor_parallel_output_grad=tensor_parallel_output_grad,
            group=group,
        )
        return gathered.transpose(0, 1).contiguous()
    chunks = [torch.empty_like(tensor) for _ in range(group.size())]
    dist.all_gather(chunks, tensor.contiguous(), group=group)
    return torch.cat(chunks, dim=1).contiguous()


def _split_context_sequence(tensor: torch.Tensor) -> torch.Tensor:
    group, world_size, rank = _cp_info()
    if group is None or world_size == 1:
        return tensor
    if tensor.shape[1] % world_size != 0:
        raise ValueError(
            f"Wan context parallelism requires sequence length {tensor.shape[1]} "
            f"to be divisible by context_parallel_size={world_size}"
        )
    return tensor.chunk(world_size, dim=1)[rank].contiguous()


def _scatter_sequence_parallel_bsh(tensor: torch.Tensor) -> torch.Tensor:
    if scatter_to_sequence_parallel_region is None:
        return tensor
    _, world_size, _ = _tp_info()
    if world_size == 1:
        return tensor
    tensor_sbh = tensor.transpose(0, 1).contiguous()
    return scatter_to_sequence_parallel_region(tensor_sbh).transpose(0, 1).contiguous()


def _gather_sequence_parallel_bsh(tensor: torch.Tensor, tensor_parallel_output_grad: bool = False) -> torch.Tensor:
    if gather_from_sequence_parallel_region is None:
        return tensor
    _, world_size, _ = _tp_info()
    if world_size == 1:
        return tensor
    tensor_sbh = tensor.transpose(0, 1).contiguous()
    gathered = gather_from_sequence_parallel_region(
        tensor_sbh,
        tensor_parallel_output_grad=tensor_parallel_output_grad,
    )
    return gathered.transpose(0, 1).contiguous()


def _sharded_state_dict_recursive(module: nn.Module, prefix="", sharded_offsets=(), metadata=None):
    if make_sharded_tensors_for_checkpoint is None or ensure_metadata_has_dp_cp_group is None:
        return module.state_dict(prefix=prefix, keep_vars=True)
    metadata = ensure_metadata_has_dp_cp_group(metadata)
    tp_group, _, _ = _tp_info()
    sharded = {}
    local_state = {}
    module._save_to_state_dict(local_state, "", keep_vars=True)
    if local_state:
        sharded.update(
            make_sharded_tensors_for_checkpoint(
                local_state,
                prefix,
                sharded_offsets=sharded_offsets,
                tp_group=tp_group,
                dp_cp_group=metadata["dp_cp_group"],
            )
        )
    for name, child in module.named_children():
        child_prefix = f"{prefix}{name}."
        if hasattr(child, "sharded_state_dict"):
            sharded.update(
                child.sharded_state_dict(
                    prefix=child_prefix,
                    sharded_offsets=sharded_offsets,
                    metadata=metadata,
                )
            )
        elif sharded_state_dict_default is not None:
            sharded.update(
                _sharded_state_dict_recursive(
                    child,
                    prefix=child_prefix,
                    sharded_offsets=sharded_offsets,
                    metadata=metadata,
                )
            )
        else:
            sharded.update(child.state_dict(prefix=child_prefix, keep_vars=True))
    return sharded


class _Linear(nn.Module):
    """Linear wrapper that uses Megatron TP layers when a config is provided."""

    def __init__(
        self,
        in_features: int,
        out_features: int,
        *,
        megatron_config=None,
        bias: bool = True,
        parallel: str = "replicated",
        gather_output: bool = True,
        input_is_parallel: bool = False,
        sequence_parallel: Optional[bool] = None,
        skip_bias_add: bool = False,
    ):
        super().__init__()
        self.parallel = parallel
        self.skip_bias_add = bool(skip_bias_add)
        self.sequence_parallel = bool(
            megatron_config is not None
            and getattr(megatron_config, "sequence_parallel", False)
            and (sequence_parallel is not False)
        )
        if megatron_config is not None and sequence_parallel is not None:
            megatron_config = copy.copy(megatron_config)
            megatron_config.sequence_parallel = bool(sequence_parallel)
            self.sequence_parallel = bool(sequence_parallel)
        self.uses_megatron_tp = megatron_config is not None and parallel != "replicated"
        self.scatter_input_for_sp_row = False
        if not self.uses_megatron_tp:
            self.linear = nn.Linear(in_features, out_features, bias=bias)
            return
        if ColumnParallelLinear is None or RowParallelLinear is None:
            raise RuntimeError("Megatron tensor-parallel layers are unavailable in this Python environment")
        if parallel == "column":
            self.linear = ColumnParallelLinear(
                in_features,
                out_features,
                config=megatron_config,
                init_method=_init_method,
                bias=bias,
                gather_output=gather_output,
                skip_bias_add=self.skip_bias_add,
            )
        elif parallel == "row":
            row_input_is_parallel = input_is_parallel
            if self.sequence_parallel and not input_is_parallel:
                row_input_is_parallel = True
                self.scatter_input_for_sp_row = True
            self.linear = RowParallelLinear(
                in_features,
                out_features,
                config=megatron_config,
                init_method=_init_method,
                bias=bias,
                input_is_parallel=row_input_is_parallel,
                skip_bias_add=self.skip_bias_add,
            )
        else:
            raise ValueError(f"unknown linear parallel mode: {parallel}")

    @property
    def weight(self):
        return self.linear.weight

    @property
    def bias(self):
        return self.linear.bias

    def forward(self, x):
        transposed_for_sp = False
        if self.uses_megatron_tp and self.sequence_parallel:
            x = x.transpose(0, 1).contiguous()
            transposed_for_sp = True
            if self.scatter_input_for_sp_row:
                x = scatter_to_tensor_model_parallel_region(x)
        if not self.uses_megatron_tp and self.skip_bias_add:
            return F.linear(x, self.weight, None), self.bias
        out = self.linear(x)
        if isinstance(out, tuple):
            x, bias = out
            if self.skip_bias_add:
                if transposed_for_sp:
                    x = x.transpose(0, 1).contiguous()
                return x, bias
            if bias is not None:
                x = x + bias
            if transposed_for_sp:
                x = x.transpose(0, 1).contiguous()
            return x
        if transposed_for_sp:
            out = out.transpose(0, 1).contiguous()
        return out

    def sharded_state_dict(self, prefix="", sharded_offsets=(), metadata=None):
        if self.uses_megatron_tp:
            return self.linear.sharded_state_dict(
                prefix=f"{prefix}linear.",
                sharded_offsets=sharded_offsets,
                metadata=metadata,
            )
        if make_sharded_tensors_for_checkpoint is None or ensure_metadata_has_dp_cp_group is None:
            return self.state_dict(prefix=prefix, keep_vars=True)
        metadata = ensure_metadata_has_dp_cp_group(metadata)
        tp_group, _, _ = _tp_info()
        return make_sharded_tensors_for_checkpoint(
            self.state_dict(prefix="", keep_vars=True),
            prefix,
            sharded_offsets=sharded_offsets,
            tp_group=tp_group,
            dp_cp_group=metadata["dp_cp_group"],
        )


def flash_attention(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, num_heads: int):
    """Attention path matching DiffSynth fallback, using PyTorch SDPA."""
    bsz, seq_len, dim = q.shape
    head_dim = dim // num_heads
    q = q.view(bsz, seq_len, num_heads, head_dim).transpose(1, 2)
    k = k.view(bsz, k.shape[1], num_heads, head_dim).transpose(1, 2)
    v = v.view(bsz, v.shape[1], num_heads, head_dim).transpose(1, 2)
    x = F.scaled_dot_product_attention(q, k, v, dropout_p=0.0, is_causal=False)
    return x.transpose(1, 2).contiguous().view(bsz, seq_len, dim)


def _attention_backend(megatron_config) -> str:
    if megatron_config is None:
        return "sdpa"
    return getattr(megatron_config, "wan_attention_backend", "te")


def _local_qkv_enabled(megatron_config) -> bool:
    if megatron_config is None:
        return False
    return bool(getattr(megatron_config, "wan_local_qkv", False))


def _te_attention_config(megatron_config, hidden_size: int, num_heads: int):
    config = copy.copy(megatron_config)
    config.hidden_size = hidden_size
    config.num_attention_heads = num_heads
    config.num_query_groups = num_heads
    config.kv_channels = hidden_size // num_heads
    config.attention_dropout = 0.0
    return config


def _cp_comm_type_for_layer(megatron_config, layer_number: int):
    cp_comm_type = getattr(megatron_config, "cp_comm_type", None)
    if isinstance(cp_comm_type, list):
        if not cp_comm_type:
            return None
        return cp_comm_type[min(layer_number - 1, len(cp_comm_type) - 1)]
    return cp_comm_type


def _split_heads_for_te(
    x: torch.Tensor,
    num_heads: int,
    *,
    hidden_size: Optional[int] = None,
    already_parallel: bool = False,
) -> torch.Tensor:
    bsz, seq_len, input_hidden_size = x.shape
    full_hidden_size = input_hidden_size if hidden_size is None else hidden_size
    if full_hidden_size % num_heads:
        raise ValueError(f"Wan TE attention requires hidden_size={full_hidden_size} divisible by heads={num_heads}")
    head_dim = full_hidden_size // num_heads
    local_heads = input_hidden_size // head_dim
    if input_hidden_size % head_dim:
        raise ValueError(f"Wan TE attention cannot split hidden={input_hidden_size} into head_dim={head_dim}")
    x = x.view(bsz, seq_len, local_heads, head_dim)
    _, tp_size, tp_rank = _tp_info()
    if tp_size > 1 and not already_parallel:
        if num_heads % tp_size:
            raise ValueError(f"Wan TE attention requires num_heads={num_heads} divisible by TP={tp_size}")
        x = x.chunk(tp_size, dim=2)[tp_rank].contiguous()
    return x.permute(1, 0, 2, 3).contiguous()


def _merge_heads_from_te(x: torch.Tensor) -> torch.Tensor:
    if x.ndim == 3:
        return x.transpose(0, 1).contiguous()
    # Some TE versions return [seq, batch, local_heads, head_dim].
    return x.permute(1, 0, 2, 3).contiguous().flatten(2)


def modulate(x: torch.Tensor, shift: torch.Tensor, scale: torch.Tensor):
    return x * (1 + scale) + shift


def sinusoidal_embedding_1d(dim: int, position: torch.Tensor):
    sinusoid = torch.outer(
        position.to(torch.float64),
        torch.pow(
            10000,
            -torch.arange(dim // 2, dtype=torch.float64, device=position.device).div(dim // 2),
        ),
    )
    x = torch.cat([torch.cos(sinusoid), torch.sin(sinusoid)], dim=1)
    return x.to(position.dtype)


def precompute_freqs_cis(dim: int, end: int = 1024, theta: float = 10000.0):
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2)[: (dim // 2)].double() / dim))
    freqs = torch.outer(torch.arange(end, device=freqs.device), freqs)
    return torch.polar(torch.ones_like(freqs), freqs)


def precompute_freqs_cis_3d(dim: int, end: int = 1024, theta: float = 10000.0):
    f_freqs_cis = precompute_freqs_cis(dim - 2 * (dim // 3), end, theta)
    h_freqs_cis = precompute_freqs_cis(dim // 3, end, theta)
    w_freqs_cis = precompute_freqs_cis(dim // 3, end, theta)
    return f_freqs_cis, h_freqs_cis, w_freqs_cis


def rope_apply(x: torch.Tensor, freqs: torch.Tensor, num_heads: int, hidden_size: Optional[int] = None):
    bsz, seq_len, dim = x.shape
    full_hidden_size = dim if hidden_size is None else hidden_size
    if full_hidden_size % num_heads:
        raise ValueError(f"Wan RoPE requires hidden_size={full_hidden_size} divisible by heads={num_heads}")
    head_dim = full_hidden_size // num_heads
    if dim % head_dim:
        raise ValueError(f"Wan RoPE cannot split hidden={dim} into head_dim={head_dim}")
    local_heads = dim // head_dim
    x = x.view(bsz, seq_len, local_heads, head_dim)
    x_out = torch.view_as_complex(x.to(torch.float64).reshape(bsz, seq_len, local_heads, -1, 2))
    freqs = freqs.to(device=x.device)
    x_out = torch.view_as_real(x_out * freqs).flatten(2)
    return x_out.to(x.dtype)


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-5):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))
        self.normalized_shape = (dim,)

    def forward(self, x):
        return F.rms_norm(x.float(), self.normalized_shape, self.weight.float(), self.eps).to(x.dtype)


class _AllReduceWithGrad(torch.autograd.Function):
    @staticmethod
    def forward(ctx, tensor: torch.Tensor, group):
        ctx.group = group
        output = tensor.contiguous().clone()
        dist.all_reduce(output, op=dist.ReduceOp.SUM, group=group)
        return output

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        grad = grad_output.contiguous().clone()
        dist.all_reduce(grad, op=dist.ReduceOp.SUM, group=ctx.group)
        return grad, None


def _all_reduce_with_grad(tensor: torch.Tensor, group) -> torch.Tensor:
    return _AllReduceWithGrad.apply(tensor, group)


class TensorParallelRMSNorm(nn.Module):
    """RMSNorm over the full hidden dimension while keeping TP-local activations.

    Wan's official Q/K norm is defined over the full hidden dimension. For
    TP-local Q/K activations, each rank computes its local sum of squares and
    the TP group all-reduces that scalar per token. The replicated full weight
    is sliced locally, so checkpoint keys stay compatible with official Wan.
    """

    def __init__(self, dim: int, eps: float = 1e-5):
        super().__init__()
        self.dim = dim
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def _local_weight(self, local_dim: int) -> torch.Tensor:
        _, tp_size, tp_rank = _tp_info()
        if local_dim == self.dim:
            return self.weight
        expected = self.dim // tp_size if tp_size > 1 else self.dim
        if local_dim != expected or self.dim % max(tp_size, 1):
            raise ValueError(
                f"TP RMSNorm expected local_dim={expected} for dim={self.dim}, "
                f"tp_size={tp_size}, got {local_dim}"
            )
        start = tp_rank * local_dim
        return self.weight[start : start + local_dim]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        local_dim = x.shape[-1]
        local_sumsq = x.float().pow(2).sum(dim=-1, keepdim=True)
        group, tp_size, _ = _tp_info()
        if group is not None and tp_size > 1 and local_dim != self.dim:
            local_sumsq = _all_reduce_with_grad(local_sumsq, group)
        elif local_dim != self.dim:
            raise RuntimeError("TP-local RMSNorm requires initialized tensor-parallel process group")
        denom = torch.rsqrt(local_sumsq / self.dim + self.eps)
        weight = self._local_weight(local_dim).float().to(device=x.device)
        return (x.float() * denom * weight).to(x.dtype)


class AttentionModule(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        context_parallel: bool = False,
        megatron_config=None,
        layer_number: int = 1,
        attention_type: str = "self",
        qkv_already_parallel: bool = False,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.context_parallel = context_parallel
        self.backend = _attention_backend(megatron_config)
        self.qkv_already_parallel = qkv_already_parallel
        if self.backend == "te":
            if TEDotProductAttention is None or AttnMaskType is None:
                raise RuntimeError(
                    "WAN attention backend 'te' requires Megatron Core TransformerEngine attention"
                )
            if hidden_size % num_heads:
                raise ValueError(f"hidden_size={hidden_size} must be divisible by num_heads={num_heads}")
            self.te_attn = TEDotProductAttention(
                config=_te_attention_config(megatron_config, hidden_size, num_heads),
                layer_number=layer_number,
                attn_mask_type=AttnMaskType.no_mask,
                attention_type=attention_type,
                attention_dropout=0.0,
                cp_comm_type=_cp_comm_type_for_layer(megatron_config, layer_number),
            )
        elif self.backend != "sdpa":
            raise ValueError(f"unknown Wan attention backend: {self.backend}")

    def forward(self, q, k, v):
        if self.backend == "te":
            q = _split_heads_for_te(
                q,
                self.num_heads,
                hidden_size=self.hidden_size,
                already_parallel=self.qkv_already_parallel,
            )
            k = _split_heads_for_te(
                k,
                self.num_heads,
                hidden_size=self.hidden_size,
                already_parallel=self.qkv_already_parallel,
            )
            v = _split_heads_for_te(
                v,
                self.num_heads,
                hidden_size=self.hidden_size,
                already_parallel=self.qkv_already_parallel,
            )
            out = self.te_attn(q, k, v, attention_mask=None, attn_mask_type=AttnMaskType.no_mask)
            return _merge_heads_from_te(out)
        if self.context_parallel:
            group, world_size, _ = _cp_info()
            if world_size > 1:
                k = _all_gather_sequence(k, group)
                v = _all_gather_sequence(v, group)
        return flash_attention(q=q, k=k, v=v, num_heads=self.num_heads)


class SelfAttention(nn.Module):
    def __init__(
        self,
        dim: int,
        num_heads: int,
        eps: float = 1e-6,
        megatron_config=None,
        layer_number: int = 1,
    ):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.attention_backend = _attention_backend(megatron_config)
        self.local_qkv = _local_qkv_enabled(megatron_config)
        if self.local_qkv and self.attention_backend != "te":
            raise ValueError("Wan local QKV path requires WAN_ATTENTION_BACKEND=te")

        # Official Wan Q/K norm is over the full hidden dimension. The default
        # path gathers Q/K/V before norm for maximum conservatism; the local QKV
        # path keeps activations TP-local and uses TensorParallelRMSNorm to
        # preserve the same full-hidden RMS denominator via TP all-reduce.
        gather_qkv = not self.local_qkv
        self.q = _Linear(dim, dim, megatron_config=megatron_config, parallel="column", gather_output=gather_qkv)
        self.k = _Linear(dim, dim, megatron_config=megatron_config, parallel="column", gather_output=gather_qkv)
        self.v = _Linear(dim, dim, megatron_config=megatron_config, parallel="column", gather_output=gather_qkv)
        self.o = _Linear(
            dim,
            dim,
            megatron_config=megatron_config,
            parallel="row",
            input_is_parallel=self.attention_backend == "te",
        )
        norm_cls = TensorParallelRMSNorm if self.local_qkv else RMSNorm
        self.norm_q = norm_cls(dim, eps=eps)
        self.norm_k = norm_cls(dim, eps=eps)
        self.attn = AttentionModule(
            dim,
            num_heads,
            context_parallel=True,
            megatron_config=megatron_config,
            layer_number=layer_number,
            attention_type="self",
            qkv_already_parallel=self.local_qkv,
        )

    def forward(self, x, freqs):
        q = self.norm_q(self.q(x))
        k = self.norm_k(self.k(x))
        v = self.v(x)
        rope_hidden_size = self.dim if self.local_qkv else None
        q = rope_apply(q, freqs, self.num_heads, hidden_size=rope_hidden_size)
        k = rope_apply(k, freqs, self.num_heads, hidden_size=rope_hidden_size)
        return self.o(self.attn(q, k, v))


class CrossAttention(nn.Module):
    def __init__(
        self,
        dim: int,
        num_heads: int,
        eps: float = 1e-6,
        has_image_input: bool = False,
        megatron_config=None,
    ):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.has_image_input = has_image_input
        self.attention_backend = _attention_backend(megatron_config)
        self.local_qkv = _local_qkv_enabled(megatron_config)
        if self.local_qkv and self.attention_backend != "te":
            raise ValueError("Wan local QKV path requires WAN_ATTENTION_BACKEND=te")

        gather_qkv = not self.local_qkv
        self.q = _Linear(dim, dim, megatron_config=megatron_config, parallel="column", gather_output=gather_qkv)
        self.k = _Linear(
            dim,
            dim,
            megatron_config=megatron_config,
            parallel="column",
            gather_output=gather_qkv,
            sequence_parallel=False,
        )
        self.v = _Linear(
            dim,
            dim,
            megatron_config=megatron_config,
            parallel="column",
            gather_output=gather_qkv,
            sequence_parallel=False,
        )
        self.o = _Linear(
            dim,
            dim,
            megatron_config=megatron_config,
            parallel="row",
            input_is_parallel=self.attention_backend == "te",
        )
        norm_cls = TensorParallelRMSNorm if self.local_qkv else RMSNorm
        self.norm_q = norm_cls(dim, eps=eps)
        self.norm_k = norm_cls(dim, eps=eps)
        if has_image_input:
            self.k_img = _Linear(
                dim,
                dim,
                megatron_config=megatron_config,
                parallel="column",
                gather_output=gather_qkv,
                sequence_parallel=False,
            )
            self.v_img = _Linear(
                dim,
                dim,
                megatron_config=megatron_config,
                parallel="column",
                gather_output=gather_qkv,
                sequence_parallel=False,
            )
            self.norm_k_img = norm_cls(dim, eps=eps)
        self.attn = AttentionModule(
            dim,
            num_heads,
            context_parallel=False,
            megatron_config=megatron_config,
            layer_number=1,
            attention_type="cross",
            qkv_already_parallel=self.local_qkv,
        )
        if has_image_input:
            self.img_attn = AttentionModule(
                dim,
                num_heads,
                context_parallel=False,
                megatron_config=megatron_config,
                layer_number=1,
                attention_type="cross",
                qkv_already_parallel=self.local_qkv,
            )

    def forward(self, x: torch.Tensor, y: torch.Tensor):
        if self.has_image_input:
            img = y[:, :257]
            ctx = y[:, 257:]
        else:
            ctx = y
        q = self.norm_q(self.q(x))
        k = self.norm_k(self.k(ctx))
        v = self.v(ctx)
        x = self.attn(q, k, v)
        if self.has_image_input:
            k_img = self.norm_k_img(self.k_img(img))
            v_img = self.v_img(img)
            x = x + self.img_attn(q, k_img, v_img)
        return self.o(x)


class GateModule(nn.Module):
    def forward(self, x, gate, residual):
        return x + gate * residual


class DiTBlock(nn.Module):
    def __init__(
        self,
        has_image_input: bool,
        dim: int,
        num_heads: int,
        ffn_dim: int,
        eps: float = 1e-6,
        megatron_config=None,
        layer_number: int = 1,
    ):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.ffn_dim = ffn_dim

        self.self_attn = SelfAttention(dim, num_heads, eps, megatron_config=megatron_config, layer_number=layer_number)
        self.cross_attn = CrossAttention(
            dim, num_heads, eps, has_image_input=has_image_input, megatron_config=megatron_config
        )
        self.norm1 = nn.LayerNorm(dim, eps=eps, elementwise_affine=False)
        self.norm2 = nn.LayerNorm(dim, eps=eps, elementwise_affine=False)
        self.norm3 = nn.LayerNorm(dim, eps=eps)
        self.ffn = nn.Sequential(
            _Linear(dim, ffn_dim, megatron_config=megatron_config, parallel="column", gather_output=False),
            nn.GELU(approximate="tanh"),
            _Linear(ffn_dim, dim, megatron_config=megatron_config, parallel="row", input_is_parallel=True),
        )
        self.modulation = nn.Parameter(torch.randn(1, 6, dim) / dim**0.5)
        self.gate = GateModule()

    def forward(self, x, context, t_mod, freqs):
        modulation = self.modulation.to(dtype=t_mod.dtype, device=t_mod.device)
        if t_mod.ndim == 4:
            modulation = modulation.unsqueeze(1) + t_mod
            shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = modulation.unbind(dim=2)
        else:
            shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = (modulation + t_mod).chunk(6, dim=1)
        input_x = modulate(self.norm1(x), shift_msa, scale_msa)
        x = self.gate(x, gate_msa, self.self_attn(input_x, freqs))
        x = x + self.cross_attn(self.norm3(x), context)
        input_x = modulate(self.norm2(x), shift_mlp, scale_mlp)
        x = self.gate(x, gate_mlp, self.ffn(input_x))
        return x


class MLP(nn.Module):
    def __init__(self, in_dim: int, out_dim: int, has_pos_emb: bool = False, megatron_config=None):
        super().__init__()
        self.proj = nn.Sequential(
            nn.LayerNorm(in_dim),
            _Linear(
                in_dim,
                in_dim,
                megatron_config=megatron_config,
                parallel="column",
                gather_output=False,
                sequence_parallel=False,
            ),
            nn.GELU(),
            _Linear(
                in_dim,
                out_dim,
                megatron_config=megatron_config,
                parallel="row",
                input_is_parallel=True,
                sequence_parallel=False,
            ),
            nn.LayerNorm(out_dim),
        )
        self.has_pos_emb = has_pos_emb
        if has_pos_emb:
            self.emb_pos = nn.Parameter(torch.zeros((1, 514, 1280)))

    def forward(self, x):
        if self.has_pos_emb:
            x = x + self.emb_pos.to(dtype=x.dtype, device=x.device)
        return self.proj(x)


class Head(nn.Module):
    def __init__(self, dim: int, out_dim: int, patch_size: Tuple[int, int, int], eps: float, megatron_config=None):
        super().__init__()
        self.dim = dim
        self.patch_size = patch_size
        self.norm = nn.LayerNorm(dim, eps=eps, elementwise_affine=False)
        self.head = _Linear(
            dim,
            out_dim * math.prod(patch_size),
            megatron_config=megatron_config,
            parallel="row",
            input_is_parallel=False,
            sequence_parallel=False,
        )
        self.modulation = nn.Parameter(torch.randn(1, 2, dim) / dim**0.5)

    def forward(self, x, t_emb):
        modulation = self.modulation.to(dtype=t_emb.dtype, device=t_emb.device)
        if t_emb.ndim == 3:
            shift, scale = (modulation.unsqueeze(1) + t_emb.unsqueeze(2)).unbind(dim=2)
        else:
            shift, scale = (modulation + t_emb.unsqueeze(1)).chunk(2, dim=1)
        return self.head(self.norm(x) * (1 + scale) + shift)


class WanModel(nn.Module):
    """DiffSynth-compatible Wan DiT core.

    The default path supports T2V (`has_image_input=False`). I2V parameter
    modules are present when enabled, but data preparation/inference scripts in
    this port currently target T2V.
    """

    _repeated_blocks = ["DiTBlock"]

    def __init__(
        self,
        cfg: WanConfig,
        megatron_config=None,
        pre_process: bool = True,
        post_process: bool = True,
    ):
        super().__init__()
        _, pp_size, pp_rank = _pp_info()
        self.dim = cfg.dim
        self.in_dim = cfg.in_dim
        self.freq_dim = cfg.freq_dim
        self.has_image_input = cfg.has_image_input
        self.patch_size = cfg.patch_size
        self.require_vae_embedding = cfg.require_vae_embedding
        self.require_clip_embedding = cfg.require_clip_embedding
        self.seperated_timestep = cfg.seperated_timestep
        self.fuse_vae_embedding_in_latents = cfg.fuse_vae_embedding_in_latents
        self.pre_process = pre_process
        self.post_process = post_process
        self.sequence_parallel = megatron_config is not None and getattr(megatron_config, "sequence_parallel", False)
        self.context_parallel = megatron_config is not None and getattr(megatron_config, "context_parallel_size", 1) > 1
        self.pipeline_payload_extra_tokens = 7  # Global timestep path: six t_mod rows plus one t row.
        self.input_tensor = None
        self.layer_start = (cfg.num_layers * pp_rank) // pp_size
        self.layer_end = (cfg.num_layers * (pp_rank + 1)) // pp_size
        if self.layer_start == self.layer_end:
            raise ValueError(
                f"Wan pipeline stage {pp_rank}/{pp_size} owns no DiT blocks; "
                f"num_layers={cfg.num_layers} must be >= pipeline size"
            )

        if self.pre_process:
            self.patch_embedding = nn.Conv3d(cfg.in_dim, cfg.dim, kernel_size=cfg.patch_size, stride=cfg.patch_size)
            self.text_embedding = nn.Sequential(
                _Linear(
                    cfg.text_dim,
                    cfg.dim,
                    megatron_config=megatron_config,
                    parallel="column",
                    gather_output=False,
                    sequence_parallel=False,
                ),
                nn.GELU(approximate="tanh"),
                _Linear(
                    cfg.dim,
                    cfg.dim,
                    megatron_config=megatron_config,
                    parallel="row",
                    input_is_parallel=True,
                    sequence_parallel=False,
                ),
            )
            self.time_embedding = nn.Sequential(
                _Linear(
                    cfg.freq_dim,
                    cfg.dim,
                    megatron_config=megatron_config,
                    parallel="column",
                    gather_output=False,
                    sequence_parallel=False,
                ),
                nn.SiLU(),
                _Linear(
                    cfg.dim,
                    cfg.dim,
                    megatron_config=megatron_config,
                    parallel="row",
                    input_is_parallel=True,
                    sequence_parallel=False,
                ),
            )
            self.time_projection = nn.Sequential(
                nn.SiLU(),
                _Linear(
                    cfg.dim,
                    cfg.dim * 6,
                    megatron_config=megatron_config,
                    parallel="column",
                    gather_output=True,
                    sequence_parallel=False,
                ),
            )
        self.blocks = nn.ModuleDict(
            {
                str(layer_idx): DiTBlock(
                    cfg.has_image_input,
                    cfg.dim,
                    cfg.num_heads,
                    cfg.ffn_dim,
                    cfg.eps,
                    megatron_config=megatron_config,
                    layer_number=layer_idx + 1,
                )
                for layer_idx in range(self.layer_start, self.layer_end)
            }
        )
        if self.post_process:
            self.head = Head(cfg.dim, cfg.out_dim, cfg.patch_size, cfg.eps, megatron_config=megatron_config)
        head_dim = cfg.dim // cfg.num_heads
        self.freqs = precompute_freqs_cis_3d(head_dim)

        if cfg.has_image_input:
            if not self.pre_process:
                raise NotImplementedError("Wan I2V pipeline parallelism needs image conditioning payload support")
            self.img_emb = MLP(1280, cfg.dim, has_pos_emb=cfg.has_image_pos_emb, megatron_config=megatron_config)
        if cfg.has_ref_conv:
            self.ref_conv = nn.Conv2d(16, cfg.dim, kernel_size=(2, 2), stride=(2, 2))

        self.has_image_pos_emb = cfg.has_image_pos_emb
        self.has_ref_conv = cfg.has_ref_conv
        self.control_adapter = None

    def set_input_tensor(self, input_tensor):
        if isinstance(input_tensor, list):
            self.input_tensor = input_tensor[0]
        else:
            self.input_tensor = input_tensor

    def patchify(self, x: torch.Tensor):
        return self.patch_embedding(x)

    def sharded_state_dict(self, prefix="", sharded_offsets=(), metadata=None):
        return _sharded_state_dict_recursive(
            self,
            prefix=prefix,
            sharded_offsets=sharded_offsets,
            metadata=metadata,
        )

    def unpatchify(self, x: torch.Tensor, grid_size: Tuple[int, int, int]):
        bsz = x.shape[0]
        f, h, w = grid_size
        pt, ph, pw = self.patch_size
        channels = x.shape[-1] // (pt * ph * pw)
        x = x.view(bsz, f, h, w, pt, ph, pw, channels)
        x = x.permute(0, 7, 1, 4, 2, 5, 3, 6).contiguous()
        return x.view(bsz, channels, f * pt, h * ph, w * pw)

    def _freqs_for_grid(self, f: int, h: int, w: int, device):
        freqs = torch.cat(
            [
                self.freqs[0][:f].view(f, 1, 1, -1).expand(f, h, w, -1),
                self.freqs[1][:h].view(1, h, 1, -1).expand(f, h, w, -1),
                self.freqs[2][:w].view(1, 1, w, -1).expand(f, h, w, -1),
            ],
            dim=-1,
        )
        return freqs.reshape(f * h * w, 1, -1).to(device)

    def _latent_grid_size(self, x: torch.Tensor) -> Tuple[int, int, int]:
        pt, ph, pw = self.patch_size
        if x.shape[2] % pt or x.shape[3] % ph or x.shape[4] % pw:
            raise ValueError(f"Latent shape {tuple(x.shape)} is not divisible by patch size {self.patch_size}")
        return x.shape[2] // pt, x.shape[3] // ph, x.shape[4] // pw

    def _time_conditioning(
        self,
        timestep: torch.Tensor,
        *,
        batch_size: int,
        grid_size: Tuple[int, int, int],
        dtype: torch.dtype,
        device: torch.device,
        fuse_vae_embedding_in_latents: bool,
    ):
        if self.seperated_timestep and fuse_vae_embedding_in_latents:
            f, h, w = grid_size
            if self.patch_size[0] != 1:
                raise NotImplementedError("Wan separated timestep currently assumes temporal patch size 1")
            timestep = timestep.reshape(-1)
            if timestep.numel() == 1:
                timestep = timestep.expand(batch_size)
            if timestep.numel() != batch_size:
                raise ValueError(f"timestep has {timestep.numel()} values for batch size {batch_size}")
            tokens_per_frame = h * w
            first = torch.zeros(batch_size, tokens_per_frame, dtype=dtype, device=device)
            rest = timestep.reshape(batch_size, 1).expand(batch_size, max(0, f - 1) * tokens_per_frame)
            timestep = torch.cat([first, rest], dim=1).reshape(-1)
            emb = sinusoidal_embedding_1d(self.freq_dim, timestep).view(batch_size, f * h * w, self.freq_dim)
            t = self.time_embedding(emb.to(dtype))
            t_mod = self.time_projection(t).unflatten(-1, (6, self.dim))
            return t, t_mod

        t = self.time_embedding(sinusoidal_embedding_1d(self.freq_dim, timestep).to(dtype))
        t_mod = self.time_projection(t).unflatten(-1, (6, self.dim))
        return t, t_mod

    def _split_token_conditioning(self, t_mod: torch.Tensor, t: torch.Tensor):
        if t_mod.ndim != 4:
            return t_mod, t
        if self.context_parallel:
            t_mod = _split_context_sequence(t_mod)
            t = _split_context_sequence(t)
        if self.sequence_parallel:
            t_mod = _scatter_sequence_parallel_bsh(t_mod)
            t = _scatter_sequence_parallel_bsh(t)
        return t_mod, t

    def _local_video_token_count(self, token_count: int) -> int:
        if self.context_parallel:
            _, cp_size, _ = _cp_info()
            if token_count % cp_size:
                raise ValueError(
                    f"Wan context parallelism requires video token count {token_count} "
                    f"to be divisible by context_parallel_size={cp_size}"
                )
            token_count //= cp_size
        if self.sequence_parallel:
            _, tp_size, _ = _tp_info()
            if token_count % tp_size:
                raise ValueError(
                    f"Wan sequence parallelism requires video token count {token_count} "
                    f"to be divisible by tensor_model_parallel_size={tp_size}"
                )
            token_count //= tp_size
        return token_count

    def _pack_pipeline_payload(self, x: torch.Tensor, context: torch.Tensor, t_mod: torch.Tensor, t: torch.Tensor):
        if t_mod.ndim == 4:
            bsz, token_count, _, dim = t_mod.shape
            if t.shape[:2] != (bsz, token_count):
                raise ValueError(f"Per-token t shape {tuple(t.shape)} is incompatible with t_mod {tuple(t_mod.shape)}")
            payload = torch.cat([x, context, t_mod.reshape(bsz, token_count * 6, dim), t], dim=1)
        else:
            payload = torch.cat([x, context, t_mod, t.unsqueeze(1)], dim=1)
        return payload.transpose(0, 1).contiguous().clone()

    def _unpack_pipeline_payload(
        self,
        payload: torch.Tensor,
        grid_size: Tuple[int, int, int],
        context_len: int,
    ):
        if payload is None:
            raise RuntimeError("Pipeline stage received no input tensor from the previous stage")
        payload = payload.transpose(0, 1).contiguous()
        global_token_count = grid_size[0] * grid_size[1] * grid_size[2]
        token_count = self._local_video_token_count(global_token_count)
        expected_global = token_count + context_len + self.pipeline_payload_extra_tokens
        expected_token = token_count + context_len + token_count * 7
        if payload.shape[1] not in (expected_global, expected_token):
            raise ValueError(
                f"Wan PP payload has {payload.shape[1]} tokens, expected {expected_global} or {expected_token} "
                f"(local_video={token_count}, global_video={global_token_count}, context={context_len})"
            )
        x = payload[:, :token_count]
        context_start = token_count
        context_end = context_start + context_len
        context = payload[:, context_start:context_end]
        if payload.shape[1] == expected_token:
            t_mod_end = context_end + token_count * 6
            t_mod = payload[:, context_end:t_mod_end].reshape(payload.shape[0], token_count, 6, payload.shape[-1])
            t = payload[:, t_mod_end : t_mod_end + token_count]
        else:
            t_mod = payload[:, context_end : context_end + 6]
            t = payload[:, context_end + 6]
        return x, context, t_mod, t

    def forward(
        self,
        x: torch.Tensor,
        timestep: torch.Tensor,
        context: torch.Tensor,
        clip_feature: Optional[torch.Tensor] = None,
        y: Optional[torch.Tensor] = None,
        fuse_vae_embedding_in_latents: bool = False,
        use_gradient_checkpointing: bool = False,
        use_gradient_checkpointing_offload: bool = False,
        **kwargs,
    ):
        del use_gradient_checkpointing_offload, kwargs
        dtype = next(self.parameters()).dtype
        x = x.to(dtype)
        context = context.to(dtype)
        if timestep.ndim == 0:
            timestep = timestep.unsqueeze(0)
        timestep = timestep.to(dtype=dtype, device=x.device)
        grid_size = self._latent_grid_size(x)

        if self.pre_process:
            context = self.text_embedding(context)

            if self.has_image_input:
                if y is None or clip_feature is None:
                    raise ValueError("WanModel has image input enabled but y/clip_feature is missing")
                x = torch.cat([x, y.to(dtype)], dim=1)
                clip_embedding = self.img_emb(clip_feature.to(dtype))
                context = torch.cat([clip_embedding, context], dim=1)

            x = self.patchify(x)
            f, h, w = x.shape[2:]
            x = x.flatten(2).transpose(1, 2).contiguous()
            t, t_mod = self._time_conditioning(
                timestep,
                batch_size=x.shape[0],
                grid_size=(f, h, w),
                dtype=dtype,
                device=x.device,
                fuse_vae_embedding_in_latents=fuse_vae_embedding_in_latents,
            )
            if self.context_parallel:
                x = _split_context_sequence(x)
            if self.sequence_parallel:
                x = _scatter_sequence_parallel_bsh(x)
            t_mod, t = self._split_token_conditioning(t_mod, t)
        else:
            f, h, w = grid_size
            x, context, t_mod, t = self._unpack_pipeline_payload(self.input_tensor, grid_size, context.shape[1])
        freqs = self._freqs_for_grid(f, h, w, x.device)
        if self.context_parallel:
            freqs = _split_context_sequence(freqs.unsqueeze(0)).squeeze(0)

        for block in self.blocks.values():
            if use_gradient_checkpointing and self.training:
                x = torch.utils.checkpoint.checkpoint(block, x, context, t_mod, freqs, use_reentrant=False)
            else:
                x = block(x, context, t_mod, freqs)

        if not self.post_process:
            return self._pack_pipeline_payload(x, context, t_mod, t)

        x = self.head(x, t)
        if self.sequence_parallel:
            x = _gather_sequence_parallel_bsh(x, tensor_parallel_output_grad=False)
        if self.context_parallel:
            x = _all_gather_sequence(x, _cp_info()[0], tensor_parallel_output_grad=False)
        return self.unpatchify(x, (f, h, w))


class WanFlowTrainingModel(MegatronModule):
    """Wan DiT plus DiffSynth FlowMatch SFT loss."""

    def __init__(
        self,
        cfg: WanConfig,
        train_timesteps: int = 1000,
        sigma_shift: float = 5.0,
        noise_scale: float = 1.0,
        min_timestep_boundary: float = 0.0,
        max_timestep_boundary: float = 1.0,
        disable_timestep_weight: bool = False,
        context_drop_prob: float = 0.0,
        gradient_checkpointing: bool = False,
        megatron_config=None,
        pre_process: bool = True,
        post_process: bool = True,
    ):
        try:
            super().__init__(config=megatron_config or _MinimalMegatronConfig())
        except TypeError:
            super().__init__()
        self.wan_config = cfg
        self.pre_process = pre_process
        self.post_process = post_process
        self.share_embeddings_and_output_weights = False
        self.dit = WanModel(
            cfg,
            megatron_config=megatron_config,
            pre_process=pre_process,
            post_process=post_process,
        )
        self.scheduler = WanFlowMatchScheduler()
        self.scheduler.set_timesteps(train_timesteps, shift=sigma_shift, training=True)
        self.noise_scale = noise_scale
        self.min_timestep_boundary = min_timestep_boundary
        self.max_timestep_boundary = max_timestep_boundary
        self.disable_timestep_weight = disable_timestep_weight
        self.context_drop_prob = context_drop_prob
        self.gradient_checkpointing = gradient_checkpointing
        self._pp_forward_step = 0
        self._pp_flow_seed = 27182818

    def set_input_tensor(self, input_tensor):
        self.input_tensor = input_tensor
        self.dit.set_input_tensor(input_tensor)

    def shared_embedding_or_output_weight(self):
        return None

    def _sample_timestep(self, device, dtype):
        n = len(self.scheduler.timesteps)
        lo = int(self.min_timestep_boundary * n)
        hi = int(self.max_timestep_boundary * n)
        hi = max(lo + 1, min(hi, n))
        timestep_id = torch.randint(lo, hi, (1,), device=device)
        _model_parallel_broadcast(timestep_id)
        return self.scheduler.timesteps.to(device=device, dtype=dtype)[timestep_id]

    def _sample_timestep_and_noise(self, input_latents, dtype):
        _, pp_size, _ = _pp_info()
        if pp_size == 1:
            timestep = self._sample_timestep(input_latents.device, dtype)
            noise = torch.randn_like(input_latents) * self.noise_scale
            _model_parallel_broadcast(noise)
            return timestep, noise

        n = len(self.scheduler.timesteps)
        lo = int(self.min_timestep_boundary * n)
        hi = max(lo + 1, min(int(self.max_timestep_boundary * n), n))
        gen = torch.Generator(device=input_latents.device)
        dp_rank = _dp_rank_without_context_parallel()
        gen.manual_seed(self._pp_flow_seed + dp_rank * 1_000_003 + self._pp_forward_step)
        self._pp_forward_step += 1
        timestep_id = torch.randint(lo, hi, (1,), device=input_latents.device, generator=gen)
        timestep = self.scheduler.timesteps.to(device=input_latents.device, dtype=dtype)[timestep_id]
        noise = torch.randn(
            input_latents.shape,
            device=input_latents.device,
            dtype=input_latents.dtype,
            generator=gen,
        ) * self.noise_scale
        return timestep, noise

    def forward(
        self,
        input_latents,
        context,
        context_mask=None,
        first_frame_latents: Optional[torch.Tensor] = None,
        fuse_vae_embedding_in_latents: bool = False,
    ):
        del context_mask
        dtype = next(self.parameters()).dtype
        input_latents = input_latents.to(dtype)
        context = context.to(dtype)
        if first_frame_latents is not None:
            first_frame_latents = first_frame_latents.to(device=input_latents.device, dtype=dtype)
            if first_frame_latents.ndim == 4:
                first_frame_latents = first_frame_latents.unsqueeze(0)
            expected = (input_latents.shape[0], input_latents.shape[1], 1, input_latents.shape[3], input_latents.shape[4])
            if tuple(first_frame_latents.shape) != expected:
                raise ValueError(
                    f"first_frame_latents shape {tuple(first_frame_latents.shape)} must match {expected}"
                )
            fuse_vae_embedding_in_latents = True
        elif fuse_vae_embedding_in_latents:
            raise ValueError("fuse_vae_embedding_in_latents requires first_frame_latents")

        if self.training and self.context_drop_prob > 0:
            drop_rand = torch.rand(1, device=context.device)
            _model_parallel_broadcast(drop_rand)
            drop = drop_rand.item() < self.context_drop_prob
            if drop:
                context = torch.zeros_like(context)

        timestep, noise = self._sample_timestep_and_noise(input_latents, dtype)
        latents = self.scheduler.add_noise(input_latents, noise, timestep)
        target = self.scheduler.training_target(input_latents, noise, timestep)
        if first_frame_latents is not None:
            latents[:, :, 0:1] = first_frame_latents

        pred = self.dit(
            latents,
            timestep=timestep,
            context=context,
            fuse_vae_embedding_in_latents=fuse_vae_embedding_in_latents,
            use_gradient_checkpointing=self.gradient_checkpointing,
        )
        if not self.post_process:
            return pred
        if first_frame_latents is not None:
            pred = pred[:, :, 1:]
            target = target[:, :, 1:]
        loss = F.mse_loss(pred.float(), target.float())
        if not self.disable_timestep_weight:
            loss = loss * self.scheduler.training_weight(timestep).float()
        return loss
