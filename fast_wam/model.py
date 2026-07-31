"""Fast-WAM training and action inference with Megatron Core parallelism.

The registered module tree intentionally mirrors the LeRobot checkpoint below
``mot.mixtures.{video,action}``.  Megatron's parallel linear wrapper adds one
``.linear`` component to linear parameter names; :mod:`fast_wam.checkpoint`
handles that mechanical difference while streaming the checkpoint.
"""

from __future__ import annotations

import hashlib
import math
import os
from typing import Any

import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F

from .mcore import (
    MegatronModule,
    TensorParallelRMSNorm,
    _Linear,
    _MinimalMegatronConfig,
    _sharded_state_dict_recursive,
    _tp_info,
    modulate,
    precompute_freqs_cis,
    rope_apply,
    sinusoidal_embedding_1d,
)

from .config import ActionExpertConfig, FastWAMConfig, VideoExpertConfig
from .scheduler import WanFlowMatchScheduler


_FLEX_ATTENTION_COMPILED = None
_FLEX_BLOCK_MASKS: dict[tuple[str, int, int, int], Any] = {}


def _debug_tensor_digest(tensor: torch.Tensor) -> str:
    value = tensor.detach().cpu().contiguous().view(torch.uint8).numpy()
    return hashlib.sha256(value.tobytes()).hexdigest()


def _fp32_layer_norm(norm: nn.LayerNorm, x: torch.Tensor) -> torch.Tensor:
    weight = norm.weight.float() if norm.weight is not None else None
    bias = norm.bias.float() if norm.bias is not None else None
    return F.layer_norm(x.float(), norm.normalized_shape, weight, bias, norm.eps).to(x.dtype)


def _training_layer_norm(
    norm: nn.LayerNorm,
    x: torch.Tensor,
    *,
    optimized: bool,
) -> torch.Tensor:
    if not optimized:
        return _fp32_layer_norm(norm, x)
    return F.layer_norm(
        x,
        norm.normalized_shape,
        norm.weight,
        norm.bias,
        norm.eps,
    )


def _training_rms_norm(
    norm: TensorParallelRMSNorm,
    x: torch.Tensor,
    *,
    optimized: bool,
) -> torch.Tensor:
    _, tp_size, _ = _tp_info()
    if not optimized or tp_size != 1:
        return norm(x)
    return F.rms_norm(x, (norm.dim,), norm.weight, norm.eps)


def _rope_apply_complex64(
    x: torch.Tensor,
    freqs: torch.Tensor,
    num_heads: int,
    *,
    hidden_size: int,
) -> torch.Tensor:
    """Apply RoPE with FP32/complex64 intermediates on the TP1 training path."""

    batch, sequence, local_dim = x.shape
    head_dim = hidden_size // num_heads
    local_heads = local_dim // head_dim
    shaped = x.view(batch, sequence, local_heads, head_dim)
    complex_value = torch.view_as_complex(
        shaped.float().reshape(batch, sequence, local_heads, -1, 2)
    )
    rotated = complex_value * freqs.to(device=x.device, dtype=torch.complex64)
    return torch.view_as_real(rotated).flatten(2).to(dtype=x.dtype)


def _linear_dtype(module: _Linear) -> torch.dtype:
    return module.weight.dtype


def _split_modulation(block: "FastWAMBlock", t_mod: torch.Tensor) -> tuple[torch.Tensor, ...]:
    chunk_dim = 2 if t_mod.ndim == 4 else 1
    values = (block.modulation.to(dtype=t_mod.dtype, device=t_mod.device) + t_mod).chunk(
        6, dim=chunk_dim
    )
    if t_mod.ndim == 4:
        values = tuple(value.squeeze(2) for value in values)
    return values


class ParallelMLP(nn.Sequential):
    def __init__(
        self,
        hidden_dim: int,
        ffn_dim: int,
        megatron_config=None,
        *,
        optimized_kernels: bool = False,
    ):
        super().__init__(
            _Linear(
                hidden_dim,
                ffn_dim,
                megatron_config=megatron_config,
                parallel="column",
                gather_output=False,
                sequence_parallel=False,
                skip_bias_add=optimized_kernels,
            ),
            nn.GELU(approximate="tanh"),
            _Linear(
                ffn_dim,
                hidden_dim,
                megatron_config=megatron_config,
                parallel="row",
                input_is_parallel=True,
                sequence_parallel=False,
            ),
        )
        self.optimized_kernels = optimized_kernels

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        if not self.optimized_kernels:
            return super().forward(value)
        value, bias = self[0](value)
        if bias is None:
            value = F.gelu(value, approximate="tanh")
        elif value.is_cuda:
            from megatron.core.fusions.fused_bias_gelu import bias_gelu_impl

            value = bias_gelu_impl(value, bias)
        else:
            value = F.gelu(value + bias, approximate="tanh")
        return self[2](value)


class ProjectedAttention(nn.Module):
    """Wan Q/K/V/O projections with TP-local heads and full-dimension QK norm."""

    def __init__(
        self,
        hidden_dim: int,
        attention_dim: int,
        num_heads: int,
        eps: float,
        megatron_config=None,
        *,
        optimized_kernels: bool = False,
    ):
        super().__init__()
        self.optimized_kernels = optimized_kernels
        if attention_dim % num_heads:
            raise ValueError("attention_dim must be divisible by num_heads")
        _, tp_size, _ = _tp_info()
        if num_heads % tp_size:
            raise ValueError(f"num_heads={num_heads} must be divisible by TP={tp_size}")
        self.dim = hidden_dim
        self.num_heads = num_heads
        self.head_dim = attention_dim // num_heads
        self.attention_dim = attention_dim
        self.q = self._column(hidden_dim, attention_dim, megatron_config)
        self.k = self._column(hidden_dim, attention_dim, megatron_config)
        self.v = self._column(hidden_dim, attention_dim, megatron_config)
        self.o = _Linear(
            attention_dim,
            hidden_dim,
            megatron_config=megatron_config,
            parallel="row",
            input_is_parallel=True,
            sequence_parallel=False,
        )
        self.norm_q = TensorParallelRMSNorm(attention_dim, eps=eps)
        self.norm_k = TensorParallelRMSNorm(attention_dim, eps=eps)

    @staticmethod
    def _column(in_dim: int, out_dim: int, megatron_config) -> _Linear:
        return _Linear(
            in_dim,
            out_dim,
            megatron_config=megatron_config,
            parallel="column",
            gather_output=False,
            sequence_parallel=False,
        )

    @property
    def local_heads(self) -> int:
        _, tp_size, _ = _tp_info()
        return self.num_heads // tp_size


class FastWAMBlock(nn.Module):
    def __init__(
        self,
        hidden_dim: int,
        attention_dim: int,
        num_heads: int,
        ffn_dim: int,
        eps: float,
        megatron_config=None,
        *,
        optimized_kernels: bool = False,
    ):
        super().__init__()
        self.optimized_kernels = optimized_kernels
        self.norm1 = nn.LayerNorm(hidden_dim, eps=eps, elementwise_affine=False)
        self.self_attn = ProjectedAttention(
            hidden_dim, attention_dim, num_heads, eps, megatron_config
        )
        self.norm3 = nn.LayerNorm(hidden_dim, eps=eps, elementwise_affine=True)
        self.cross_attn = ProjectedAttention(
            hidden_dim, attention_dim, num_heads, eps, megatron_config
        )
        self.norm2 = nn.LayerNorm(hidden_dim, eps=eps, elementwise_affine=False)
        self.ffn = ParallelMLP(
            hidden_dim,
            ffn_dim,
            megatron_config,
            optimized_kernels=optimized_kernels,
        )
        self.modulation = nn.Parameter(torch.randn(1, 6, hidden_dim) / hidden_dim**0.5)

    def project_self_attention(
        self,
        x: torch.Tensor,
        freqs: torch.Tensor,
        *,
        optimized_kernels: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        attn = self.self_attn
        optimized_kernels = optimized_kernels and self.optimized_kernels
        q = _training_rms_norm(
            attn.norm_q,
            attn.q(x),
            optimized=optimized_kernels,
        )
        k = _training_rms_norm(
            attn.norm_k,
            attn.k(x),
            optimized=optimized_kernels,
        )
        v = attn.v(x)
        if optimized_kernels:
            q = _rope_apply_complex64(
                q,
                freqs,
                attn.num_heads,
                hidden_size=attn.attention_dim,
            )
            k = _rope_apply_complex64(
                k,
                freqs,
                attn.num_heads,
                hidden_size=attn.attention_dim,
            )
        else:
            q = rope_apply(q, freqs, attn.num_heads, hidden_size=attn.attention_dim)
            k = rope_apply(k, freqs, attn.num_heads, hidden_size=attn.attention_dim)
        return q, k, v

    def apply_cross_attention(
        self,
        x: torch.Tensor,
        context: torch.Tensor,
        context_mask: torch.Tensor | None,
        *,
        fp32_attention: bool,
        optimized_kernels: bool = False,
    ) -> torch.Tensor:
        attn = self.cross_attn
        optimized_kernels = optimized_kernels and self.optimized_kernels
        q = _training_rms_norm(
            attn.norm_q,
            attn.q(x),
            optimized=optimized_kernels,
        )
        k = _training_rms_norm(
            attn.norm_k,
            attn.k(context),
            optimized=optimized_kernels,
        )
        v = attn.v(context)
        if context_mask is not None and context_mask.ndim == 3:
            context_mask = context_mask.unsqueeze(1)
        mixed = _masked_attention(
            q, k, v, attn.local_heads, context_mask, fp32_attention=fp32_attention
        )
        return attn.o(mixed.to(dtype=_linear_dtype(attn.o)))


class VideoHead(nn.Module):
    def __init__(
        self,
        hidden_dim: int,
        out_dim: int,
        patch_size: tuple[int, int, int],
        eps: float,
        megatron_config=None,
    ):
        super().__init__()
        self.norm = nn.LayerNorm(hidden_dim, eps=eps, elementwise_affine=False)
        self.head = _Linear(
            hidden_dim,
            out_dim * math.prod(patch_size),
            megatron_config=megatron_config,
            parallel="row",
            input_is_parallel=False,
            sequence_parallel=False,
        )
        self.modulation = nn.Parameter(torch.randn(1, 2, hidden_dim) / hidden_dim**0.5)

    def forward(self, tokens: torch.Tensor, time_embedding: torch.Tensor) -> torch.Tensor:
        if time_embedding.ndim == 3:
            modulation = self.modulation.unsqueeze(0).to(
                dtype=time_embedding.dtype, device=time_embedding.device
            )
            shift, scale = (modulation + time_embedding.unsqueeze(2)).chunk(2, dim=2)
            normalized = self.norm(tokens) * (1.0 + scale.squeeze(2)) + shift.squeeze(2)
        else:
            shift, scale = (
                self.modulation.to(
                    dtype=time_embedding.dtype, device=time_embedding.device
                )
                + time_embedding
            ).chunk(2, dim=1)
            normalized = self.norm(tokens) * (1.0 + scale) + shift
        return self.head(normalized)


def _embedding_mlp(in_dim: int, hidden_dim: int, activation: nn.Module, megatron_config):
    return nn.Sequential(
        _Linear(
            in_dim,
            hidden_dim,
            megatron_config=megatron_config,
            parallel="column",
            gather_output=False,
            sequence_parallel=False,
        ),
        activation,
        _Linear(
            hidden_dim,
            hidden_dim,
            megatron_config=megatron_config,
            parallel="row",
            input_is_parallel=True,
            sequence_parallel=False,
        ),
    )


class VideoExpert(nn.Module):
    def __init__(
        self,
        cfg: VideoExpertConfig,
        megatron_config=None,
        *,
        optimized_kernels: bool = False,
    ):
        super().__init__()
        self.optimized_kernels = optimized_kernels
        self.hidden_dim = cfg.hidden_dim
        self.num_heads = cfg.num_heads
        self.attn_head_dim = cfg.attn_head_dim
        self.patch_size = cfg.patch_size
        self.patch_embedding = nn.Conv3d(
            cfg.in_dim, cfg.hidden_dim, kernel_size=cfg.patch_size, stride=cfg.patch_size
        )
        self.text_embedding = _embedding_mlp(
            cfg.text_dim, cfg.hidden_dim, nn.GELU(approximate="tanh"), megatron_config
        )
        self.time_embedding = _embedding_mlp(
            cfg.freq_dim, cfg.hidden_dim, nn.SiLU(), megatron_config
        )
        self.time_projection = nn.Sequential(
            nn.SiLU(),
            _Linear(
                cfg.hidden_dim,
                cfg.hidden_dim * 6,
                megatron_config=megatron_config,
                parallel="column",
                gather_output=True,
                sequence_parallel=False,
            ),
        )
        attention_dim = cfg.num_heads * cfg.attn_head_dim
        self.blocks = nn.ModuleList(
            [
                FastWAMBlock(
                    cfg.hidden_dim,
                    attention_dim,
                    cfg.num_heads,
                    cfg.ffn_dim,
                    cfg.eps,
                    megatron_config,
                    optimized_kernels=optimized_kernels,
                )
                for _ in range(cfg.num_layers)
            ]
        )
        self.head = VideoHead(
            cfg.hidden_dim, cfg.out_dim, cfg.patch_size, cfg.eps, megatron_config
        )
        self.freq_dim = cfg.freq_dim
        # Match Wan's asymmetric 3-D RoPE split exactly.  This is equivalent to
        # MCore's helper for the released head_dim=128 and also covers tiny tests.
        spatial_dim = 2 * (cfg.attn_head_dim // 6)
        self.freqs = (
            precompute_freqs_cis(cfg.attn_head_dim - 2 * spatial_dim),
            precompute_freqs_cis(spatial_dim),
            precompute_freqs_cis(spatial_dim),
        )

    def _freqs_for_grid(self, f: int, h: int, w: int, device: torch.device) -> torch.Tensor:
        freqs = torch.cat(
            [
                self.freqs[0][:f].view(f, 1, 1, -1).expand(f, h, w, -1),
                self.freqs[1][:h].view(1, h, 1, -1).expand(f, h, w, -1),
                self.freqs[2][:w].view(1, 1, w, -1).expand(f, h, w, -1),
            ],
            dim=-1,
        )
        return freqs.reshape(f * h * w, 1, -1).to(device)

    def pre_dit(
        self,
        latents: torch.Tensor,
        timestep: torch.Tensor,
        context: torch.Tensor,
        context_mask: torch.Tensor | None,
    ) -> dict[str, Any]:
        if latents.ndim != 5:
            raise ValueError(f"latents must be [B,C,T,H,W], got {tuple(latents.shape)}")
        dtype = self.patch_embedding.weight.dtype
        latents = latents.to(dtype=dtype)
        context = context.to(dtype=dtype)
        bsz, _, _, _, _ = latents.shape
        patched = self.patch_embedding(latents)
        _, _, f, h, w = patched.shape
        if timestep.ndim != 1 or timestep.shape[0] not in (1, bsz):
            raise ValueError(
                f"timestep must be [1] or [B], got {tuple(timestep.shape)} for B={bsz}"
            )
        if timestep.shape[0] == 1 and bsz > 1:
            if self.training:
                raise ValueError("training requires one video timestep per sample")
            timestep = timestep.expand(bsz)
        tokens_per_frame = h * w
        token_t = timestep.to(dtype=dtype).view(bsz, 1, 1).expand(
            bsz, f, tokens_per_frame
        ).clone()
        # Wan2.2 TI2V keeps the first VAE latent step clean.
        token_t[:, 0, :] = 0
        token_t = token_t.reshape(-1)
        token_t_emb = sinusoidal_embedding_1d(self.freq_dim, token_t).to(dtype=dtype)
        t = self.time_embedding(token_t_emb).reshape(bsz, f * h * w, self.hidden_dim)
        t_mod = self.time_projection(t).unflatten(2, (6, self.hidden_dim)).to(dtype=dtype)
        tokens = patched.permute(0, 2, 3, 4, 1).reshape(bsz, f * h * w, self.hidden_dim)
        context = self.text_embedding(context)
        mask = (
            None
            if context_mask is None
            else context_mask.unsqueeze(1).expand(-1, tokens.shape[1], -1)
        )
        freqs = self._freqs_for_grid(f, h, w, tokens.device)
        if self.training and self.optimized_kernels:
            freqs = freqs.to(torch.complex64)
        return {
            "tokens": tokens,
            "freqs": freqs,
            "t": t,
            "t_mod": t_mod,
            "context": context,
            "context_mask": mask,
            "meta": {
                "grid_size": (f, h, w),
                "tokens_per_frame": tokens_per_frame,
                "batch_size": bsz,
            },
        }

    def post_dit(self, tokens: torch.Tensor, state: dict[str, Any]) -> torch.Tensor:
        f, h, w = state["meta"]["grid_size"]
        patches = self.head(tokens, state["t"])
        pt, ph, pw = self.patch_size
        out_channels = patches.shape[-1] // (pt * ph * pw)
        patches = patches.view(
            patches.shape[0], f, h, w, out_channels, pt, ph, pw
        )
        return (
            patches.permute(0, 4, 1, 5, 2, 6, 3, 7)
            .reshape(patches.shape[0], out_channels, f * pt, h * ph, w * pw)
            .contiguous()
        )


class ActionExpert(nn.Module):
    def __init__(
        self,
        cfg: ActionExpertConfig,
        megatron_config=None,
        *,
        optimized_kernels: bool = False,
    ):
        super().__init__()
        self.optimized_kernels = optimized_kernels
        self.hidden_dim = cfg.hidden_dim
        self.action_dim = cfg.action_dim
        self.num_heads = cfg.num_heads
        self.attn_head_dim = cfg.attn_head_dim
        self.action_encoder = _Linear(
            cfg.action_dim,
            cfg.hidden_dim,
            megatron_config=megatron_config,
            parallel="column",
            gather_output=True,
            sequence_parallel=False,
        )
        self.text_embedding = _embedding_mlp(
            cfg.text_dim, cfg.hidden_dim, nn.GELU(approximate="tanh"), megatron_config
        )
        self.time_embedding = _embedding_mlp(
            cfg.freq_dim, cfg.hidden_dim, nn.SiLU(), megatron_config
        )
        self.time_projection = nn.Sequential(
            nn.SiLU(),
            _Linear(
                cfg.hidden_dim,
                cfg.hidden_dim * 6,
                megatron_config=megatron_config,
                parallel="column",
                gather_output=True,
                sequence_parallel=False,
            ),
        )
        attention_dim = cfg.num_heads * cfg.attn_head_dim
        self.blocks = nn.ModuleList(
            [
                FastWAMBlock(
                    cfg.hidden_dim,
                    attention_dim,
                    cfg.num_heads,
                    cfg.ffn_dim,
                    cfg.eps,
                    megatron_config,
                    optimized_kernels=optimized_kernels,
                )
                for _ in range(cfg.num_layers)
            ]
        )
        self.head = _Linear(
            cfg.hidden_dim,
            cfg.action_dim,
            megatron_config=megatron_config,
            parallel="row",
            input_is_parallel=False,
            sequence_parallel=False,
        )
        self.freq_dim = cfg.freq_dim
        self.freqs = precompute_freqs_cis(cfg.attn_head_dim, end=1024)

    def pre_dit(
        self,
        action: torch.Tensor,
        timestep: torch.Tensor,
        context: torch.Tensor,
        context_mask: torch.Tensor | None,
    ) -> dict[str, Any]:
        dtype = self.action_encoder.weight.dtype
        action = action.to(dtype=dtype)
        context = context.to(dtype=dtype)
        t_emb = sinusoidal_embedding_1d(self.freq_dim, timestep).to(dtype=dtype)
        t = self.time_embedding(t_emb)
        t_mod = self.time_projection(t).unflatten(1, (6, self.hidden_dim))
        tokens = self.action_encoder(action)
        context = self.text_embedding(context)
        mask = (
            None
            if context_mask is None
            else context_mask.unsqueeze(1).expand(-1, action.shape[1], -1)
        )
        freqs = self.freqs[: action.shape[1]].view(action.shape[1], 1, -1).to(action.device)
        if self.training and self.optimized_kernels:
            freqs = freqs.to(torch.complex64)
        return {
            "tokens": tokens,
            "freqs": freqs,
            "t_mod": t_mod,
            "context": context,
            "context_mask": mask,
        }

    def post_dit(self, tokens: torch.Tensor) -> torch.Tensor:
        return self.head(tokens)


def _masked_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    local_heads: int,
    mask: torch.Tensor | None,
    *,
    fp32_attention: bool,
    backend: str = "sdpa",
) -> torch.Tensor:
    bsz, q_len, local_dim = q.shape
    head_dim = local_dim // local_heads
    q = q.view(bsz, q_len, local_heads, head_dim).transpose(1, 2)
    k = k.view(bsz, k.shape[1], local_heads, head_dim).transpose(1, 2)
    v = v.view(bsz, v.shape[1], local_heads, head_dim).transpose(1, 2)
    output_dtype = v.dtype
    if fp32_attention:
        q, k, v = q.float(), k.float(), v.float()
    else:
        q, k = q.to(v.dtype), k.to(v.dtype)
    if backend == "sdpa":
        out = F.scaled_dot_product_attention(q, k, v, attn_mask=mask, dropout_p=0.0)
    elif backend == "flex":
        try:
            from torch.nn.attention.flex_attention import (
                create_block_mask,
                flex_attention,
            )
        except ImportError as exc:  # pragma: no cover - guarded by runtime version
            raise RuntimeError("FlexAttention is unavailable in this PyTorch build") from exc
        if mask is None:
            block_mask = None
        else:
            bool_mask = mask.to(device=q.device, dtype=torch.bool)

            def mask_mod(batch, head, query_index, key_index):
                del batch, head
                return bool_mask[query_index, key_index]

            cache_key = (
                str(q.device),
                q.shape[-2],
                k.shape[-2],
                bool_mask.data_ptr(),
            )
            block_mask = _FLEX_BLOCK_MASKS.get(cache_key)
            if block_mask is None:
                block_mask = create_block_mask(
                    mask_mod,
                    B=None,
                    H=None,
                    Q_LEN=q.shape[-2],
                    KV_LEN=k.shape[-2],
                    device=str(q.device),
                )
                _FLEX_BLOCK_MASKS[cache_key] = block_mask
        global _FLEX_ATTENTION_COMPILED
        use_compile = q.is_cuda and os.environ.get("FAST_WAM_DISABLE_COMPILED_FLEX") != "1"
        if use_compile:
            if _FLEX_ATTENTION_COMPILED is None:
                _FLEX_ATTENTION_COMPILED = torch.compile(
                    flex_attention,
                    dynamic=False,
                )
            attention_op = _FLEX_ATTENTION_COMPILED
        else:
            attention_op = flex_attention
        out = attention_op(q, k, v, block_mask=block_mask)
    else:
        raise ValueError(f"Unsupported attention backend: {backend}")
    return out.transpose(1, 2).contiguous().reshape(bsz, q_len, local_dim).to(output_dtype)


class MoT(nn.Module):
    def __init__(
        self,
        video: VideoExpert,
        action: ActionExpert,
        *,
        fp32_attention: bool,
        training_attention_backend: str,
        optimized_kernels: bool,
    ):
        super().__init__()
        self.mixtures = nn.ModuleDict({"video": video, "action": action})
        self.fp32_attention = fp32_attention
        self.training_attention_backend = training_attention_backend
        self.optimized_kernels = optimized_kernels

    def _prefill_layer(
        self,
        block: FastWAMBlock,
        x: torch.Tensor,
        freqs: torch.Tensor,
        t_mod: torch.Tensor,
        context: torch.Tensor,
        context_mask: torch.Tensor,
        self_attention_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = _split_modulation(
            block, t_mod
        )
        q, k, v = block.project_self_attention(
            modulate(_fp32_layer_norm(block.norm1, x), shift_msa, scale_msa), freqs
        )
        attn = block.self_attn
        mixed = _masked_attention(
            q,
            k,
            v,
            attn.local_heads,
            self_attention_mask,
            fp32_attention=self.fp32_attention,
        )
        x = x + gate_msa * attn.o(mixed.to(dtype=_linear_dtype(attn.o)))
        x = x + block.apply_cross_attention(
            _fp32_layer_norm(block.norm3, x),
            context,
            context_mask,
            fp32_attention=self.fp32_attention,
        )
        mlp_input = modulate(_fp32_layer_norm(block.norm2, x), shift_mlp, scale_mlp)
        return x + gate_mlp * block.ffn(mlp_input), k, v

    def prefill_video_cache(self, state: dict[str, Any]) -> list[dict[str, torch.Tensor]]:
        x = state["tokens"]
        cache = []
        attention_mask = torch.ones(
            (x.shape[1], x.shape[1]), dtype=torch.bool, device=x.device
        )
        for block in self.mixtures["video"].blocks:
            x, k, v = self._prefill_layer(
                block,
                x,
                state["freqs"],
                state["t_mod"],
                state["context"],
                state["context_mask"],
                attention_mask,
            )
            cache.append({"k": k, "v": v})
        return cache

    def forward_action_with_video_cache(
        self, state: dict[str, Any], video_cache: list[dict[str, torch.Tensor]]
    ) -> torch.Tensor:
        x = state["tokens"]
        action_attention_mask = torch.ones(
            (x.shape[1], video_cache[0]["k"].shape[1] + x.shape[1]),
            dtype=torch.bool,
            device=x.device,
        )
        for block, cached in zip(self.mixtures["action"].blocks, video_cache, strict=True):
            shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = _split_modulation(
                block, state["t_mod"]
            )
            q, k, v = block.project_self_attention(
                modulate(_fp32_layer_norm(block.norm1, x), shift_msa, scale_msa),
                state["freqs"],
            )
            attn = block.self_attn
            mixed = _masked_attention(
                q,
                torch.cat([cached["k"], k], dim=1),
                torch.cat([cached["v"], v], dim=1),
                attn.local_heads,
                action_attention_mask,
                fp32_attention=self.fp32_attention,
            )
            x = x + gate_msa * attn.o(mixed.to(dtype=_linear_dtype(attn.o)))
            x = x + block.apply_cross_attention(
                _fp32_layer_norm(block.norm3, x),
                state["context"],
                state["context_mask"],
                fp32_attention=self.fp32_attention,
            )
            mlp_input = modulate(_fp32_layer_norm(block.norm2, x), shift_mlp, scale_mlp)
            x = x + gate_mlp * block.ffn(mlp_input)
        return x

    def forward_joint(
        self,
        video_state: dict[str, Any],
        action_state: dict[str, Any],
        attention_mask: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Run the official layer-wise mixed Video/Action attention graph."""

        video_tokens = video_state["tokens"]
        action_tokens = action_state["tokens"]
        video_length = video_tokens.shape[1]
        expected = video_length + action_tokens.shape[1]
        if self.training_attention_backend != "structured_sdpa" and (
            attention_mask is None
            or attention_mask.shape != (expected, expected)
        ):
            raise ValueError(
                f"joint attention mask must be {(expected, expected)}, "
                f"got {None if attention_mask is None else tuple(attention_mask.shape)}"
            )
        tokens_per_frame = int(video_state["meta"]["tokens_per_frame"])
        if not 0 < tokens_per_frame <= video_length:
            raise ValueError(
                f"invalid first-frame token count {tokens_per_frame} "
                f"for video length {video_length}"
            )

        for video_block, action_block in zip(
            self.mixtures["video"].blocks,
            self.mixtures["action"].blocks,
            strict=True,
        ):
            expert_payloads = []
            for block, tokens, state in (
                (video_block, video_tokens, video_state),
                (action_block, action_tokens, action_state),
            ):
                (
                    shift_msa,
                    scale_msa,
                    gate_msa,
                    shift_mlp,
                    scale_mlp,
                    gate_mlp,
                ) = _split_modulation(block, state["t_mod"])
                optimized = self.optimized_kernels and self.training
                q, k, v = block.project_self_attention(
                    modulate(
                        _training_layer_norm(
                            block.norm1,
                            tokens,
                            optimized=optimized,
                        ),
                        shift_msa,
                        scale_msa,
                    ),
                    state["freqs"],
                    optimized_kernels=optimized,
                )
                expert_payloads.append(
                    (
                        block,
                        tokens,
                        state,
                        q,
                        k,
                        v,
                        gate_msa,
                        shift_mlp,
                        scale_mlp,
                        gate_mlp,
                    )
                )

            if self.training_attention_backend == "structured_sdpa":
                video_q, video_k, video_v = expert_payloads[0][3:6]
                action_q, action_k, action_v = expert_payloads[1][3:6]
                clean_mixed = _masked_attention(
                    video_q[:, :tokens_per_frame],
                    video_k[:, :tokens_per_frame],
                    video_v[:, :tokens_per_frame],
                    video_block.self_attn.local_heads,
                    None,
                    fp32_attention=False,
                )
                future_mixed = _masked_attention(
                    video_q[:, tokens_per_frame:],
                    video_k,
                    video_v,
                    video_block.self_attn.local_heads,
                    None,
                    fp32_attention=False,
                )
                action_mixed = _masked_attention(
                    action_q,
                    torch.cat(
                        [video_k[:, :tokens_per_frame], action_k],
                        dim=1,
                    ),
                    torch.cat(
                        [video_v[:, :tokens_per_frame], action_v],
                        dim=1,
                    ),
                    action_block.self_attn.local_heads,
                    None,
                    fp32_attention=False,
                )
                mixed_by_expert = (
                    torch.cat([clean_mixed, future_mixed], dim=1),
                    action_mixed,
                )
            else:
                q = torch.cat([payload[3] for payload in expert_payloads], dim=1)
                k = torch.cat([payload[4] for payload in expert_payloads], dim=1)
                v = torch.cat([payload[5] for payload in expert_payloads], dim=1)
                mixed = _masked_attention(
                    q,
                    k,
                    v,
                    video_block.self_attn.local_heads,
                    attention_mask,
                    fp32_attention=False,
                    backend=self.training_attention_backend,
                )
                mixed_by_expert = (
                    mixed[:, :video_length],
                    mixed[:, video_length:],
                )

            updated = []
            for payload, mixed_slice in zip(
                expert_payloads,
                mixed_by_expert,
                strict=True,
            ):
                (
                    block,
                    residual,
                    state,
                    _q,
                    _k,
                    _v,
                    gate_msa,
                    shift_mlp,
                    scale_mlp,
                    gate_mlp,
                ) = payload
                x = residual + gate_msa * block.self_attn.o(
                    mixed_slice.to(dtype=_linear_dtype(block.self_attn.o))
                )
                x = x + block.apply_cross_attention(
                    _training_layer_norm(
                        block.norm3,
                        x,
                        optimized=optimized,
                    ),
                    state["context"],
                    state["context_mask"],
                    fp32_attention=False,
                    optimized_kernels=optimized,
                )
                x = x + gate_mlp * block.ffn(
                    modulate(
                        _training_layer_norm(
                            block.norm2,
                            x,
                            optimized=optimized,
                        ),
                        shift_mlp,
                        scale_mlp,
                    )
                )
                updated.append(x)
            video_tokens, action_tokens = updated
        return video_tokens, action_tokens


class FastWAMModel(MegatronModule):
    """Fast-WAM model supporting official joint training plus action inference."""

    def __init__(self, cfg: FastWAMConfig, megatron_config=None):
        try:
            super().__init__(config=megatron_config or _MinimalMegatronConfig())
        except TypeError:
            super().__init__()
        self.fast_wam_config = cfg
        optimized_kernels = cfg.training_kernel_mode == "optimized"
        video = VideoExpert(
            cfg.video,
            megatron_config,
            optimized_kernels=optimized_kernels,
        )
        action = ActionExpert(
            cfg.action,
            megatron_config,
            optimized_kernels=optimized_kernels,
        )
        self.mot = MoT(
            video,
            action,
            fp32_attention=cfg.fp32_attention,
            training_attention_backend=cfg.training_attention_backend,
            optimized_kernels=optimized_kernels,
        )
        self.proprio_encoder = _Linear(
            cfg.proprio_dim,
            cfg.video.text_dim,
            megatron_config=megatron_config,
            parallel="column",
            gather_output=True,
            sequence_parallel=False,
        )
        self.scheduler = WanFlowMatchScheduler(
            cfg.num_train_timesteps, shift=cfg.action_train_shift
        )
        self.video_train_scheduler = WanFlowMatchScheduler(
            cfg.num_train_timesteps, shift=cfg.video_train_shift
        )
        self.action_train_scheduler = WanFlowMatchScheduler(
            cfg.num_train_timesteps, shift=cfg.action_train_shift
        )
        self.share_embeddings_and_output_weights = False

    @property
    def video_expert(self) -> VideoExpert:
        return self.mot.mixtures["video"]

    @property
    def action_expert(self) -> ActionExpert:
        return self.mot.mixtures["action"]

    def shared_embedding_or_output_weight(self):
        return None

    def set_input_tensor(self, input_tensor):
        # Required by Megatron's encoder/decoder schedule even with PP=1.
        self.input_tensor = input_tensor

    def sharded_state_dict(self, prefix="", sharded_offsets=(), metadata=None):
        return _sharded_state_dict_recursive(
            self, prefix=prefix, sharded_offsets=sharded_offsets, metadata=metadata
        )

    def _append_proprio(
        self, context: torch.Tensor, context_mask: torch.Tensor, proprio: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        token = self.proprio_encoder(proprio.to(dtype=context.dtype)).unsqueeze(1)
        mask = torch.ones(
            (context_mask.shape[0], 1), dtype=torch.bool, device=context_mask.device
        )
        return torch.cat([context, token], dim=1), torch.cat([context_mask, mask], dim=1)

    @staticmethod
    def _tp_leader_value(factory, empty_factory) -> torch.Tensor:
        """Generate stochastic training inputs once per TP replica."""

        group, tp_size, tp_rank = _tp_info()
        if group is None or tp_size == 1:
            return factory()
        value = factory() if tp_rank == 0 else empty_factory()
        dist.broadcast(value, src=dist.get_global_rank(group, 0), group=group)
        return value

    def build_training_attention_mask(
        self,
        video_sequence_length: int,
        action_sequence_length: int,
        video_tokens_per_frame: int,
        *,
        device: torch.device,
    ) -> torch.Tensor:
        """Build Fast-WAM's no-future-leakage MoT training mask."""

        total = video_sequence_length + action_sequence_length
        mask = torch.zeros((total, total), dtype=torch.bool, device=device)
        # First-frame queries cannot read future video; future-video queries are
        # bidirectional over the whole video branch.
        mask[:video_tokens_per_frame, :video_tokens_per_frame] = True
        mask[video_tokens_per_frame:video_sequence_length, :video_sequence_length] = True
        # Action queries read the clean first-frame anchor and all action tokens.
        mask[video_sequence_length:, :video_tokens_per_frame] = True
        mask[video_sequence_length:, video_sequence_length:] = True
        return mask

    def _video_loss_per_sample(
        self,
        prediction: torch.Tensor,
        target: torch.Tensor,
        image_is_pad: torch.Tensor | None,
    ) -> torch.Tensor:
        token_loss = F.mse_loss(
            prediction.float(), target.float(), reduction="none"
        ).mean(dim=(1, 3, 4))
        if image_is_pad is None:
            return token_loss.mean(dim=1)
        factor = self.fast_wam_config.temporal_downsample_factor
        if (image_is_pad.shape[1] - 1) % factor:
            raise ValueError(
                "image padding length is incompatible with the VAE temporal factor"
            )
        latent_pad = image_is_pad[:, 1:].view(image_is_pad.shape[0], -1, factor).all(dim=2)
        if latent_pad.shape[1] != token_loss.shape[1]:
            raise ValueError(
                f"video padding/loss mismatch: {latent_pad.shape[1]} vs {token_loss.shape[1]}"
            )
        valid = (~latent_pad).to(dtype=token_loss.dtype, device=token_loss.device)
        return (token_loss * valid).sum(dim=1) / valid.sum(dim=1).clamp(min=1.0)

    def training_loss_encoded(
        self,
        input_latents: torch.Tensor,
        action: torch.Tensor,
        context: torch.Tensor,
        context_mask: torch.Tensor,
        proprio: torch.Tensor,
        *,
        image_is_pad: torch.Tensor | None = None,
        action_is_pad: torch.Tensor | None = None,
        noise_video: torch.Tensor | None = None,
        noise_action: torch.Tensor | None = None,
        timestep_video: torch.Tensor | None = None,
        timestep_action: torch.Tensor | None = None,
        stochastic_seed: int | None = None,
        return_debug_tensors: bool = False,
        context_is_dense: bool = False,
    ) -> tuple[torch.Tensor, dict[str, Any]]:
        """Compute the official weighted video/action FlowMatch objective."""

        if input_latents.ndim != 5:
            raise ValueError("input_latents must be [B,C,T,H,W]")
        if action.ndim != 3:
            raise ValueError("action must be [B,T,D]")
        batch_size = input_latents.shape[0]
        dtype = self.proprio_encoder.weight.dtype
        device = self.proprio_encoder.weight.device
        input_latents = input_latents.to(device=device, dtype=dtype)
        action = action.to(device=device, dtype=dtype)
        context = context.to(device=device, dtype=dtype)
        context_mask = context_mask.to(device=device, dtype=torch.bool)
        if proprio.ndim == 3:
            proprio = proprio[:, 0]
        proprio = proprio.to(device=device, dtype=dtype)
        context, context_mask = self._append_proprio(context, context_mask, proprio)
        attention_context_mask = None if context_is_dense else context_mask

        generator = None
        if stochastic_seed is not None:
            generator = torch.Generator(device=device)
            generator.manual_seed(int(stochastic_seed))

        # Preserve the upstream RNG call order exactly:
        # video noise -> video timestep -> action noise -> action timestep.
        if noise_video is None:
            noise_video = self._tp_leader_value(
                lambda: torch.randn(
                    input_latents.shape,
                    device=device,
                    dtype=dtype,
                    generator=generator,
                ),
                lambda: torch.empty_like(input_latents),
            )
        else:
            noise_video = noise_video.to(device=device, dtype=dtype)
        if timestep_video is None:
            timestep_video = self._tp_leader_value(
                lambda: self.video_train_scheduler.sample_training_t(
                    batch_size, device, dtype, generator=generator
                ),
                lambda: torch.empty(batch_size, device=device, dtype=dtype),
            )
        else:
            timestep_video = timestep_video.to(device=device, dtype=dtype)
        if noise_action is None:
            noise_action = self._tp_leader_value(
                lambda: torch.randn(
                    action.shape,
                    device=device,
                    dtype=dtype,
                    generator=generator,
                ),
                lambda: torch.empty_like(action),
            )
        else:
            noise_action = noise_action.to(device=device, dtype=dtype)
        if timestep_action is None:
            timestep_action = self._tp_leader_value(
                lambda: self.action_train_scheduler.sample_training_t(
                    batch_size, device, dtype, generator=generator
                ),
                lambda: torch.empty(batch_size, device=device, dtype=dtype),
            )
        else:
            timestep_action = timestep_action.to(device=device, dtype=dtype)

        noisy_video = self.video_train_scheduler.add_noise(
            input_latents, noise_video, timestep_video
        )
        noisy_video[:, :, :1] = input_latents[:, :, :1]
        target_video = self.video_train_scheduler.training_target(
            input_latents, noise_video, timestep_video
        )
        noisy_action = self.action_train_scheduler.add_noise(
            action, noise_action, timestep_action
        )
        target_action = self.action_train_scheduler.training_target(
            action, noise_action, timestep_action
        )

        video_state = self.video_expert.pre_dit(
            noisy_video,
            timestep_video,
            context,
            attention_context_mask,
        )
        action_state = self.action_expert.pre_dit(
            noisy_action,
            timestep_action,
            context,
            attention_context_mask,
        )
        needs_attention_mask = (
            self.fast_wam_config.training_attention_backend != "structured_sdpa"
            or return_debug_tensors
        )
        attention_mask = (
            self.build_training_attention_mask(
                video_state["tokens"].shape[1],
                action_state["tokens"].shape[1],
                video_state["meta"]["tokens_per_frame"],
                device=device,
            )
            if needs_attention_mask
            else None
        )
        video_tokens, action_tokens = self.mot.forward_joint(
            video_state, action_state, attention_mask
        )
        prediction_video = self.video_expert.post_dit(video_tokens, video_state)[:, :, 1:]
        prediction_action = self.action_expert.post_dit(action_tokens)

        target_video = target_video[:, :, 1:]
        if image_is_pad is not None:
            image_is_pad = image_is_pad.to(device=device, dtype=torch.bool)
        video_loss_per_sample = self._video_loss_per_sample(
            prediction_video, target_video, image_is_pad
        )
        video_weight = self.video_train_scheduler.training_weight(
            timestep_video
        ).to(device=device, dtype=video_loss_per_sample.dtype)
        loss_video = (video_loss_per_sample * video_weight).mean()

        action_token_loss = F.mse_loss(
            prediction_action.float(), target_action.float(), reduction="none"
        ).mean(dim=2)
        if action_is_pad is None:
            action_loss_per_sample = action_token_loss.mean(dim=1)
        else:
            action_is_pad = action_is_pad.to(device=device, dtype=torch.bool)
            valid = (~action_is_pad).to(dtype=action_token_loss.dtype)
            action_loss_per_sample = (
                (action_token_loss * valid).sum(dim=1)
                / valid.sum(dim=1).clamp(min=1.0)
            )
        action_weight = self.action_train_scheduler.training_weight(
            timestep_action
        ).to(device=device, dtype=action_loss_per_sample.dtype)
        loss_action = (action_loss_per_sample * action_weight).mean()

        cfg = self.fast_wam_config
        loss = cfg.loss_lambda_video * loss_video + cfg.loss_lambda_action * loss_action
        metrics: dict[str, Any] = {
            "loss_video": cfg.loss_lambda_video * loss_video.detach(),
            "loss_action": cfg.loss_lambda_action * loss_action.detach(),
        }
        if return_debug_tensors:
            assert attention_mask is not None
            metrics["_debug_stochastic_digests"] = {
                "noise_video": _debug_tensor_digest(noise_video),
                "timestep_video": _debug_tensor_digest(timestep_video),
                "noise_action": _debug_tensor_digest(noise_action),
                "timestep_action": _debug_tensor_digest(timestep_action),
                "attention_mask": _debug_tensor_digest(attention_mask),
            }
        return loss, metrics

    @torch.no_grad()
    def infer_action_encoded(
        self,
        first_frame_latents: torch.Tensor,
        context: torch.Tensor,
        context_mask: torch.Tensor,
        proprio: torch.Tensor,
        *,
        seed: int | None = None,
        num_inference_steps: int | None = None,
        sigma_shift: float | None = None,
    ) -> torch.Tensor:
        """Return a normalized action chunk from already encoded Wan inputs."""

        self.eval()
        cfg = self.fast_wam_config
        if first_frame_latents.ndim != 5 or first_frame_latents.shape[2] != 1:
            raise ValueError("first_frame_latents must be [B,C,1,H,W]")
        if context.ndim != 3 or context_mask.ndim != 2 or proprio.ndim != 2:
            raise ValueError("context/context_mask/proprio must be [B,L,D]/[B,L]/[B,P]")
        if first_frame_latents.shape[0] != 1:
            raise ValueError("Each TP replica handles one environment at a time; use DP for batching.")
        dtype = self.proprio_encoder.weight.dtype
        device = self.proprio_encoder.weight.device
        first_frame_latents = first_frame_latents.to(device=device, dtype=dtype)
        context = context.to(device=device, dtype=dtype)
        context_mask = context_mask.to(device=device, dtype=torch.bool)
        proprio = proprio.to(device=device, dtype=dtype)
        context, context_mask = self._append_proprio(context, context_mask, proprio)

        video_timestep = torch.zeros((1,), device=device, dtype=dtype)
        video_state = self.video_expert.pre_dit(
            first_frame_latents, video_timestep, context, context_mask
        )
        video_cache = self.mot.prefill_video_cache(video_state)

        noise_seed = cfg.inference_seed if seed is None else seed
        generator = torch.Generator(device="cpu").manual_seed(noise_seed)
        action = torch.randn(
            (1, cfg.action_horizon, cfg.action.action_dim),
            generator=generator,
            device="cpu",
            dtype=torch.float32,
        ).to(device=device, dtype=dtype)
        tp_group, tp_size, _ = _tp_info()
        if tp_group is not None and tp_size > 1:
            dist.broadcast(action, src=dist.get_global_rank(tp_group, 0), group=tp_group)

        steps = cfg.num_inference_steps if num_inference_steps is None else num_inference_steps
        shift = cfg.sigma_shift if sigma_shift is None else sigma_shift
        self.scheduler.set_timesteps(steps, shift=shift)
        for timestep in self.scheduler.timesteps:
            t = timestep.to(device=device, dtype=dtype).reshape(1)
            action_state = self.action_expert.pre_dit(action, t, context, context_mask)
            tokens = self.mot.forward_action_with_video_cache(action_state, video_cache)
            prediction = self.action_expert.post_dit(tokens)
            action = self.scheduler.step(prediction, timestep, action)
        return action[0].detach().float().cpu()

    @torch.no_grad()
    def infer_action(self, *args, **kwargs) -> torch.Tensor:
        """Public inference alias; inputs are frozen-component encoded tensors."""

        return self.infer_action_encoded(*args, **kwargs)

    def forward(self, *args, **kwargs):
        if "input_latents" in kwargs or (args and isinstance(args[0], torch.Tensor)):
            return self.training_loss_encoded(*args, **kwargs)
        return self.infer_action(*args, **kwargs)
