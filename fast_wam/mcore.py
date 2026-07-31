"""Runtime selection for Megatron Core primitives with a CPU test fallback."""

from __future__ import annotations

import os

import torch
import torch.nn as nn


_USE_MCORE = os.environ.get("FAST_WAM_DISABLE_MCORE") != "1" and (
    torch.cuda.is_available() or os.environ.get("FAST_WAM_FORCE_MCORE") == "1"
)

if _USE_MCORE:
    from wan.model.wan_dit import (  # noqa: F401
        MegatronModule,
        TensorParallelRMSNorm,
        _Linear,
        _MinimalMegatronConfig,
        _sharded_state_dict_recursive,
        _tp_info,
        modulate,
        precompute_freqs_cis,
        precompute_freqs_cis_3d,
        rope_apply,
        sinusoidal_embedding_1d,
    )
else:

    class _MinimalMegatronConfig:
        pass

    MegatronModule = nn.Module

    class _Linear(nn.Module):
        def __init__(
            self,
            in_features,
            out_features,
            *,
            bias=True,
            parallel="replicated",
            skip_bias_add=False,
            **_,
        ):
            super().__init__()
            self.parallel = parallel
            self.skip_bias_add = bool(skip_bias_add)
            self.linear = nn.Linear(in_features, out_features, bias=bias)

        @property
        def weight(self):
            return self.linear.weight

        @property
        def bias(self):
            return self.linear.bias

        def forward(self, x):
            if self.skip_bias_add:
                return torch.nn.functional.linear(x, self.weight, None), self.bias
            return self.linear(x)

    class TensorParallelRMSNorm(nn.Module):
        def __init__(self, dim: int, eps: float = 1e-5):
            super().__init__()
            self.dim = dim
            self.eps = eps
            self.weight = nn.Parameter(torch.ones(dim))

        def forward(self, x):
            normed = x.float() * torch.rsqrt(x.float().pow(2).mean(dim=-1, keepdim=True) + self.eps)
            return (normed * self.weight.float()).to(x.dtype)

    def _tp_info():
        return None, 1, 0

    def _sharded_state_dict_recursive(module, prefix="", sharded_offsets=(), metadata=None):
        del sharded_offsets, metadata
        return module.state_dict(prefix=prefix, keep_vars=True)

    def modulate(x, shift, scale):
        return x * (1 + scale) + shift

    def sinusoidal_embedding_1d(dim: int, position: torch.Tensor):
        sinusoid = torch.outer(
            position.to(torch.float64),
            torch.pow(
                10000,
                -torch.arange(dim // 2, dtype=torch.float64, device=position.device).div(dim // 2),
            ),
        )
        return torch.cat([torch.cos(sinusoid), torch.sin(sinusoid)], dim=1).to(position.dtype)

    def precompute_freqs_cis(dim: int, end: int = 1024, theta: float = 10000.0):
        inv = 1.0 / (theta ** (torch.arange(0, dim, 2)[: dim // 2].double() / dim))
        angles = torch.outer(torch.arange(end, device=inv.device), inv)
        return torch.polar(torch.ones_like(angles), angles)

    def precompute_freqs_cis_3d(dim: int, end: int = 1024, theta: float = 10000.0):
        return (
            precompute_freqs_cis(dim - 2 * (dim // 3), end, theta),
            precompute_freqs_cis(dim // 3, end, theta),
            precompute_freqs_cis(dim // 3, end, theta),
        )

    def rope_apply(x, freqs, num_heads, hidden_size=None):
        bsz, seq_len, dim = x.shape
        full_hidden = dim if hidden_size is None else hidden_size
        head_dim = full_hidden // num_heads
        local_heads = dim // head_dim
        x = x.view(bsz, seq_len, local_heads, head_dim)
        x_complex = torch.view_as_complex(x.to(torch.float64).reshape(bsz, seq_len, local_heads, -1, 2))
        return torch.view_as_real(x_complex * freqs.to(x.device)).flatten(2).to(x.dtype)


__all__ = [
    "MegatronModule",
    "TensorParallelRMSNorm",
    "_Linear",
    "_MinimalMegatronConfig",
    "_sharded_state_dict_recursive",
    "_tp_info",
    "modulate",
    "precompute_freqs_cis",
    "precompute_freqs_cis_3d",
    "rope_apply",
    "sinusoidal_embedding_1d",
]
