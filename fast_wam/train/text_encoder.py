"""Exact DiffSynth UMT5 encoder used by the official Fast-WAM recipe."""

from __future__ import annotations

import math
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F


class _GELU(nn.Module):
    def forward(self, value):
        return 0.5 * value * (
            1.0
            + torch.tanh(
                math.sqrt(2.0 / math.pi)
                * (value + 0.044715 * torch.pow(value, 3.0))
            )
        )


class _T5LayerNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1.0e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, value):
        value = value * torch.rsqrt(
            value.float().pow(2).mean(dim=-1, keepdim=True) + self.eps
        )
        if self.weight.dtype in (torch.float16, torch.bfloat16):
            value = value.type_as(self.weight)
        return self.weight * value


class _T5Attention(nn.Module):
    """T5 attention without query scaling, matching DiffSynth exactly."""

    def __init__(
        self,
        dim: int,
        attention_dim: int,
        num_heads: int,
        dropout: float,
    ):
        super().__init__()
        if attention_dim % num_heads:
            raise ValueError("attention_dim must be divisible by num_heads")
        self.num_heads = num_heads
        self.head_dim = attention_dim // num_heads
        self.q = nn.Linear(dim, attention_dim, bias=False)
        self.k = nn.Linear(dim, attention_dim, bias=False)
        self.v = nn.Linear(dim, attention_dim, bias=False)
        self.o = nn.Linear(attention_dim, dim, bias=False)
        self.dropout = nn.Dropout(dropout)

    def forward(self, value, *, mask=None, position_bias=None):
        batch = value.shape[0]
        query = self.q(value).view(
            batch, -1, self.num_heads, self.head_dim
        )
        key = self.k(value).view(
            batch, -1, self.num_heads, self.head_dim
        )
        result = self.v(value).view(
            batch, -1, self.num_heads, self.head_dim
        )
        attention_bias = value.new_zeros(
            batch,
            self.num_heads,
            query.shape[1],
            key.shape[1],
        )
        if position_bias is not None:
            attention_bias += position_bias
        if mask is not None:
            mask = mask.view(batch, 1, 1, -1)
            attention_bias.masked_fill_(
                mask == 0,
                torch.finfo(value.dtype).min,
            )
        attention = (
            torch.einsum("binc,bjnc->bnij", query, key) + attention_bias
        )
        attention = F.softmax(attention.float(), dim=-1).type_as(attention)
        value = torch.einsum("bnij,bjnc->binc", attention, result)
        value = value.reshape(batch, -1, self.num_heads * self.head_dim)
        return self.dropout(self.o(value))


class _T5FeedForward(nn.Module):
    def __init__(self, dim: int, ffn_dim: int, dropout: float):
        super().__init__()
        self.gate = nn.Sequential(
            nn.Linear(dim, ffn_dim, bias=False),
            _GELU(),
        )
        self.fc1 = nn.Linear(dim, ffn_dim, bias=False)
        self.fc2 = nn.Linear(ffn_dim, dim, bias=False)
        self.dropout = nn.Dropout(dropout)

    def forward(self, value):
        value = self.fc1(value) * self.gate(value)
        value = self.dropout(value)
        value = self.fc2(value)
        return self.dropout(value)


class _T5RelativeEmbedding(nn.Module):
    def __init__(
        self,
        num_buckets: int,
        num_heads: int,
        *,
        max_distance: int = 128,
    ):
        super().__init__()
        self.num_buckets = num_buckets
        self.max_distance = max_distance
        self.embedding = nn.Embedding(num_buckets, num_heads)

    def _bucket(self, relative_position):
        buckets_per_direction = self.num_buckets // 2
        buckets = (
            (relative_position > 0).long() * buckets_per_direction
        )
        relative_position = torch.abs(relative_position)
        max_exact = buckets_per_direction // 2
        large = max_exact + (
            torch.log(relative_position.float() / max_exact)
            / math.log(self.max_distance / max_exact)
            * (buckets_per_direction - max_exact)
        ).long()
        large = torch.min(
            large,
            torch.full_like(large, buckets_per_direction - 1),
        )
        return buckets + torch.where(
            relative_position < max_exact,
            relative_position,
            large,
        )

    def forward(self, query_length: int, key_length: int):
        device = self.embedding.weight.device
        relative = (
            torch.arange(key_length, device=device).unsqueeze(0)
            - torch.arange(query_length, device=device).unsqueeze(1)
        )
        value = self.embedding(self._bucket(relative))
        return value.permute(2, 0, 1).unsqueeze(0).contiguous()


class _T5Block(nn.Module):
    def __init__(
        self,
        dim: int,
        attention_dim: int,
        ffn_dim: int,
        num_heads: int,
        num_buckets: int,
        dropout: float,
    ):
        super().__init__()
        self.norm1 = _T5LayerNorm(dim)
        self.attn = _T5Attention(
            dim,
            attention_dim,
            num_heads,
            dropout,
        )
        self.norm2 = _T5LayerNorm(dim)
        self.ffn = _T5FeedForward(dim, ffn_dim, dropout)
        self.pos_embedding = _T5RelativeEmbedding(
            num_buckets,
            num_heads,
        )

    def forward(self, value, mask):
        position = self.pos_embedding(value.shape[1], value.shape[1])
        value = value + self.attn(
            self.norm1(value),
            mask=mask,
            position_bias=position,
        )
        return value + self.ffn(self.norm2(value))


class WanT5Encoder(nn.Module):
    """The 24-layer, 4096-wide text encoder from official Fast-WAM."""

    def __init__(
        self,
        vocab_size: int = 256384,
        dim: int = 4096,
        attention_dim: int = 4096,
        ffn_dim: int = 10240,
        num_heads: int = 64,
        num_layers: int = 24,
        num_buckets: int = 32,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.token_embedding = nn.Embedding(vocab_size, dim)
        self.dropout = nn.Dropout(dropout)
        self.blocks = nn.ModuleList(
            [
                _T5Block(
                    dim,
                    attention_dim,
                    ffn_dim,
                    num_heads,
                    num_buckets,
                    dropout,
                )
                for _ in range(num_layers)
            ]
        )
        self.norm = _T5LayerNorm(dim)

    def forward(self, ids, mask):
        value = self.dropout(self.token_embedding(ids))
        for block in self.blocks:
            value = block(value, mask)
        return self.dropout(self.norm(value))

    @classmethod
    def from_pretrained(
        cls,
        checkpoint: str | Path,
        *,
        device: torch.device,
        dtype: torch.dtype,
    ) -> "WanT5Encoder":
        path = Path(checkpoint).expanduser().resolve()
        if not path.is_file():
            raise FileNotFoundError(path)
        # Avoid allocating and randomly initializing a second 11 GB copy.
        with torch.device("meta"):
            model = cls()
        state = torch.load(
            path,
            map_location="cpu",
            weights_only=True,
            mmap=True,
        )
        incompatible = model.load_state_dict(
            state,
            strict=True,
            assign=True,
        )
        if incompatible.missing_keys or incompatible.unexpected_keys:
            raise RuntimeError(
                f"{path}: text checkpoint mismatch: {incompatible}"
            )
        return (
            model.to(device=device, dtype=dtype)
            .eval()
            .requires_grad_(False)
        )
