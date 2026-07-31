"""Wan2.2-TI2V-5B VAE encoder used by the Fast-WAM training path.

This is the encoder-only subset of the official Fast-WAM ``WanVideoVAE38``.
Training never decodes video, so omitting the decoder both removes an
unnecessary leader-rank memory cost and avoids depending on a particular
``diffusers`` development revision.  Module names and operations intentionally
match the official implementation so the released ``Wan2.2_VAE.pth`` encoder
weights load without conversion.
"""

from __future__ import annotations

from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange


_CACHE_T = 2
_MEAN = (
    -0.2289, -0.0052, -0.1323, -0.2339, -0.2799, 0.0174, 0.1838, 0.1557,
    -0.1382, 0.0542, 0.2813, 0.0891, 0.1570, -0.0098, 0.0375, -0.1825,
    -0.2246, -0.1207, -0.0698, 0.5109, 0.2665, -0.2108, -0.2158, 0.2502,
    -0.2055, -0.0322, 0.1109, 0.1567, -0.0729, 0.0899, -0.2799, -0.1230,
    -0.0313, -0.1649, 0.0117, 0.0723, -0.2839, -0.2083, -0.0520, 0.3748,
    0.0152, 0.1957, 0.1433, -0.2944, 0.3573, -0.0548, -0.1681, -0.0667,
)
_STD = (
    0.4765, 1.0364, 0.4514, 1.1677, 0.5313, 0.4990, 0.4818, 0.5013,
    0.8158, 1.0344, 0.5894, 1.0901, 0.6885, 0.6165, 0.8454, 0.4978,
    0.5759, 0.3523, 0.7135, 0.6804, 0.5833, 1.4146, 0.8986, 0.5659,
    0.7069, 0.5338, 0.4889, 0.4917, 0.4069, 0.4999, 0.6866, 0.4093,
    0.5709, 0.6065, 0.6415, 0.4944, 0.5726, 1.2042, 0.5458, 1.6887,
    0.3971, 1.0600, 0.3943, 0.5537, 0.5444, 0.4089, 0.7468, 0.7744,
)


class _CausalConv3d(nn.Conv3d):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._causal_padding = (
            self.padding[2],
            self.padding[2],
            self.padding[1],
            self.padding[1],
            2 * self.padding[0],
            0,
        )
        self.padding = (0, 0, 0)

    def forward(self, value, cache=None):
        padding = list(self._causal_padding)
        if cache is not None and padding[4] > 0:
            value = torch.cat([cache.to(value.device), value], dim=2)
            padding[4] -= cache.shape[2]
        return super().forward(F.pad(value, padding))


class _RMSNorm(nn.Module):
    def __init__(self, dim: int, *, images: bool):
        super().__init__()
        broadcast = (1, 1) if images else (1, 1, 1)
        self.scale = dim**0.5
        self.gamma = nn.Parameter(torch.ones(dim, *broadcast))

    def forward(self, value):
        return F.normalize(value, dim=1) * self.scale * self.gamma


class _ResidualBlock(nn.Module):
    def __init__(self, in_dim: int, out_dim: int, dropout: float = 0.0):
        super().__init__()
        self.residual = nn.Sequential(
            _RMSNorm(in_dim, images=False),
            nn.SiLU(),
            _CausalConv3d(in_dim, out_dim, 3, padding=1),
            _RMSNorm(out_dim, images=False),
            nn.SiLU(),
            nn.Dropout(dropout),
            _CausalConv3d(out_dim, out_dim, 3, padding=1),
        )
        self.shortcut = (
            _CausalConv3d(in_dim, out_dim, 1)
            if in_dim != out_dim
            else nn.Identity()
        )

    def forward(self, value, feature_cache, feature_index):
        residual = self.shortcut(value)
        for layer in self.residual:
            if isinstance(layer, _CausalConv3d):
                index = feature_index[0]
                cache = value[:, :, -_CACHE_T:].clone()
                previous = feature_cache[index]
                if cache.shape[2] < 2 and previous is not None:
                    cache = torch.cat(
                        [previous[:, :, -1].unsqueeze(2).to(cache.device), cache],
                        dim=2,
                    )
                value = layer(value, previous)
                feature_cache[index] = cache
                feature_index[0] += 1
            else:
                value = layer(value)
        return value + residual, feature_cache, feature_index


class _AttentionBlock(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.norm = _RMSNorm(dim, images=True)
        self.to_qkv = nn.Conv2d(dim, dim * 3, 1)
        self.proj = nn.Conv2d(dim, dim, 1)
        nn.init.zeros_(self.proj.weight)

    def forward(self, value):
        identity = value
        batch, channels, frames, height, width = value.shape
        value = rearrange(value, "b c t h w -> (b t) c h w")
        value = self.norm(value)
        query, key, result = (
            self.to_qkv(value)
            .reshape(batch * frames, 1, channels * 3, -1)
            .permute(0, 1, 3, 2)
            .contiguous()
            .chunk(3, dim=-1)
        )
        value = F.scaled_dot_product_attention(query, key, result)
        value = (
            value.squeeze(1)
            .permute(0, 2, 1)
            .reshape(batch * frames, channels, height, width)
        )
        value = self.proj(value)
        return rearrange(
            value,
            "(b t) c h w -> b c t h w",
            b=batch,
            t=frames,
        ) + identity


class _Resample38(nn.Module):
    def __init__(self, dim: int, mode: str):
        super().__init__()
        if mode not in {"downsample2d", "downsample3d"}:
            raise ValueError(f"Unsupported encoder resample mode {mode!r}")
        self.mode = mode
        self.resample = nn.Sequential(
            nn.ZeroPad2d((0, 1, 0, 1)),
            nn.Conv2d(dim, dim, 3, stride=(2, 2)),
        )
        if mode == "downsample3d":
            self.time_conv = _CausalConv3d(
                dim,
                dim,
                (3, 1, 1),
                stride=(2, 1, 1),
                padding=(0, 0, 0),
            )

    def forward(self, value, feature_cache, feature_index):
        batch, channels, frames, height, width = value.shape
        value = rearrange(value, "b c t h w -> (b t) c h w")
        value = self.resample(value)
        value = rearrange(
            value,
            "(b t) c h w -> b c t h w",
            b=batch,
            t=frames,
        )
        if self.mode == "downsample3d":
            index = feature_index[0]
            if feature_cache[index] is None:
                feature_cache[index] = value.clone()
                feature_index[0] += 1
            else:
                cache = value[:, :, -1:].clone()
                value = self.time_conv(
                    torch.cat(
                        [feature_cache[index][:, :, -1:], value],
                        dim=2,
                    )
                )
                feature_cache[index] = cache
                feature_index[0] += 1
        return value, feature_cache, feature_index


class _AverageDownsample3d(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        temporal_factor: int,
        spatial_factor: int,
    ):
        super().__init__()
        self.out_channels = out_channels
        self.temporal_factor = temporal_factor
        self.spatial_factor = spatial_factor
        self.factor = temporal_factor * spatial_factor * spatial_factor
        if in_channels * self.factor % out_channels:
            raise ValueError("Invalid average-downsample channel ratio")
        self.group_size = in_channels * self.factor // out_channels

    def forward(self, value):
        temporal_pad = (
            self.temporal_factor - value.shape[2] % self.temporal_factor
        ) % self.temporal_factor
        value = F.pad(value, (0, 0, 0, 0, temporal_pad, 0))
        batch, channels, frames, height, width = value.shape
        value = value.view(
            batch,
            channels,
            frames // self.temporal_factor,
            self.temporal_factor,
            height // self.spatial_factor,
            self.spatial_factor,
            width // self.spatial_factor,
            self.spatial_factor,
        )
        value = value.permute(0, 1, 3, 5, 7, 2, 4, 6).contiguous()
        value = value.view(
            batch,
            channels * self.factor,
            frames // self.temporal_factor,
            height // self.spatial_factor,
            width // self.spatial_factor,
        )
        return value.view(
            batch,
            self.out_channels,
            self.group_size,
            value.shape[2],
            value.shape[3],
            value.shape[4],
        ).mean(dim=2)


class _DownResidualBlock(nn.Module):
    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        *,
        temporal_downsample: bool,
        downsample: bool,
        count: int = 2,
    ):
        super().__init__()
        self.avg_shortcut = _AverageDownsample3d(
            in_dim,
            out_dim,
            2 if temporal_downsample else 1,
            2 if downsample else 1,
        )
        layers = []
        for _ in range(count):
            layers.append(_ResidualBlock(in_dim, out_dim))
            in_dim = out_dim
        if downsample:
            layers.append(
                _Resample38(
                    out_dim,
                    "downsample3d" if temporal_downsample else "downsample2d",
                )
            )
        self.downsamples = nn.Sequential(*layers)

    def forward(self, value, feature_cache, feature_index):
        shortcut = value.clone()
        for layer in self.downsamples:
            value, feature_cache, feature_index = layer(
                value,
                feature_cache,
                feature_index,
            )
        return (
            value + self.avg_shortcut(shortcut),
            feature_cache,
            feature_index,
        )


class _Encoder3d38(nn.Module):
    def __init__(self, dim: int = 160, latent_dim: int = 96):
        super().__init__()
        dims = [dim, dim, dim * 2, dim * 4, dim * 4]
        temporal = (False, True, True)
        self.conv1 = _CausalConv3d(12, dims[0], 3, padding=1)
        self.downsamples = nn.Sequential(
            *[
                _DownResidualBlock(
                    in_dim,
                    out_dim,
                    temporal_downsample=(
                        temporal[index] if index < len(temporal) else False
                    ),
                    downsample=index != len(dims) - 2,
                )
                for index, (in_dim, out_dim) in enumerate(
                    zip(dims[:-1], dims[1:], strict=True)
                )
            ]
        )
        self.middle = nn.Sequential(
            _ResidualBlock(dims[-1], dims[-1]),
            _AttentionBlock(dims[-1]),
            _ResidualBlock(dims[-1], dims[-1]),
        )
        self.head = nn.Sequential(
            _RMSNorm(dims[-1], images=False),
            nn.SiLU(),
            _CausalConv3d(dims[-1], latent_dim, 3, padding=1),
        )

    def forward(self, value, feature_cache, feature_index):
        index = feature_index[0]
        cache = value[:, :, -_CACHE_T:].clone()
        previous = feature_cache[index]
        if cache.shape[2] < 2 and previous is not None:
            cache = torch.cat(
                [previous[:, :, -1].unsqueeze(2).to(cache.device), cache],
                dim=2,
            )
        value = self.conv1(value, previous)
        feature_cache[index] = cache
        feature_index[0] += 1

        for layer in self.downsamples:
            value, feature_cache, feature_index = layer(
                value,
                feature_cache,
                feature_index,
            )
        for layer in self.middle:
            if isinstance(layer, _ResidualBlock):
                value, feature_cache, feature_index = layer(
                    value,
                    feature_cache,
                    feature_index,
                )
            else:
                value = layer(value)
        for layer in self.head:
            if isinstance(layer, _CausalConv3d):
                index = feature_index[0]
                cache = value[:, :, -_CACHE_T:].clone()
                previous = feature_cache[index]
                if cache.shape[2] < 2 and previous is not None:
                    cache = torch.cat(
                        [
                            previous[:, :, -1].unsqueeze(2).to(cache.device),
                            cache,
                        ],
                        dim=2,
                    )
                value = layer(value, previous)
                feature_cache[index] = cache
                feature_index[0] += 1
            else:
                value = layer(value)
        return value, feature_cache, feature_index


def _patchify(value):
    return rearrange(
        value,
        "b c f (h q) (w r) -> b (c r q) f h w",
        q=2,
        r=2,
    )


def _causal_convolution_count(module: nn.Module) -> int:
    return sum(isinstance(item, _CausalConv3d) for item in module.modules())


class WanVideoVAE38Encoder(nn.Module):
    """Frozen encoder with the exact official standardized-latent output."""

    temporal_downsample_factor = 4
    spatial_downsample_factor = 16
    latent_dim = 48

    def __init__(self):
        super().__init__()
        self.encoder = _Encoder3d38()
        self.conv1 = _CausalConv3d(96, 96, 1)
        self.register_buffer(
            "latent_mean",
            torch.tensor(_MEAN).view(1, 48, 1, 1, 1),
            persistent=False,
        )
        self.register_buffer(
            "latent_inverse_std",
            torch.tensor([1.0 / value for value in _STD]).view(
                1, 48, 1, 1, 1
            ),
            persistent=False,
        )
        self._clear_cache()

    def _clear_cache(self):
        count = _causal_convolution_count(self.encoder)
        self._encoder_feature_index = [0]
        self._encoder_feature_cache = [None] * count

    @classmethod
    def from_pretrained(
        cls,
        checkpoint: str | Path,
        *,
        device: torch.device,
        dtype: torch.dtype,
    ) -> "WanVideoVAE38Encoder":
        path = Path(checkpoint).expanduser().resolve()
        if not path.is_file():
            raise FileNotFoundError(path)
        model = cls()
        state = torch.load(path, map_location="cpu", weights_only=True)
        if "model_state" in state:
            state = state["model_state"]
        expected = set(model.state_dict())
        encoder_state = {
            key: value
            for key, value in state.items()
            if key in expected
        }
        missing = sorted(expected - set(encoder_state))
        if missing:
            raise RuntimeError(
                f"{path}: missing {len(missing)} VAE encoder tensors: {missing[:16]}"
            )
        model.load_state_dict(encoder_state, strict=True)
        return model.to(device=device, dtype=dtype).eval().requires_grad_(False)

    @torch.no_grad()
    def encode_normalized_video(self, video: torch.Tensor) -> torch.Tensor:
        if video.ndim != 5 or video.shape[1] != 3:
            raise ValueError(f"video must be [B,3,T,H,W], got {tuple(video.shape)}")
        if video.shape[2] % 4 != 1:
            raise ValueError(f"Wan video length must satisfy T % 4 == 1, got {video.shape[2]}")
        dtype = self.conv1.weight.dtype
        device = self.conv1.weight.device
        value = _patchify(video.to(device=device, dtype=dtype))
        self._clear_cache()
        output = None
        iterations = 1 + (value.shape[2] - 1) // 4
        for index in range(iterations):
            self._encoder_feature_index = [0]
            chunk = (
                value[:, :, :1]
                if index == 0
                else value[:, :, 1 + 4 * (index - 1) : 1 + 4 * index]
            )
            encoded, self._encoder_feature_cache, self._encoder_feature_index = (
                self.encoder(
                    chunk,
                    self._encoder_feature_cache,
                    self._encoder_feature_index,
                )
            )
            output = encoded if output is None else torch.cat([output, encoded], dim=2)
        mean, _ = self.conv1(output).chunk(2, dim=1)
        result = (
            mean - self.latent_mean.to(device=mean.device, dtype=mean.dtype)
        ) * self.latent_inverse_std.to(device=mean.device, dtype=mean.dtype)
        self._clear_cache()
        return result
