"""Official Fast-WAM continuous FlowMatch training and inference schedule."""

from __future__ import annotations

import numpy as np
import torch


class WanFlowMatchScheduler:
    def __init__(
        self,
        num_train_timesteps: int = 1000,
        *,
        shift: float = 5.0,
        eps: float = 1.0e-10,
    ):
        if num_train_timesteps <= 0:
            raise ValueError("num_train_timesteps must be positive")
        if shift <= 0:
            raise ValueError("shift must be positive")
        self.num_train_timesteps = int(num_train_timesteps)
        self.shift = float(shift)
        self.eps = float(eps)
        self.sigmas: torch.Tensor | None = None
        self.timesteps: torch.Tensor | None = None
        self._y_min, self._weight_norm_const = self._precompute_training_weight_stats()

    @staticmethod
    def _phi(value: torch.Tensor, shift: float) -> torch.Tensor:
        return shift * value / (1.0 + (shift - 1.0) * value)

    def _precompute_training_weight_stats(self) -> tuple[float, float]:
        steps = self.num_train_timesteps
        grid = torch.linspace(1.0, 0.0, steps + 1, dtype=torch.float64)[:-1]
        timestep = self._phi(grid, self.shift) * float(steps)
        weight = torch.exp(-2.0 * ((timestep - (steps / 2.0)) / steps) ** 2)
        y_min = float(weight.min().item())
        norm_const = float((weight - y_min).mean().item())
        return y_min, norm_const

    def sample_training_t(
        self,
        batch_size: int,
        device: torch.device | str,
        dtype: torch.dtype,
        *,
        generator: torch.Generator | None = None,
    ) -> torch.Tensor:
        if batch_size <= 0:
            raise ValueError("batch_size must be positive")
        uniform = torch.rand(
            (batch_size,), device=device, dtype=torch.float32, generator=generator
        )
        return (
            self._phi(uniform, self.shift) * float(self.num_train_timesteps)
        ).to(dtype=dtype)

    def training_weight(self, timestep: torch.Tensor) -> torch.Tensor:
        value = timestep.float()
        steps = float(self.num_train_timesteps)
        weight = torch.exp(-2.0 * ((value - (steps / 2.0)) / steps) ** 2)
        return (weight - self._y_min) / (self._weight_norm_const + self.eps)

    def add_noise(
        self,
        original_samples: torch.Tensor,
        noise: torch.Tensor,
        timestep: torch.Tensor,
    ) -> torch.Tensor:
        sigma = (timestep / float(self.num_train_timesteps)).to(
            device=original_samples.device, dtype=original_samples.dtype
        )
        if sigma.ndim:
            sigma = sigma.view(-1, *([1] * (original_samples.ndim - 1)))
        return (1.0 - sigma) * original_samples + sigma * noise

    @staticmethod
    def training_target(
        sample: torch.Tensor, noise: torch.Tensor, timestep: torch.Tensor
    ) -> torch.Tensor:
        del timestep
        return noise - sample

    def set_timesteps(self, num_inference_steps: int, *, shift: float | None = None):
        if num_inference_steps <= 0:
            raise ValueError("num_inference_steps must be positive")
        shift = self.shift if shift is None else float(shift)
        sigma = np.linspace(1.0, 0.0, num_inference_steps + 1)[:num_inference_steps]
        sigma = shift * sigma / (1.0 + (shift - 1.0) * sigma)
        self.sigmas = torch.as_tensor(sigma, dtype=torch.float32)
        self.timesteps = self.sigmas * self.num_train_timesteps

    def step(self, model_output, timestep, sample):
        if self.timesteps is None or self.sigmas is None:
            raise RuntimeError("set_timesteps must be called before step")
        index = torch.argmin((self.timesteps - timestep.detach().float().cpu()).abs())
        sigma = self.sigmas[index].to(device=sample.device, dtype=sample.dtype)
        sigma_next = (
            self.sigmas[index + 1].to(device=sample.device, dtype=sample.dtype)
            if index + 1 < len(self.sigmas)
            else torch.zeros((), device=sample.device, dtype=sample.dtype)
        )
        return sample + model_output * (sigma_next - sigma)
