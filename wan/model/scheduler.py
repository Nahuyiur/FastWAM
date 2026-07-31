"""FlowMatch scheduler logic matching DiffSynth's Wan template."""

from __future__ import annotations

import torch


class WanFlowMatchScheduler:
    """Small DiffSynth-compatible FlowMatchScheduler for Wan.

    DiffSynth uses sigma in [1, 0], timesteps = sigma * 1000, and training
    target = noise - clean_latents.
    """

    def __init__(self, num_train_timesteps: int = 1000):
        self.num_train_timesteps = num_train_timesteps
        self.sigmas = None
        self.timesteps = None
        self.training = False
        self.linear_timesteps_weights = None

    @staticmethod
    def _wan_sigmas(num_inference_steps: int, denoising_strength: float, shift: float):
        sigma_min = 0.0
        sigma_max = 1.0
        sigma_start = sigma_min + (sigma_max - sigma_min) * denoising_strength
        sigmas = torch.linspace(sigma_start, sigma_min, num_inference_steps + 1)[:-1]
        if shift is not None and shift != 1.0:
            sigmas = shift * sigmas / (1 + (shift - 1) * sigmas)
        return sigmas

    def set_timesteps(
        self,
        num_inference_steps: int = 1000,
        denoising_strength: float = 1.0,
        shift: float = 5.0,
        training: bool = False,
    ):
        self.sigmas = self._wan_sigmas(num_inference_steps, denoising_strength, shift)
        self.timesteps = self.sigmas * self.num_train_timesteps
        self.training = training
        if training:
            self.set_training_weight()

    def set_training_weight(self):
        steps = self.num_train_timesteps
        x = self.timesteps
        y = torch.exp(-2 * ((x - steps / 2) / steps) ** 2)
        y_shifted = y - y.min()
        weights = y_shifted * (steps / y_shifted.sum().clamp(min=1e-12))
        if len(self.timesteps) != steps:
            weights = weights * (len(self.timesteps) / steps)
            if len(weights) > 1:
                weights = weights + weights[1]
        self.linear_timesteps_weights = weights

    def _timestep_id(self, timestep):
        if isinstance(timestep, torch.Tensor):
            timestep_cpu = timestep.detach().float().cpu()
        else:
            timestep_cpu = torch.tensor(float(timestep))
        return torch.argmin((self.timesteps - timestep_cpu).abs())

    def add_noise(self, original_samples, noise, timestep):
        timestep_id = self._timestep_id(timestep)
        sigma = self.sigmas[timestep_id].to(dtype=original_samples.dtype, device=original_samples.device)
        return (1 - sigma) * original_samples + sigma * noise

    def training_target(self, sample, noise, timestep):
        return noise - sample

    def training_weight(self, timestep):
        if self.linear_timesteps_weights is None:
            return torch.tensor(1.0, device=timestep.device if isinstance(timestep, torch.Tensor) else None)
        timestep_id = self._timestep_id(timestep)
        weight = self.linear_timesteps_weights[timestep_id]
        if isinstance(timestep, torch.Tensor):
            weight = weight.to(dtype=timestep.dtype, device=timestep.device)
        return weight

    def step(self, model_output, timestep, sample):
        timestep_id = self._timestep_id(timestep)
        sigma = self.sigmas[timestep_id].to(dtype=sample.dtype, device=sample.device)
        if timestep_id + 1 >= len(self.timesteps):
            sigma_next = torch.tensor(0.0, dtype=sample.dtype, device=sample.device)
        else:
            sigma_next = self.sigmas[timestep_id + 1].to(dtype=sample.dtype, device=sample.device)
        return sample + model_output * (sigma_next - sigma)
