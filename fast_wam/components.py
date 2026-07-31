"""Frozen Wan2.2 VAE and UMT5 components used only on TP leader ranks."""

from __future__ import annotations

from pathlib import Path

import torch


class WanVAEEncoder:
    """Frozen Diffusers Wan2.2 VAE with the official standardized-latent contract."""

    def __init__(self, vae, *, device: torch.device, dtype: torch.dtype):
        self.vae = vae.to(device=device, dtype=dtype).eval().requires_grad_(False)
        self.device = device
        self.dtype = dtype
        self.latents_mean = torch.tensor(
            self.vae.config.latents_mean, device=device, dtype=torch.float32
        ).view(1, 48, 1, 1, 1)
        self.latents_std = torch.tensor(
            self.vae.config.latents_std, device=device, dtype=torch.float32
        ).view(1, 48, 1, 1, 1)

    @classmethod
    def from_pretrained(
        cls,
        assets: str | Path,
        *,
        device: torch.device,
        dtype: torch.dtype,
    ) -> "WanVAEEncoder":
        try:
            from diffusers import AutoencoderKLWan
        except ImportError as exc:  # pragma: no cover
            raise RuntimeError("Fast-WAM video encoding requires diffusers") from exc
        vae = AutoencoderKLWan.from_pretrained(
            str(assets),
            subfolder="vae",
            torch_dtype=dtype,
            local_files_only=True,
        )
        return cls(vae, device=device, dtype=dtype)

    @torch.no_grad()
    def encode_normalized_video(self, video: torch.Tensor) -> torch.Tensor:
        """Encode `[B,3,4n+1,H,W]` video already normalized to `[-1,1]`."""

        if video.ndim != 5 or video.shape[1] != 3:
            raise ValueError(f"video must be [B,3,T,H,W], got {tuple(video.shape)}")
        if video.shape[2] % 4 != 1:
            raise ValueError(f"Wan video length must satisfy T % 4 == 1, got {video.shape[2]}")
        raw = self.vae.encode(
            video.to(device=self.device, dtype=self.dtype)
        ).latent_dist.mode().float()
        return ((raw - self.latents_mean) / self.latents_std).to(self.dtype)

    @torch.no_grad()
    def encode_image(self, image: torch.Tensor) -> torch.Tensor:
        """Encode one `[0,1]` image as standardized Wan latent mean."""

        if image.ndim == 3:
            image = image.unsqueeze(0)
        if image.ndim != 4 or image.shape[:2] != (1, 3):
            raise ValueError(f"image must be [1,3,H,W], got {tuple(image.shape)}")
        normalized = (image.to(device=self.device, dtype=self.dtype) * 2.0 - 1.0).unsqueeze(2)
        return self.encode_normalized_video(normalized)


class WanTextEncoder:
    """Frozen UMT5 encoder/tokenizer used to build the official prompt cache."""

    def __init__(
        self,
        text_encoder,
        tokenizer,
        *,
        device: torch.device,
        dtype: torch.dtype,
    ):
        self.text_encoder = (
            text_encoder.to(device=device, dtype=dtype).eval().requires_grad_(False)
        )
        self.tokenizer = tokenizer
        self.device = device
        self.dtype = dtype

    @classmethod
    def from_pretrained(
        cls,
        assets: str | Path,
        tokenizer: str | Path,
        *,
        device: torch.device,
        dtype: torch.dtype,
    ) -> "WanTextEncoder":
        try:
            from transformers import AutoTokenizer, UMT5EncoderModel
        except ImportError as exc:  # pragma: no cover
            raise RuntimeError("Fast-WAM text encoding requires transformers") from exc
        encoder = UMT5EncoderModel.from_pretrained(
            str(assets),
            subfolder="text_encoder",
            torch_dtype=dtype,
            local_files_only=True,
        )
        tokenizer_obj = AutoTokenizer.from_pretrained(
            str(tokenizer),
            local_files_only=True,
        )
        return cls(
            encoder,
            tokenizer_obj,
            device=device,
            dtype=dtype,
        )

    @torch.no_grad()
    def encode(
        self,
        prompts: str | list[str],
        *,
        max_length: int = 128,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if isinstance(prompts, str):
            prompts = [prompts]
        tokens = self.tokenizer(
            prompts,
            padding="max_length",
            truncation=True,
            max_length=max_length,
            add_special_tokens=True,
            return_tensors="pt",
        )
        ids = tokens.input_ids.to(self.device)
        mask = tokens.attention_mask.to(device=self.device, dtype=torch.bool)
        context = self.text_encoder(input_ids=ids, attention_mask=mask).last_hidden_state
        return context, mask


class WanFrozenComponents:
    """LeRobot-compatible image and prompt encoders.

    These modules deliberately live outside :class:`FastWAMModel`, keeping
    frozen weights out of DDP/DCP and allowing non-leader TP ranks to avoid the
    VAE/UMT5 memory cost entirely.
    """

    def __init__(self, vae, text_encoder, tokenizer, *, device: torch.device, dtype: torch.dtype):
        self._vae_encoder = WanVAEEncoder(vae, device=device, dtype=dtype)
        self._text_encoder = WanTextEncoder(
            text_encoder,
            tokenizer,
            device=device,
            dtype=dtype,
        )
        self.vae = self._vae_encoder.vae
        self.text_encoder = self._text_encoder.text_encoder
        self.tokenizer = self._text_encoder.tokenizer
        self.device = device
        self.dtype = dtype
        self.latents_mean = self._vae_encoder.latents_mean
        self.latents_std = self._vae_encoder.latents_std

    @classmethod
    def from_pretrained(
        cls,
        assets: str | Path,
        tokenizer: str | Path,
        *,
        device: torch.device,
        dtype: torch.dtype,
    ) -> "WanFrozenComponents":
        try:
            from diffusers import AutoencoderKLWan
            from transformers import AutoTokenizer, UMT5EncoderModel
        except ImportError as exc:  # pragma: no cover
            raise RuntimeError("Fast-WAM preprocessing requires diffusers and transformers") from exc
        assets = str(assets)
        vae = AutoencoderKLWan.from_pretrained(
            assets, subfolder="vae", torch_dtype=dtype, local_files_only=True
        )
        text_encoder = UMT5EncoderModel.from_pretrained(
            assets, subfolder="text_encoder", torch_dtype=dtype, local_files_only=True
        )
        tokenizer_obj = AutoTokenizer.from_pretrained(str(tokenizer), local_files_only=True)
        return cls(vae, text_encoder, tokenizer_obj, device=device, dtype=dtype)

    @torch.no_grad()
    def encode_image(self, image: torch.Tensor) -> torch.Tensor:
        """Encode one [0,1] image as standardized Wan latent mean."""

        return self._vae_encoder.encode_image(image)

    @torch.no_grad()
    def encode_normalized_video(self, video: torch.Tensor) -> torch.Tensor:
        return self._vae_encoder.encode_normalized_video(video)

    @torch.no_grad()
    def encode_prompt(self, prompt: str, *, max_length: int = 128) -> tuple[torch.Tensor, torch.Tensor]:
        context, source_mask = self._text_encoder.encode(prompt, max_length=max_length)
        context = context.clone()
        context[~source_mask] = 0
        return context, torch.ones_like(source_mask, dtype=torch.bool)


def prepare_camera_image(
    images: dict[str, torch.Tensor], image_size: tuple[int, int] = (224, 448)
) -> torch.Tensor:
    """Resize sorted cameras and concatenate them along width exactly as LeRobot."""

    if not images:
        raise ValueError("At least one camera image is required")
    keys = sorted(images)
    per_camera = (image_size[0], image_size[1] // len(keys))
    resized = []
    for key in keys:
        image = images[key]
        if image.ndim == 3:
            image = image.unsqueeze(0)
        if image.ndim != 4 or image.shape[1] != 3:
            raise ValueError(f"{key} must be [B,3,H,W], got {tuple(image.shape)}")
        if tuple(image.shape[-2:]) != per_camera:
            image = torch.nn.functional.interpolate(
                image,
                size=per_camera,
                mode="bilinear",
                align_corners=False,
                antialias=True,
            )
        resized.append(image)
    return torch.cat(resized, dim=-1)
