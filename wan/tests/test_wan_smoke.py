import json
import tempfile
from pathlib import Path

import torch

from wan.data.dataset import WanJsonlDataset, wan_collate
from wan.model.checkpoint import (
    adapt_official_state_to_model,
    convert_diffusers_wan_state_dict,
    load_official_wan_checkpoint,
)
from wan.model.config import PRESETS, WanConfig
from wan.model.scheduler import WanFlowMatchScheduler
from wan.model.wan_dit import WanFlowTrainingModel, WanModel


def test_tiny_wan_forward_shape():
    cfg = PRESETS["tiny"]
    model = WanModel(cfg)
    x = torch.randn(1, cfg.in_dim, 5, 16, 16)
    context = torch.randn(1, 8, cfg.text_dim)
    timestep = torch.tensor([500.0])
    y = model(x, timestep=timestep, context=context)
    assert y.shape == x.shape


def test_scheduler_matches_wan_endpoints():
    scheduler = WanFlowMatchScheduler()
    scheduler.set_timesteps(4, shift=5.0, training=True)
    assert scheduler.sigmas[0] == 1
    assert scheduler.sigmas[-1] > 0
    clean = torch.zeros(1, 1, 1, 1, 1)
    noise = torch.ones_like(clean)
    noised = scheduler.add_noise(clean, noise, scheduler.timesteps[0])
    assert torch.allclose(noised, noise)
    assert torch.allclose(scheduler.training_target(clean, noise, scheduler.timesteps[0]), noise)


def test_tiny_flow_loss_backward():
    cfg = PRESETS["tiny"]
    model = WanFlowTrainingModel(cfg, train_timesteps=16, disable_timestep_weight=True)
    x = torch.randn(1, cfg.in_dim, 5, 16, 16)
    context = torch.randn(1, 8, cfg.text_dim)
    loss = model(x, context)
    loss.backward()
    assert torch.isfinite(loss)
    assert model.dit.patch_embedding.weight.grad is not None


def test_ti2v_5b_preset_matches_diffsynth_dimensions():
    cfg = PRESETS["ti2v-5b"]
    assert cfg.in_dim == 48
    assert cfg.out_dim == 48
    assert cfg.dim == 3072
    assert cfg.ffn_dim == 14336
    assert cfg.num_heads == 24
    assert cfg.num_layers == 30
    assert cfg.seperated_timestep
    assert cfg.fuse_vae_embedding_in_latents
    assert not cfg.require_clip_embedding
    assert not cfg.require_vae_embedding


def test_tiny_separated_timestep_forward_and_first_frame_loss():
    base = PRESETS["tiny"]
    cfg = WanConfig(**{**base.__dict__, "seperated_timestep": True, "fuse_vae_embedding_in_latents": True})
    model = WanFlowTrainingModel(cfg, train_timesteps=16, disable_timestep_weight=True)
    x = torch.randn(1, cfg.in_dim, 5, 8, 8)
    context = torch.randn(1, 8, cfg.text_dim)
    first = x[:, :, 0:1].clone()
    y = model.dit(x, timestep=torch.tensor([500.0]), context=context, fuse_vae_embedding_in_latents=True)
    assert y.shape == x.shape
    loss = model(x, context, first_frame_latents=first)
    loss.backward()
    assert torch.isfinite(loss)
    assert model.dit.patch_embedding.weight.grad is not None


def test_diffusers_key_converter_maps_repeated_blocks():
    state = {
        "blocks.3.attn1.to_q.weight": torch.randn(4, 4),
        "blocks.3.attn2.to_out.0.bias": torch.randn(4),
        "blocks.3.ffn.net.0.proj.weight": torch.randn(8, 4),
        "condition_embedder.time_proj.bias": torch.randn(24),
        "proj_out.weight": torch.randn(4, 4),
    }
    converted = convert_diffusers_wan_state_dict(state)
    assert "blocks.3.self_attn.q.weight" in converted
    assert "blocks.3.cross_attn.o.bias" in converted
    assert "blocks.3.ffn.0.weight" in converted
    assert "time_projection.1.bias" in converted
    assert "head.head.weight" in converted


def test_official_state_adapts_to_wrapped_linear_keys():
    cfg = PRESETS["tiny"]
    model = WanModel(cfg)
    state = {}
    for key, value in model.state_dict().items():
        if ".linear." in key:
            official_key = key.replace(".linear.", ".")
            state[official_key] = torch.randn_like(value)
        elif not key.endswith("_extra_state"):
            state[key] = torch.randn_like(value) if value.is_floating_point() else value.clone()
    adapted, unexpected = adapt_official_state_to_model(state, model)
    assert not unexpected
    assert "blocks.0.self_attn.q.linear.weight" in adapted
    assert "text_embedding.0.linear.weight" in adapted
    assert "head.head.linear.weight" in adapted


def test_official_checkpoint_dir_loads_incrementally():
    cfg = PRESETS["tiny"]
    source = WanModel(cfg)
    target = WanModel(cfg)
    official_state = {}
    for key, value in source.state_dict().items():
        if key.endswith("_extra_state"):
            continue
        official_key = key.replace(".linear.", ".")
        official_state[official_key] = value.clone()
    items = list(official_state.items())

    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        torch.save(dict(items[: len(items) // 2]), root / "model-00001.pt")
        torch.save(dict(items[len(items) // 2 :]), root / "model-00002.pt")
        missing, unexpected = load_official_wan_checkpoint(target, root, strict=True)

    assert not missing
    assert not unexpected


def test_jsonl_dataset_reads_sample_and_torch_shard():
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        sample = {
            "input_latents": torch.randn(4, 5, 8, 8),
            "context": torch.randn(16, 64),
            "first_frame_latents": torch.randn(4, 1, 8, 8),
            "fuse_vae_embedding_in_latents": True,
            "prompt": "sample prompt",
            "video_path": "sample.mp4",
        }
        sample_path = root / "sample.pt"
        torch.save(sample, sample_path)

        shard_path = root / "shard.pt"
        torch.save(
            {
                "input_latents": torch.stack([sample["input_latents"], sample["input_latents"] + 1], dim=0),
                "context": torch.stack([sample["context"], sample["context"] + 1], dim=0),
                "first_frame_latents": torch.stack(
                    [sample["first_frame_latents"], sample["first_frame_latents"] + 1],
                    dim=0,
                ),
            },
            shard_path,
        )

        manifest = root / "data.jsonl"
        with open(manifest, "w", encoding="utf-8") as f:
            f.write(json.dumps({"sample_path": str(sample_path)}) + "\n")
            f.write(
                json.dumps(
                    {
                        "shard_path": str(shard_path),
                        "index": 1,
                        "first_frame_latents_key": "first_frame_latents",
                        "fuse_vae_embedding_in_latents": True,
                    }
                )
                + "\n"
            )

        ds = WanJsonlDataset(str(manifest), shard_cache_size=1)
        assert torch.allclose(ds[0]["input_latents"], sample["input_latents"])
        assert torch.allclose(ds[1]["context"], sample["context"] + 1)
        assert torch.allclose(ds[1]["first_frame_latents"], sample["first_frame_latents"] + 1)
        batch = wan_collate([ds[0], ds[1]])
        assert batch["first_frame_latents"].shape == (2, 4, 1, 8, 8)
        assert batch["fuse_vae_embedding_in_latents"].item()
