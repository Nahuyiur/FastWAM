from __future__ import annotations

import importlib.util
import json
import sys
import types
from dataclasses import replace
from pathlib import Path

import pytest
import torch

from fast_wam.checkpoint import (
    _slice_for_rank,
    _unexpected_dcp_keys,
    load_lerobot_checkpoint,
)
from fast_wam.components import prepare_camera_image
from fast_wam.config import FastWAMConfig
from fast_wam.model import FastWAMModel, _fast_wam_training_mask_mod
from fast_wam.policy import MinMaxStats
from fast_wam.scheduler import WanFlowMatchScheduler
from fast_wam.train.initialization import resize_action_backbone_tensor
from fast_wam.train.data import LATENT_SHAPE, LatentCache
from fast_wam.train.sampler import (
    OfficialEpochBatchSampler,
    OfficialValidationBatchSampler,
)


def _reference_modules():
    root = Path(__file__).resolve().parents[3] / "lerobot/src/lerobot/policies/fastwam/wan"
    if not root.is_dir():
        source_root = Path(__file__).resolve().parents[2] / "src"
        if str(source_root) not in sys.path:
            sys.path.insert(0, str(source_root))
        from fastwam.models.wan22 import action_dit, mot, wan_video_dit
        from fastwam.models.wan22.schedulers import scheduler_continuous

        class FastWAMJointMaskReference:
            @torch.no_grad()
            def _build_mot_attention_mask(
                self,
                video_seq_len,
                action_seq_len,
                video_tokens_per_frame,
                device,
            ):
                total_seq_len = video_seq_len + action_seq_len
                mask = torch.zeros(
                    (total_seq_len, total_seq_len),
                    dtype=torch.bool,
                    device=device,
                )
                mask[:video_seq_len, :video_seq_len] = (
                    self.video_expert.build_video_to_video_mask(
                        video_seq_len=video_seq_len,
                        video_tokens_per_frame=video_tokens_per_frame,
                        device=device,
                    )
                )
                mask[video_seq_len:, video_seq_len:] = True
                mask[video_seq_len:, :video_seq_len] = True
                return mask

        modular = types.SimpleNamespace(
            ActionDiT=action_dit.ActionDiT,
            MoT=mot.MoT,
            FastWAMJoint=FastWAMJointMaskReference,
            WanContinuousFlowMatchScheduler=(
                scheduler_continuous.WanContinuousFlowMatchScheduler
            ),
        )
        return wan_video_dit, modular
    package = types.ModuleType("_fastwam_reference")
    package.__path__ = [str(root)]
    sys.modules[package.__name__] = package

    def load(name, filename):
        spec = importlib.util.spec_from_file_location(name, root / filename)
        module = importlib.util.module_from_spec(spec)
        sys.modules[name] = module
        spec.loader.exec_module(module)
        return module

    load("_fastwam_reference.model", "model.py")
    video = load("_fastwam_reference.video_dit", "video_dit.py")
    components = types.ModuleType("_fastwam_reference.components")
    components.WAN22_DIFFUSERS_MODEL_ID = ""
    components.WAN_T5_TOKENIZER = ""
    for name in (
        "build_wan_tokenizer",
        "load_pretrained_wan_text_encoder",
        "load_pretrained_wan_vae",
        "load_wan_video_dit",
        "resolve_wan_dit_paths",
    ):
        setattr(components, name, lambda *args, **kwargs: None)
    sys.modules[components.__name__] = components
    modular = load("_fastwam_reference.modular", "modular.py")
    modular.WanContinuousFlowMatchScheduler = (
        video.WanContinuousFlowMatchScheduler
    )
    return video, modular


def _reference_compatible_tiny_config(*, joint: bool = True):
    """Use a head dimension whose legacy 3D RoPE split is exactly representable."""

    cfg = FastWAMConfig.tiny()
    return replace(
        cfg,
        video=replace(cfg.video, attn_head_dim=24),
        action=replace(cfg.action, attn_head_dim=24),
        joint_action_video_attention=joint,
    )


def _copy_reference_weights(ours, video, action):
    source = {
        **{f"mot.mixtures.video.{key}": value for key, value in video.state_dict().items()},
        **{f"mot.mixtures.action.{key}": value for key, value in action.state_dict().items()},
    }
    for name, parameter in ours.named_parameters():
        if name.startswith("proprio_encoder"):
            continue
        source_name = name.replace(".linear.weight", ".weight").replace(".linear.bias", ".bias")
        parameter.data.copy_(source[source_name])


def test_tiny_action_only_inference_matches_lerobot_fastwam():
    video_module, modular = _reference_modules()
    torch.manual_seed(7)
    cfg = _reference_compatible_tiny_config(joint=False)
    ours = FastWAMModel(cfg)
    v, a = cfg.video, cfg.action
    video = video_module.WanVideoDiT(
        has_image_input=False,
        hidden_dim=v.hidden_dim,
        in_dim=v.in_dim,
        ffn_dim=v.ffn_dim,
        out_dim=v.out_dim,
        text_dim=v.text_dim,
        freq_dim=v.freq_dim,
        eps=v.eps,
        patch_size=v.patch_size,
        num_heads=v.num_heads,
        attn_head_dim=v.attn_head_dim,
        num_layers=v.num_layers,
        seperated_timestep=True,
        fuse_vae_embedding_in_latents=True,
        video_attention_mask_mode="first_frame_causal",
        action_conditioned=False,
    )
    action = modular.ActionDiT(
        hidden_dim=a.hidden_dim,
        action_dim=a.action_dim,
        ffn_dim=a.ffn_dim,
        text_dim=a.text_dim,
        freq_dim=a.freq_dim,
        eps=a.eps,
        num_heads=a.num_heads,
        attn_head_dim=a.attn_head_dim,
        num_layers=a.num_layers,
    )
    _copy_reference_weights(ours, video, action)
    reference_proprio = torch.nn.Linear(cfg.proprio_dim, cfg.video.text_dim)
    reference_proprio.weight.data.copy_(ours.proprio_encoder.weight)
    reference_proprio.bias.data.copy_(ours.proprio_encoder.bias)
    mot = modular.MoT({"video": video, "action": action}, mot_checkpoint_mixed_attn=False)

    latents = torch.randn(1, v.in_dim, 1, 4, 8)
    context = torch.randn(1, 5, v.text_dim)
    context_mask = torch.ones(1, 5, dtype=torch.bool)
    proprio = torch.randn(1, cfg.proprio_dim)
    reference_context = torch.cat([context, reference_proprio(proprio).unsqueeze(1)], dim=1)
    reference_mask = torch.cat([context_mask, torch.ones(1, 1, dtype=torch.bool)], dim=1)
    video_pre = video.pre_dit(
        latents,
        torch.zeros(1),
        reference_context,
        reference_mask,
        action=None,
        fuse_vae_embedding_in_latents=True,
    )
    video_len = video_pre["tokens"].shape[1]
    attention_mask = torch.ones(
        video_len + cfg.action_horizon, video_len + cfg.action_horizon, dtype=torch.bool
    )
    cache = mot.prefill_video_cache(
        video_pre["tokens"],
        video_pre["freqs"],
        video_pre["t_mod"],
        {"context": video_pre["context"], "mask": video_pre["context_mask"]},
        attention_mask[:video_len, :video_len],
    )
    generator = torch.Generator("cpu").manual_seed(cfg.inference_seed)
    reference = torch.randn(
        (1, cfg.action_horizon, a.action_dim), generator=generator, dtype=torch.float32
    )
    scheduler = modular.WanContinuousFlowMatchScheduler(shift=cfg.sigma_shift)
    timesteps, deltas = scheduler.build_inference_schedule(
        cfg.num_inference_steps, reference.device, reference.dtype
    )
    for timestep, delta in zip(timesteps, deltas, strict=True):
        action_pre = action.pre_dit(reference, timestep.reshape(1), reference_context, reference_mask)
        tokens = mot.forward_action_with_video_cache(
            action_pre["tokens"],
            action_pre["freqs"],
            action_pre["t_mod"],
            {"context": action_pre["context"], "mask": action_pre["context_mask"]},
            cache,
            attention_mask,
            video_len,
        )
        reference = scheduler.step(action.post_dit(tokens, action_pre), delta, reference)

    actual = ours.infer_action_only_encoded(latents, context, context_mask, proprio)
    torch.testing.assert_close(actual, reference[0], atol=1.0e-5, rtol=1.0e-5)


def test_tiny_joint_inference_matches_robocasa_fastwam_joint():
    video_module, modular = _reference_modules()
    torch.manual_seed(11)
    cfg = _reference_compatible_tiny_config()
    ours = FastWAMModel(cfg)
    v, a = cfg.video, cfg.action
    video = video_module.WanVideoDiT(
        has_image_input=False,
        hidden_dim=v.hidden_dim,
        in_dim=v.in_dim,
        ffn_dim=v.ffn_dim,
        out_dim=v.out_dim,
        text_dim=v.text_dim,
        freq_dim=v.freq_dim,
        eps=v.eps,
        patch_size=v.patch_size,
        num_heads=v.num_heads,
        attn_head_dim=v.attn_head_dim,
        num_layers=v.num_layers,
        seperated_timestep=True,
        fuse_vae_embedding_in_latents=True,
        video_attention_mask_mode="first_frame_causal",
        action_conditioned=False,
    )
    action_expert = modular.ActionDiT(
        hidden_dim=a.hidden_dim,
        action_dim=a.action_dim,
        ffn_dim=a.ffn_dim,
        text_dim=a.text_dim,
        freq_dim=a.freq_dim,
        eps=a.eps,
        num_heads=a.num_heads,
        attn_head_dim=a.attn_head_dim,
        num_layers=a.num_layers,
    )
    _copy_reference_weights(ours, video, action_expert)
    reference_proprio = torch.nn.Linear(cfg.proprio_dim, cfg.video.text_dim)
    reference_proprio.weight.data.copy_(ours.proprio_encoder.weight)
    reference_proprio.bias.data.copy_(ours.proprio_encoder.bias)
    reference_mot = modular.MoT(
        {"video": video, "action": action_expert}, mot_checkpoint_mixed_attn=False
    )

    generator = torch.Generator(device="cpu").manual_seed(31)
    first_frame = torch.randn(1, v.in_dim, 1, 4, 8, generator=generator)
    context = torch.randn(1, 5, v.text_dim, generator=generator)
    context_mask = torch.ones(1, 5, dtype=torch.bool)
    proprio = torch.randn(1, cfg.proprio_dim, generator=generator)
    reference_context = torch.cat(
        [context, reference_proprio(proprio).unsqueeze(1)], dim=1
    )
    reference_mask = torch.cat(
        [context_mask, torch.ones(1, 1, dtype=torch.bool)], dim=1
    )
    video_generator = torch.Generator(device="cpu").manual_seed(cfg.inference_seed)
    action_generator = torch.Generator(device="cpu").manual_seed(cfg.inference_seed)
    reference_video = torch.randn(
        1,
        v.in_dim,
        (cfg.num_video_frames - 1) // cfg.temporal_downsample_factor + 1,
        4,
        8,
        generator=video_generator,
    )
    reference_action = torch.randn(
        1,
        cfg.action_horizon,
        a.action_dim,
        generator=action_generator,
    )
    reference_video[:, :, :1] = first_frame
    video_scheduler = modular.WanContinuousFlowMatchScheduler(shift=cfg.sigma_shift)
    action_scheduler = modular.WanContinuousFlowMatchScheduler(shift=cfg.sigma_shift)
    video_timesteps, video_deltas = video_scheduler.build_inference_schedule(
        cfg.num_inference_steps, reference_video.device, reference_video.dtype
    )
    action_timesteps, action_deltas = action_scheduler.build_inference_schedule(
        cfg.num_inference_steps, reference_action.device, reference_action.dtype
    )
    joint_owner = types.SimpleNamespace(video_expert=video)
    for video_t, video_delta, action_t, action_delta in zip(
        video_timesteps,
        video_deltas,
        action_timesteps,
        action_deltas,
        strict=True,
    ):
        video_state = video.pre_dit(
            reference_video,
            video_t.reshape(1),
            reference_context,
            reference_mask,
            fuse_vae_embedding_in_latents=True,
        )
        action_state = action_expert.pre_dit(
            reference_action,
            action_t.reshape(1),
            reference_context,
            reference_mask,
        )
        attention_mask = modular.FastWAMJoint._build_mot_attention_mask(
            joint_owner,
            video_state["tokens"].shape[1],
            action_state["tokens"].shape[1],
            video_state["meta"]["tokens_per_frame"],
            reference_video.device,
        )
        output = reference_mot(
            embeds_all={"video": video_state["tokens"], "action": action_state["tokens"]},
            attention_mask=attention_mask,
            freqs_all={"video": video_state["freqs"], "action": action_state["freqs"]},
            context_all={
                "video": {"context": video_state["context"], "mask": video_state["context_mask"]},
                "action": {"context": action_state["context"], "mask": action_state["context_mask"]},
            },
            t_mod_all={"video": video_state["t_mod"], "action": action_state["t_mod"]},
        )
        video_prediction = video.post_dit(output["video"], video_state)
        action_prediction = action_expert.post_dit(output["action"], action_state)
        reference_video = video_scheduler.step(
            video_prediction, video_delta, reference_video
        )
        reference_action = action_scheduler.step(
            action_prediction, action_delta, reference_action
        )
        reference_video[:, :, :1] = first_frame

    actual = ours.infer_action_encoded(first_frame, context, context_mask, proprio)
    torch.testing.assert_close(actual, reference_action[0], atol=2.0e-5, rtol=2.0e-5)


def test_streaming_checkpoint_round_trip(tmp_path):
    safetensors = pytest.importorskip("safetensors.torch")
    source = FastWAMModel(FastWAMConfig.tiny())
    state = {
        f"model.{name.replace('.linear.weight', '.weight').replace('.linear.bias', '.bias')}": value
        for name, value in source.state_dict().items()
    }
    safetensors.save_file(state, tmp_path / "model.safetensors")
    target = FastWAMModel(FastWAMConfig.tiny())
    load_lerobot_checkpoint(target, tmp_path, strict=True)
    for expected, actual in zip(source.parameters(), target.parameters(), strict=True):
        torch.testing.assert_close(actual, expected)


def test_training_dcp_common_state_is_not_a_model_mismatch():
    keys = [
        "args",
        "checkpoint_version",
        "iteration",
        "optimizer",
        "opt_param_scheduler",
        "num_floating_point_operations_so_far",
        "content_metadata",
        "unexpected.model.key",
    ]
    assert _unexpected_dcp_keys(keys) == ["unexpected.model.key"]


def _fixed_training_inputs(cfg):
    generator = torch.Generator(device="cpu").manual_seed(23)
    batch_size = 2
    return {
        "input_latents": torch.randn(
            batch_size, cfg.video.in_dim, 3, 4, 8, generator=generator
        ),
        "action": torch.randn(
            batch_size,
            cfg.action_horizon,
            cfg.action.action_dim,
            generator=generator,
        ),
        "context": torch.randn(
            batch_size, 5, cfg.video.text_dim, generator=generator
        ),
        "context_mask": torch.ones(batch_size, 5, dtype=torch.bool),
        "proprio": torch.randn(batch_size, cfg.proprio_dim, generator=generator),
        "image_is_pad": torch.zeros(batch_size, 9, dtype=torch.bool),
        "action_is_pad": torch.zeros(
            batch_size, cfg.action_horizon, dtype=torch.bool
        ),
        "noise_video": torch.randn(
            batch_size, cfg.video.in_dim, 3, 4, 8, generator=generator
        ),
        "noise_action": torch.randn(
            batch_size,
            cfg.action_horizon,
            cfg.action.action_dim,
            generator=generator,
        ),
        "timestep_video": torch.tensor([125.0, 725.0]),
        "timestep_action": torch.tensor([250.0, 850.0]),
    }


def test_tiny_training_matches_reference_and_backward():
    video_module, modular = _reference_modules()
    torch.manual_seed(17)
    cfg = _reference_compatible_tiny_config()
    ours = FastWAMModel(cfg)
    v, a = cfg.video, cfg.action
    video = video_module.WanVideoDiT(
        has_image_input=False,
        hidden_dim=v.hidden_dim,
        in_dim=v.in_dim,
        ffn_dim=v.ffn_dim,
        out_dim=v.out_dim,
        text_dim=v.text_dim,
        freq_dim=v.freq_dim,
        eps=v.eps,
        patch_size=v.patch_size,
        num_heads=v.num_heads,
        attn_head_dim=v.attn_head_dim,
        num_layers=v.num_layers,
        seperated_timestep=True,
        fuse_vae_embedding_in_latents=True,
        video_attention_mask_mode="first_frame_causal",
        action_conditioned=False,
    )
    action_expert = modular.ActionDiT(
        hidden_dim=a.hidden_dim,
        action_dim=a.action_dim,
        ffn_dim=a.ffn_dim,
        text_dim=a.text_dim,
        freq_dim=a.freq_dim,
        eps=a.eps,
        num_heads=a.num_heads,
        attn_head_dim=a.attn_head_dim,
        num_layers=a.num_layers,
    )
    _copy_reference_weights(ours, video, action_expert)
    reference_proprio = torch.nn.Linear(cfg.proprio_dim, cfg.video.text_dim)
    reference_proprio.weight.data.copy_(ours.proprio_encoder.weight)
    reference_proprio.bias.data.copy_(ours.proprio_encoder.bias)
    reference_mot = modular.MoT(
        {"video": video, "action": action_expert}, mot_checkpoint_mixed_attn=False
    )

    values = _fixed_training_inputs(cfg)
    context = torch.cat(
        [
            values["context"],
            reference_proprio(values["proprio"]).unsqueeze(1),
        ],
        dim=1,
    )
    context_mask = torch.cat(
        [
            values["context_mask"],
            torch.ones(values["context_mask"].shape[0], 1, dtype=torch.bool),
        ],
        dim=1,
    )
    scheduler_video = modular.WanContinuousFlowMatchScheduler(
        shift=cfg.video_train_shift
    )
    scheduler_action = modular.WanContinuousFlowMatchScheduler(
        shift=cfg.action_train_shift
    )
    noisy_video = scheduler_video.add_noise(
        values["input_latents"],
        values["noise_video"],
        values["timestep_video"],
    )
    noisy_video[:, :, :1] = values["input_latents"][:, :, :1]
    target_video = scheduler_video.training_target(
        values["input_latents"],
        values["noise_video"],
        values["timestep_video"],
    )
    noisy_action = scheduler_action.add_noise(
        values["action"],
        values["noise_action"],
        values["timestep_action"],
    )
    target_action = scheduler_action.training_target(
        values["action"],
        values["noise_action"],
        values["timestep_action"],
    )
    video_state = video.pre_dit(
        noisy_video,
        values["timestep_video"],
        context,
        context_mask,
        fuse_vae_embedding_in_latents=True,
    )
    action_state = action_expert.pre_dit(
        noisy_action,
        values["timestep_action"],
        context,
        context_mask,
    )
    joint_owner = types.SimpleNamespace(video_expert=video)
    attention_mask = modular.FastWAMJoint._build_mot_attention_mask(
        joint_owner,
        video_state["tokens"].shape[1],
        action_state["tokens"].shape[1],
        video_state["meta"]["tokens_per_frame"],
        noisy_video.device,
    )
    output = reference_mot(
        embeds_all={
            "video": video_state["tokens"],
            "action": action_state["tokens"],
        },
        attention_mask=attention_mask,
        freqs_all={
            "video": video_state["freqs"],
            "action": action_state["freqs"],
        },
        context_all={
            "video": {
                "context": video_state["context"],
                "mask": video_state["context_mask"],
            },
            "action": {
                "context": action_state["context"],
                "mask": action_state["context_mask"],
            },
        },
        t_mod_all={
            "video": video_state["t_mod"],
            "action": action_state["t_mod"],
        },
    )
    prediction_video = video.post_dit(output["video"], video_state)[:, :, 1:]
    prediction_action = action_expert.post_dit(output["action"], action_state)
    reference_video_per_sample = torch.nn.functional.mse_loss(
        prediction_video.float(),
        target_video[:, :, 1:].float(),
        reduction="none",
    ).mean(dim=(1, 2, 3, 4))
    reference_video_loss = (
        reference_video_per_sample
        * scheduler_video.training_weight(values["timestep_video"])
    ).mean()
    reference_action_per_sample = torch.nn.functional.mse_loss(
        prediction_action.float(), target_action.float(), reduction="none"
    ).mean(dim=(1, 2))
    reference_action_loss = (
        reference_action_per_sample
        * scheduler_action.training_weight(values["timestep_action"])
    ).mean()

    reference_loss = reference_video_loss + reference_action_loss
    reference_optimizer = torch.optim.AdamW(
        list(video.parameters())
        + list(action_expert.parameters())
        + list(reference_proprio.parameters()),
        lr=3.0e-4,
        betas=(0.9, 0.95),
        eps=1.0e-8,
        weight_decay=1.0e-2,
    )
    actual_optimizer = torch.optim.AdamW(
        ours.parameters(),
        lr=3.0e-4,
        betas=(0.9, 0.95),
        eps=1.0e-8,
        weight_decay=1.0e-2,
    )
    actual, metrics = ours.training_loss_encoded(**values)
    torch.testing.assert_close(metrics["loss_video"], reference_video_loss)
    torch.testing.assert_close(metrics["loss_action"], reference_action_loss)
    torch.testing.assert_close(
        actual, reference_loss
    )
    reference_loss.backward()
    actual.backward()
    reference_parameters = {
        **{
            f"mot.mixtures.video.{name}": parameter
            for name, parameter in video.named_parameters()
        },
        **{
            f"mot.mixtures.action.{name}": parameter
            for name, parameter in action_expert.named_parameters()
        },
        **{
            f"proprio_encoder.{name}": parameter
            for name, parameter in reference_proprio.named_parameters()
        },
    }
    actual_parameters = dict(ours.named_parameters())
    assert len(actual_parameters) == len(reference_parameters)
    for actual_name, actual_parameter in actual_parameters.items():
        reference_name = actual_name.replace(".linear.weight", ".weight").replace(
            ".linear.bias", ".bias"
        )
        reference_parameter = reference_parameters[reference_name]
        assert (actual_parameter.grad is None) == (reference_parameter.grad is None)
        if actual_parameter.grad is not None:
            torch.testing.assert_close(
                actual_parameter.grad,
                reference_parameter.grad,
                atol=5.0e-5,
                rtol=5.0e-5,
            )

    reference_optimizer.step()
    actual_optimizer.step()
    for actual_name, actual_parameter in actual_parameters.items():
        reference_name = actual_name.replace(".linear.weight", ".weight").replace(
            ".linear.bias", ".bias"
        )
        torch.testing.assert_close(
            actual_parameter,
            reference_parameters[reference_name],
            atol=5.0e-5,
            rtol=5.0e-5,
        )


@pytest.mark.parametrize(
    ("video_length", "action_length", "first_frame_length", "joint"),
    (
        (12, 8, 4, False),
        (20, 8, 4, False),
        (20, 8, 8, False),
        (12, 8, 4, True),
        (20, 8, 4, True),
        (20, 8, 8, True),
    ),
)
def test_fast_wam_flex_mask_predicate_matches_dense_contract(
    video_length,
    action_length,
    first_frame_length,
    joint,
):
    model = FastWAMModel(
        replace(FastWAMConfig.tiny(), joint_action_video_attention=joint)
    )
    expected = model.build_training_attention_mask(
        video_length,
        action_length,
        first_frame_length,
        device=torch.device("cpu"),
    )
    total = video_length + action_length
    query_index = torch.arange(total).view(total, 1)
    key_index = torch.arange(total).view(1, total)
    actual = _fast_wam_training_mask_mod(
        video_length,
        first_frame_length,
        joint,
    )(None, None, query_index, key_index)
    torch.testing.assert_close(actual, expected)


def test_robocasa_joint_action_queries_see_all_video_tokens():
    model = FastWAMModel(
        replace(FastWAMConfig.tiny(), joint_action_video_attention=True)
    )
    video_length = 12
    action_length = 8
    first_frame_length = 4
    mask = model.build_training_attention_mask(
        video_length,
        action_length,
        first_frame_length,
        device=torch.device("cpu"),
    )
    assert mask[video_length:, :video_length].all()
    assert mask[video_length:, video_length:].all()
    assert not mask[:video_length, video_length:].any()


def test_original_fastwam_action_queries_only_see_clean_video_tokens():
    model = FastWAMModel(FastWAMConfig.tiny())
    video_length = 12
    action_length = 8
    first_frame_length = 4
    mask = model.build_training_attention_mask(
        video_length,
        action_length,
        first_frame_length,
        device=torch.device("cpu"),
    )
    action_rows = mask[video_length:]
    assert action_rows[:, :first_frame_length].all()
    assert not action_rows[:, first_frame_length:video_length].any()
    assert action_rows[:, video_length:].all()


@pytest.mark.parametrize("joint", (False, True))
def test_flex_training_matches_structured_sdpa_for_all_gradients(joint):
    cfg = replace(
        FastWAMConfig.tiny(),
        training_attention_backend="structured_sdpa",
        training_kernel_mode="optimized",
        joint_action_video_attention=joint,
    )
    values = _fixed_training_inputs(cfg)
    torch.manual_seed(31)
    reference = FastWAMModel(cfg)
    expected, expected_metrics = reference.training_loss_encoded(**values)
    expected.backward()

    flex = FastWAMModel(replace(cfg, training_attention_backend="flex"))
    flex.load_state_dict(reference.state_dict(), strict=True)
    actual, actual_metrics = flex.training_loss_encoded(**values)
    actual.backward()
    torch.testing.assert_close(actual, expected, atol=1.0e-5, rtol=1.0e-5)
    for key in ("loss_video", "loss_action"):
        torch.testing.assert_close(
            actual_metrics[key],
            expected_metrics[key],
            atol=1.0e-5,
            rtol=1.0e-5,
        )
    reference_parameters = dict(reference.named_parameters())
    flex_parameters = dict(flex.named_parameters())
    assert flex_parameters.keys() == reference_parameters.keys()
    for name, reference_parameter in reference_parameters.items():
        flex_parameter = flex_parameters[name]
        assert (flex_parameter.grad is None) == (reference_parameter.grad is None), name
        if reference_parameter.grad is not None:
            torch.testing.assert_close(
                flex_parameter.grad,
                reference_parameter.grad,
                atol=5.0e-5,
                rtol=5.0e-5,
                msg=lambda message, name=name: f"{name}: {message}",
            )


@pytest.mark.parametrize("joint", (False, True))
def test_optimized_structured_training_matches_reference_and_backward(joint):
    cfg = replace(
        FastWAMConfig.tiny(),
        joint_action_video_attention=joint,
    )
    values = _fixed_training_inputs(cfg)
    values["context_is_dense"] = True
    torch.manual_seed(37)
    reference = FastWAMModel(cfg)
    expected, _ = reference.training_loss_encoded(**values)
    expected.backward()

    optimized_cfg = replace(
        cfg,
        training_attention_backend="structured_sdpa",
        training_kernel_mode="optimized",
    )
    optimized = FastWAMModel(optimized_cfg)
    optimized.load_state_dict(reference.state_dict(), strict=True)
    actual, _ = optimized.training_loss_encoded(**values)
    actual.backward()
    torch.testing.assert_close(actual, expected, atol=2.0e-5, rtol=2.0e-5)
    torch.testing.assert_close(
        optimized.video_expert.patch_embedding.weight.grad,
        reference.video_expert.patch_embedding.weight.grad,
        atol=5.0e-5,
        rtol=5.0e-5,
    )
    assert optimized.state_dict().keys() == reference.state_dict().keys()


def test_latent_cache_contract_and_mmap(tmp_path):
    values = torch.arange(
        2 * torch.tensor(LATENT_SHAPE).prod().item(),
        dtype=torch.int64,
    ).to(torch.bfloat16).view(2, *LATENT_SHAPE)
    shard = tmp_path / "latents-00000.bf16"
    shard.write_bytes(values.view(torch.uint8).numpy().tobytes())
    manifest = {
        "version": 1,
        "complete": True,
        "num_samples": 2,
        "dtype": "bfloat16",
        "sample_shape": list(LATENT_SHAPE),
        "samples_per_shard": 2,
        "dataset_fingerprint": "dataset",
        "preprocessing_fingerprint": "preprocess",
        "shards": [
            {
                "id": 0,
                "file": shard.name,
                "start": 0,
                "count": 2,
                "size_bytes": shard.stat().st_size,
            }
        ],
    }
    (tmp_path / "manifest.json").write_text(json.dumps(manifest))
    cache = LatentCache(
        tmp_path,
        expected_samples=2,
        dataset_digest="dataset",
        preprocessing_digest="preprocess",
    )
    torch.testing.assert_close(cache[0], values[0], atol=0, rtol=0)
    torch.testing.assert_close(cache[1], values[1], atol=0, rtol=0)


def test_stochastic_seed_is_independent_of_global_rng():
    cfg = FastWAMConfig.tiny()
    values = _fixed_training_inputs(cfg)
    for name in (
        "noise_video",
        "noise_action",
        "timestep_video",
        "timestep_action",
    ):
        del values[name]
    torch.manual_seed(47)
    model = FastWAMModel(cfg)

    first, first_metrics = model.training_loss_encoded(
        **values, stochastic_seed=42
    )
    torch.randn(97)
    second, second_metrics = model.training_loss_encoded(
        **values, stochastic_seed=42
    )
    torch.testing.assert_close(second, first, atol=0.0, rtol=0.0)
    torch.testing.assert_close(
        second_metrics["loss_video"],
        first_metrics["loss_video"],
        atol=0.0,
        rtol=0.0,
    )
    torch.testing.assert_close(
        second_metrics["loss_action"],
        first_metrics["loss_action"],
        atol=0.0,
        rtol=0.0,
    )
    different, _ = model.training_loss_encoded(
        **values, stochastic_seed=43
    )
    assert not torch.equal(different, first)


@pytest.mark.parametrize("world", [2, 4])
def test_checkpoint_tp_slice(world):
    full = torch.arange(32).reshape(8, 4)
    target = torch.empty(8 // world, 4)
    for rank in range(world):
        actual = _slice_for_rank(
            full, target, name="weight", tp_rank=rank, tp_world=world
        )
        torch.testing.assert_close(actual, full.chunk(world, dim=0)[rank])


def test_scheduler_and_processing_contract():
    scheduler = WanFlowMatchScheduler()
    scheduler.set_timesteps(10, shift=5.0)
    assert scheduler.timesteps[0] == 1000
    assert scheduler.timesteps[-1] > 0
    images = {
        "observation.images.image2": torch.zeros(1, 3, 8, 8),
        "observation.images.image": torch.ones(1, 3, 8, 8),
    }
    prepared = prepare_camera_image(images, (16, 32))
    assert prepared.shape == (1, 3, 16, 32)
    assert torch.all(prepared[..., :16] == 1)
    stats = MinMaxStats(
        torch.zeros(2), torch.full((2,), 2.0), torch.zeros(3), torch.ones(3)
    )
    torch.testing.assert_close(stats.normalize_state(torch.tensor([[0.0, 2.0]])), torch.tensor([[-1.0, 1.0]]))
    action = stats.unnormalize_action(torch.tensor([[-1.0, 0.0, 1.0]]))
    torch.testing.assert_close(action, torch.tensor([[0.0, 0.5, -1.0]]))


def test_official_epoch_sampler_and_resume_contract():
    samplers = [
        OfficialEpochBatchSampler(
            dataset_size=13,
            consumed_samples=0,
            micro_batch_size=2,
            data_parallel_rank=rank,
            data_parallel_size=3,
            seed=42,
        )
        for rank in range(3)
    ]
    batches = [list(sampler) for sampler in samplers]
    permutation = torch.randperm(
        13,
        generator=torch.Generator(device="cpu").manual_seed(42),
    ).tolist()
    permutation.extend(permutation[: 18 - len(permutation)])
    for step in range(3):
        combined = sum((batches[rank][step] for rank in range(3)), [])
        assert combined == permutation[step * 6 : (step + 1) * 6]

    resumed = OfficialEpochBatchSampler(
        dataset_size=13,
        consumed_samples=6,
        micro_batch_size=2,
        data_parallel_rank=1,
        data_parallel_size=3,
        seed=42,
    )
    assert list(resumed) == batches[1][1:]


def test_official_validation_sampler_contract():
    sampler = OfficialValidationBatchSampler(
        dataset_size=277713,
        consumed_samples=8,
        data_parallel_rank=3,
        data_parallel_size=8,
        eval_interval=200,
    )
    actual = list(sampler)[0][0]
    expected = int(
        torch.randint(
            0,
            277713,
            (1,),
            generator=torch.Generator(device="cpu").manual_seed(403),
        ).item()
    )
    assert actual == expected


def test_action_backbone_resize_alpha_contract():
    source = torch.arange(24, dtype=torch.float32).reshape(2, 3, 4)
    actual = resize_action_backbone_tensor(source, (3, 2, 2))

    expected = source
    for dimension, size in enumerate((3, 2, 2)):
        if expected.shape[dimension] == size:
            continue
        permutation = [
            index for index in range(expected.ndim) if index != dimension
        ] + [dimension]
        inverse = [0] * expected.ndim
        for index, value in enumerate(permutation):
            inverse[value] = index
        value = expected.permute(*permutation).contiguous()
        prefix = value.shape[:-1]
        value = torch.nn.functional.interpolate(
            value.reshape(-1, 1, value.shape[-1]),
            size=size,
            mode="linear",
            align_corners=True,
        ).reshape(*prefix, size)
        expected = value.permute(*inverse).contiguous()
    expected *= (4.0 / 2.0) ** 0.5
    torch.testing.assert_close(actual, expected, atol=0, rtol=0)
