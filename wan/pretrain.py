"""Megatron training entry for Wan FlowMatch SFT.

Inputs are pre-encoded Wan latents and text context. This mirrors DiffSynth's
Wan training loss while using Megatron's distributed training loop and DCP.
"""

from __future__ import annotations

import functools
import os
from functools import partial

import torch

from megatron.core.enums import ModelType
from megatron.training import get_args, get_timers, inprocess_restart, pretrain, print_rank_0
from megatron.training.arguments import core_transformer_config_from_args, parse_and_validate_args
from wan.data.dataset import WanJsonlDataset, WanOverfitDataset, wan_collate
from wan.model.checkpoint import load_official_wan_checkpoint
from wan.model.config import wan_config_from_args
from wan.model.wan_dit import WanFlowTrainingModel
from wan.patches import wan_extra_args

# pytorch-25.09 currently ships an nvidia-resiliency-ext build whose async
# checkpoint API may not match Megatron's NVRx integration. Follow DiT's guard.
try:
    from nvidia_resiliency_ext.checkpointing.async_ckpt.filesystem_async import (
        get_write_results_queue as _nvrx_get_write_results_queue,
    )

    _nvrx_get_write_results_queue
    _nvrx_compatible = True
except Exception:
    _nvrx_compatible = False

if not _nvrx_compatible:
    try:
        import megatron.core.dist_checkpointing.strategies.torch as _torch_strat

        _torch_strat.HAVE_NVRX = False
    except Exception:
        pass


def model_provider(pre_process=True, post_process=True, **kwargs):
    del kwargs
    args = get_args()
    _validate_parallel_args(args)
    cfg = wan_config_from_args(args)
    megatron_config = core_transformer_config_from_args(args)
    megatron_config.wan_attention_backend = args.wan_attention_backend
    megatron_config.wan_local_qkv = bool(args.wan_local_qkv)
    gradient_checkpointing = _wan_gradient_checkpointing_from_args(args)
    model = WanFlowTrainingModel(
        cfg=cfg,
        train_timesteps=args.wan_train_timesteps,
        sigma_shift=args.wan_sigma_shift,
        noise_scale=args.wan_noise_scale,
        min_timestep_boundary=args.wan_min_timestep_boundary,
        max_timestep_boundary=args.wan_max_timestep_boundary,
        disable_timestep_weight=args.wan_disable_timestep_weight,
        context_drop_prob=args.wan_context_drop_prob,
        gradient_checkpointing=gradient_checkpointing,
        megatron_config=megatron_config,
        pre_process=pre_process,
        post_process=post_process,
    )
    if gradient_checkpointing:
        print_rank_0(
            "[Wan] Activation recompute enabled: checkpointing each Wan DiT block "
            f"(wan_gradient_checkpointing={args.wan_gradient_checkpointing}, "
            f"recompute_granularity={getattr(args, 'recompute_granularity', None)})"
        )

    if args.wan_load_official_ckpt:
        missing, unexpected = load_official_wan_checkpoint(
            model,
            args.wan_load_official_ckpt,
            strict=args.wan_strict_load,
        )
        print_rank_0(
            f"[Wan] Loaded official checkpoint {args.wan_load_official_ckpt}; "
            f"missing={len(missing)}, unexpected={len(unexpected)}"
        )
        if missing[:8]:
            print_rank_0(f"[Wan] first missing keys: {missing[:8]}")
        if unexpected[:8]:
            print_rank_0(f"[Wan] first unexpected keys: {unexpected[:8]}")

    n_params = sum(p.numel() for p in model.parameters()) / 1e6
    print_rank_0(
        f"Wan model: preset={args.wan_preset}, params={n_params:.1f}M, "
        f"dim={cfg.dim}, layers={cfg.num_layers}, heads={cfg.num_heads}, "
        f"latent_channels={cfg.in_dim}, patch={cfg.patch_size}"
    )
    print_rank_0(
        f"[Wan] attention backend={args.wan_attention_backend}, "
        f"local_qkv={args.wan_local_qkv}, "
        f"cp_comm_type={getattr(megatron_config, 'cp_comm_type', None)}"
    )
    return model


def _validate_parallel_args(args):
    unsupported = []
    if getattr(args, "virtual_pipeline_model_parallel_size", None) is not None:
        unsupported.append("--virtual-pipeline-model-parallel-size")
    if getattr(args, "hierarchical_context_parallel_sizes", None):
        unsupported.append("--hierarchical-context-parallel-sizes")
    if getattr(args, "fp8", None) is not None:
        unsupported.append("--fp8")
    if getattr(args, "fp4", None) is not None:
        unsupported.append("--fp4")
    if getattr(args, "use_megatron_fsdp", False):
        unsupported.append("--use-megatron-fsdp")
    if getattr(args, "use_torch_fsdp2", False):
        unsupported.append("--use-torch-fsdp2")
    recompute_granularity = getattr(args, "recompute_granularity", None)
    if recompute_granularity not in (None, "full"):
        unsupported.append(f"--recompute-granularity={recompute_granularity}")
    if unsupported:
        raise NotImplementedError(
            "Wan Megatron port currently supports TP/SP/PP/CP/DDP/distributed optimizer/DCP/full block recompute, "
            f"but not {', '.join(unsupported)}. Refusing to run instead of silently "
            "falling back to incorrect full-replica behavior."
        )


def _wan_gradient_checkpointing_from_args(args):
    recompute_granularity = getattr(args, "recompute_granularity", None)
    return bool(args.wan_gradient_checkpointing or recompute_granularity == "full")


def get_batch(data_iterator):
    if data_iterator is not None:
        data = next(data_iterator)
    else:
        data = None
    if data is None:
        return None
    batch = {
        "input_latents": data["input_latents"].cuda(non_blocking=True),
        "context": data["context"].cuda(non_blocking=True),
        "context_mask": data["context_mask"].cuda(non_blocking=True),
        "fuse_vae_embedding_in_latents": bool(data.get("fuse_vae_embedding_in_latents", False)),
    }
    if data.get("first_frame_latents") is not None:
        batch["first_frame_latents"] = data["first_frame_latents"].cuda(non_blocking=True)
    return batch


def loss_func(output_tensor):
    loss = output_tensor.float()
    return loss, {"mse loss": loss.clone().detach().view(1)}


def forward_step(data_iterator, model):
    timers = get_timers()
    timers("batch-generator", log_level=2).start()
    batch = get_batch(data_iterator)
    timers("batch-generator").stop()
    if batch is None:
        return None, None
    loss = model(
        input_latents=batch["input_latents"],
        context=batch["context"],
        context_mask=batch["context_mask"],
        first_frame_latents=batch.get("first_frame_latents"),
        fuse_vae_embedding_in_latents=batch.get("fuse_vae_embedding_in_latents", False),
    )
    return loss, partial(loss_func)


def train_valid_test_datasets_provider(train_val_test_num_samples):
    args = get_args()
    num_samples = train_val_test_num_samples[0]
    if args.wan_data_path:
        train_ds = WanJsonlDataset(
            args.wan_data_path,
            num_samples=num_samples,
            shard_cache_size=args.wan_shard_cache_size,
        )
    else:
        if not args.wan_sample_path:
            raise ValueError("--wan-sample-path or --wan-data-path is required")
        train_ds = WanOverfitDataset(args.wan_sample_path, num_samples=num_samples)
    return train_ds, None, None


def _patch_data_loader_collate():
    import megatron.training.datasets.data_samplers as samplers_module

    _orig_build = samplers_module.build_pretraining_data_loader

    @functools.wraps(_orig_build)
    def _patched_build(dataset, consumed_samples):
        loader = _orig_build(dataset, consumed_samples)
        if loader is not None:
            loader.collate_fn = wan_collate
        return loader

    samplers_module.build_pretraining_data_loader = _patched_build
    try:
        import megatron.training.training as training_module

        if hasattr(training_module, "build_pretraining_data_loader"):
            training_module.build_pretraining_data_loader = _patched_build
    except ImportError:
        pass


if __name__ == "__main__":
    train_valid_test_datasets_provider.is_distributed = True

    # Keep proxy leakage from breaking wandb/HF downloads in pyxis jobs.
    for key in ("HTTP_PROXY", "HTTPS_PROXY", "ALL_PROXY", "NO_PROXY", "http_proxy", "https_proxy", "all_proxy", "no_proxy"):
        os.environ.pop(key, None)

    pretrain_fn, store = inprocess_restart.maybe_wrap_for_inprocess_restart(pretrain)
    parse_and_validate_args(
        extra_args_provider=wan_extra_args,
        args_defaults={"tokenizer_type": "NullTokenizer"},
    )

    _orig_pretrain = pretrain_fn

    def _patched_pretrain(datasets_provider, *args_p, **kwargs_p):
        patched = [False]
        orig_provider = datasets_provider

        def _wrapped_provider(*a, **kw):
            if not patched[0]:
                _patch_data_loader_collate()
                patched[0] = True
            return orig_provider(*a, **kw)

        _wrapped_provider.is_distributed = getattr(datasets_provider, "is_distributed", False)
        return _orig_pretrain(_wrapped_provider, *args_p, **kwargs_p)

    _patched_pretrain(
        train_valid_test_datasets_provider,
        model_provider,
        ModelType.encoder_or_decoder,
        forward_step,
        store=store,
    )
