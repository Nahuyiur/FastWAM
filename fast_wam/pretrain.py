"""Megatron Core entry for the official Fast-WAM LIBERO training recipe."""

from __future__ import annotations

import functools
import hashlib
import os
from dataclasses import replace
from functools import partial
from typing import Literal

import torch
import torch.distributed as dist
from torch.utils.data import Dataset

from megatron.core import parallel_state
from megatron.core.enums import ModelType
from megatron.training import (
    get_args,
    get_timers,
    inprocess_restart,
    pretrain,
    print_rank_0,
)
from megatron.training.arguments import (
    core_transformer_config_from_args,
    parse_and_validate_args,
)
from megatron.training.utils import average_losses_across_data_parallel_group

from .checkpoint import load_megatron_dcp
from .config import FastWAMConfig
from .model import FastWAMModel
from .train.data import LiberoTrainingDataset, libero_collate
from .train.sampler import (
    OfficialEpochBatchSampler,
    OfficialValidationBatchSampler,
)
from .train.vae import WanVideoVAE38Encoder


_VAE: WanVideoVAE38Encoder | None = None
_STOCHASTIC_KEY: tuple[bool, int] | None = None
_STOCHASTIC_MICROBATCH = 0


class _DatasetView(Dataset):
    """Tag a shared dataset as train or validation for the loader patch."""

    def __init__(
        self,
        dataset: LiberoTrainingDataset,
        role: Literal["train", "valid"],
    ):
        self.dataset = dataset
        self.fast_wam_role = role

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, index):
        return self.dataset[index]


class _FastWAMDataLoader(torch.utils.data.DataLoader):
    """Keep DataLoader bookkeeping RNG from perturbing diffusion RNG.

    The official trainer owns only a train DataLoader; validation samples are
    drawn directly.  Megatron owns both train and validation DataLoaders, and
    constructing each iterator normally draws a ``base_seed`` from the global
    Torch generator.  Validation therefore uses a private generator.  A
    mid-epoch resume also uses a private generator for its first train iterator
    because the uninterrupted run already created that iterator before the
    checkpoint.  Later epoch transitions go back to the global generator,
    exactly like the official train DataLoader.
    """

    def __init__(
        self,
        *args,
        first_iterator_generator: torch.Generator | None = None,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self._first_iterator_generator = first_iterator_generator

    def __iter__(self):
        if self._first_iterator_generator is None:
            return super().__iter__()
        original_generator = self.generator
        self.generator = self._first_iterator_generator
        try:
            iterator = super().__iter__()
        finally:
            self.generator = original_generator
            self._first_iterator_generator = None
        return iterator


def fast_wam_extra_args(parser):
    group = parser.add_argument_group(title="fast-wam-training")
    group.add_argument("--fast-wam-initial-dcp", type=str, default=None)
    group.add_argument("--fast-wam-dataset-root", type=str, required=True)
    group.add_argument("--fast-wam-stats-path", type=str, required=True)
    group.add_argument("--fast-wam-text-cache", type=str, required=True)
    group.add_argument("--fast-wam-vae-checkpoint", type=str, default=None)
    group.add_argument(
        "--fast-wam-latent-cache",
        type=str,
        default=None,
        help="Validated offline BF16 latent cache; bypasses video decode and VAE.",
    )
    group.add_argument(
        "--fast-wam-attention-backend",
        choices=("flex", "sdpa", "structured_sdpa"),
        default="sdpa",
    )
    group.add_argument(
        "--fast-wam-kernel-mode",
        choices=("reference", "optimized"),
        default="reference",
        help=(
            "Use BF16-native norms, complex64 RoPE, and bias-GELU fusion. "
            "The reference mode preserves the accepted implementation."
        ),
    )
    group.add_argument(
        "--fast-wam-joint-action-video-attention",
        action="store_true",
        help=(
            "Use RoboCasa FastWAMJoint semantics: action queries attend to all "
            "jointly denoised video tokens instead of only the clean first frame."
        ),
    )
    group.add_argument("--fast-wam-parquet-cache-size", type=int, default=4)
    group.add_argument(
        "--fast-wam-action-dim",
        type=int,
        default=7,
        help="Action dimension. LIBERO uses 7; RoboCasa ACG uses 12.",
    )
    group.add_argument(
        "--fast-wam-proprio-dim",
        type=int,
        default=8,
        help="Proprioception dimension. LIBERO uses 8; RoboCasa ACG uses 16.",
    )
    group.add_argument(
        "--fast-wam-log-batch-indices",
        action="store_true",
        help="Log per-rank sample indices and stochastic seeds for parity smoke tests.",
    )
    group.add_argument(
        "--fast-wam-log-full-model-digest",
        action="store_true",
        help="Hash all live parameters on rank 0; expensive and intended only for DCP diagnosis.",
    )
    group.add_argument(
        "--fast-wam-debug-repeat-forward",
        action="store_true",
        help="Run one no-grad cold probe before the real forward to diagnose backend state.",
    )
    group.add_argument(
        "--fast-wam-no-validate-release",
        action="store_false",
        dest="fast_wam_validate_release",
    )
    group.set_defaults(fast_wam_validate_release=True)
    return parser


def model_config_from_args(args=None) -> FastWAMConfig:
    """Build the model contract without changing the upstream LIBERO defaults."""

    args = get_args() if args is None else args
    base = FastWAMConfig()
    return replace(
        base,
        action=replace(base.action, action_dim=int(args.fast_wam_action_dim)),
        proprio_dim=int(args.fast_wam_proprio_dim),
        training_attention_backend=args.fast_wam_attention_backend,
        training_kernel_mode=args.fast_wam_kernel_mode,
        joint_action_video_attention=bool(
            args.fast_wam_joint_action_video_attention
        ),
    )


def _validate_args(args) -> None:
    unsupported = []
    if args.pipeline_model_parallel_size != 1:
        unsupported.append("--pipeline-model-parallel-size")
    if args.context_parallel_size != 1:
        unsupported.append("--context-parallel-size")
    if getattr(args, "sequence_parallel", False):
        unsupported.append("--sequence-parallel")
    if getattr(args, "virtual_pipeline_model_parallel_size", None) is not None:
        unsupported.append("--virtual-pipeline-model-parallel-size")
    if getattr(args, "fp8", None) is not None:
        unsupported.append("--fp8")
    if getattr(args, "fp4", None) is not None:
        unsupported.append("--fp4")
    if getattr(args, "recompute_granularity", None) is not None:
        unsupported.append("--recompute-granularity")
    if unsupported:
        raise NotImplementedError(
            "Fast-WAM training currently supports TP+DP only; unsupported: "
            + ", ".join(unsupported)
        )
    if not args.bf16:
        raise ValueError("The official Fast-WAM recipe requires --bf16")
    if args.seed != 42:
        raise ValueError("The deterministic Fast-WAM recipe requires --seed 42")
    if args.fast_wam_latent_cache is None and args.fast_wam_vae_checkpoint is None:
        raise ValueError(
            "Provide --fast-wam-latent-cache or --fast-wam-vae-checkpoint"
        )


def model_provider(pre_process=True, post_process=True, **kwargs):
    del kwargs
    if not pre_process or not post_process:
        raise NotImplementedError("Fast-WAM pipeline parallelism is not implemented")
    args = get_args()
    _validate_args(args)
    # PPU NCCL initializes process-group communicators lazily.  Initializing
    # the distributed-optimizer group after a 6B model, FP32 gradient buffer,
    # optimizer state, activations, and VAE are resident can fail because the
    # communicator itself needs device memory.  Warm the exact group used by
    # reduce-scatter before allocating the model; this changes neither the
    # collective nor its arithmetic.
    optimizer_group = (
        parallel_state.get_intra_distributed_optimizer_instance_group()
    )
    dist.barrier(
        group=optimizer_group,
        device_ids=[torch.cuda.current_device()],
    )
    torch.cuda.synchronize()
    config = model_config_from_args(args)
    megatron_config = core_transformer_config_from_args(args)
    model = FastWAMModel(config, megatron_config)
    # A normal Megatron --load resume supersedes the deterministic initial DCP.
    if args.load is None:
        if not args.fast_wam_initial_dcp:
            raise ValueError(
                "--fast-wam-initial-dcp is required for a fresh training run"
            )
        load_megatron_dcp(model, args.fast_wam_initial_dcp)
        print_rank_0(
            f"[Fast-WAM] loaded initial Wan->Action DCP: {args.fast_wam_initial_dcp}"
        )
    else:
        print_rank_0(
            f"[Fast-WAM] Megatron resume requested from {args.load}; "
            "skipping initial DCP load"
        )
    total = sum(parameter.numel() for parameter in model.parameters())
    print_rank_0(
        f"[Fast-WAM] model params(local shards included)={total / 1e9:.3f}B, "
        f"attention={args.fast_wam_attention_backend}, "
        f"kernels={args.fast_wam_kernel_mode}"
    )
    # Upstream calls set_global_seed(42) after constructing/loading the full
    # model and offsets it by process rank.  A DP rank is the corresponding
    # logical process when TP>1; all TP ranks must draw the same noise.
    logical_rank = parallel_state.get_data_parallel_rank(
        with_context_parallel=False
    )
    torch.manual_seed(args.seed + logical_rank)
    torch.cuda.manual_seed(args.seed + logical_rank)
    return model


def _vae() -> WanVideoVAE38Encoder | None:
    global _VAE
    if parallel_state.get_tensor_model_parallel_rank() != 0:
        return None
    if _VAE is None:
        args = get_args()
        device_index = torch.cuda.current_device()
        # The official trainer loads VAE weights before resetting its training
        # RNG.  Our leader-only lazy load must therefore not advance the noise
        # stream established in model_provider.
        with torch.random.fork_rng(devices=[device_index]):
            _VAE = WanVideoVAE38Encoder.from_pretrained(
                args.fast_wam_vae_checkpoint,
                device=torch.device("cuda", device_index),
                dtype=torch.bfloat16,
            )
        print_rank_0(
            f"[Fast-WAM] loaded frozen Wan VAE encoder from "
            f"{args.fast_wam_vae_checkpoint}"
        )
    return _VAE


@torch.no_grad()
def _encode_video_for_tp(video: torch.Tensor) -> torch.Tensor:
    group = parallel_state.get_tensor_model_parallel_group()
    tp_rank = parallel_state.get_tensor_model_parallel_rank()
    if tp_rank == 0:
        encoder = _vae()
        assert encoder is not None
        latents = encoder.encode_normalized_video(
            video.cuda(non_blocking=True)
        ).contiguous()
    else:
        batch, _, frames, height, width = video.shape
        latent_frames = (frames - 1) // 4 + 1
        spatial_factor = WanVideoVAE38Encoder.spatial_downsample_factor
        latents = torch.empty(
            (
                batch,
                48,
                latent_frames,
                height // spatial_factor,
                width // spatial_factor,
            ),
            device=torch.cuda.current_device(),
            dtype=torch.bfloat16,
        )
    if group.size() > 1:
        dist.broadcast(
            latents,
            src=dist.get_global_rank(group, 0),
            group=group,
        )
    return latents


@torch.no_grad()
def _cached_latents_for_tp(value: torch.Tensor) -> torch.Tensor:
    group = parallel_state.get_tensor_model_parallel_group()
    tp_rank = parallel_state.get_tensor_model_parallel_rank()
    if tp_rank == 0:
        latents = value.cuda(non_blocking=True).contiguous()
    else:
        latents = torch.empty(
            value.shape,
            device=torch.cuda.current_device(),
            dtype=torch.bfloat16,
        )
    if group.size() > 1:
        dist.broadcast(
            latents,
            src=dist.get_global_rank(group, 0),
            group=group,
        )
    return latents


def get_batch(data_iterator):
    data = next(data_iterator) if data_iterator is not None else None
    if data is None:
        return None
    context_mask = data["context_mask"]
    context_is_dense = bool(context_mask.all())
    if "input_latents" in data:
        input_latents = _cached_latents_for_tp(data["input_latents"])
    else:
        input_latents = _encode_video_for_tp(data["video"])
    return {
        "_sample_indices": data["idx"],
        "input_latents": input_latents,
        "action": data["action"].cuda(non_blocking=True),
        "context": data["context"].cuda(non_blocking=True),
        "context_mask": context_mask.cuda(non_blocking=True),
        "context_is_dense": context_is_dense,
        "proprio": data["proprio"].cuda(non_blocking=True),
        "image_is_pad": data["image_is_pad"].cuda(non_blocking=True),
        "action_is_pad": data["action_is_pad"].cuda(non_blocking=True),
    }


def loss_func(output_tensor):
    loss, metrics = output_tensor
    averaged = average_losses_across_data_parallel_group(
        [
            loss,
            metrics["loss_video"],
            metrics["loss_action"],
        ]
    )
    reduced = {
        "loss": averaged[0].float().view(1),
        "video loss": averaged[1].float().view(1),
        "action loss": averaged[2].float().view(1),
    }
    return loss.float(), reduced


def _next_stochastic_seed(*, training: bool) -> int:
    """Return a resume-stable per-step/per-DP/per-microbatch seed.

    Official Fast-WAM samples four tensors from the rank-local CUDA generator.
    Megatron and DeepSpeed checkpoint/restore that generator at different
    points relative to DataLoader iterator construction.  Keying a private
    generator by the logical training position preserves the exact four-draw
    distribution and order while making a resumed step independent of those
    framework bookkeeping draws.
    """

    global _STOCHASTIC_KEY, _STOCHASTIC_MICROBATCH
    args = get_args()
    iteration = int(args.curr_iteration)
    key = (training, iteration)
    if key != _STOCHASTIC_KEY:
        _STOCHASTIC_KEY = key
        _STOCHASTIC_MICROBATCH = 0
    else:
        _STOCHASTIC_MICROBATCH += 1
    dp_rank = parallel_state.get_data_parallel_rank(
        with_context_parallel=False
    )
    phase_offset = 0 if training else 1_000_000_000_000_000
    return (
        int(args.seed)
        + phase_offset
        + iteration * 1_000_000_000
        + _STOCHASTIC_MICROBATCH * 1_000_000
        + dp_rank
    )


def _parameter_probe(model) -> str:
    """Return a cheap deterministic probe of the live BF16 model parameters."""

    probes = []
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad or parameter.numel() == 0:
            continue
        flat = parameter.detach().view(-1)
        positions = sorted({0, flat.numel() // 2, flat.numel() - 1})
        values = [float(flat[position].float().item()) for position in positions]
        probes.append(f"{name}:{values}")
        if len(probes) == 8:
            break
    return ";".join(probes)


def _tensor_digest(tensor: torch.Tensor) -> str:
    value = tensor.detach().cpu().contiguous().view(torch.uint8).numpy()
    return hashlib.sha256(value.tobytes()).hexdigest()


def _model_digest(model) -> str:
    digest = hashlib.sha256()
    for name, parameter in model.named_parameters():
        digest.update(name.encode("utf-8"))
        value = parameter.detach().cpu().contiguous().view(torch.uint8).numpy()
        digest.update(value.tobytes())
    return digest.hexdigest()


def forward_step(data_iterator, model):
    timers = get_timers()
    timers("batch-generator", log_level=2).start()
    batch = get_batch(data_iterator)
    timers("batch-generator").stop()
    if batch is None:
        return None, None
    sample_indices = batch.pop("_sample_indices")
    stochastic_seed = _next_stochastic_seed(
        training=bool(model.training)
    )
    if get_args().fast_wam_log_batch_indices:
        print(
            f"[Fast-WAM][rank={dist.get_rank()}] "
            f"iteration={get_args().curr_iteration} "
            f"training={bool(model.training)} "
            f"seed={stochastic_seed} "
            f"indices={sample_indices.tolist()}",
            flush=True,
        )
        if dist.get_rank() == 0:
            print(
                f"[Fast-WAM][parameter-probe] "
                f"iteration={get_args().curr_iteration} "
                f"training={bool(model.training)} "
                f"{_parameter_probe(model)}",
                flush=True,
            )
        print(
            f"[Fast-WAM][batch-digest][rank={dist.get_rank()}] "
            f"iteration={get_args().curr_iteration} "
            f"latents={_tensor_digest(batch['input_latents'])} "
            f"action={_tensor_digest(batch['action'])} "
            f"context={_tensor_digest(batch['context'])} "
            f"proprio={_tensor_digest(batch['proprio'])}",
            flush=True,
        )
        if (
            dist.get_rank() == 0
            and bool(model.training)
            and get_args().fast_wam_log_full_model_digest
        ):
            print(
                f"[Fast-WAM][model-digest] "
                f"iteration={get_args().curr_iteration} "
                f"sha256={_model_digest(model)}",
                flush=True,
            )
    batch["stochastic_seed"] = stochastic_seed
    batch["return_debug_tensors"] = bool(get_args().fast_wam_log_batch_indices)
    if bool(model.training) and get_args().fast_wam_debug_repeat_forward:
        probe_batch = dict(batch)
        probe_batch["return_debug_tensors"] = False
        with torch.no_grad():
            probe_loss, probe_metrics = model(**probe_batch)
        print(
            f"[Fast-WAM][repeat-forward-cold][rank={dist.get_rank()}] "
            f"iteration={get_args().curr_iteration} "
            f"loss={float(probe_loss.detach()):.9f} "
            f"video={float(probe_metrics['loss_video'].detach()):.9f} "
            f"action={float(probe_metrics['loss_action'].detach()):.9f}",
            flush=True,
        )
    output = model(**batch)
    debug_digests = output[1].pop("_debug_stochastic_digests", None)
    if debug_digests is not None:
        print(
            f"[Fast-WAM][stochastic-digest][rank={dist.get_rank()}] "
            f"iteration={get_args().curr_iteration} "
            + " ".join(
                f"{name}={digest}"
                for name, digest in debug_digests.items()
            ),
            flush=True,
        )
        print(
            f"[Fast-WAM][local-loss][rank={dist.get_rank()}] "
            f"iteration={get_args().curr_iteration} "
            f"loss={float(output[0].detach()):.9f} "
            f"video={float(output[1]['loss_video'].detach()):.9f} "
            f"action={float(output[1]['loss_action'].detach()):.9f}",
            flush=True,
        )
    return output, partial(loss_func)


def train_valid_test_datasets_provider(train_val_test_num_samples):
    del train_val_test_num_samples
    args = get_args()
    config = model_config_from_args(args)
    dataset = LiberoTrainingDataset(
        root=args.fast_wam_dataset_root,
        stats_path=args.fast_wam_stats_path,
        text_cache_dir=args.fast_wam_text_cache,
        config=config,
        parquet_cache_size=args.fast_wam_parquet_cache_size,
        validate_official_release=args.fast_wam_validate_release,
        latent_cache=args.fast_wam_latent_cache,
    )
    print_rank_0(
        f"[Fast-WAM] LIBERO dataset: frames={len(dataset)}, "
        f"episodes={len(dataset.episodes)}, "
        f"latents={'cached' if dataset.latent_cache is not None else 'online'}"
    )
    # One validation loss every 200 steps preserves the upstream training RNG
    # consumption. Video/action rollout metrics are handled by the final 2k gate.
    return _DatasetView(dataset, "train"), _DatasetView(dataset, "valid"), None


def _build_fast_wam_loader(dataset, consumed_samples):
    if dataset is None:
        return None
    args = get_args()
    dp_rank = parallel_state.get_data_parallel_rank(with_context_parallel=False)
    dp_size = parallel_state.get_data_parallel_world_size(with_context_parallel=False)
    role = getattr(dataset, "fast_wam_role", "train")
    loader_generator = None
    first_iterator_generator = None
    if role == "valid":
        batch_sampler = OfficialValidationBatchSampler(
            dataset_size=len(dataset),
            consumed_samples=consumed_samples,
            data_parallel_rank=dp_rank,
            data_parallel_size=dp_size,
            eval_interval=args.eval_interval,
        )
        # Upstream validation samples directly from the dataset and therefore
        # never consumes the training generator for DataLoader bookkeeping.
        loader_generator = torch.Generator(device="cpu").manual_seed(
            args.seed + 100_000 + dp_rank
        )
    elif role == "train":
        batch_sampler = OfficialEpochBatchSampler(
            dataset_size=len(dataset),
            consumed_samples=consumed_samples,
            micro_batch_size=args.micro_batch_size,
            data_parallel_rank=dp_rank,
            data_parallel_size=dp_size,
            seed=args.seed,
        )
        # When resuming within an epoch, the uninterrupted run's iterator was
        # created before the checkpoint.  Suppress only the resume-time extra
        # base-seed draw; subsequent epoch transitions use global RNG as in the
        # official DataLoader.
        if consumed_samples % batch_sampler.padded_epoch_size:
            first_iterator_generator = torch.Generator(
                device="cpu"
            ).manual_seed(args.seed + 200_000 + dp_rank)
    else:
        raise ValueError(f"Unknown Fast-WAM dataset role {role!r}")
    return _FastWAMDataLoader(
        dataset,
        batch_sampler=batch_sampler,
        num_workers=args.num_workers,
        pin_memory=True,
        persistent_workers=args.num_workers > 0,
        collate_fn=libero_collate,
        generator=loader_generator,
        first_iterator_generator=first_iterator_generator,
    )


def _patch_data_loader() -> None:
    import megatron.training.datasets.data_samplers as samplers_module

    samplers_module.build_pretraining_data_loader = _build_fast_wam_loader
    try:
        import megatron.training.training as training_module

        if hasattr(training_module, "build_pretraining_data_loader"):
            training_module.build_pretraining_data_loader = _build_fast_wam_loader
    except ImportError:
        pass


if __name__ == "__main__":
    train_valid_test_datasets_provider.is_distributed = True
    pretrain_fn, store = inprocess_restart.maybe_wrap_for_inprocess_restart(pretrain)
    parse_and_validate_args(
        extra_args_provider=fast_wam_extra_args,
        args_defaults={"tokenizer_type": "NullTokenizer"},
    )
    original_pretrain = pretrain_fn

    @functools.wraps(original_pretrain)
    def patched_pretrain(datasets_provider, *args, **kwargs):
        patched = [False]

        def wrapped_provider(*provider_args, **provider_kwargs):
            if not patched[0]:
                _patch_data_loader()
                patched[0] = True
            return datasets_provider(*provider_args, **provider_kwargs)

        wrapped_provider.is_distributed = getattr(
            datasets_provider,
            "is_distributed",
            False,
        )
        return original_pretrain(wrapped_provider, *args, **kwargs)

    patched_pretrain(
        train_valid_test_datasets_provider,
        model_provider,
        ModelType.encoder_or_decoder,
        forward_step,
        store=store,
    )
