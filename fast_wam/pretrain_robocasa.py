"""Megatron Core training entry for the RoboCasa ACG Fast-WAM baseline."""

from __future__ import annotations

import functools
from pathlib import Path

import torch
from torch.utils.data._utils.collate import default_collate

from megatron.core.enums import ModelType
from megatron.training import inprocess_restart, pretrain, print_rank_0
from megatron.training.arguments import parse_and_validate_args

from . import pretrain as common
from .train.robocasa_data import build_robocasa_datasets


def robocasa_extra_args(parser):
    parser = common.fast_wam_extra_args(parser)
    group = parser.add_argument_group(title="fast-wam-robocasa")
    group.add_argument(
        "--fast-wam-robocasa-repo-root",
        type=str,
        default=str(Path(__file__).resolve().parents[1]),
    )
    group.add_argument(
        "--fast-wam-robocasa-task-config",
        type=str,
        default="robocasa_acg_v1_fastwam_8gpu",
    )
    group.add_argument("--fast-wam-robocasa-train-latent-cache", type=str, default=None)
    group.add_argument("--fast-wam-robocasa-valid-latent-cache", type=str, default=None)
    group.add_argument("--fast-wam-robocasa-train-webdataset", type=str, default=None)
    group.add_argument("--fast-wam-robocasa-valid-webdataset", type=str, default=None)
    group.add_argument("--fast-wam-robocasa-train-index-file", type=str, default=None)
    group.add_argument("--fast-wam-robocasa-valid-index-file", type=str, default=None)
    return parser


def train_valid_test_datasets_provider(train_val_test_num_samples):
    del train_val_test_num_samples
    args = common.get_args()
    train_dataset, valid_dataset, cfg = build_robocasa_datasets(
        args.fast_wam_robocasa_repo_root,
        args.fast_wam_robocasa_task_config,
        train_latent_cache=args.fast_wam_robocasa_train_latent_cache,
        valid_latent_cache=args.fast_wam_robocasa_valid_latent_cache,
        train_webdataset=args.fast_wam_robocasa_train_webdataset,
        valid_webdataset=args.fast_wam_robocasa_valid_webdataset,
        train_index_file=args.fast_wam_robocasa_train_index_file,
        valid_index_file=args.fast_wam_robocasa_valid_index_file,
    )
    expected_action = int(args.fast_wam_action_dim)
    expected_proprio = int(args.fast_wam_proprio_dim)
    sample = train_dataset[0]
    if sample["action"].shape[-1] != expected_action:
        raise ValueError(
            f"RoboCasa action contract mismatch: dataset={sample['action'].shape[-1]} "
            f"model={expected_action}"
        )
    if sample["proprio"].shape[-1] != expected_proprio:
        raise ValueError(
            f"RoboCasa proprio contract mismatch: dataset={sample['proprio'].shape[-1]} "
            f"model={expected_proprio}"
        )
    print_rank_0(
        "[Fast-WAM][RoboCasa] "
        f"train_windows={len(train_dataset)} valid_windows={len(valid_dataset)} "
        f"task_config={args.fast_wam_robocasa_task_config} "
        f"action_dim={expected_action} proprio_dim={expected_proprio} "
        f"cameras={list(cfg.data.train.camera_keys)} "
        f"input={'webdataset' if args.fast_wam_robocasa_train_webdataset else 'ordinary'} "
        f"latents={'cached' if ('input_latents' in sample) else 'online'}"
    )
    return (
        common._DatasetView(train_dataset, "train"),
        common._DatasetView(valid_dataset, "valid"),
        None,
    )


def _patch_data_loader() -> None:
    # The baseline dataset already returns fixed-shape tensors compatible with
    # default_collate; reuse the accepted Megatron sampler/loader semantics.
    common.libero_collate = default_collate
    common._patch_data_loader()


def _patch_torch_optimizer_dcp_load() -> None:
    """Initialize native Torch Adam state before Megatron builds a load template.

    Upstream DistributedOptimizer assumes Apex/TE param-group steps exist when
    constructing the torch-dist sharded state.  With its supported native
    Torch fallback, a fresh optimizer has no per-parameter state yet and the
    upstream template builder asserts before it can load a checkpoint.
    """

    from megatron.core.optimizer.distrib_optimizer import DistributedOptimizer

    if getattr(DistributedOptimizer, "_fast_wam_torch_load_patch", False):
        return
    original = DistributedOptimizer.sharded_state_dict

    @functools.wraps(original)
    def wrapped(self, model_sharded_state_dict, is_loading=False, *args, **kwargs):
        if is_loading and len(self.optimizer.state) == 0:
            self._init_optimizer_states_with_dummy_values()
        return original(
            self,
            model_sharded_state_dict,
            is_loading,
            *args,
            **kwargs,
        )

    DistributedOptimizer.sharded_state_dict = wrapped
    DistributedOptimizer._fast_wam_torch_load_patch = True


if __name__ == "__main__":
    train_valid_test_datasets_provider.is_distributed = True
    pretrain_fn, store = inprocess_restart.maybe_wrap_for_inprocess_restart(pretrain)
    parse_and_validate_args(
        extra_args_provider=robocasa_extra_args,
        args_defaults={"tokenizer_type": "NullTokenizer"},
    )
    _patch_torch_optimizer_dcp_load()
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

    try:
        patched_pretrain(
            train_valid_test_datasets_provider,
            common.model_provider,
            ModelType.encoder_or_decoder,
            common.forward_step,
            store=store,
        )
    finally:
        if torch.distributed.is_available() and torch.distributed.is_initialized():
            torch.distributed.destroy_process_group()
