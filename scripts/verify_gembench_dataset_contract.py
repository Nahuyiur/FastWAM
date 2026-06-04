import argparse
import hashlib
import os
from pathlib import Path

import torch
from torch.utils.data import DataLoader

PROJECT_ROOT = Path(__file__).resolve().parents[1]
import sys

if str(PROJECT_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT / "src"))

from fastwam.datasets.gembench import GEMBenchKeystepsDataset
from fastwam.datasets.gembench.instructions import instruction_for_taskvar
from fastwam.datasets.lerobot.robot_video_dataset import DEFAULT_PROMPT
from fastwam.trainer import Wan22Trainer


EXPECTED_KEYS = {
    "video",
    "action",
    "proprio",
    "prompt",
    "context",
    "context_mask",
    "image_is_pad",
    "action_is_pad",
    "proprio_is_pad",
    "action_dim_is_pad",
    "proprio_dim_is_pad",
    "taskvar",
    "episode_key",
}


def _assert(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def _as_taskvar_arg(taskvars: str | None) -> str | None:
    if taskvars is None:
        return None
    taskvars = taskvars.strip()
    if taskvars.lower() in {"", "all", "local"}:
        return None
    return taskvars or None


def _denormalize_sample(dataset: GEMBenchKeystepsDataset, sample: dict) -> tuple[torch.Tensor, torch.Tensor]:
    batch = {
        "action": sample["action"].unsqueeze(0).to(torch.float32),
        "state": sample["proprio"].unsqueeze(0).to(torch.float32),
    }
    batch = dataset.processor.action_state_merger.backward(batch)
    batch = dataset.processor.normalizer.backward(batch)
    action = batch["action"]["default"].squeeze(0)
    state = batch["state"]["default"].squeeze(0)
    return action, state


def _check_all_text_cache(dataset: GEMBenchKeystepsDataset) -> None:
    _assert(dataset.text_embedding_cache_dir is not None, "text_embedding_cache_dir must be set")
    cache_dir = Path(dataset.text_embedding_cache_dir)
    missing: list[Path] = []
    checked = 0
    seen_prompts = set()
    for taskvar in dataset.taskvars:
        instruction = instruction_for_taskvar(
            taskvar,
            instruction_map=dataset.instruction_map,
            instruction_index=dataset.instruction_index,
        )
        prompt = DEFAULT_PROMPT.format(task=instruction)
        if prompt in seen_prompts:
            continue
        seen_prompts.add(prompt)
        hashed = hashlib.sha256(prompt.encode("utf-8")).hexdigest()
        path = cache_dir / f"{hashed}.t5_len{dataset.context_len}.{dataset.text_encoder_id}.pt"
        if not path.exists():
            missing.append(path)
        checked += 1
    _assert(not missing, "missing text cache files: " + ", ".join(str(p) for p in missing[:8]))
    print(f"text_cache_ok prompts={checked} cache_dir={cache_dir}")


def _check_sample(dataset: GEMBenchKeystepsDataset, dataset_idx: int, *, atol: float) -> None:
    taskvar, episode_key = dataset.index[dataset_idx]
    episode = dataset.store.get(taskvar, episode_key)
    raw_action, raw_proprio = dataset._episode_action_proprio(episode)
    sample = dataset[dataset_idx]

    missing = EXPECTED_KEYS.difference(sample)
    _assert(not missing, f"sample missing keys: {sorted(missing)}")
    _assert(sample["taskvar"] == taskvar, f"taskvar mismatch at idx={dataset_idx}")
    _assert(sample["episode_key"] == episode_key.decode("ascii"), f"episode_key mismatch at idx={dataset_idx}")
    _assert(tuple(sample["video"].shape) == (3, dataset.num_video_frames, dataset.video_size[0], dataset.video_size[1]), tuple(sample["video"].shape))
    _assert(tuple(sample["action"].shape) == (dataset.action_horizon, 8), tuple(sample["action"].shape))
    _assert(tuple(sample["proprio"].shape) == (dataset.action_horizon, 8), tuple(sample["proprio"].shape))
    _assert(tuple(sample["context"].shape) == (dataset.context_len, dataset.text_dim), tuple(sample["context"].shape))
    _assert(tuple(sample["context_mask"].shape) == (dataset.context_len,), tuple(sample["context_mask"].shape))
    _assert(tuple(sample["image_is_pad"].shape) == (dataset.num_video_frames,), tuple(sample["image_is_pad"].shape))
    _assert(tuple(sample["action_is_pad"].shape) == (dataset.action_horizon,), tuple(sample["action_is_pad"].shape))
    _assert(tuple(sample["proprio_is_pad"].shape) == (dataset.action_horizon,), tuple(sample["proprio_is_pad"].shape))
    _assert(sample["video"].dtype == torch.float32, sample["video"].dtype)
    _assert(sample["context_mask"].dtype == torch.bool, sample["context_mask"].dtype)
    _assert(sample["video"].min().item() >= -1.001 and sample["video"].max().item() <= 1.001, "video must be normalized to [-1, 1]")
    _assert(not bool(sample["image_is_pad"].any().item()), "GEMBench v1 should not pad image frames")
    _assert(not bool(sample["action_is_pad"].any().item()), "GEMBench v1 should not pad actions")
    _assert(not bool(sample["proprio_is_pad"].any().item()), "GEMBench v1 should not pad proprio")
    _assert(int(sample["context_mask"].sum().item()) == dataset.context_len, "FastWAM expects all text mask entries enabled after zeroing pads")

    denorm_action, denorm_proprio = _denormalize_sample(dataset, sample)
    raw_action_t = torch.as_tensor(raw_action, dtype=torch.float32)
    raw_proprio_t = torch.as_tensor(raw_proprio, dtype=torch.float32)
    action_err = (denorm_action - raw_action_t).abs().max().item()
    proprio_err = (denorm_proprio - raw_proprio_t).abs().max().item()
    _assert(action_err <= atol, f"action normalization round-trip error {action_err:.6g} > {atol}")
    _assert(proprio_err <= atol, f"proprio normalization round-trip error {proprio_err:.6g} > {atol}")

    eval_sample = Wan22Trainer._to_batched_eval_sample(sample)
    _assert(tuple(eval_sample["video"].shape) == (1, 3, dataset.num_video_frames, dataset.video_size[0], dataset.video_size[1]), tuple(eval_sample["video"].shape))
    _assert(eval_sample["action_horizon"] == dataset.action_horizon, eval_sample["action_horizon"])

    print(
        "sample_ok",
        f"idx={dataset_idx}",
        f"taskvar={taskvar}",
        f"episode={sample['episode_key']}",
        f"action_roundtrip={action_err:.3g}",
        f"proprio_roundtrip={proprio_err:.3g}",
    )


def _balanced_indices(dataset: GEMBenchKeystepsDataset, limit: int) -> list[int]:
    out: list[int] = []
    seen = set()
    for idx, (taskvar, _) in enumerate(dataset.index):
        if taskvar in seen:
            continue
        out.append(idx)
        seen.add(taskvar)
        if len(out) >= limit:
            return out
    used = set(out)
    for idx in range(len(dataset)):
        if idx in used:
            continue
        out.append(idx)
        if len(out) >= limit:
            return out
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="Verify GEMBenchKeystepsDataset against the FastWAM trainer contract.")
    parser.add_argument("--root", default=os.environ.get("GEMBENCH_ROOT", "/mnt/yuhan/datasets/GEMBench"))
    parser.add_argument("--taskvars", default=None, help="Comma-separated taskvars. Defaults to all complete local taskvars.")
    parser.add_argument("--cache-dir", default="data/text_embeds_cache/gembench_keysteps_bbox")
    parser.add_argument("--pretrained-norm-stats", default="data/gembench_keysteps_bbox_dataset_stats.json")
    parser.add_argument("--norm-default-mode", default="z-score")
    parser.add_argument("--stats-scan-limit", type=int, default=0)
    parser.add_argument("--num-samples", type=int, default=6)
    parser.add_argument("--max-episodes-per-taskvar", type=int, default=4)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--atol", type=float, default=1e-5)
    args = parser.parse_args()

    train = GEMBenchKeystepsDataset(
        root=args.root,
        taskvars=_as_taskvar_arg(args.taskvars),
        text_embedding_cache_dir=args.cache_dir,
        pretrained_norm_stats=args.pretrained_norm_stats,
        norm_default_mode=args.norm_default_mode,
        stats_scan_limit=args.stats_scan_limit,
        max_episodes_per_taskvar=args.max_episodes_per_taskvar,
        val_set_proportion=0.25,
        is_training_set=True,
    )
    val = GEMBenchKeystepsDataset(
        root=args.root,
        taskvars=_as_taskvar_arg(args.taskvars),
        text_embedding_cache_dir=args.cache_dir,
        pretrained_norm_stats=args.pretrained_norm_stats,
        norm_default_mode=args.norm_default_mode,
        stats_scan_limit=args.stats_scan_limit,
        max_episodes_per_taskvar=args.max_episodes_per_taskvar,
        val_set_proportion=0.25,
        is_training_set=False,
    )
    train_keys = set(train.index)
    val_keys = set(val.index)
    _assert(train_keys.isdisjoint(val_keys), "train/val deterministic split must be disjoint")
    print(f"split_ok train={len(train)} val={len(val)} taskvars={train.taskvars}")
    _check_all_text_cache(train)

    for idx in _balanced_indices(train, min(args.num_samples, len(train))):
        _check_sample(train, idx, atol=args.atol)

    loader = DataLoader(train, batch_size=args.batch_size, num_workers=args.num_workers, shuffle=False)
    batch = next(iter(loader))
    _assert(tuple(batch["video"].shape[1:]) == (3, train.num_video_frames, train.video_size[0], train.video_size[1]), tuple(batch["video"].shape))
    _assert(tuple(batch["action"].shape[1:]) == (train.action_horizon, 8), tuple(batch["action"].shape))
    _assert(tuple(batch["context"].shape[1:]) == (train.context_len, train.text_dim), tuple(batch["context"].shape))
    print(
        "dataloader_ok",
        f"batch_video={tuple(batch['video'].shape)}",
        f"batch_action={tuple(batch['action'].shape)}",
        f"num_workers={args.num_workers}",
    )
    print("contract_ok")


if __name__ == "__main__":
    main()
