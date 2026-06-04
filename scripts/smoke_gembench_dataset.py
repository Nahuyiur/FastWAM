import argparse
import os
from pathlib import Path

import numpy as np
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parents[1]
import sys
if str(PROJECT_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT / "src"))

from fastwam.datasets.gembench.dataset import GEMBenchKeystepsDataset
from fastwam.utils.video_io import save_mp4


def _balanced_sample_indices(dataset: GEMBenchKeystepsDataset, limit: int) -> list[int]:
    selected: list[int] = []
    seen_taskvars = set()
    for idx, (taskvar, _) in enumerate(dataset.index):
        if taskvar in seen_taskvars:
            continue
        selected.append(idx)
        seen_taskvars.add(taskvar)
        if len(selected) >= limit:
            return selected
    selected_set = set(selected)
    for idx in range(len(dataset)):
        if idx in selected_set:
            continue
        selected.append(idx)
        if len(selected) >= limit:
            return selected
    return selected


def _frame_to_pil(video, t: int) -> Image.Image:
    frame = ((video[:, t].detach().cpu().clamp(-1, 1) + 1.0) * 127.5).to(dtype=None).numpy()
    frame = np.asarray(frame.transpose(1, 2, 0), dtype=np.uint8)
    return Image.fromarray(frame)


def main():
    parser = argparse.ArgumentParser(description="Smoke-test GEMBenchKeystepsDataset shapes and optional video output.")
    parser.add_argument("--root", default=os.environ.get("GEMBENCH_ROOT", "/mnt/yuhan/datasets/GEMBench"))
    parser.add_argument("--taskvars", default="close_fridge+0,push_button+0,open_door+0")
    parser.add_argument("--num-samples", type=int, default=3)
    parser.add_argument("--cache-dir", default="data/text_embeds_cache/gembench_keysteps_bbox")
    parser.add_argument("--allow-missing-text-embeds", action="store_true")
    parser.add_argument("--output-video", default=None)
    args = parser.parse_args()

    dataset = GEMBenchKeystepsDataset(
        root=args.root,
        taskvars=args.taskvars,
        max_episodes_per_taskvar=max(args.num_samples, 1),
        text_embedding_cache_dir=args.cache_dir,
        allow_missing_text_embeds=args.allow_missing_text_embeds,
        val_set_proportion=0.0,
        is_training_set=True,
    )
    print(f"dataset_len={len(dataset)} taskvars={dataset.taskvars}")
    first_video = None
    sample_indices = _balanced_sample_indices(dataset, min(args.num_samples, len(dataset)))
    for i, dataset_idx in enumerate(sample_indices):
        sample = dataset[dataset_idx]
        if first_video is None:
            first_video = sample["video"]
        print(
            "sample",
            i,
            "dataset_idx=",
            dataset_idx,
            "taskvar=",
            sample["taskvar"],
            "episode=",
            sample["episode_key"],
            "video=",
            tuple(sample["video"].shape),
            "action=",
            tuple(sample["action"].shape),
            "proprio=",
            tuple(sample["proprio"].shape),
            "context=",
            tuple(sample["context"].shape),
            "context_mask_sum=",
            int(sample["context_mask"].sum().item()),
        )
    if args.output_video and first_video is not None:
        frames = [_frame_to_pil(first_video, t) for t in range(first_video.shape[1])]
        save_mp4(frames, args.output_video, fps=4)
        print(f"saved_video={args.output_video}")


if __name__ == "__main__":
    main()
