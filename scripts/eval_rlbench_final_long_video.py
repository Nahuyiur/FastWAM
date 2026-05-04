#!/usr/bin/env python3
"""Generate long final-eval videos for the four RLBench FastWAM checkpoints.

This script is intentionally eval-only. It does not call the training loop and
does not modify the existing trainer/evaluate implementation.
"""

from __future__ import annotations

import argparse
import gc
import json
from pathlib import Path

import numpy as np
import torch
from hydra import compose, initialize
from hydra.utils import instantiate
from omegaconf import OmegaConf
from PIL import Image

from fastwam.runtime import _mixed_precision_to_model_dtype, _normalize_mixed_precision
from fastwam.trainer import Wan22Trainer
from fastwam.utils import misc
from fastwam.utils.config_resolvers import register_default_resolvers
from fastwam.utils.video_io import save_mp4


RLBENCH_TASKS = [
    "rlbench_shape_3cam224_1e-4",
    "rlbench_original_3cam224_1e-4",
    "rlbench_color_3cam224_1e-4",
    "rlbench_color_shape_3cam224_1e-4",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run long-video eval from final RLBench FastWAM checkpoints."
    )
    parser.add_argument(
        "--tasks",
        nargs="*",
        default=list(RLBENCH_TASKS),
        choices=list(RLBENCH_TASKS),
        help="Hydra task names to evaluate. Default: all four RLBench tasks.",
    )
    parser.add_argument(
        "--runs-root",
        default="runs",
        help="Root directory that contains task run folders. Default: runs.",
    )
    parser.add_argument(
        "--num-frames",
        type=int,
        default=129,
        help="Raw observation/action window length. Default 129 yields 33 output frames at ratio 4.",
    )
    parser.add_argument(
        "--action-video-freq-ratio",
        type=int,
        default=4,
        help="Temporal subsampling ratio used for video frames.",
    )
    parser.add_argument(
        "--videos-per-task",
        type=int,
        default=4,
        help="How many episode-start samples to render per task.",
    )
    parser.add_argument("--fps", type=int, default=8, help="Output video FPS.")
    parser.add_argument(
        "--device",
        default="auto",
        help="Torch device. Use 'auto' for cuda:0 when available, otherwise cpu.",
    )
    parser.add_argument(
        "--output-subdir",
        default="eval_final_long",
        help="Subdirectory under each run directory for generated videos.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Regenerate videos even if the output file already exists.",
    )
    return parser.parse_args()


def validate_window(num_frames: int, ratio: int) -> None:
    if num_frames <= 1:
        raise ValueError("--num-frames must be greater than 1")
    if ratio <= 0:
        raise ValueError("--action-video-freq-ratio must be positive")
    if (num_frames - 1) % ratio != 0:
        raise ValueError(
            f"num_frames-1 must be divisible by ratio, got num_frames={num_frames}, ratio={ratio}"
        )
    output_transitions = (num_frames - 1) // ratio
    if output_transitions % 4 != 0:
        raise ValueError(
            "The number of output video transitions must be divisible by 4; "
            f"got (num_frames-1)/ratio={output_transitions}"
        )


def resolve_device(device: str) -> str:
    if device != "auto":
        return device
    return "cuda:0" if torch.cuda.is_available() else "cpu"


def compose_task_cfg(task_name: str, run_dir: Path, num_frames: int, ratio: int):
    overrides = [
        f"task={task_name}",
        f"output_dir={run_dir}",
        "wandb.enabled=false",
        "num_workers=0",
        f"data.train.num_frames={num_frames}",
        f"data.val.num_frames={num_frames}",
        f"data.train.action_video_freq_ratio={ratio}",
        f"data.val.action_video_freq_ratio={ratio}",
    ]
    with initialize(version_base="1.3", config_path="../configs"):
        cfg = compose(config_name="train", overrides=overrides)
    return OmegaConf.create(OmegaConf.to_container(cfg, resolve=True))


def resolve_latest_checkpoint(task_name: str, runs_root: Path) -> Path:
    ckpts = sorted((runs_root / task_name).glob("*/checkpoints/weights/step_*.pt"))
    if not ckpts:
        raise FileNotFoundError(f"No checkpoints found for task={task_name} under {runs_root}")

    def _step(path: Path) -> int:
        return int(path.stem.split("_")[-1])

    return max(ckpts, key=_step)


def choose_episode_start_indices(val_ds, count: int) -> list[int]:
    base_ds = val_ds.lerobot_dataset
    starts = getattr(base_ds, "episode_data_index", {}).get("from")
    if starts is None or len(starts) == 0:
        return np.linspace(0, len(val_ds) - 1, count, dtype=int).tolist()
    starts = starts.detach().cpu().numpy().astype(int)
    count = min(max(int(count), 1), len(starts))
    positions = np.linspace(0, len(starts) - 1, count, dtype=int)
    return starts[positions].tolist()


def gt_frames_from_video_tensor(video0: torch.Tensor) -> list[Image.Image]:
    gt_tensor = ((video0.detach().float().cpu().clamp(-1.0, 1.0) + 1.0) * 0.5).contiguous()
    frames = []
    for t in range(gt_tensor.shape[1]):
        arr = (gt_tensor[:, t].permute(1, 2, 0).numpy() * 255.0).astype(np.uint8)
        frames.append(Image.fromarray(arr).convert("RGB"))
    return frames


def stitch_pred_gt(pred_frames: list[Image.Image], gt_frames: list[Image.Image]) -> list[Image.Image]:
    stitched = []
    for pred, gt in zip(pred_frames, gt_frames):
        pred = pred.convert("RGB")
        gt = gt.convert("RGB")
        height = max(pred.height, gt.height)
        canvas = Image.new("RGB", (pred.width + gt.width, height), (0, 0, 0))
        canvas.paste(pred, (0, 0))
        canvas.paste(gt, (pred.width, 0))
        stitched.append(canvas)
    return stitched


def render_task(
    *,
    task_name: str,
    runs_root: Path,
    device: str,
    num_frames: int,
    ratio: int,
    videos_per_task: int,
    fps: int,
    output_subdir: str,
    overwrite: bool,
) -> dict:
    checkpoint = resolve_latest_checkpoint(task_name, runs_root)
    run_dir = checkpoint.parents[2]
    output_dir = run_dir / output_subdir
    output_dir.mkdir(parents=True, exist_ok=True)
    print(
        f"[long-eval] start task={task_name} ckpt={checkpoint} "
        f"num_frames={num_frames} ratio={ratio} device={device}",
        flush=True,
    )

    cfg = compose_task_cfg(task_name, run_dir, num_frames, ratio)
    misc.register_work_dir(str(run_dir))

    model_dtype = _mixed_precision_to_model_dtype(_normalize_mixed_precision(cfg.mixed_precision))
    model = instantiate(cfg.model, model_dtype=model_dtype, device=device)
    model.load_checkpoint(str(checkpoint))
    model.eval()

    stats_path = run_dir / "dataset_stats.json"
    val_ds = instantiate(cfg.data.val, pretrained_norm_stats=str(stats_path))
    indices = choose_episode_start_indices(val_ds, videos_per_task)
    step_tag = checkpoint.stem
    manifest = {
        "task": task_name,
        "checkpoint": str(checkpoint),
        "run_dir": str(run_dir),
        "output_dir": str(output_dir),
        "num_frames": int(num_frames),
        "action_video_freq_ratio": int(ratio),
        "output_video_frames": len(range(0, num_frames, ratio)),
        "fps": int(fps),
        "val_len": len(val_ds),
        "indices": indices,
        "videos": [],
    }

    for sample_i, idx in enumerate(indices):
        out_path = output_dir / (
            f"{step_tag}_long_nf{num_frames}_pred_gt_sample_{sample_i:02d}_idx_{idx:06d}.mp4"
        )
        if out_path.exists() and out_path.stat().st_size > 0 and not overwrite:
            print(f"[long-eval] skip existing {out_path}", flush=True)
            manifest["videos"].append(
                {"sample": sample_i, "idx": int(idx), "path": str(out_path), "bytes": out_path.stat().st_size}
            )
            continue

        print(f"[long-eval] infer task={task_name} sample={sample_i} idx={idx}", flush=True)
        sample = Wan22Trainer._to_batched_eval_sample(val_ds[int(idx)])
        video0 = sample["video"][0]
        action = sample["action"][0] if sample["action"] is not None else None
        proprio = sample["proprio"][0, 0] if sample["proprio"] is not None else None
        input_image = video0[:, 0].unsqueeze(0)
        _, output_frames, _, _ = video0.shape

        infer_kwargs = {
            "input_image": input_image,
            "num_frames": output_frames,
            "action": action,
            "action_horizon": sample["action_horizon"],
            "proprio": proprio,
            "text_cfg_scale": 1.0,
            "action_cfg_scale": 1.0,
            "num_inference_steps": int(cfg.eval_num_inference_steps),
            "seed": 12900 + sample_i,
            "tiled": False,
        }
        if sample["context"] is not None:
            infer_kwargs["prompt"] = None
            infer_kwargs["context"] = sample["context"][0]
            infer_kwargs["context_mask"] = sample["context_mask"][0]
        else:
            infer_kwargs["prompt"] = sample["prompt"][0]

        with torch.no_grad():
            pred = model.infer(**infer_kwargs)

        stitched = stitch_pred_gt(
            [frame.convert("RGB") for frame in pred["video"]],
            gt_frames_from_video_tensor(video0),
        )
        save_mp4(stitched, str(out_path), fps=fps)
        record = {
            "sample": sample_i,
            "idx": int(idx),
            "path": str(out_path),
            "bytes": out_path.stat().st_size,
            "frames": len(stitched),
        }
        manifest["videos"].append(record)
        print(f"[long-eval] saved {out_path} bytes={record['bytes']} frames={record['frames']}", flush=True)

        del sample, video0, pred, stitched
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    manifest_path = output_dir / f"{step_tag}_long_nf{num_frames}_eval_manifest.json"
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=True, indent=2), encoding="utf-8")
    print(f"[long-eval] done task={task_name} manifest={manifest_path}", flush=True)

    del model, val_ds, cfg
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return manifest


def main() -> None:
    args = parse_args()
    validate_window(args.num_frames, args.action_video_freq_ratio)
    register_default_resolvers()
    device = resolve_device(args.device)

    manifests = []
    runs_root = Path(args.runs_root)
    for task_name in args.tasks:
        manifests.append(
            render_task(
                task_name=task_name,
                runs_root=runs_root,
                device=device,
                num_frames=args.num_frames,
                ratio=args.action_video_freq_ratio,
                videos_per_task=args.videos_per_task,
                fps=args.fps,
                output_subdir=args.output_subdir,
                overwrite=args.overwrite,
            )
        )

    summary_path = Path("runs") / f"rlbench_long_eval_nf{args.num_frames}_manifest.json"
    summary_path.write_text(json.dumps(manifests, ensure_ascii=True, indent=2), encoding="utf-8")
    print(f"[long-eval] all_done summary={summary_path}", flush=True)


if __name__ == "__main__":
    main()
