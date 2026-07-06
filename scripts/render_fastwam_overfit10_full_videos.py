#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

import numpy as np
import torch
from hydra import compose, initialize_config_dir
from hydra.utils import instantiate
from omegaconf import OmegaConf
from PIL import Image, ImageDraw


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from fastwam.runtime import _mixed_precision_to_model_dtype, _normalize_mixed_precision
from fastwam.utils.config_resolvers import register_default_resolvers
from fastwam.utils.video_io import save_mp4
from fastwam.utils.video_metrics import pil_frames_to_video_tensor, video_psnr, video_ssim


def step_from_checkpoint(path: Path) -> int:
    match = re.search(r"step_(\d+)\.pt$", path.name)
    return int(match.group(1)) if match else 0


def tensor_video_to_pils(video: torch.Tensor) -> list[Image.Image]:
    if video.ndim != 4 or video.shape[0] != 3:
        raise ValueError(f"Expected video [3,T,H,W], got {tuple(video.shape)}")
    video_01 = ((video.detach().cpu().float().clamp(-1.0, 1.0) + 1.0) * 0.5).contiguous()
    frames: list[Image.Image] = []
    for t in range(video_01.shape[1]):
        arr = (video_01[:, t].permute(1, 2, 0).numpy() * 255.0).round().astype(np.uint8)
        frames.append(Image.fromarray(arr, mode="RGB"))
    return frames


def label_frame(frame: Image.Image, text: str, fill: tuple[int, int, int] = (255, 255, 255)) -> Image.Image:
    out = frame.convert("RGB").copy()
    draw = ImageDraw.Draw(out)
    pad = 5
    bbox = draw.textbbox((pad, pad), text)
    draw.rectangle((bbox[0] - 3, bbox[1] - 3, bbox[2] + 3, bbox[3] + 3), fill=(0, 0, 0))
    draw.text((pad, pad), text, fill=fill)
    return out


def hcat(frames: list[Image.Image]) -> Image.Image:
    widths, heights = zip(*(f.size for f in frames))
    canvas = Image.new("RGB", (sum(widths), max(heights)), (0, 0, 0))
    x = 0
    for frame in frames:
        canvas.paste(frame.convert("RGB"), (x, 0))
        x += frame.size[0]
    return canvas


def load_cfg(task: str, config_dir: Path):
    with initialize_config_dir(version_base="1.3", config_dir=str(config_dir.resolve())):
        return compose(config_name="train", overrides=[f"task={task}"])


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", type=Path, default=Path.cwd())
    parser.add_argument("--task", default="robocasa_acg_v1_fastwam_overfit10")
    parser.add_argument("--config-dir", type=Path, default=Path("configs"))
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--split", choices=["train", "val"], default="train")
    parser.add_argument("--num-windows", type=int, default=10)
    parser.add_argument("--num-inference-steps", type=int, default=10)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--fps", type=int, default=8)
    args = parser.parse_args()

    register_default_resolvers()
    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    cfg = load_cfg(args.task, args.config_dir)
    precision = _normalize_mixed_precision(str(cfg.get("mixed_precision", "bf16")))
    dtype = _mixed_precision_to_model_dtype(precision)
    model = instantiate(cfg.model, model_dtype=dtype, device=str(args.device))
    model.load_checkpoint(str(args.checkpoint.expanduser().resolve()))
    model.eval()

    dataset = instantiate(cfg.data[args.split])
    n = min(int(args.num_windows), len(dataset))
    if n <= 0:
        raise ValueError("No windows selected")

    first_sample = dataset[0]
    episode_index = int(first_sample["episode_index"])
    episode_pos, _start = dataset.windows[0]
    episode = dataset.episodes[episode_pos]
    _states, _actions, timestamps = dataset._load_episode_arrays(episode)

    # Full raw episode, resized and concatenated in the same two-view format as training.
    full_gt_video = dataset._load_video(episode, timestamps)
    full_gt_frames = [
        label_frame(frame, f"GT full episode {episode_index} | frame {i+1}/{full_gt_video.shape[1]}")
        for i, frame in enumerate(tensor_video_to_pils(full_gt_video))
    ]
    full_gt_path = output_dir / f"episode_{episode_index:06d}_gt_full_two_view.mp4"
    save_mp4(full_gt_frames, str(full_gt_path), fps=int(args.fps))

    all_compare_frames: list[Image.Image] = []
    all_pred_frames: list[Image.Image] = []
    all_gt_window_frames: list[Image.Image] = []
    per_window: list[dict] = []

    with torch.no_grad():
        for i in range(n):
            sample = dataset[i]
            gt_video = sample["video"].detach().cpu().float()
            gt_frames = tensor_video_to_pils(gt_video)
            out = model.infer_joint(
                prompt=None,
                input_image=gt_video[:, 0].unsqueeze(0),
                num_video_frames=int(gt_video.shape[1]),
                action_horizon=int(sample["action"].shape[0]),
                action=None,
                proprio=sample["proprio"][0].detach().cpu().float(),
                context=sample["context"],
                context_mask=sample["context_mask"],
                text_cfg_scale=1.0,
                num_inference_steps=int(args.num_inference_steps),
                seed=int(args.seed) + i,
                rand_device="cpu",
                tiled=False,
                test_action_with_infer_action=False,
            )
            pred_frames = [frame.convert("RGB") for frame in out["video"]]
            pred_video = pil_frames_to_video_tensor(pred_frames)
            gt_video_01 = ((gt_video.clamp(-1.0, 1.0) + 1.0) * 0.5).contiguous()
            psnr = float(video_psnr(pred=pred_video, target=gt_video_01))
            ssim = float(video_ssim(pred=pred_video, target=gt_video_01))
            window_start = int(sample["window_start"])

            for t, (pred, gt) in enumerate(zip(pred_frames, gt_frames)):
                label = f"window {i:02d} start={window_start} local_frame={t:02d} | PSNR={psnr:.2f} SSIM={ssim:.3f}"
                pred_l = label_frame(pred, "PRED " + label, fill=(255, 230, 80))
                gt_l = label_frame(gt, "GT " + label, fill=(140, 220, 255))
                all_compare_frames.append(hcat([pred_l, gt_l]))
                all_pred_frames.append(pred_l)
                all_gt_window_frames.append(gt_l)

            per_window.append(
                {
                    "idx": i,
                    "episode_index": int(sample["episode_index"]),
                    "window_start": window_start,
                    "frames": int(gt_video.shape[1]),
                    "psnr_gt": psnr,
                    "ssim_gt": ssim,
                }
            )

    compare_path = output_dir / f"episode_{episode_index:06d}_overfit10_pred_vs_gt_all_windows.mp4"
    pred_path = output_dir / f"episode_{episode_index:06d}_overfit10_pred_all_windows.mp4"
    gt_window_path = output_dir / f"episode_{episode_index:06d}_overfit10_gt_all_windows.mp4"
    save_mp4(all_compare_frames, str(compare_path), fps=int(args.fps))
    save_mp4(all_pred_frames, str(pred_path), fps=int(args.fps))
    save_mp4(all_gt_window_frames, str(gt_window_path), fps=int(args.fps))

    summary = {
        "checkpoint": str(args.checkpoint.expanduser().resolve()),
        "global_step": step_from_checkpoint(args.checkpoint),
        "split": args.split,
        "episode_index": episode_index,
        "episode_length_frames": int(full_gt_video.shape[1]),
        "num_windows": n,
        "window_video_frames": int(dataset.num_frames),
        "frame_offsets": list(dataset.frame_offsets),
        "fps": int(args.fps),
        "note": "Predicted windows are open-loop FastWAM chunks, each conditioned on that window's GT first frame; this is not online closed-loop RoboCasa env rollout.",
        "videos": {
            "gt_full_episode": str(full_gt_path),
            "pred_vs_gt_all_windows": str(compare_path),
            "pred_all_windows": str(pred_path),
            "gt_all_windows": str(gt_window_path),
        },
        "metrics": {
            "psnr_gt_mean": float(np.mean([r["psnr_gt"] for r in per_window])),
            "ssim_gt_mean": float(np.mean([r["ssim_gt"] for r in per_window])),
        },
        "windows": per_window,
    }
    summary_path = output_dir / "summary_full_videos.json"
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
