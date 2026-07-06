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


def pil_to_model_tensor(frame: Image.Image) -> torch.Tensor:
    arr = np.asarray(frame.convert("RGB"), dtype=np.float32) / 255.0
    tensor = torch.from_numpy(arr).permute(2, 0, 1).contiguous()
    return (tensor * 2.0 - 1.0).unsqueeze(0)


def label_frame(frame: Image.Image, text: str, fill: tuple[int, int, int]) -> Image.Image:
    out = frame.convert("RGB").copy()
    draw = ImageDraw.Draw(out)
    x, y = 5, 5
    bbox = draw.textbbox((x, y), text)
    draw.rectangle((bbox[0] - 3, bbox[1] - 3, bbox[2] + 3, bbox[3] + 3), fill=(0, 0, 0))
    draw.text((x, y), text, fill=fill)
    return out


def hcat(left: Image.Image, right: Image.Image) -> Image.Image:
    left = left.convert("RGB")
    right = right.convert("RGB")
    canvas = Image.new("RGB", (left.width + right.width, max(left.height, right.height)), (0, 0, 0))
    canvas.paste(left, (0, 0))
    canvas.paste(right, (left.width, 0))
    return canvas


def load_cfg(task: str, config_dir: Path):
    with initialize_config_dir(version_base="1.3", config_dir=str(config_dir.resolve())):
        return compose(config_name="train", overrides=[f"task={task}"])


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", default="robocasa_acg_v1_fastwam_overfit10")
    parser.add_argument("--config-dir", type=Path, default=Path("configs"))
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--chunk-stride", type=int, default=32)
    parser.add_argument("--num-chunks", type=int, default=6)
    parser.add_argument("--num-inference-steps", type=int, default=10)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--fps", type=int, default=8)
    args = parser.parse_args()

    register_default_resolvers()
    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    cfg = load_cfg(args.task, args.config_dir)
    # The overfit task used max_samples=10/window_stride=2, which creates
    # overlapping independent windows. For a real continuing visualization,
    # use the first episode with non-overlapping 32-step chunks.
    data_cfg = OmegaConf.create(OmegaConf.to_container(cfg.data.train, resolve=True))
    data_cfg.max_episodes = 1
    data_cfg.max_samples = None
    data_cfg.max_windows_per_episode = None
    data_cfg.window_stride = int(args.chunk_stride)
    data_cfg.is_training_set = False
    dataset = instantiate(data_cfg)
    num_chunks = min(int(args.num_chunks), len(dataset))
    if num_chunks <= 0:
        raise ValueError("No chunks available for autoregressive visualization")

    precision = _normalize_mixed_precision(str(cfg.get("mixed_precision", "bf16")))
    dtype = _mixed_precision_to_model_dtype(precision)
    model = instantiate(cfg.model, model_dtype=dtype, device=str(args.device))
    model.load_checkpoint(str(args.checkpoint.expanduser().resolve()))
    model.eval()

    first_sample = dataset[0]
    episode_index = int(first_sample["episode_index"])
    pred_frames = [tensor_video_to_pils(first_sample["video"])[0]]
    gt_timeline_frames = [tensor_video_to_pils(first_sample["video"])[0]]
    current_image = first_sample["video"][:, 0].unsqueeze(0)
    rows: list[dict] = []

    with torch.no_grad():
        for chunk_idx in range(num_chunks):
            sample = dataset[chunk_idx]
            gt_video = sample["video"].detach().cpu().float()
            gt_frames = tensor_video_to_pils(gt_video)
            window_start = int(sample["window_start"])
            out = model.infer_joint(
                prompt=None,
                input_image=current_image,
                num_video_frames=int(gt_video.shape[1]),
                action_horizon=int(sample["action"].shape[0]),
                action=None,
                proprio=sample["proprio"][0].detach().cpu().float(),
                context=sample["context"],
                context_mask=sample["context_mask"],
                text_cfg_scale=1.0,
                num_inference_steps=int(args.num_inference_steps),
                seed=int(args.seed) + chunk_idx,
                rand_device="cpu",
                tiled=False,
                test_action_with_infer_action=False,
            )
            chunk_pred_frames = [frame.convert("RGB") for frame in out["video"]]
            pred_frames.extend(chunk_pred_frames[1:])
            gt_timeline_frames.extend(gt_frames[1:])
            current_image = pil_to_model_tensor(chunk_pred_frames[-1])
            rows.append(
                {
                    "chunk_idx": chunk_idx,
                    "episode_index": int(sample["episode_index"]),
                    "window_start": window_start,
                    "gt_abs_frames": [window_start + int(v) for v in dataset.frame_offsets],
                }
            )

    pred_video = pil_frames_to_video_tensor(pred_frames)
    gt_video_01 = pil_frames_to_video_tensor(gt_timeline_frames)
    psnr = float(video_psnr(pred=pred_video, target=gt_video_01))
    ssim = float(video_ssim(pred=pred_video, target=gt_video_01))

    compare_frames: list[Image.Image] = []
    for frame_idx, (pred, gt) in enumerate(zip(pred_frames, gt_timeline_frames)):
        pred_l = label_frame(pred, f"PRED autoreg frame={frame_idx:03d}", (255, 230, 80))
        gt_l = label_frame(gt, f"GT timeline frame={frame_idx:03d}", (140, 220, 255))
        compare_frames.append(hcat(pred_l, gt_l))

    prefix = f"episode_{episode_index:06d}_autoreg_stride{int(args.chunk_stride)}_chunks{num_chunks}"
    pred_path = output_dir / f"{prefix}_pred.mp4"
    gt_path = output_dir / f"{prefix}_gt_timeline.mp4"
    compare_path = output_dir / f"{prefix}_pred_vs_gt.mp4"
    save_mp4(pred_frames, str(pred_path), fps=int(args.fps))
    save_mp4(gt_timeline_frames, str(gt_path), fps=int(args.fps))
    save_mp4(compare_frames, str(compare_path), fps=int(args.fps))

    summary = {
        "eval_type": "fastwam_robocasa_overfit_autoregressive_visualization",
        "checkpoint": str(args.checkpoint.expanduser().resolve()),
        "global_step": step_from_checkpoint(args.checkpoint),
        "episode_index": episode_index,
        "chunk_stride": int(args.chunk_stride),
        "num_chunks": num_chunks,
        "frames": len(pred_frames),
        "fps": int(args.fps),
        "psnr_gt_timeline": psnr,
        "ssim_gt_timeline": ssim,
        "important_note": (
            "This is autoregressive in image space: only the first frame is GT; each later chunk uses the previous "
            "predicted last frame. Proprio/state is still taken from the GT dataset at each chunk start, because "
            "there is no simulator state propagation in this open-loop visual diagnostic."
        ),
        "videos": {
            "pred": str(pred_path),
            "gt_timeline": str(gt_path),
            "pred_vs_gt": str(compare_path),
        },
        "chunks": rows,
    }
    summary_path = output_dir / f"{prefix}_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
