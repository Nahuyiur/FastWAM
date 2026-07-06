#!/usr/bin/env python3
"""Small RoboCasa ACG open-loop WAM visual smoke test.

This is intentionally separate from training and official online evaluation.
It samples RoboCasa val_id windows, conditions FastWAM on the first frame plus
cached text/proprio context, predicts the short video chunk, and writes
prediction-only plus prediction/GT side-by-side MP4 files.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch
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


def _step_from_checkpoint(path: Path) -> int:
    match = re.search(r"step_(\d+)\.pt$", path.name)
    return int(match.group(1)) if match else 0


def _tensor_frame_to_pil(frame: torch.Tensor) -> Image.Image:
    if frame.ndim != 3 or frame.shape[0] != 3:
        raise ValueError(f"Expected [3,H,W], got {tuple(frame.shape)}")
    array = ((frame.detach().cpu().float().clamp(-1.0, 1.0) + 1.0) * 127.5)
    array = array.permute(1, 2, 0).numpy().round().astype(np.uint8)
    return Image.fromarray(array, mode="RGB")


def _label_frame(frame: Image.Image, text: str, *, fill: tuple[int, int, int]) -> Image.Image:
    out = frame.convert("RGB").copy()
    draw = ImageDraw.Draw(out)
    x, y = 5, 5
    bbox = draw.textbbox((x, y), text)
    draw.rectangle((bbox[0] - 3, bbox[1] - 3, bbox[2] + 3, bbox[3] + 3), fill=(0, 0, 0))
    draw.text((x, y), text, fill=fill)
    return out


def _side_by_side_frames(
    pred_01: torch.Tensor,
    gt_01: torch.Tensor,
    *,
    episode_index: int,
    window_start: int,
) -> list[Image.Image]:
    if pred_01.shape != gt_01.shape:
        raise ValueError(f"Video shape mismatch: pred={tuple(pred_01.shape)} gt={tuple(gt_01.shape)}")
    frames = []
    for idx in range(pred_01.shape[1]):
        pred_array = (pred_01[:, idx].permute(1, 2, 0).clamp(0.0, 1.0).numpy() * 255.0).round().astype(np.uint8)
        gt_array = (gt_01[:, idx].permute(1, 2, 0).clamp(0.0, 1.0).numpy() * 255.0).round().astype(np.uint8)
        pred = _label_frame(
            Image.fromarray(pred_array, mode="RGB"),
            f"PRED | ep={episode_index} start={window_start} f={idx}",
            fill=(255, 230, 80),
        )
        gt = _label_frame(
            Image.fromarray(gt_array, mode="RGB"),
            f"GT matched | ep={episode_index} start={window_start} f={idx}",
            fill=(140, 220, 255),
        )
        canvas = Image.new("RGB", (pred.width + gt.width, max(pred.height, gt.height)), (0, 0, 0))
        canvas.paste(pred, (0, 0))
        canvas.paste(gt, (pred.width, 0))
        frames.append(canvas)
    return frames


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames: list[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _write_video_index(output_dir: Path, rows: list[dict[str, Any]]) -> None:
    html = [
        "<html><head><meta charset='utf-8'><title>RoboCasa ACG GT-matched WAM videos</title>",
        "<style>body{font-family:sans-serif} video{width:720px;max-width:100%}.card{margin:16px 0}</style>",
        "</head><body><h1>RoboCasa ACG GT-matched WAM diagnostics</h1>",
        "<p>Each card compares a FastWAM prediction against the exact RoboCasa365 dataset window GT.</p>",
    ]
    for row in rows:
        rel = Path(row["pred_gt_video_path"]).relative_to(output_dir)
        html.append("<div class='card'>")
        html.append(
            "<h3>"
            f"sample {row['sample_idx']} | episode {row['episode_index']} | "
            f"window {row['window_start']} | PSNR {row['psnr_gt']:.2f} | SSIM {row['ssim_gt']:.3f}"
            "</h3>"
        )
        html.append(f"<video controls src='{rel.as_posix()}'></video>")
        html.append(f"<p>{row['prompt']}</p>")
        html.append("</div>")
    html.append("</body></html>")
    (output_dir / "video_index.html").write_text("\n".join(html), encoding="utf-8")


def _latest_checkpoint(run_dir: Path) -> Path:
    candidates = sorted((run_dir / "checkpoints" / "weights").glob("step_*.pt"))
    if not candidates:
        raise FileNotFoundError(f"No step_*.pt found under {run_dir / 'checkpoints' / 'weights'}")
    return max(candidates, key=_step_from_checkpoint)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--split", choices=("train", "val"), default="val")
    parser.add_argument("--num-samples", type=int, default=5)
    parser.add_argument("--num-inference-steps", type=int, default=10)
    parser.add_argument("--seed", type=int, default=47)
    parser.add_argument("--video-fps", type=int, default=8)
    parser.add_argument("--tiled", action="store_true")
    parser.add_argument(
        "--episode-index",
        type=int,
        action="append",
        default=None,
        help="Optional RoboCasa365 episode_index filter. Can be passed multiple times.",
    )
    parser.add_argument(
        "--window-start",
        type=int,
        action="append",
        default=None,
        help="Optional window_start filter. Can be passed multiple times.",
    )
    return parser.parse_args()


def main() -> None:
    register_default_resolvers()
    args = parse_args()
    run_dir = args.run_dir.expanduser().resolve()
    cfg_path = run_dir / "config.yaml"
    checkpoint = (args.checkpoint or _latest_checkpoint(run_dir)).expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    cfg = OmegaConf.load(cfg_path)
    mixed_precision = _normalize_mixed_precision(str(cfg.get("mixed_precision", "bf16")))
    model_dtype = _mixed_precision_to_model_dtype(mixed_precision)
    model_cfg = OmegaConf.create(OmegaConf.to_container(cfg.model, resolve=True))
    model = instantiate(model_cfg, model_dtype=model_dtype, device=str(args.device))
    model.load_checkpoint(str(checkpoint))
    model = model.to(args.device).eval()

    data_cfg = OmegaConf.create(OmegaConf.to_container(cfg.data[args.split], resolve=True))
    dataset = instantiate(data_cfg)
    if len(dataset) <= 0:
        raise ValueError("RoboCasa WAM smoke dataset is empty.")

    rng = np.random.default_rng(int(args.seed))
    if args.episode_index or args.window_start:
        wanted_episodes = None if args.episode_index is None else {int(v) for v in args.episode_index}
        wanted_starts = None if args.window_start is None else {int(v) for v in args.window_start}
        sample_indices = []
        for idx in range(len(dataset)):
            episode_pos, start = dataset.windows[idx]
            episode = dataset.episodes[episode_pos]
            if wanted_episodes is not None and int(episode.episode_index) not in wanted_episodes:
                continue
            if wanted_starts is not None and int(start) not in wanted_starts:
                continue
            sample_indices.append(idx)
            if len(sample_indices) >= int(args.num_samples):
                break
        if not sample_indices:
            raise ValueError(
                f"No dataset windows matched episode_index={args.episode_index} window_start={args.window_start}"
            )
    elif hasattr(dataset, "windows"):
        by_episode: dict[int, list[int]] = {}
        for idx, (episode_pos, _start) in enumerate(dataset.windows):
            by_episode.setdefault(int(episode_pos), []).append(idx)
        episode_positions = list(by_episode)
        rng.shuffle(episode_positions)
        sample_indices = [int(rng.choice(by_episode[pos])) for pos in episode_positions[: int(args.num_samples)]]
    else:
        sample_indices = rng.choice(len(dataset), size=min(int(args.num_samples), len(dataset)), replace=False).tolist()
    rows: list[dict[str, Any]] = []

    with torch.no_grad():
        for out_idx, dataset_idx in enumerate(sample_indices):
            sample = dataset[int(dataset_idx)]
            gt_video = sample["video"].detach().cpu().float()
            first_frame = gt_video[:, 0].unsqueeze(0)
            proprio = sample.get("proprio")
            if proprio is not None:
                proprio = proprio[0].detach().cpu().float()
            context = sample.get("context")
            context_mask = sample.get("context_mask")
            prompt = None if context is not None else sample.get("prompt")

            out = model.infer_joint(
                prompt=prompt,
                input_image=first_frame,
                num_video_frames=int(gt_video.shape[1]),
                action_horizon=int(sample["action"].shape[0]),
                action=None,
                proprio=proprio,
                context=context,
                context_mask=context_mask,
                text_cfg_scale=1.0,
                num_inference_steps=int(args.num_inference_steps),
                seed=int(args.seed) + out_idx,
                rand_device="cpu",
                tiled=bool(args.tiled),
                test_action_with_infer_action=False,
            )
            pred_frames = [frame.convert("RGB") for frame in out["video"]]
            pred_video = pil_frames_to_video_tensor(pred_frames)
            gt_video_01 = ((gt_video.clamp(-1.0, 1.0) + 1.0) * 0.5).contiguous()
            psnr = float(video_psnr(pred=pred_video, target=gt_video_01))
            ssim = float(video_ssim(pred=pred_video, target=gt_video_01))
            episode_index = int(sample.get("episode_index", -1))
            window_start = int(sample.get("window_start", -1))

            pred_path = output_dir / f"sample_{out_idx:02d}_pred.mp4"
            pred_gt_path = output_dir / f"sample_{out_idx:02d}_pred_gt.mp4"
            save_mp4(pred_frames, str(pred_path), fps=int(args.video_fps))
            save_mp4(
                _side_by_side_frames(
                    pred_video,
                    gt_video_01,
                    episode_index=episode_index,
                    window_start=window_start,
                ),
                str(pred_gt_path),
                fps=int(args.video_fps),
            )

            action = out.get("action")
            action_shape = list(action.shape) if isinstance(action, torch.Tensor) else None
            rows.append(
                {
                    "eval_protocol": "dataset_window_gt_matched_wam",
                    "gt_matched": True,
                    "sample_idx": out_idx,
                    "dataset_idx": int(dataset_idx),
                    "episode_index": episode_index,
                    "window_start": window_start,
                    "prompt": str(sample.get("prompt", "")),
                    "gt_video_layout": "two_camera_horizontal(robot0_agentview_left|robot0_eye_in_hand)",
                    "frames": int(gt_video.shape[1]),
                    "pred_action_shape": action_shape,
                    "psnr_gt": psnr,
                    "ssim_gt": ssim,
                    "pred_video_path": str(pred_path),
                    "pred_gt_video_path": str(pred_gt_path),
                }
            )

    summary = {
        "eval_type": "robocasa_acg_open_loop_wam_smoke",
        "eval_protocol": "dataset_window_gt_matched_wam",
        "gt_matched": True,
        "protocol_note": (
            "This diagnostic compares FastWAM predictions against the exact RoboCasa365 "
            "dataset window identified by episode_index/window_start. It is not an online "
            "RoboCasa success-rate rollout."
        ),
        "checkpoint": str(checkpoint),
        "global_step": _step_from_checkpoint(checkpoint),
        "split": args.split,
        "num_samples": len(rows),
        "num_inference_steps": int(args.num_inference_steps),
        "psnr_gt_mean": float(np.mean([r["psnr_gt"] for r in rows])) if rows else None,
        "ssim_gt_mean": float(np.mean([r["ssim_gt"] for r in rows])) if rows else None,
        "samples": rows,
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    _write_csv(output_dir / "eval_manifest.csv", rows)
    _write_video_index(output_dir, rows)
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
