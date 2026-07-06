#!/usr/bin/env python3
"""Autoregressive long-horizon RoboCasa ACG WAM visualization.

This evaluates FastWAM as a visual dynamics model, not as an online RoboCasa
policy. Only the first frame is GT. Later chunks use the previous predicted
last frame as the next visual condition, while text/proprio context is still
read from the matched RoboCasa dataset chunk.
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


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
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


def write_video_index(path: Path, rows: list[dict[str, Any]]) -> None:
    html = [
        "<html><head><meta charset='utf-8'><title>RoboCasa ACG Autoreg WAM</title>",
        "<style>body{font-family:sans-serif} video{width:960px;max-width:100%}.card{margin:20px 0}</style>",
        "</head><body><h1>RoboCasa ACG Autoregressive WAM Diagnostics</h1>",
        "<p>Only the first frame is GT. Later chunks use the previous predicted last frame. "
        "Proprio/text context is oracle dataset context at each chunk start.</p>",
    ]
    for row in rows:
        rel = Path(row["pred_vs_gt_video_path"]).relative_to(path.parent)
        html.append("<div class='card'>")
        html.append(
            f"<h3>{row['split']} episode {row['episode_index']} | chunks {row['num_chunks']} | "
            f"frames {row['frames']} | PSNR {row['psnr_gt_timeline']:.2f} | "
            f"SSIM {row['ssim_gt_timeline']:.3f}</h3>"
        )
        html.append(f"<video controls src='{rel.as_posix()}'></video>")
        html.append(f"<p>{row['prompt']}</p>")
        html.append("</div>")
    html.append("</body></html>")
    path.write_text("\n".join(html), encoding="utf-8")


def load_cfg(task: str, config_dir: Path):
    with initialize_config_dir(version_base="1.3", config_dir=str(config_dir.resolve())):
        return compose(config_name="train", overrides=[f"task={task}"])


def select_episode_positions(dataset: Any, args: argparse.Namespace) -> list[int]:
    if args.episode_index:
        wanted = {int(v) for v in args.episode_index}
        positions = [idx for idx, episode in enumerate(dataset.episodes) if int(episode.episode_index) in wanted]
        missing = sorted(wanted - {int(dataset.episodes[idx].episode_index) for idx in positions})
        if missing:
            raise ValueError(f"Requested episode_index values not found in split {args.split}: {missing}")
        return positions

    candidates: list[tuple[int, int, int]] = []
    for idx, episode in enumerate(dataset.episodes):
        valid_count = max(0, int(episode.length) - int(dataset.action_horizon))
        chunks = 0 if valid_count <= 0 else ((valid_count - 1) // int(args.chunk_stride) + 1)
        candidates.append((chunks, int(episode.length), idx))
    candidates.sort(reverse=True)
    positions = [idx for chunks, _length, idx in candidates if chunks >= int(args.min_chunks)]
    if not positions:
        positions = [idx for _chunks, _length, idx in candidates[: int(args.num_episodes)]]
    return positions[: int(args.num_episodes)]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", default="robocasa_acg_v1_fastwam_8gpu")
    parser.add_argument("--config-dir", type=Path, default=Path("configs"))
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--split", choices=["train", "val"], default="val")
    parser.add_argument("--num-episodes", type=int, default=2)
    parser.add_argument("--episode-index", type=int, action="append", default=None)
    parser.add_argument("--min-chunks", type=int, default=8)
    parser.add_argument("--chunk-stride", type=int, default=32)
    parser.add_argument("--num-chunks", type=int, default=12)
    parser.add_argument("--num-inference-steps", type=int, default=20)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--fps", type=int, default=8)
    args = parser.parse_args()

    register_default_resolvers()
    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    cfg = load_cfg(args.task, args.config_dir)
    data_cfg = OmegaConf.create(OmegaConf.to_container(cfg.data[args.split], resolve=True))
    data_cfg.max_samples = None
    data_cfg.max_windows_per_episode = None
    data_cfg.max_episodes = None
    data_cfg.is_training_set = False
    dataset = instantiate(data_cfg)

    index_by_episode_start: dict[tuple[int, int], int] = {}
    for dataset_idx, (episode_pos, start) in enumerate(dataset.windows):
        index_by_episode_start[(int(episode_pos), int(start))] = int(dataset_idx)

    precision = _normalize_mixed_precision(str(cfg.get("mixed_precision", "bf16")))
    dtype = _mixed_precision_to_model_dtype(precision)
    model = instantiate(cfg.model, model_dtype=dtype, device=str(args.device))
    model.load_checkpoint(str(args.checkpoint.expanduser().resolve()))
    model.eval()

    rows: list[dict[str, Any]] = []
    with torch.no_grad():
        for episode_order, episode_pos in enumerate(select_episode_positions(dataset, args)):
            episode = dataset.episodes[episode_pos]
            valid_count = max(0, int(episode.length) - int(dataset.action_horizon))
            starts = list(range(0, valid_count, int(args.chunk_stride)))[: int(args.num_chunks)]
            starts = [start for start in starts if (int(episode_pos), int(start)) in index_by_episode_start]
            if not starts:
                continue

            first_sample = dataset[index_by_episode_start[(int(episode_pos), int(starts[0]))]]
            first_gt_frames = tensor_video_to_pils(first_sample["video"].detach().cpu().float())
            pred_frames = [first_gt_frames[0]]
            gt_timeline_frames = [first_gt_frames[0]]
            current_image = first_sample["video"][:, 0].unsqueeze(0)
            chunk_rows: list[dict[str, Any]] = []
            prompt = str(first_sample.get("prompt", ""))

            for chunk_idx, start in enumerate(starts):
                sample = dataset[index_by_episode_start[(int(episode_pos), int(start))]]
                gt_video = sample["video"].detach().cpu().float()
                gt_frames = tensor_video_to_pils(gt_video)
                out = model.infer_joint(
                    prompt=None if sample.get("context") is not None else sample.get("prompt"),
                    input_image=current_image,
                    num_video_frames=int(gt_video.shape[1]),
                    action_horizon=int(sample["action"].shape[0]),
                    action=None,
                    proprio=sample["proprio"][0].detach().cpu().float(),
                    context=sample.get("context"),
                    context_mask=sample.get("context_mask"),
                    text_cfg_scale=1.0,
                    num_inference_steps=int(args.num_inference_steps),
                    seed=int(args.seed) + episode_order * 1000 + chunk_idx,
                    rand_device="cpu",
                    tiled=False,
                    test_action_with_infer_action=False,
                )
                chunk_pred_frames = [frame.convert("RGB") for frame in out["video"]]
                chunk_pred_video = pil_frames_to_video_tensor(chunk_pred_frames)
                chunk_gt_video_01 = ((gt_video.clamp(-1.0, 1.0) + 1.0) * 0.5).contiguous()
                chunk_rows.append(
                    {
                        "chunk_idx": chunk_idx,
                        "window_start": int(start),
                        "frames": int(gt_video.shape[1]),
                        "gt_abs_frames": [int(start) + int(v) for v in dataset.frame_offsets],
                        "psnr_gt": float(video_psnr(pred=chunk_pred_video, target=chunk_gt_video_01)),
                        "ssim_gt": float(video_ssim(pred=chunk_pred_video, target=chunk_gt_video_01)),
                    }
                )
                pred_frames.extend(chunk_pred_frames[1:])
                gt_timeline_frames.extend(gt_frames[1:])
                current_image = pil_to_model_tensor(chunk_pred_frames[-1])

            pred_video = pil_frames_to_video_tensor(pred_frames)
            gt_video_01 = pil_frames_to_video_tensor(gt_timeline_frames)
            psnr = float(video_psnr(pred=pred_video, target=gt_video_01))
            ssim = float(video_ssim(pred=pred_video, target=gt_video_01))

            prefix = (
                f"{args.split}_episode_{int(episode.episode_index):06d}_"
                f"autoreg_stride{int(args.chunk_stride)}_chunks{len(starts)}"
            )
            pred_path = output_dir / f"{prefix}_pred.mp4"
            gt_path = output_dir / f"{prefix}_gt_timeline.mp4"
            compare_path = output_dir / f"{prefix}_pred_vs_gt.mp4"
            compare_frames = [
                hcat(
                    label_frame(pred, f"PRED autoreg frame={idx:03d}", (255, 230, 80)),
                    label_frame(gt, f"GT timeline frame={idx:03d}", (140, 220, 255)),
                )
                for idx, (pred, gt) in enumerate(zip(pred_frames, gt_timeline_frames))
            ]
            save_mp4(pred_frames, str(pred_path), fps=int(args.fps))
            save_mp4(gt_timeline_frames, str(gt_path), fps=int(args.fps))
            save_mp4(compare_frames, str(compare_path), fps=int(args.fps))

            row = {
                "eval_type": "robocasa_acg_autoregressive_wam",
                "eval_protocol": "autoreg_oracle_state",
                "split": str(args.split),
                "episode_index": int(episode.episode_index),
                "episode_length": int(episode.length),
                "prompt": prompt,
                "chunk_stride": int(args.chunk_stride),
                "num_chunks": len(starts),
                "frames": len(pred_frames),
                "fps": int(args.fps),
                "psnr_gt_timeline": psnr,
                "ssim_gt_timeline": ssim,
                "pred_video_path": str(pred_path),
                "gt_timeline_video_path": str(gt_path),
                "pred_vs_gt_video_path": str(compare_path),
            }
            rows.append(row)
            (output_dir / f"{prefix}_chunks.json").write_text(
                json.dumps(chunk_rows, indent=2, ensure_ascii=False),
                encoding="utf-8",
            )

    summary = {
        "eval_type": "robocasa_acg_autoregressive_wam",
        "eval_protocol": "autoreg_oracle_state",
        "protocol_note": (
            "Only the first frame is GT. Each later chunk uses the previous predicted last frame as visual input. "
            "Text/proprio context is still oracle RoboCasa dataset context at each chunk start; this is not an "
            "online simulator rollout or success-rate eval."
        ),
        "checkpoint": str(args.checkpoint.expanduser().resolve()),
        "global_step": step_from_checkpoint(args.checkpoint),
        "task": str(args.task),
        "split": str(args.split),
        "num_episodes": len(rows),
        "num_inference_steps": int(args.num_inference_steps),
        "chunk_stride": int(args.chunk_stride),
        "requested_num_chunks": int(args.num_chunks),
        "fps": int(args.fps),
        "psnr_gt_timeline_mean": float(np.mean([r["psnr_gt_timeline"] for r in rows])) if rows else None,
        "ssim_gt_timeline_mean": float(np.mean([r["ssim_gt_timeline"] for r in rows])) if rows else None,
        "episodes": rows,
    }
    (output_dir / "summary_autoreg_wam.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    write_csv(output_dir / "eval_manifest.csv", rows)
    write_video_index(output_dir / "video_index.html", rows)
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
