from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image

from fastwam.utils.video_io import save_mp4
from fastwam.utils.video_metrics import pil_frames_to_video_tensor, video_psnr, video_ssim


def _tensor_frame_to_pil(frame: torch.Tensor) -> Image.Image:
    if frame.ndim != 3 or frame.shape[0] != 3:
        raise ValueError(f"Expected frame [3,H,W], got {tuple(frame.shape)}")
    array = ((frame.detach().cpu().float().clamp(-1.0, 1.0) + 1.0) * 127.5)
    array = array.permute(1, 2, 0).numpy().round().astype(np.uint8)
    return Image.fromarray(array, mode="RGB")


def _pil_frame_to_model_tensor(frame: Image.Image) -> torch.Tensor:
    array = np.asarray(frame.convert("RGB"), dtype=np.float32) / 255.0
    tensor = torch.from_numpy(array).permute(2, 0, 1).contiguous()
    return (tensor * 2.0 - 1.0).unsqueeze(0)


def _side_by_side_frames(left: torch.Tensor, right: torch.Tensor) -> list[Image.Image]:
    if left.shape != right.shape:
        raise ValueError(f"Video shape mismatch: left={tuple(left.shape)} right={tuple(right.shape)}")
    stitched = torch.cat([left, right], dim=3).contiguous()
    frames: list[Image.Image] = []
    for idx in range(stitched.shape[1]):
        array = (stitched[:, idx].permute(1, 2, 0).clamp(0.0, 1.0).numpy() * 255.0).round().astype(np.uint8)
        frames.append(Image.fromarray(array, mode="RGB"))
    return frames


def _action_jsonable(action: torch.Tensor) -> list[list[list[float]]]:
    return action.detach().cpu().float().numpy().round(6).tolist()


@torch.no_grad()
def run_autoregressive_open_loop_wam_eval(
    *,
    model: Any,
    dataset: Any,
    output_dir: str | Path,
    global_step: int,
    num_samples: int = 1,
    rollout_chunks: int = 4,
    chunk_stride: int = 32,
    num_inference_steps: int = 10,
    seed: int = 42,
    save_video: bool = True,
    video_fps: int = 8,
    tiled: bool = False,
) -> dict[str, Any]:
    """Run a FastWAM self-rollout diagnostic over GEMBench 9V32 windows.

    The model predicts each 32-action/9-frame chunk with `action=None`, then
    feeds the chunk's predicted last frame into the next chunk. GT actions are
    used only for shape/diagnostic alignment, not for video conditioning.
    """
    if not hasattr(dataset, "sample_autoreg_sequence"):
        raise TypeError("open-loop WAM eval requires a dataset with sample_autoreg_sequence(...).")

    num_samples = int(num_samples)
    rollout_chunks = int(rollout_chunks)
    chunk_stride = int(chunk_stride)
    if num_samples <= 0:
        raise ValueError(f"num_samples must be positive, got {num_samples}")
    if rollout_chunks <= 0:
        raise ValueError(f"rollout_chunks must be positive, got {rollout_chunks}")

    step_dir = Path(output_dir).expanduser() / f"step_{int(global_step):06d}"
    step_dir.mkdir(parents=True, exist_ok=True)

    sample_summaries: list[dict[str, Any]] = []
    psnr_values: list[float] = []
    ssim_values: list[float] = []
    video_paths: list[str] = []

    for sample_idx in range(num_samples):
        sequence = dataset.sample_autoreg_sequence(
            num_chunks=rollout_chunks,
            stride=chunk_stride,
            seed=int(seed) + sample_idx,
        )
        gt_video = sequence["gt_video_sequence"].detach().cpu().float()
        first_input = sequence["video"][0, :, 0].detach().cpu().float()
        current_image = first_input.unsqueeze(0)
        context = sequence.get("context")
        context_mask = sequence.get("context_mask")
        prompt = None if context is not None else sequence["prompt"]

        pred_frames = [_tensor_frame_to_pil(first_input)]
        pred_actions: list[torch.Tensor] = []

        for chunk_idx, sample in enumerate(sequence["samples"]):
            proprio = sample["proprio"][0] if sample.get("proprio") is not None else None
            infer_kwargs = {
                "prompt": prompt,
                "input_image": current_image,
                "num_video_frames": 9,
                "action_horizon": 32,
                "action": None,
                "proprio": proprio,
                "context": context,
                "context_mask": context_mask,
                "text_cfg_scale": 1.0,
                "num_inference_steps": int(num_inference_steps),
                "seed": int(seed) + sample_idx * 1000 + chunk_idx,
                "rand_device": "cpu",
                "tiled": bool(tiled),
                "test_action_with_infer_action": False,
            }
            out = model.infer_joint(**infer_kwargs)
            chunk_video = out["video"]
            if len(chunk_video) != 9:
                raise ValueError(f"infer_joint returned {len(chunk_video)} frames; expected 9.")
            action = out.get("action")
            if not isinstance(action, torch.Tensor) or tuple(action.shape) != (32, 8):
                raise ValueError(f"infer_joint action must be Tensor [32,8], got {type(action)} {getattr(action, 'shape', None)}")
            pred_actions.append(action.detach().cpu().float())
            pred_frames.extend(frame.convert("RGB") for frame in chunk_video[1:])
            current_image = _pil_frame_to_model_tensor(chunk_video[-1])

        pred_video = pil_frames_to_video_tensor(pred_frames)
        gt_video_01 = ((gt_video.clamp(-1.0, 1.0) + 1.0) * 0.5).contiguous()
        if pred_video.shape != gt_video_01.shape:
            raise ValueError(
                "Open-loop prediction/GT shape mismatch: "
                f"pred={tuple(pred_video.shape)} gt={tuple(gt_video_01.shape)}"
            )
        psnr = float(video_psnr(pred=pred_video, target=gt_video_01))
        ssim = float(video_ssim(pred=pred_video, target=gt_video_01))
        psnr_values.append(psnr)
        ssim_values.append(ssim)

        action_tensor = torch.stack(pred_actions, dim=0)
        action_path = step_dir / f"sample_{sample_idx:02d}_pred_actions.json"
        action_path.write_text(
            json.dumps(
                {
                    "taskvar": sequence["taskvar"],
                    "episode_key": sequence["episode_key"],
                    "base_start": int(sequence["base_start"]),
                    "window_starts": [int(v) for v in sequence["window_starts"]],
                    "shape": list(action_tensor.shape),
                    "normalized_pred_actions": _action_jsonable(action_tensor),
                },
                ensure_ascii=True,
                indent=2,
            ),
            encoding="utf-8",
        )

        pred_path = None
        pred_gt_path = None
        if save_video:
            pred_path = step_dir / f"sample_{sample_idx:02d}_pred.mp4"
            pred_gt_path = step_dir / f"sample_{sample_idx:02d}_pred_gt.mp4"
            save_mp4(pred_frames, str(pred_path), fps=video_fps)
            save_mp4(_side_by_side_frames(pred_video, gt_video_01), str(pred_gt_path), fps=video_fps)
            video_paths.append(str(pred_gt_path))

        sample_summaries.append(
            {
                "sample_idx": sample_idx,
                "taskvar": sequence["taskvar"],
                "episode_key": sequence["episode_key"],
                "base_start": int(sequence["base_start"]),
                "window_starts": [int(v) for v in sequence["window_starts"]],
                "frames": int(pred_video.shape[1]),
                "pred_action_shape": list(action_tensor.shape),
                "psnr_gt": psnr,
                "ssim_gt": ssim,
                "pred_video_path": str(pred_path) if pred_path is not None else None,
                "pred_gt_video_path": str(pred_gt_path) if pred_gt_path is not None else None,
                "pred_actions_path": str(action_path),
            }
        )

    summary = {
        "eval_type": "fastwam_autoregressive_open_loop_wam",
        "official_full_score": False,
        "write_official_preds": False,
        "global_step": int(global_step),
        "num_samples": num_samples,
        "rollout_chunks": rollout_chunks,
        "chunk_stride": chunk_stride,
        "num_inference_steps": int(num_inference_steps),
        "expected_frames": 1 + 8 * rollout_chunks,
        "proprio_source": "dataset_current_at_chunk_start",
        "future_action_conditioning": False,
        "metrics": {
            "psnr_gt_mean": float(np.mean(psnr_values)),
            "ssim_gt_mean": float(np.mean(ssim_values)),
        },
        "samples": sample_summaries,
    }
    summary_path = step_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, ensure_ascii=True, indent=2), encoding="utf-8")

    return {
        "error": 0.0,
        "num_samples": num_samples,
        "rollout_chunks": rollout_chunks,
        "frames": 1 + 8 * rollout_chunks,
        "psnr_gt_mean": float(np.mean(psnr_values)),
        "ssim_gt_mean": float(np.mean(ssim_values)),
        "summary_path": str(summary_path),
        "video_paths": video_paths,
        "samples": sample_summaries,
    }
