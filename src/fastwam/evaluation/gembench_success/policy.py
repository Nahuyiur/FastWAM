from __future__ import annotations

import inspect
import logging
import re
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import torch
from hydra import compose, initialize_config_dir
from hydra.core.global_hydra import GlobalHydra
from hydra.utils import instantiate
from omegaconf import OmegaConf
from PIL import Image

from fastwam.datasets.gembench.normalization import GEMBenchProcessorShim, load_or_create_stats
from fastwam.datasets.lerobot.robot_video_dataset import DEFAULT_PROMPT

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parents[4]


def _mixed_precision_to_dtype(mixed_precision: str) -> torch.dtype:
    key = str(mixed_precision).strip().lower()
    if key == "no":
        return torch.float32
    if key == "fp16":
        return torch.float16
    if key == "bf16":
        return torch.bfloat16
    raise ValueError(f"Unsupported mixed precision: {mixed_precision}")


def _compose_train_cfg(task_name: str):
    if GlobalHydra.instance().is_initialized():
        GlobalHydra.instance().clear()
    with initialize_config_dir(version_base="1.3", config_dir=str(PROJECT_ROOT / "configs")):
        return compose(config_name="train.yaml", overrides=[f"task={task_name}"])


def _checkpoint_run_dir(checkpoint: Path) -> Path | None:
    parts = checkpoint.parts
    if len(parts) >= 4 and parts[-4:-2] == ("checkpoints", "weights"):
        return checkpoint.parents[2]
    for parent in checkpoint.parents:
        if (parent / "config.yaml").exists():
            return parent
    return None


def _load_run_cfg(checkpoint: Path):
    run_dir = _checkpoint_run_dir(checkpoint)
    if run_dir is None:
        return None, None
    config_path = run_dir / "config.yaml"
    if not config_path.exists():
        return run_dir, None
    return run_dir, OmegaConf.load(config_path)


def _latest_checkpoint(runs_root: Path) -> Path:
    candidates = sorted(runs_root.glob("gembench*/checkpoints/weights/step_*.pt"))
    if not candidates:
        candidates = sorted(runs_root.glob("**/checkpoints/weights/step_*.pt"))
    if not candidates:
        raise FileNotFoundError(f"No FastWAM checkpoint found under {runs_root}")

    def step(path: Path) -> int:
        match = re.search(r"step_(\d+)\.pt$", path.name)
        return int(match.group(1)) if match else -1

    return max(candidates, key=step)


def _resize_rgb(image: np.ndarray, size_wh: tuple[int, int]) -> np.ndarray:
    pil = Image.fromarray(np.asarray(image, dtype=np.uint8), mode="RGB")
    return np.asarray(pil.resize(size_wh, resample=Image.BILINEAR), dtype=np.uint8)


def _normalize_quaternion(quat: np.ndarray) -> np.ndarray:
    quat = np.asarray(quat, dtype=np.float32)
    norm = float(np.linalg.norm(quat))
    if not np.isfinite(norm) or norm < 1e-6:
        return np.asarray([0.0, 0.0, 0.0, 1.0], dtype=np.float32)
    return (quat / norm).astype(np.float32)


class FastWAMGEMBenchPolicy:
    """FastWAM policy adapter for official GEMBench simulator evaluation.

    The official GEMBench evaluator calls a policy once per high-level
    environment step and expects one 8D action:
    ``[x, y, z, qx, qy, qz, qw, gripper_open]``.
    FastWAM predicts a chunk, so this adapter caches the chunk and emits
    ``replan_steps`` actions before querying the model again.
    """

    def __init__(
        self,
        *,
        checkpoint: str | None = None,
        task_name: str = "gembench_keysteps_bbox_3cam224_1e-4",
        device: str = "cuda",
        num_inference_steps: int = 10,
        replan_steps: int = 1,
        model_seed: int = -1,
        stats_path: str | None = None,
        min_z: float | None = None,
    ):
        self.task_name = str(task_name)
        self.device = str(device)
        self.num_inference_steps = int(num_inference_steps)
        self.replan_steps = max(1, int(replan_steps))
        self.model_seed = int(model_seed)
        self.min_z = None if min_z is None else float(min_z)
        ckpt = Path(checkpoint).expanduser().resolve() if checkpoint else _latest_checkpoint(PROJECT_ROOT / "runs")
        if not ckpt.exists():
            raise FileNotFoundError(f"FastWAM checkpoint not found: {ckpt}")
        self.run_dir, run_cfg = _load_run_cfg(ckpt)
        if run_cfg is not None:
            self.cfg = run_cfg
            logger.info("Loaded GEMBench eval config from checkpoint run: %s", self.run_dir / "config.yaml")
        else:
            if checkpoint is not None:
                logger.warning(
                    "Could not find config.yaml next to checkpoint %s; falling back to task config %s.",
                    ckpt,
                    self.task_name,
                )
            self.cfg = _compose_train_cfg(self.task_name)

        model_cfg = OmegaConf.create(OmegaConf.to_container(self.cfg.model, resolve=True))
        model_cfg.load_text_encoder = True
        dtype = _mixed_precision_to_dtype(str(self.cfg.get("mixed_precision", "bf16")))

        logger.info("Loading FastWAM checkpoint for GEMBench success eval: %s", ckpt)
        self.model = instantiate(model_cfg, model_dtype=dtype, device=self.device)
        self.model.load_checkpoint(str(ckpt))
        self.model = self.model.to(self.device).eval()
        self.checkpoint = ckpt

        data_train = self.cfg.data.train
        self.camera_order = [str(x) for x in data_train.camera_order]
        self.camera_width = int(data_train.video_size[1]) // len(self.camera_order)
        self.camera_height = int(data_train.video_size[0])
        self.video_width = int(data_train.video_size[1])
        self.video_height = int(data_train.video_size[0])
        self.action_horizon = int(data_train.action_horizon)

        resolved_stats = stats_path or data_train.get("pretrained_norm_stats")
        if resolved_stats is None:
            raise ValueError("GEMBench success eval requires explicit action/state normalization stats.")
        stats_file = Path(str(resolved_stats)).expanduser()
        if not stats_file.is_absolute():
            stats_file = PROJECT_ROOT / stats_file
        if not stats_file.exists():
            raise FileNotFoundError(
                f"GEMBench normalization stats not found: {stats_file}. "
                "Do not run simulator eval with implicit default stats."
            )
        stats = load_or_create_stats(str(stats_file), action_dim=8, state_dim=8, save_if_missing=False)
        self.processor = GEMBenchProcessorShim(
            stats,
            action_dim=8,
            proprio_dim=8,
            norm_default_mode=str(data_train.get("norm_default_mode", "-2.0/2.0")),
        )
        self.pending_actions: list[np.ndarray] = []
        self.replan_count = 0

    @property
    def actioner_checkpoint_value(self) -> str:
        return str(self.checkpoint)

    def reset_episode(self) -> None:
        self.pending_actions.clear()
        self.replan_count = 0

    def set_model_seed(self, seed: int | None) -> None:
        self.model_seed = -1 if seed is None else int(seed)
        self.replan_count = 0

    def _obs_image_tensor(self, obs_state_dict: dict[str, Any]) -> torch.Tensor:
        if "rgb_by_name" in obs_state_dict:
            by_name = obs_state_dict["rgb_by_name"]
            frames = [np.asarray(by_name[name], dtype=np.uint8) for name in self.camera_order]
        else:
            rgb = np.asarray(obs_state_dict["rgb"], dtype=np.uint8)
            if rgb.ndim != 4 or rgb.shape[-1] != 3:
                raise ValueError(f"Expected obs rgb [N,H,W,3], got {rgb.shape}")
            if len(rgb) < len(self.camera_order):
                raise ValueError(f"Need {len(self.camera_order)} cameras, got {len(rgb)}")
            frames = [rgb[i] for i in range(len(self.camera_order))]
        frames = [_resize_rgb(frame, (self.camera_width, self.camera_height)) for frame in frames]
        image = np.concatenate(frames, axis=1)
        if image.shape[:2] != (self.video_height, self.video_width):
            raise ValueError(f"Unexpected concatenated image shape: {image.shape}")
        tensor = torch.from_numpy(image).permute(2, 0, 1).unsqueeze(0)
        tensor = tensor.to(device=self.model.device, dtype=self.model.torch_dtype)
        return tensor * (2.0 / 255.0) - 1.0

    def _normalize_proprio(self, gripper: np.ndarray) -> torch.Tensor:
        gripper = np.asarray(gripper, dtype=np.float32)
        if gripper.shape != (8,):
            raise ValueError(f"Expected GEMBench gripper state [8], got {gripper.shape}")
        batch = {"state": {"default": torch.from_numpy(gripper).to(torch.float32).unsqueeze(0)}}
        batch = self.processor.normalizer.forward(batch)
        proprio = batch["state"]["default"]
        return proprio.to(device=self.model.device, dtype=self.model.torch_dtype)

    def _denormalize_actions(self, action: torch.Tensor) -> np.ndarray:
        if action.ndim == 3:
            action = action[0]
        if action.ndim != 2 or action.shape[-1] != 8:
            raise ValueError(f"Expected action [T,8] or [1,T,8], got {tuple(action.shape)}")
        normalizer = self.processor.normalizer.normalizers["action"]["default"]
        out = normalizer.backward(action.to(dtype=torch.float32, device="cpu")).numpy()
        out = np.asarray(out, dtype=np.float32)
        out[:, 3:7] = np.stack([_normalize_quaternion(row[3:7]) for row in out], axis=0)
        out[:, 7] = (out[:, 7] > 0.5).astype(np.float32)
        if self.min_z is not None:
            out[:, 2] = np.maximum(out[:, 2], self.min_z)
        xyz_min = out[:, :3].min(axis=0)
        xyz_max = out[:, :3].max(axis=0)
        logger.info(
            "Predicted GEMBench action chunk stats: horizon=%d xyz_min=%s xyz_max=%s gripper_open_mean=%.3f",
            out.shape[0],
            np.array2string(xyz_min, precision=3),
            np.array2string(xyz_max, precision=3),
            float(out[:, 7].mean()),
        )
        return out

    def predict_chunk(
        self,
        *,
        obs_state_dict: dict[str, Any],
        instructions: Sequence[str],
    ) -> np.ndarray:
        instruction = str(instructions[0]) if instructions else ""
        prompt = DEFAULT_PROMPT.format(task=instruction)
        image = self._obs_image_tensor(obs_state_dict)
        proprio = self._normalize_proprio(np.asarray(obs_state_dict["gripper"], dtype=np.float32))
        seed = None if self.model_seed < 0 else self.model_seed + self.replan_count
        infer_kwargs = {
            "prompt": prompt,
            "input_image": image,
            "action_horizon": self.action_horizon,
            "proprio": proprio,
            "negative_prompt": "",
            "text_cfg_scale": 1.0,
            "num_inference_steps": self.num_inference_steps,
            "sigma_shift": None,
            "seed": seed,
            "rand_device": "cpu",
            "tiled": False,
        }
        if "num_video_frames" in inspect.signature(self.model.infer_action).parameters:
            infer_kwargs["num_video_frames"] = int(self.cfg.data.train.num_video_frames)
        with torch.no_grad():
            pred = self.model.infer_action(**infer_kwargs)
        self.replan_count += 1
        return self._denormalize_actions(pred["action"])

    def predict(
        self,
        *,
        task_str: str | None = None,
        variation: int | None = None,
        step_id: int | None = None,
        obs_state_dict: dict[str, Any],
        episode_id: int | str | None = None,
        instructions: Sequence[str] | None = None,
    ) -> dict[str, np.ndarray]:
        del task_str, variation, step_id, episode_id
        if not self.pending_actions:
            chunk = self.predict_chunk(
                obs_state_dict=obs_state_dict,
                instructions=list(instructions or []),
            )
            n = min(self.replan_steps, len(chunk))
            self.pending_actions = [np.asarray(chunk[i], dtype=np.float32) for i in range(n)]
        return {"action": self.pending_actions.pop(0)}
