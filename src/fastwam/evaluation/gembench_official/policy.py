from __future__ import annotations

import hashlib
import inspect
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf
from PIL import Image

from fastwam.datasets.gembench.normalization import GEMBenchProcessorShim
from fastwam.datasets.gembench.policy_local_frame import (
    PolicyLocalFrameConfig,
    action_local_to_world,
    compute_policy_local_frame,
)
from fastwam.datasets.lerobot.robot_video_dataset import DEFAULT_PROMPT
from fastwam.datasets.lerobot.utils.normalizer import load_dataset_stats_from_json
from fastwam.runtime import _mixed_precision_to_model_dtype
from fastwam.utils import misc
from fastwam.utils.config_resolvers import register_default_resolvers

from .common import OFFICIAL_CAMERA_NAMES, PROJECT_ROOT, resolve_existing_path


class GEMBenchOfficialActioner:
    """FastWAM adapter for the official GEMBench single-action contract."""

    OFFICIAL_RLBENCH_TABLE_HEIGHT = 0.7505
    OFFICIAL_RLBENCH_MIN_ACTION_Z_MARGIN = 0.005

    def __init__(
        self,
        *,
        cfg: DictConfig,
        run_dir: Path,
        checkpoint: Path,
        device: str,
        mixed_precision: str | None,
        num_inference_steps: int,
        relation_mode: str,
        observation_camera_names: tuple[str, ...] = OFFICIAL_CAMERA_NAMES,
        rand_device: str = "cpu",
        model_seed: int = -1,
        tiled: bool = False,
        chunk_action_horizon: int | None = None,
        min_chunk_action_horizon: int | None = None,
    ):
        register_default_resolvers()
        self.cfg = cfg
        self.run_dir = Path(run_dir).resolve()
        self.checkpoint = Path(checkpoint).resolve()
        self.device = str(device)
        self.mixed_precision = mixed_precision or str(cfg.get("mixed_precision", "bf16"))
        self.num_inference_steps = int(num_inference_steps) if int(num_inference_steps) > 0 else int(
            cfg.get("eval_num_inference_steps", 10)
        )
        self.relation_mode = "none" if str(relation_mode) == "auto" else str(relation_mode)
        self.observation_camera_names = tuple(str(name) for name in observation_camera_names)
        self.rand_device = str(rand_device)
        self.model_seed = int(model_seed)
        self.tiled = bool(tiled)
        self.replans = 0

        data_train = cfg.data.train
        self.policy_contract = cfg.get("policy_contract", {}) or {}
        contract_action_horizon = self.policy_contract.get("action_horizon")
        self.training_action_horizon = int(data_train.get("action_horizon", 8))
        self.action_horizon = (
            int(contract_action_horizon) if contract_action_horizon is not None else self.training_action_horizon
        )
        self.policy_vgm_auxiliary_action_horizon = self.policy_contract.get("policy_vgm_auxiliary_action_horizon")
        auto_chunk_candidates = [int(self.action_horizon), int(self.training_action_horizon)]
        if self.policy_vgm_auxiliary_action_horizon is not None:
            auto_chunk_candidates.append(int(self.policy_vgm_auxiliary_action_horizon))
        if min_chunk_action_horizon is not None and int(min_chunk_action_horizon) > 0:
            auto_chunk_candidates.append(int(min_chunk_action_horizon))
        self.chunk_action_horizon = (
            int(chunk_action_horizon)
            if chunk_action_horizon is not None and int(chunk_action_horizon) > 0
            else (
                max(auto_chunk_candidates)
                if min_chunk_action_horizon is not None
                and int(min_chunk_action_horizon) > int(self.action_horizon)
                else int(self.action_horizon)
            )
        )
        self.executed_action_index = 0
        self.video_size = [int(v) for v in data_train.get("video_size", [224, 672])]
        self.camera_order = [str(v) for v in data_train.get("camera_order", ["front", "wrist", "left_shoulder"])]
        self.norm_default_mode = str(data_train.get("norm_default_mode", "-2.0/2.0"))
        self.policy_target_frame = str(
            self.policy_contract.get("policy_target_frame", data_train.get("policy_target_frame", "world"))
        )
        if self.policy_target_frame not in ("world", "official_pcd_local"):
            raise ValueError(
                f"Unsupported policy_target_frame={self.policy_target_frame!r}; "
                "expected 'world' or 'official_pcd_local'."
            )
        self.policy_local_eval_config = PolicyLocalFrameConfig(
            enabled=(self.policy_target_frame == "official_pcd_local"),
            xyz_shift=str(
                self.policy_contract.get(
                    "policy_local_xyz_shift", data_train.get("policy_local_xyz_shift", "center")
                )
            ),
            xyz_norm=bool(
                self.policy_contract.get(
                    "policy_local_xyz_norm", data_train.get("policy_local_xyz_norm", False)
                )
            ),
            rm_table=bool(
                self.policy_contract.get(
                    "policy_local_rm_table", data_train.get("policy_local_rm_table", True)
                )
            ),
            rm_robot=str(
                self.policy_contract.get(
                    "policy_local_eval_rm_robot",
                    self.policy_contract.get(
                        "policy_local_rm_robot", data_train.get("policy_local_rm_robot", "none")
                    ),
                )
            ),
            num_points=int(
                self.policy_contract.get(
                    "policy_local_num_points", data_train.get("policy_local_num_points", 4096)
                )
            ),
            sample_seed=int(
                self.policy_contract.get(
                    "policy_local_sample_seed", data_train.get("policy_local_sample_seed", 0)
                )
            ),
            voxel_size=float(
                self.policy_contract.get(
                    "policy_local_eval_voxel_size", data_train.get("policy_local_eval_voxel_size", 0.01)
                )
            ),
            require_open3d=bool(
                self.policy_contract.get(
                    "policy_local_require_open3d", data_train.get("policy_local_require_open3d", False)
                )
            ),
        )
        frame_offsets = data_train.get("frame_offsets")
        self.num_video_frames = (
            len(frame_offsets) if frame_offsets is not None else int(data_train.get("num_video_frames", 9))
        )
        self.action_video_freq_ratio = int(data_train.get("action_video_freq_ratio", 4))
        self.min_action_z = self.OFFICIAL_RLBENCH_TABLE_HEIGHT + self.OFFICIAL_RLBENCH_MIN_ACTION_Z_MARGIN

        self.processor = self._build_processor(data_train)
        self.model = self._load_model()
        self.has_relation = False
        if getattr(self.model, "relation_expert", None) is not None:
            raise ValueError(
                "This FastWAM repo actioner is intentionally relation-free. "
                "Use a relation-capable evaluator for relation checkpoints."
            )
        if self.relation_mode != "none":
            raise ValueError(
                f"FastWAM official eval only supports --relation-mode=none/auto, got {self.relation_mode!r}."
            )

    @classmethod
    def from_run_dir(
        cls,
        *,
        run_dir: Path,
        checkpoint: Path,
        device: str,
        mixed_precision: str | None,
        num_inference_steps: int,
        relation_mode: str,
        observation_camera_names: tuple[str, ...] = OFFICIAL_CAMERA_NAMES,
        rand_device: str = "cpu",
        model_seed: int = -1,
        tiled: bool = False,
        chunk_action_horizon: int | None = None,
        min_chunk_action_horizon: int | None = None,
    ) -> "GEMBenchOfficialActioner":
        run_dir = Path(run_dir).resolve()
        cfg_path = run_dir / "config.yaml"
        if not cfg_path.exists():
            raise FileNotFoundError(f"Missing run config: {cfg_path}")
        cfg = OmegaConf.load(cfg_path)
        misc.register_work_dir(str(run_dir))
        return cls(
            cfg=cfg,
            run_dir=run_dir,
            checkpoint=checkpoint,
            device=device,
            mixed_precision=mixed_precision,
            num_inference_steps=num_inference_steps,
            relation_mode=relation_mode,
            observation_camera_names=observation_camera_names,
            rand_device=rand_device,
            model_seed=model_seed,
            tiled=tiled,
            chunk_action_horizon=chunk_action_horizon,
            min_chunk_action_horizon=min_chunk_action_horizon,
        )

    def _build_processor(self, data_train: DictConfig) -> GEMBenchProcessorShim:
        stats_path_raw = data_train.get("pretrained_norm_stats")
        if stats_path_raw is None:
            raise ValueError("Official GEMBench eval requires data.train.pretrained_norm_stats for action/proprio scaling.")
        stats_path = resolve_existing_path(
            stats_path_raw,
            label="pretrained_norm_stats",
            bases=[self.run_dir, PROJECT_ROOT],
        )
        self.stats_path = stats_path
        processor_cfg = data_train.get("processor", {})
        return GEMBenchProcessorShim(
            load_dataset_stats_from_json(str(stats_path)),
            action_dim=int(processor_cfg.get("action_output_dim", 8)),
            proprio_dim=int(processor_cfg.get("proprio_output_dim", 8)),
            norm_default_mode=self.norm_default_mode,
        )

    def _load_model(self):
        model_cfg = OmegaConf.create(OmegaConf.to_container(self.cfg.model, resolve=True))
        model_cfg.load_text_encoder = True
        dtype = _mixed_precision_to_model_dtype(self.mixed_precision)
        start = time.time()
        print(f"[gembench-actioner] instantiate_model start device={self.device} dtype={dtype}", flush=True)
        model = instantiate(model_cfg, model_dtype=dtype, device=self.device)
        print(f"[gembench-actioner] instantiate_model done seconds={time.time() - start:.1f}", flush=True)
        start = time.time()
        print(f"[gembench-actioner] load_checkpoint start checkpoint={self.checkpoint}", flush=True)
        model.load_checkpoint(str(self.checkpoint))
        print(f"[gembench-actioner] load_checkpoint done seconds={time.time() - start:.1f}", flush=True)
        start = time.time()
        model = model.to(self.device).eval()
        print(f"[gembench-actioner] model_to_eval done seconds={time.time() - start:.1f}", flush=True)
        return model

    def _camera_index(self, name: str) -> int:
        try:
            return self.observation_camera_names.index(name)
        except ValueError as exc:
            raise ValueError(
                f"Training camera {name!r} is not present in observation cameras {self.observation_camera_names}."
            ) from exc

    @staticmethod
    def _as_uint8_rgb(image: np.ndarray) -> np.ndarray:
        arr = np.asarray(image)
        if arr.dtype != np.uint8:
            if np.issubdtype(arr.dtype, np.floating) and float(np.nanmax(arr)) <= 1.0:
                arr = arr * 255.0
            arr = np.clip(arr, 0, 255).astype(np.uint8)
        if arr.ndim != 3 or arr.shape[-1] != 3:
            raise ValueError(f"Expected RGB image [H,W,3], got {arr.shape}")
        return arr

    def _obs_image_tensor(self, obs_state_dict: dict[str, Any]) -> torch.Tensor:
        rgb = np.asarray(obs_state_dict.get("rgb"))
        if rgb.ndim != 4:
            raise ValueError(f"Official observation must contain rgb [C,H,W,3], got {rgb.shape}")
        camera_h = int(self.video_size[0])
        camera_w = int(self.video_size[1]) // len(self.camera_order)
        frames = []
        for camera in self.camera_order:
            image = self._as_uint8_rgb(rgb[self._camera_index(camera)])
            pil = Image.fromarray(image, mode="RGB").resize((camera_w, camera_h), resample=Image.BILINEAR)
            frames.append(np.asarray(pil, dtype=np.uint8))
        cat = np.concatenate(frames, axis=1)
        tensor = torch.from_numpy(cat).permute(2, 0, 1).unsqueeze(0).to(device=self.device, dtype=self.model.torch_dtype)
        return tensor * (2.0 / 255.0) - 1.0

    def _normalize_proprio(self, gripper: np.ndarray) -> torch.Tensor:
        proprio = torch.as_tensor(np.asarray(gripper, dtype=np.float32), dtype=torch.float32).reshape(1, -1)
        batch = {"state": {"default": proprio}}
        batch = self.processor.normalizer.forward(batch)
        return batch["state"]["default"].reshape(-1)

    def _denormalize_action_chunk(self, action: torch.Tensor) -> np.ndarray:
        if action.ndim == 2:
            action = action.unsqueeze(0)
        action_cpu = action.detach().to(dtype=torch.float32, device="cpu")
        dummy_state = torch.zeros(
            (*action_cpu.shape[:-1], int(self.processor.proprio_output_dim)),
            dtype=torch.float32,
        )
        batch = {
            "action": {"default": action_cpu},
            "state": {"default": dummy_state},
        }
        batch = self.processor.normalizer.backward(batch)
        out = batch["action"]["default"][0].numpy().astype(np.float32)
        return out

    def _postprocess_action(self, action: np.ndarray) -> np.ndarray:
        out = np.asarray(action, dtype=np.float32).copy()
        out[2] = max(float(out[2]), float(self.min_action_z))
        quat = out[3:7]
        norm = float(np.linalg.norm(quat))
        if np.isfinite(norm) and norm > 1.0e-6:
            out[3:7] = quat / norm
        else:
            out[3:7] = np.asarray([0.0, 0.0, 0.0, 1.0], dtype=np.float32)
        out[7] = 1.0 if float(out[7]) > 0.5 else 0.0
        return out

    def _policy_local_frame(self, obs_state_dict: dict[str, Any], *, taskvar: str, episode_id: int, step_id: int) -> dict[str, Any] | None:
        if self.policy_target_frame != "official_pcd_local":
            return None
        pc = np.asarray(obs_state_dict.get("pc"), dtype=np.float64)
        if pc.ndim != 4 or pc.shape[-1] != 3:
            raise ValueError(
                "policy_target_frame='official_pcd_local' requires obs_state_dict['pc'] with shape [C,H,W,3], "
                f"got {pc.shape}"
            )
        rgb = obs_state_dict.get("rgb")
        gripper = np.asarray(obs_state_dict.get("gripper"), dtype=np.float32)
        arm_links_info = obs_state_dict.get("arm_links_info")
        return compute_policy_local_frame(
            xyz=pc,
            rgb=None if rgb is None else np.asarray(rgb),
            ee_pose=gripper,
            arm_links_info=arm_links_info,
            config=self.policy_local_eval_config,
            sample_seed=int(self.policy_local_eval_config.sample_seed)
            + int(episode_id) * 1009
            + int(step_id) * 9176,
        )

    def _prediction_seed(self, *, taskvar: str, episode_id: int, step_id: int) -> int | None:
        if self.model_seed < 0:
            return None
        material = f"{self.model_seed}:{taskvar}:{int(episode_id)}:{int(step_id)}".encode("utf-8")
        return int.from_bytes(hashlib.sha256(material).digest()[:8], "little") % (2**63 - 1)

    @staticmethod
    def _instruction_text(instructions: Any) -> str:
        if isinstance(instructions, (list, tuple)) and instructions:
            return str(instructions[0])
        return str(instructions)

    def predict_chunk(
        self,
        *,
        task_str: str,
        variation: int,
        step_id: int,
        obs_state_dict: dict[str, Any],
        episode_id: int,
        instructions: Any,
        return_predicted_video: bool = False,
    ) -> dict[str, Any]:
        instruction = self._instruction_text(instructions)
        taskvar = f"{task_str}+{int(variation)}"
        image = self._obs_image_tensor(obs_state_dict)
        gripper = np.asarray(obs_state_dict.get("gripper"), dtype=np.float32)
        if gripper.shape[-1] != 8:
            raise ValueError(f"Official observation gripper state must be 8D, got {gripper.shape}")
        proprio = self._normalize_proprio(gripper)
        relation_summary = {"mode": "none", "valid_edges": None}

        seed = self._prediction_seed(taskvar=taskvar, episode_id=episode_id, step_id=step_id)
        requested_action_horizon = int(self.chunk_action_horizon)
        infer_kwargs = {
            "prompt": DEFAULT_PROMPT.format(task=instruction),
            "input_image": image,
            "action_horizon": requested_action_horizon,
            "proprio": proprio,
            "negative_prompt": "",
            "text_cfg_scale": 1.0,
            "num_inference_steps": self.num_inference_steps,
            "sigma_shift": None,
            "seed": seed,
            "rand_device": self.rand_device,
            "tiled": self.tiled,
        }
        infer_fn = self.model.infer_action
        if return_predicted_video:
            infer_joint = getattr(self.model, "infer_joint", None)
            if infer_joint is None:
                raise ValueError("return_predicted_video=True requires model.infer_joint(...).")
            infer_fn = infer_joint
            infer_kwargs["num_video_frames"] = self.num_video_frames
            if "test_action_with_infer_action" in inspect.signature(infer_joint).parameters:
                infer_kwargs["test_action_with_infer_action"] = False
        elif "num_video_frames" in inspect.signature(self.model.infer_action).parameters:
            infer_kwargs["num_video_frames"] = self.num_video_frames
        with torch.no_grad():
            pred = infer_fn(**infer_kwargs)
        self.replans += 1
        normalized_action = pred["action"].detach().to(dtype=torch.float32, device="cpu")
        if normalized_action.ndim == 2:
            normalized_chunk = normalized_action.numpy().astype(np.float32)
        elif normalized_action.ndim == 3 and normalized_action.shape[0] == 1:
            normalized_chunk = normalized_action[0].numpy().astype(np.float32)
        else:
            raise ValueError(
                "Official GEMBench actioner expects a normalized action chunk [T,8] or [1,T,8], "
                f"got {tuple(normalized_action.shape)}"
            )
        if normalized_chunk.ndim != 2 or normalized_chunk.shape[-1] != 8 or normalized_chunk.shape[0] < 1:
            raise ValueError(f"Normalized action chunk must be non-empty [T,8], got {normalized_chunk.shape}")
        model_denormalized_action = self._denormalize_action_chunk(normalized_action)
        if (
            model_denormalized_action.ndim != 2
            or model_denormalized_action.shape[-1] != 8
            or model_denormalized_action.shape[0] < 1
        ):
            raise ValueError(
                f"Denormalized action chunk must be non-empty [T,8], got {model_denormalized_action.shape}"
            )
        if int(model_denormalized_action.shape[0]) != int(requested_action_horizon):
            raise ValueError(
                f"Predicted action chunk horizon={model_denormalized_action.shape[0]} does not match "
                f"requested chunk_action_horizon={requested_action_horizon}."
            )
        policy_local_frame = self._policy_local_frame(
            obs_state_dict,
            taskvar=taskvar,
            episode_id=episode_id,
            step_id=step_id,
        )
        if policy_local_frame is None:
            world_denormalized_action = model_denormalized_action
            policy_frame_summary = {"mode": "world"}
        else:
            world_denormalized_action = action_local_to_world(model_denormalized_action, policy_local_frame)
            policy_frame_summary = {
                "mode": "official_pcd_local",
                "centroid": [float(v) for v in np.asarray(policy_local_frame["centroid"]).reshape(3).tolist()],
                "radius": float(policy_local_frame["radius"]),
                "xyz_shift": policy_local_frame["xyz_shift"],
                "xyz_norm": bool(policy_local_frame["xyz_norm"]),
                "rm_table": bool(policy_local_frame["rm_table"]),
                "rm_robot": str(policy_local_frame["rm_robot"]),
                "voxel_size": float(policy_local_frame["voxel_size"]),
                "voxel_filter": str(policy_local_frame["voxel_filter"]),
                "robot_filter": str(policy_local_frame["robot_filter"]),
                "points_used": int(policy_local_frame["points_used"]),
            }
        executed_chunk = np.stack([self._postprocess_action(action) for action in world_denormalized_action], axis=0)
        z_delta = executed_chunk[:, 2] - world_denormalized_action[:, 2]
        denorm_delta = model_denormalized_action - normalized_chunk
        return {
            "action_chunk": executed_chunk.astype(np.float32),
            "normalized_action_chunk": normalized_chunk.astype(np.float32),
            "denormalized_action_chunk": world_denormalized_action.astype(np.float32),
            "model_denormalized_action_chunk": model_denormalized_action.astype(np.float32),
            "instruction": instruction,
            "step_id": int(step_id),
            "relation": relation_summary,
            "policy_target_frame": self.policy_target_frame,
            "policy_local_frame": policy_frame_summary,
            "chunk_horizon": int(world_denormalized_action.shape[0]),
            "chunk_action_horizon": int(requested_action_horizon),
            "policy_action_horizon": int(self.action_horizon),
            "training_action_horizon": int(self.training_action_horizon),
            "policy_vgm_auxiliary_action_horizon": (
                None
                if self.policy_vgm_auxiliary_action_horizon is None
                else int(self.policy_vgm_auxiliary_action_horizon)
            ),
            "normalization": {
                "stats_path": str(self.stats_path),
                "norm_default_mode": self.norm_default_mode,
                "max_abs_denorm_delta": float(np.max(np.abs(denorm_delta))),
                "normalized_min": float(np.min(normalized_chunk)),
                "normalized_max": float(np.max(normalized_chunk)),
                "denormalized_min": float(np.min(world_denormalized_action)),
                "denormalized_max": float(np.max(world_denormalized_action)),
                "model_denormalized_min": float(np.min(model_denormalized_action)),
                "model_denormalized_max": float(np.max(model_denormalized_action)),
                "world_denormalized_min": float(np.min(world_denormalized_action)),
                "world_denormalized_max": float(np.max(world_denormalized_action)),
            },
            "postprocess": {
                "table_height": float(self.OFFICIAL_RLBENCH_TABLE_HEIGHT),
                "min_action_z": float(self.min_action_z),
                "z_clamped_count": int(np.count_nonzero(z_delta > 0.0)),
                "max_z_clamp_delta": float(np.max(z_delta)) if z_delta.size else 0.0,
            },
            "num_inference_steps": self.num_inference_steps,
            "num_video_frames": self.num_video_frames,
            "action_video_freq_ratio": self.action_video_freq_ratio,
            "predicted_video": pred.get("video") if return_predicted_video else None,
        }

    def predict(
        self,
        *,
        task_str: str,
        variation: int,
        step_id: int,
        obs_state_dict: dict[str, Any],
        episode_id: int,
        instructions: Any,
    ) -> dict[str, Any]:
        output = self.predict_chunk(
            task_str=task_str,
            variation=variation,
            step_id=step_id,
            obs_state_dict=obs_state_dict,
            episode_id=episode_id,
            instructions=instructions,
            return_predicted_video=False,
        )
        executed_index = int(self.executed_action_index)
        action_chunk = np.asarray(output["action_chunk"], dtype=np.float32)
        normalized_chunk = np.asarray(output["normalized_action_chunk"], dtype=np.float32)
        denormalized_chunk = np.asarray(output["denormalized_action_chunk"], dtype=np.float32)
        model_denormalized_chunk = np.asarray(output["model_denormalized_action_chunk"], dtype=np.float32)
        if executed_index >= int(action_chunk.shape[0]):
            raise ValueError(
                f"executed_action_index={executed_index} is outside denormalized action chunk "
                f"with horizon={action_chunk.shape[0]}"
            )
        return {
            "action": action_chunk[executed_index],
            "normalized_action": normalized_chunk[executed_index],
            "denormalized_action": denormalized_chunk[executed_index],
            "model_denormalized_action": model_denormalized_chunk[executed_index],
            "instruction": output["instruction"],
            "step_id": int(step_id),
            "relation": output["relation"],
            "policy_target_frame": output["policy_target_frame"],
            "policy_local_frame": output["policy_local_frame"],
            "chunk_horizon": int(output["chunk_horizon"]),
            "chunk_action_horizon": int(output["chunk_action_horizon"]),
            "executed_action_index": executed_index,
            "policy_action_horizon": int(output["policy_action_horizon"]),
            "training_action_horizon": int(output["training_action_horizon"]),
            "policy_vgm_auxiliary_action_horizon": output["policy_vgm_auxiliary_action_horizon"],
            "normalization": output["normalization"],
            "postprocess": output["postprocess"],
            "num_inference_steps": output["num_inference_steps"],
            "num_video_frames": output["num_video_frames"],
            "action_video_freq_ratio": output["action_video_freq_ratio"],
        }
