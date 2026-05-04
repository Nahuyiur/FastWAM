#!/usr/bin/env python3
"""Run true RLBench success-rate evaluation for the RLBench FastWAM runs.

This script is intentionally standalone. It does not modify FastWAM training or
offline eval code; it only loads trained checkpoints and executes predicted
actions in the RLBench simulator.
"""

from __future__ import annotations

import argparse
import inspect
import json
import os
import re
import sys
import time
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import imageio.v2 as imageio
import numpy as np
import torch
from hydra import compose, initialize_config_dir
from hydra.core.global_hydra import GlobalHydra
from hydra.utils import instantiate
from omegaconf import OmegaConf
from PIL import Image


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
YUHAN_ROOT = Path(os.environ.get("YUHAN_ROOT", "/mnt/world_foundational_model/yuhan"))
DATA_ROOT = Path(
    os.environ.get(
        "RLBENCH_PICK_LIFT_ROOT",
        str(YUHAN_ROOT / "data/rlbench_pick_lift_color_shape"),
    )
)
RAW_TEST_ROOT = DATA_ROOT / "raw/test"
RLBENCH_ROOT = Path(os.environ.get("RLBENCH_ROOT", YUHAN_ROOT / "RLBench"))
PYREP_SITE = Path(
    os.environ.get(
        "RLBENCH_PYREP_SITE",
        str(YUHAN_ROOT / "miniconda3/envs/gembench/lib/python3.10/site-packages"),
    )
)

for path in [
    PROJECT_ROOT,
    SRC_ROOT,
    YUHAN_ROOT / "rlbench_lerobot_tools/stubs",
    RLBENCH_ROOT,
    PYREP_SITE,
]:
    path_str = str(path)
    if path_str not in sys.path:
        sys.path.insert(0, path_str)

from fastwam.datasets.lerobot.processors.fastwam_processor import FastWAMProcessor
from fastwam.datasets.lerobot.robot_video_dataset import DEFAULT_PROMPT
from fastwam.datasets.lerobot.utils.normalizer import load_dataset_stats_from_json


SHAPE_NAMES = ["cube", "cylinder", "triangular prism", "star", "moon"]
VARIANT_NAMES = [
    "original",
    "color_changed",
    "shape_changed",
    "color_and_shape_changed",
]
TASK_TO_VARIANT = {
    "rlbench_original_3cam224_1e-4": "original",
    "rlbench_color_3cam224_1e-4": "color_changed",
    "rlbench_shape_3cam224_1e-4": "shape_changed",
    "rlbench_color_shape_3cam224_1e-4": "color_and_shape_changed",
}


@dataclass(frozen=True)
class TrialSpec:
    trial_idx: int
    source: str
    variant_type: str
    variant_id: int
    variation_index: int
    shape: str
    shape_index: int
    color: str
    color_index: int
    instruction: str
    seed: int


def _now_slug() -> str:
    return datetime.now().strftime("%Y-%m-%d_%H-%M-%S")


def _mixed_precision_to_dtype(mixed_precision: str) -> torch.dtype:
    key = str(mixed_precision).strip().lower()
    if key == "no":
        return torch.float32
    if key == "fp16":
        return torch.float16
    if key == "bf16":
        return torch.bfloat16
    raise ValueError(f"Unsupported mixed precision: {mixed_precision}")


def _compose_cfg(task_name: str):
    if GlobalHydra.instance().is_initialized():
        GlobalHydra.instance().clear()
    with initialize_config_dir(version_base="1.3", config_dir=str(PROJECT_ROOT / "configs")):
        return compose(config_name="train.yaml", overrides=[f"task={task_name}"])


def _resolve_run(task_name: str) -> tuple[Path, Path, Path]:
    task_root = PROJECT_ROOT / "runs" / task_name
    ckpts = sorted(task_root.glob("*/checkpoints/weights/step_*.pt"))
    if not ckpts:
        raise FileNotFoundError(f"No checkpoints found under {task_root}")

    def _step(path: Path) -> int:
        match = re.search(r"step_(\d+)\.pt$", path.name)
        return int(match.group(1)) if match else -1

    ckpt = max(ckpts, key=_step)
    run_dir = ckpt.parents[2]
    stats_path = run_dir / "dataset_stats.json"
    if not stats_path.exists():
        raise FileNotFoundError(f"Missing dataset_stats.json: {stats_path}")
    return run_dir, ckpt, stats_path


def _load_policy(task_name: str, device: str, num_inference_steps: int):
    cfg = _compose_cfg(task_name)
    model_cfg = OmegaConf.create(OmegaConf.to_container(cfg.model, resolve=True))
    model_cfg.load_text_encoder = True
    dtype = _mixed_precision_to_dtype(str(cfg.get("mixed_precision", "bf16")))

    run_dir, ckpt, stats_path = _resolve_run(task_name)
    print(
        f"[load] task={task_name} ckpt={ckpt} stats={stats_path} device={device} dtype={dtype}",
        flush=True,
    )
    model = instantiate(model_cfg, model_dtype=dtype, device=device)
    model.load_checkpoint(str(ckpt))
    model = model.to(device).eval()

    processor: FastWAMProcessor = instantiate(cfg.data.train.processor).eval()
    processor.set_normalizer_from_stats(load_dataset_stats_from_json(str(stats_path)))

    action_horizon = int(cfg.data.train.num_frames) - 1
    video_frames = (
        (int(cfg.data.train.num_frames) - 1)
        // int(cfg.data.train.action_video_freq_ratio)
        + 1
    )
    if num_inference_steps <= 0:
        num_inference_steps = int(cfg.get("eval_num_inference_steps", 10))

    return {
        "cfg": cfg,
        "model": model,
        "processor": processor,
        "run_dir": run_dir,
        "ckpt": ckpt,
        "stats_path": stats_path,
        "action_horizon": action_horizon,
        "video_frames": video_frames,
        "num_inference_steps": num_inference_steps,
    }


def _resize_rgb(image: np.ndarray, size_wh: tuple[int, int]) -> np.ndarray:
    pil = Image.fromarray(image.astype(np.uint8), mode="RGB")
    return np.asarray(pil.resize(size_wh, resample=Image.BILINEAR), dtype=np.uint8)


def _normalize_state(state: np.ndarray, processor: FastWAMProcessor) -> torch.Tensor:
    state_meta = processor.shape_meta["state"]
    if len(state_meta) != 1:
        raise ValueError("Expected a single merged RLBench state key.")
    state_key = state_meta[0]["key"]
    batch = {
        "state": {
            state_key: torch.as_tensor(state, dtype=torch.float32).unsqueeze(0)
        }
    }
    batch = processor.action_state_transform(batch)
    batch = processor.normalizer.forward(batch)
    return batch["state"][state_key]


def _denormalize_action(action: torch.Tensor, processor: FastWAMProcessor) -> np.ndarray:
    if action.ndim == 2:
        action = action.unsqueeze(0)
    if action.ndim != 3:
        raise ValueError(f"Expected action [B,T,D], got {tuple(action.shape)}")
    action_meta = processor.shape_meta["action"]
    if len(action_meta) != 1:
        raise ValueError("Expected a single merged RLBench action key.")
    action_key = action_meta[0]["key"]
    normalizer = processor.normalizer.normalizers["action"][action_key]
    return normalizer.backward(action.to(dtype=torch.float32, device="cpu")).numpy()


def _obs_state(obs: Any) -> np.ndarray:
    return np.concatenate(
        [
            np.asarray(obs.joint_positions, dtype=np.float32),
            np.asarray([obs.gripper_open], dtype=np.float32),
        ],
        axis=0,
    )


def _obs_image_tensor(obs: Any, device: str, dtype: torch.dtype) -> torch.Tensor:
    front = _resize_rgb(np.asarray(obs.front_rgb), (224, 224))
    wrist = _resize_rgb(np.asarray(obs.wrist_rgb), (224, 224))
    overhead = _resize_rgb(np.asarray(obs.overhead_rgb), (224, 224))
    rgb = np.concatenate([front, wrist, overhead], axis=1)
    tensor = torch.from_numpy(rgb).permute(2, 0, 1).unsqueeze(0).to(
        device=device,
        dtype=dtype,
    )
    return tensor * (2.0 / 255.0) - 1.0


def _obs_video_frame(obs: Any) -> np.ndarray:
    return np.concatenate(
        [
            np.asarray(obs.front_rgb, dtype=np.uint8),
            np.asarray(obs.wrist_rgb, dtype=np.uint8),
            np.asarray(obs.overhead_rgb, dtype=np.uint8),
        ],
        axis=1,
    )


def _predict_action_chunk(
    policy: dict[str, Any],
    obs: Any,
    instruction: str,
    seed: int | None,
) -> np.ndarray:
    model = policy["model"]
    processor = policy["processor"]
    image = _obs_image_tensor(obs, device=str(model.device), dtype=model.torch_dtype)
    proprio = _normalize_state(_obs_state(obs), processor)
    prompt = DEFAULT_PROMPT.format(task=instruction)
    infer_kwargs = {
        "prompt": prompt,
        "input_image": image,
        "action_horizon": int(policy["action_horizon"]),
        "proprio": proprio,
        "negative_prompt": "",
        "text_cfg_scale": 1.0,
        "num_inference_steps": int(policy["num_inference_steps"]),
        "sigma_shift": None,
        "seed": seed,
        "rand_device": "cpu",
        "tiled": False,
    }
    if "num_video_frames" in inspect.signature(model.infer_action).parameters:
        infer_kwargs["num_video_frames"] = int(policy["video_frames"])
    with torch.no_grad():
        pred = model.infer_action(**infer_kwargs)
    action = _denormalize_action(pred["action"], processor)[0]
    action = np.asarray(action, dtype=np.float32)
    action[:, -1] = (action[:, -1] > 0.5).astype(np.float32)
    return action


def _load_raw_test_specs(variant_type: str) -> list[TrialSpec]:
    specs: list[TrialSpec] = []
    paths = sorted(RAW_TEST_ROOT.glob(f"*_{variant_type}_*.json"))
    for path in paths:
        data = json.loads(path.read_text())
        if str(data["variant_type"]) != variant_type:
            continue
        specs.append(
            TrialSpec(
                trial_idx=len(specs),
                source="raw_test",
                variant_type=str(data["variant_type"]),
                variant_id=int(data["variant_id"]),
                variation_index=int(data["variation_index"]),
                shape=str(data["shape"]),
                shape_index=int(data["shape_index"]),
                color=str(data["color"]),
                color_index=int(data["color_index"]),
                instruction=str(data["instruction"]),
                seed=int(data["seed"]),
            )
        )
    return specs


def _encode_variation(variant_id: int, color_index: int, shape_index: int) -> int:
    return variant_id * 100 + color_index * len(SHAPE_NAMES) + shape_index


def _fresh_spec(variant_type: str, local_idx: int, seed_base: int, colors: list[Any]) -> TrialSpec:
    variant_id = VARIANT_NAMES.index(variant_type)
    blue_index = next(i for i, (name, _) in enumerate(colors) if name == "blue")
    non_blue = [i for i, (name, _) in enumerate(colors) if name != "blue"]
    non_cube = list(range(1, len(SHAPE_NAMES)))

    if variant_type == "original":
        shape_index = 0
        color_index = blue_index
    elif variant_type == "color_changed":
        shape_index = 0
        color_index = non_blue[local_idx % len(non_blue)]
    elif variant_type == "shape_changed":
        shape_index = non_cube[local_idx % len(non_cube)]
        color_index = blue_index
    else:
        shape_index = non_cube[local_idx % len(non_cube)]
        color_index = non_blue[
            (local_idx // len(non_cube) + local_idx) % len(non_blue)
        ]

    color_name, _ = colors[color_index]
    shape_name = SHAPE_NAMES[shape_index]
    return TrialSpec(
        trial_idx=local_idx,
        source="fresh_holdout",
        variant_type=variant_type,
        variant_id=variant_id,
        variation_index=_encode_variation(variant_id, color_index, shape_index),
        shape=shape_name,
        shape_index=shape_index,
        color=color_name,
        color_index=color_index,
        instruction=f"pick up the {color_name} {shape_name} and lift it up to the target",
        seed=seed_base + variant_id * 1000 + local_idx,
    )


def _build_trial_specs(
    variant_type: str,
    trials: int,
    seed_base: int,
    colors: list[Any],
) -> list[TrialSpec]:
    specs = _load_raw_test_specs(variant_type)
    if len(specs) >= trials:
        return [
            TrialSpec(**{**asdict(spec), "trial_idx": i})
            for i, spec in enumerate(specs[:trials])
        ]
    out = [
        TrialSpec(**{**asdict(spec), "trial_idx": i})
        for i, spec in enumerate(specs)
    ]
    for i in range(len(out), trials):
        out.append(_fresh_spec(variant_type, i, seed_base=seed_base, colors=colors))
    return out


def _make_obs_config(renderer: str):
    from pyrep.const import RenderMode
    from rlbench import ObservationConfig

    obs_config = ObservationConfig()
    obs_config.set_all(False)
    render_mode = RenderMode.OPENGL3 if renderer == "opengl3" else RenderMode.OPENGL
    for cam in [
        obs_config.front_camera,
        obs_config.wrist_camera,
        obs_config.overhead_camera,
    ]:
        cam.rgb = True
        cam.depth = False
        cam.point_cloud = False
        cam.mask = False
        cam.image_size = [256, 256]
        cam.render_mode = render_mode
    obs_config.joint_positions = True
    obs_config.gripper_open = True
    obs_config.gripper_pose = False
    obs_config.joint_velocities = False
    obs_config.joint_forces = False
    return obs_config


def _make_task_class():
    from pyrep.objects import Dummy
    from pyrep.objects.proximity_sensor import ProximitySensor
    from pyrep.objects.shape import Shape
    from rlbench.backend.conditions import DetectedCondition, GraspedCondition
    from rlbench.backend.spawn_boundary import SpawnBoundary
    from rlbench.const import colors
    from rlbench.tasks.pick_and_lift_small import PickAndLiftSmall

    class RuntimePickAndLiftColorShape(PickAndLiftSmall):
        def __init__(self, pyrep, robot):
            super().__init__(pyrep, robot, name="pick_and_lift_small")

        def init_task(self) -> None:
            self._shapes = [Shape(ob.replace(" ", "_")) for ob in SHAPE_NAMES]
            self._grasp_points = [
                Dummy("%s_grasp_point" % ob.replace(" ", "_"))
                for ob in SHAPE_NAMES
            ]
            self._w1 = Dummy("waypoint1")
            self.register_graspable_objects(self._shapes)
            self.boundary = SpawnBoundary([Shape("pick_and_lift_boundary")])
            self.success_detector = ProximitySensor("pick_and_lift_success")

        def init_episode(self, index: int) -> list[str]:
            variant_id = index // 100
            local = index % 100
            color_index = local // len(SHAPE_NAMES)
            shape_index = local % len(SHAPE_NAMES)
            color_name, rgb = colors[color_index]
            shape_name = SHAPE_NAMES[shape_index]

            neutral = (0.5, 0.5, 0.5)
            for i, shape in enumerate(self._shapes):
                shape.set_color(rgb if i == shape_index else neutral)

            target_shape = self._shapes[shape_index]
            self.register_success_conditions(
                [
                    GraspedCondition(self.robot.gripper, target_shape),
                    DetectedCondition(target_shape, self.success_detector),
                ]
            )
            self.boundary.clear()
            self.boundary.sample(
                self.success_detector,
                min_rotation=(0.0, 0.0, 0.0),
                max_rotation=(0.0, 0.0, 0.0),
            )
            for shape in self._shapes:
                self.boundary.sample(shape, min_distance=0.1)
            self._w1.set_pose(self._grasp_points[shape_index].get_pose())

            if variant_id == 0:
                return ["pick up the blue cube and lift it up to the target"]
            return [
                f"pick up the {color_name} {shape_name} and lift it up to the target"
            ]

        def variation_count(self) -> int:
            return len(VARIANT_NAMES) * len(colors) * len(SHAPE_NAMES)

    return RuntimePickAndLiftColorShape


def _make_env(renderer: str):
    from rlbench.action_modes.action_mode import MoveArmThenGripper
    from rlbench.action_modes.arm_action_modes import JointPosition
    from rlbench.action_modes.gripper_action_modes import Discrete
    from rlbench.environment import Environment

    env = Environment(
        action_mode=MoveArmThenGripper(JointPosition(True), Discrete()),
        obs_config=_make_obs_config(renderer),
        headless=True,
    )
    env.launch()
    task_env = env.get_task(_make_task_class())
    return env, task_env


def _run_trial(
    task_env: Any,
    policy: dict[str, Any],
    spec: TrialSpec,
    args: argparse.Namespace,
    video_dir: Path,
) -> dict[str, Any]:
    start = time.time()
    np.random.seed(spec.seed)
    task_env.set_variation(spec.variation_index)
    # Match the data-generation reset order: one reset for descriptions, then
    # get_demos(live_demos=True) internally reset again before recording.
    task_env.reset()
    descriptions, obs = task_env.reset()
    instruction = spec.instruction or descriptions[0]

    frames: list[np.ndarray] = []
    if args.save_video:
        frames.append(_obs_video_frame(obs))

    pending: list[np.ndarray] = []
    success = False
    reward = 0.0
    done = False
    error = None
    steps = 0
    replans = 0
    try:
        for step_idx in range(int(args.max_steps)):
            if not pending:
                chunk = _predict_action_chunk(
                    policy,
                    obs,
                    instruction=instruction,
                    seed=None if args.model_seed < 0 else int(args.model_seed) + replans,
                )
                n = min(int(args.replan_steps), len(chunk))
                pending = [np.asarray(chunk[i], dtype=np.float32) for i in range(n)]
                replans += 1
            action = pending.pop(0)
            obs, reward, done = task_env.step(action)
            steps = step_idx + 1
            if args.save_video and (step_idx % int(args.video_stride) == 0):
                frames.append(_obs_video_frame(obs))
            if reward > 0 or done:
                success = bool(reward > 0)
                break
    except Exception as exc:  # RLBench can raise invalid action / sim errors.
        error = f"{type(exc).__name__}: {exc}"

    video_path = None
    if args.save_video:
        video_path = video_dir / f"{spec.trial_idx:03d}_{spec.source}_{spec.color}_{spec.shape}.mp4"
        imageio.mimwrite(str(video_path), frames, fps=int(args.video_fps), macro_block_size=1)

    return {
        **asdict(spec),
        "task": args.task,
        "success": bool(success),
        "reward": float(reward),
        "done": bool(done),
        "steps": int(steps),
        "replans": int(replans),
        "seconds": float(time.time() - start),
        "instruction": instruction,
        "video_path": str(video_path) if video_path is not None else None,
        "error": error,
    }


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=True) + "\n")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", required=True, choices=sorted(TASK_TO_VARIANT))
    parser.add_argument("--trials", type=int, default=20)
    parser.add_argument("--output-root", default=None)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--renderer", choices=["opengl", "opengl3"], default="opengl")
    parser.add_argument("--max-steps", type=int, default=240)
    parser.add_argument("--replan-steps", type=int, default=8)
    parser.add_argument("--num-inference-steps", type=int, default=10)
    parser.add_argument("--seed-base", type=int, default=20270505)
    parser.add_argument("--model-seed", type=int, default=-1)
    parser.add_argument("--save-video", action="store_true", default=True)
    parser.add_argument("--no-save-video", dest="save_video", action="store_false")
    parser.add_argument("--video-fps", type=int, default=8)
    parser.add_argument("--video-stride", type=int, default=2)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for this success eval, but torch.cuda.is_available() is false.")

    output_root = (
        Path(args.output_root)
        if args.output_root is not None
        else PROJECT_ROOT / "runs" / "rlbench_success_eval_20" / _now_slug()
    )
    task_dir = output_root / args.task
    video_dir = task_dir / "videos"
    task_dir.mkdir(parents=True, exist_ok=True)
    video_dir.mkdir(parents=True, exist_ok=True)

    from rlbench.const import colors

    variant_type = TASK_TO_VARIANT[args.task]
    specs = _build_trial_specs(
        variant_type=variant_type,
        trials=int(args.trials),
        seed_base=int(args.seed_base),
        colors=colors,
    )
    _write_json(task_dir / "trial_specs.json", [asdict(spec) for spec in specs])

    policy = _load_policy(
        task_name=args.task,
        device=args.device,
        num_inference_steps=int(args.num_inference_steps),
    )
    env, task_env = _make_env(renderer=str(args.renderer))
    results_path = task_dir / "results.jsonl"
    successes = 0
    results: list[dict[str, Any]] = []
    try:
        with results_path.open("a") as f:
            for spec in specs:
                result = _run_trial(task_env, policy, spec, args, video_dir)
                results.append(result)
                successes += int(result["success"])
                f.write(json.dumps(result, ensure_ascii=True) + "\n")
                f.flush()
                print(
                    "[trial] "
                    f"task={args.task} idx={spec.trial_idx:02d}/{len(specs)} "
                    f"variant={variant_type} success={result['success']} "
                    f"steps={result['steps']} rate={successes}/{len(results)} "
                    f"error={result['error']}",
                    flush=True,
                )
    finally:
        env.shutdown()

    summary = {
        "task": args.task,
        "variant_type": variant_type,
        "trials": len(results),
        "successes": successes,
        "success_rate": float(successes / max(len(results), 1)),
        "results_path": str(results_path),
        "output_dir": str(task_dir),
        "checkpoint": str(policy["ckpt"]),
        "dataset_stats": str(policy["stats_path"]),
        "renderer": args.renderer,
        "max_steps": int(args.max_steps),
        "replan_steps": int(args.replan_steps),
        "num_inference_steps": int(args.num_inference_steps),
    }
    _write_json(task_dir / "summary.json", summary)
    print("[summary] " + json.dumps(summary, ensure_ascii=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
