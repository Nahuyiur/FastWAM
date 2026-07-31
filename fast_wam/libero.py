"""Small LIBERO adapter shared by reference and Megatron acceptance runners."""

from __future__ import annotations

from typing import Any

import numpy as np
import torch
import torch.distributed as dist


SUITE_MAX_STEPS = {
    "libero_spatial": 280,
    "libero_object": 280,
    "libero_goal": 300,
    "libero_10": 520,
}


def make_env(case: dict[str, Any]):
    from libero.libero import benchmark
    from lerobot.envs.libero import LiberoEnv

    suite_name = case["suite"]
    suite = benchmark.get_benchmark_dict()[suite_name]()
    return LiberoEnv(
        task_suite=suite,
        task_id=int(case["task_id"]),
        task_suite_name=suite_name,
        obs_type="pixels_agent_pos",
        observation_width=256,
        observation_height=256,
        init_states=True,
        episode_index=int(case["init_state_id"]),
        n_envs=1,
        num_steps_wait=int(case.get("num_steps_wait", 10)),
        control_mode="relative",
        episode_length=int(case.get("episode_length", SUITE_MAX_STEPS[suite_name])),
    )


def _axis_angle(quaternion: torch.Tensor) -> torch.Tensor:
    quaternion = quaternion.float().reshape(1, 4)
    w = quaternion[:, 3].clamp(-1.0, 1.0)
    denominator = torch.sqrt(torch.clamp(1.0 - w * w, min=0.0))
    result = torch.zeros((1, 3), dtype=torch.float32)
    valid = denominator > 1.0e-10
    if valid.any():
        angle = 2.0 * torch.acos(w[valid])
        axis = quaternion[valid, :3] / denominator[valid].unsqueeze(1)
        result[valid] = axis * angle.unsqueeze(1)
    return result


def preprocess_observation(observation: dict[str, Any]) -> tuple[dict[str, torch.Tensor], torch.Tensor]:
    images = {}
    for key, value in observation["pixels"].items():
        image = torch.from_numpy(np.asarray(value)).permute(2, 0, 1).float().div_(255.0)
        images[f"observation.images.{key}"] = torch.flip(image, dims=[1, 2]).unsqueeze(0)
    robot = observation["robot_state"]
    position = torch.as_tensor(robot["eef"]["pos"], dtype=torch.float32).reshape(1, 3)
    quaternion = torch.as_tensor(robot["eef"]["quat"], dtype=torch.float32)
    gripper = torch.as_tensor(robot["gripper"]["qpos"], dtype=torch.float32).reshape(1, 2)
    return images, torch.cat([position, _axis_angle(quaternion), gripper], dim=-1)


def rollout(policy, case: dict[str, Any], n_action_steps: int) -> tuple[bool, int]:
    """Run one case while keeping all ranks in its TP group synchronized."""

    env = make_env(case) if policy.is_tp_leader else None
    observation = None
    success = False
    step = 0
    if policy.is_tp_leader:
        observation, _ = env.reset(seed=int(case.get("seed", 0)))
    group = __import__("megatron.core", fromlist=["parallel_state"]).parallel_state.get_tensor_model_parallel_group()
    max_steps_tensor = torch.tensor(
        [env._max_episode_steps if policy.is_tp_leader else 0],
        dtype=torch.int64,
        device=policy.model.proprio_encoder.weight.device,
    )
    dist.broadcast(max_steps_tensor, src=dist.get_global_rank(group, 0), group=group)
    max_steps = int(max_steps_tensor.item())
    try:
        while step < max_steps and not success:
            if policy.is_tp_leader:
                images, state = preprocess_observation(observation)
                task = env.task_description
            else:
                images = state = task = None
            chunk = policy.predict_action_chunk(images, state, task)
            if policy.is_tp_leader:
                for action in chunk[:n_action_steps]:
                    observation, _, terminated, _, info = env.step(action.numpy())
                    step += 1
                    success = bool(info.get("is_success", False))
                    if terminated or success or step >= max_steps:
                        break
            flag = torch.tensor(
                [success, step],
                dtype=torch.int64,
                device=policy.model.proprio_encoder.weight.device,
            )
            dist.broadcast(flag, src=dist.get_global_rank(group, 0), group=group)
            success, step = bool(flag[0].item()), int(flag[1].item())
    finally:
        if env is not None:
            env.close()
    return success, step
