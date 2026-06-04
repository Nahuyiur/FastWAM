from __future__ import annotations

import json
import logging
import random
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable, Sequence

import imageio.v2 as imageio
import numpy as np
import torch

logger = logging.getLogger(__name__)


@dataclass
class EpisodeResult:
    task: str
    variation: int
    episode: str
    success: bool
    reward: float
    steps: int
    error: str | None
    video_path: str | None = None


def split_taskvar(taskvar: str) -> tuple[str, int]:
    if "+" not in taskvar:
        raise ValueError(f"Expected taskvar '<task>+<variation>', got {taskvar!r}")
    task, variation = taskvar.rsplit("+", 1)
    return task, int(variation)


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=True) + "\n")
            f.flush()


def set_eval_seed(seed: int) -> None:
    random.seed(int(seed))
    np.random.seed(int(seed))
    torch.manual_seed(int(seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(seed))


class Mover:
    """Official GEMBench motion wrapper adapted from robot-3dlotus.

    It first moves the arm target while keeping the previous gripper state,
    retries pose planning, and then applies the gripper open/close command when
    the target is close enough.
    """

    def __init__(self, task: Any, max_tries: int = 10):
        self._task = task
        self._last_action: np.ndarray | None = None
        self._step_id = 0
        self._max_tries = int(max_tries)

    def reset(self, ee_pose: np.ndarray) -> None:
        self._last_action = np.asarray(ee_pose, dtype=np.float32).copy()
        self._step_id = 0

    def __call__(self, action: np.ndarray, verbose: bool = False):
        action = np.asarray(action, dtype=np.float32).copy()
        if self._last_action is None:
            self._last_action = action.copy()
        change_gripper = ((self._last_action[-1] > 0.5) and (action[-1] < 0.5)) or (
            (self._last_action[-1] < 0.5) and (action[-1] > 0.5)
        )
        target = action.copy()
        action[7] = self._last_action[7].copy()

        obs = None
        reward = 0.0
        terminate = False
        criteria = (False,)
        dist_pos = float("inf")
        for try_id in range(self._max_tries):
            obs, reward, terminate = self._task.step(action)
            pos = np.asarray(obs.gripper_pose[:3], dtype=np.float32)
            rot = np.asarray(obs.gripper_pose[3:7], dtype=np.float32)
            dist_pos = float(np.sqrt(np.square(target[:3] - pos).sum()))
            dist_rot = float(np.sqrt(np.square(target[3:7] - rot).sum()))
            criteria = (dist_pos < 2e-2,) if change_gripper else (dist_pos < 5e-2,)
            if all(criteria) or reward == 1:
                break
            if verbose:
                logger.info(
                    "Mover retry step=%d try=%d dist_pos=%.3f dist_rot=%.3f",
                    self._step_id,
                    try_id,
                    dist_pos,
                    dist_rot,
                )

        action = target
        if not reward and change_gripper and all(criteria):
            obs, reward, terminate = self._task.step(action)
        if self._max_tries > 0 and not all(criteria):
            logger.debug("Mover step=%d failed dist_pos=%.3f", self._step_id, dist_pos)
        self._step_id += 1
        self._last_action = action.copy()
        return obs, reward, terminate


class GEMBenchSimulator:
    def __init__(
        self,
        *,
        microstep_data_dir: str | Path,
        image_size: Sequence[int] = (256, 256),
        cameras: Sequence[str] = ("front", "wrist", "left_shoulder"),
        headless: bool = True,
        renderer: str = "opengl",
    ):
        self.microstep_data_dir = Path(microstep_data_dir)
        self.image_size = [int(image_size[0]), int(image_size[1])]
        self.cameras = tuple(str(x) for x in cameras)
        self.headless = bool(headless)
        self.renderer = str(renderer)
        self.env = None

    def _make_obs_config(self):
        from pyrep.const import RenderMode
        from rlbench import ObservationConfig
        from rlbench.observation_config import CameraConfig

        unused = CameraConfig()
        unused.set_all(False)
        mode = RenderMode.OPENGL3 if self.renderer == "opengl3" else RenderMode.OPENGL
        used = CameraConfig(
            rgb=True,
            point_cloud=False,
            depth=False,
            mask=False,
            render_mode=mode,
            image_size=self.image_size,
        )
        kwargs = {
            "front_camera": used if "front" in self.cameras else unused,
            "left_shoulder_camera": used if "left_shoulder" in self.cameras else unused,
            "right_shoulder_camera": used if "right_shoulder" in self.cameras else unused,
            "wrist_camera": used if "wrist" in self.cameras else unused,
            "overhead_camera": used if "overhead" in self.cameras else unused,
            "joint_forces": False,
            "joint_positions": False,
            "joint_velocities": False,
            "task_low_dim_state": False,
            "gripper_touch_forces": False,
            "gripper_pose": True,
            "gripper_open": True,
            "gripper_matrix": True,
            "gripper_joint_positions": True,
        }
        return ObservationConfig(**kwargs)

    def launch(self):
        from rlbench.action_modes.action_mode import MoveArmThenGripper
        from rlbench.action_modes.arm_action_modes import EndEffectorPoseViaPlanning
        from rlbench.action_modes.gripper_action_modes import Discrete
        from rlbench.environment import Environment

        action_mode = MoveArmThenGripper(
            arm_action_mode=EndEffectorPoseViaPlanning(collision_checking=False),
            gripper_action_mode=Discrete(),
        )
        self.env = Environment(
            action_mode,
            str(self.microstep_data_dir),
            self._make_obs_config(),
            headless=self.headless,
        )
        self.env.launch()
        return self

    def shutdown(self) -> None:
        if self.env is not None:
            self.env.shutdown()
            self.env = None

    def get_task(self, task_str: str):
        from rlbench.backend.utils import task_file_to_task_class

        if self.env is None:
            raise RuntimeError("Simulator is not launched.")
        return self.env.get_task(task_file_to_task_class(task_str))

    def get_demo(self, task_str: str, variation: int, episode_index: int):
        if self.env is None:
            raise RuntimeError("Simulator is not launched.")
        demos = self.env.get_demos(
            task_name=task_str,
            variation_number=int(variation),
            amount=1,
            from_episode_number=int(episode_index),
            random_selection=False,
            load_images=False,
        )
        return demos[0]

    def observation_dict(self, obs: Any) -> dict[str, Any]:
        rgb_by_name = {name: np.asarray(getattr(obs, f"{name}_rgb"), dtype=np.uint8) for name in self.cameras}
        return {
            "rgb": np.stack([rgb_by_name[name] for name in self.cameras], axis=0),
            "rgb_by_name": rgb_by_name,
            "gripper": np.concatenate([np.asarray(obs.gripper_pose, dtype=np.float32), [obs.gripper_open]]).astype(np.float32),
        }


def _episode_indices(episodes_dir: Path) -> list[int]:
    if not episodes_dir.exists():
        return []
    indices = []
    for path in episodes_dir.iterdir():
        if path.name.startswith("episode"):
            try:
                indices.append(int(path.name[len("episode") :]))
            except ValueError:
                pass
    return sorted(indices)


def _video_frame(obs_state_dict: dict[str, Any], cameras: Sequence[str]) -> np.ndarray:
    return np.concatenate(
        [np.asarray(obs_state_dict["rgb_by_name"][name], dtype=np.uint8) for name in cameras],
        axis=1,
    )


def evaluate_taskvar(
    *,
    policy: Any,
    taskvar: str,
    microstep_data_dir: str | Path,
    num_demos: int = 20,
    max_steps: int = 25,
    max_tries: int = 10,
    image_size: Sequence[int] = (256, 256),
    cameras: Sequence[str] = ("front", "wrist", "left_shoulder"),
    headless: bool = True,
    renderer: str = "opengl",
    record_video: bool = False,
    video_dir: str | Path | None = None,
    video_fps: int = 8,
    seed: int | None = None,
    keep_going_on_error: bool = False,
) -> tuple[float, list[EpisodeResult]]:
    from pyrep.errors import ConfigurationPathError, IKError
    from rlbench.backend.exceptions import InvalidActionError

    task_str, variation = split_taskvar(taskvar)
    if seed is not None:
        set_eval_seed(int(seed))
    microstep_data_dir = Path(microstep_data_dir)
    episodes_dir = microstep_data_dir / task_str / f"variation{variation}" / "episodes"
    episode_indices = _episode_indices(episodes_dir)
    if not episode_indices:
        logger.warning("Skipping %s because no microstep episodes exist under %s", taskvar, episodes_dir)
        return 0.0, []
    episode_indices = episode_indices[: int(num_demos)]

    sim = GEMBenchSimulator(
        microstep_data_dir=microstep_data_dir,
        image_size=image_size,
        cameras=cameras,
        headless=headless,
        renderer=renderer,
    ).launch()
    results: list[EpisodeResult] = []
    successes = 0
    try:
        task = sim.get_task(task_str)
        task.set_variation(int(variation))
        mover = Mover(task, max_tries=max_tries)
        for local_i, episode_index in enumerate(episode_indices):
            if hasattr(policy, "reset_episode"):
                policy.reset_episode()
            reward = 0.0
            steps = 0
            error = None
            video_frames: list[np.ndarray] = []
            try:
                demo = sim.get_demo(task_str, variation, episode_index)
                instructions, obs = task.reset_to_demo(demo)
                obs_state_dict = sim.observation_dict(obs)
                mover.reset(obs_state_dict["gripper"])
                if record_video:
                    video_frames.append(_video_frame(obs_state_dict, cameras))
                for step_id in range(int(max_steps)):
                    output = policy.predict(
                        task_str=task_str,
                        variation=variation,
                        step_id=step_id,
                        obs_state_dict=obs_state_dict,
                        episode_id=episode_index,
                        instructions=instructions,
                    )
                    action = output.get("action")
                    if action is None:
                        break
                    obs, reward, terminate = mover(action, verbose=False)
                    steps = step_id + 1
                    obs_state_dict = sim.observation_dict(obs)
                    if record_video:
                        video_frames.append(_video_frame(obs_state_dict, cameras))
                    if reward == 1:
                        break
                    if terminate:
                        logger.info("The episode has terminated!")
            except (IKError, ConfigurationPathError, InvalidActionError) as exc:
                error = f"{type(exc).__name__}: {exc}"
            except Exception as exc:
                if not keep_going_on_error:
                    raise
                error = f"{type(exc).__name__}: {exc}"
                logger.exception("GEMBench episode failed: %s episode=%s", taskvar, episode_index)

            success = bool(reward == 1)
            successes += int(success)
            video_path = None
            if record_video and video_dir is not None and video_frames:
                path = Path(video_dir) / taskvar / f"episode{episode_index}_SR{int(success)}.mp4"
                path.parent.mkdir(parents=True, exist_ok=True)
                imageio.mimwrite(str(path), video_frames, fps=int(video_fps), macro_block_size=1)
                video_path = str(path)
            results.append(
                EpisodeResult(
                    task=task_str,
                    variation=variation,
                    episode=f"episode{episode_index}",
                    success=success,
                    reward=float(reward),
                    steps=int(steps),
                    error=error,
                    video_path=video_path,
                )
            )
            logger.info(
                "[gembench-eval] %s demo=%s/%s reward=%s success=%s sr=%.2f",
                taskvar,
                local_i + 1,
                len(episode_indices),
                reward,
                success,
                successes / max(len(results), 1),
            )
    finally:
        sim.shutdown()

    return successes / max(len(results), 1), results
