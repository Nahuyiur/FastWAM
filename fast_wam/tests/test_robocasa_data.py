from __future__ import annotations

import json
import sys
import types
from types import SimpleNamespace

import numpy as np
import torch

from fast_wam.train.robocasa_data import RoboCasaLatentDataset


class _FakeRoboCasaDataset:
    def __init__(self):
        self.episodes = [SimpleNamespace(task_text="open the drawer", episode_index=7)]
        self.windows = [(0, 0)]
        self.action_horizon = 2
        self.shape_meta = {
            "action": [{"raw_shape": 3}],
            "state": [{"raw_shape": 4}],
        }
        self.num_frames = 3
        self.max_getitem_retry = 1
        self.video_calls = 0

    def __len__(self):
        return 1

    def _load_episode_arrays(self, episode):
        del episode
        return (
            np.zeros((2, 4), dtype=np.float32),
            np.zeros((2, 3), dtype=np.float32),
            np.zeros((2,), dtype=np.float32),
        )

    def _normalize_action_state(self, action, proprio):
        return (
            action,
            proprio,
            torch.zeros(3, dtype=torch.bool),
            torch.zeros(4, dtype=torch.bool),
        )

    def _get_cached_text_context(self, prompt):
        assert prompt == "Task: open the drawer"
        return torch.zeros(2, 5), torch.ones(2, dtype=torch.bool)

    def _load_video(self, *args, **kwargs):
        self.video_calls += 1
        raise AssertionError("latent-cache reads must not decode video")


def _install_prompt_module(monkeypatch):
    names = (
        "fastwam",
        "fastwam.datasets",
        "fastwam.datasets.lerobot",
        "fastwam.datasets.lerobot.robot_video_dataset",
    )
    for name in names:
        monkeypatch.setitem(sys.modules, name, types.ModuleType(name))
    sys.modules[names[-1]].DEFAULT_PROMPT = "Task: {task}"


def test_latent_dataset_never_decodes_video(tmp_path, monkeypatch):
    _install_prompt_module(monkeypatch)
    latent = torch.tensor([2.0], dtype=torch.bfloat16)
    (tmp_path / "latents-00000.bf16").write_bytes(
        latent.view(torch.uint16).numpy().tobytes()
    )
    (tmp_path / "manifest.json").write_text(
        json.dumps(
            {
                "complete": True,
                "num_samples": 1,
                "sample_shape": [1],
                "shards": [
                    {
                        "id": 0,
                        "start": 0,
                        "count": 1,
                        "file": "latents-00000.bf16",
                    }
                ],
            }
        )
    )
    base = _FakeRoboCasaDataset()
    sample = RoboCasaLatentDataset(base, tmp_path)[0]
    assert base.video_calls == 0
    assert "video" not in sample
    assert sample["input_latents"].dtype == torch.bfloat16
    assert sample["input_latents"].item() == 2.0
    assert sample["action"].shape == (2, 3)
    assert sample["proprio"].shape == (2, 4)
