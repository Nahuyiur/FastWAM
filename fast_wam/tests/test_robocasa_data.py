from __future__ import annotations

import json
import sys
import tarfile
import types
from types import SimpleNamespace

import numpy as np
import torch

from fast_wam.train.prepare_robocasa_webdataset import (
    _add_bytes,
    _npy_bytes,
    _npz_bytes,
    _scan_shard,
)
from fast_wam.train.robocasa_data import RoboCasaLatentDataset
from fast_wam.train.robocasa_webdataset import FORMAT_VERSION, RoboCasaWebDataset


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


def _build_webdataset(tmp_path, mode: str) -> RoboCasaWebDataset:
    context_dir = tmp_path / "contexts"
    context_dir.mkdir()
    (context_dir / "task.npz").write_bytes(
        _npz_bytes(
            {
                "context": torch.arange(10, dtype=torch.float32).view(2, 5),
                "context_mask": torch.ones(2, dtype=torch.bool),
            }
        )
    )
    shard = tmp_path / "shard-00000.tar"
    with tarfile.open(shard, mode="w", format=tarfile.USTAR_FORMAT) as archive:
        descriptor = json.dumps(
            {
                "logical_index": 0,
                "source_index": 17,
                "context_id": "task",
                "mode": mode,
            }
        ).encode()
        _add_bytes(archive, "000000000.json", descriptor)
        _add_bytes(
            archive,
            "000000000.metadata.npz",
            _npz_bytes(
                {
                    "action": torch.zeros(2, 3),
                    "proprio": torch.zeros(2, 4),
                    "image_is_pad": torch.zeros(3, dtype=torch.bool),
                    "action_is_pad": torch.zeros(2, dtype=torch.bool),
                    "episode_index": torch.tensor(7),
                    "window_start": torch.tensor(11),
                }
            ),
        )
        if mode == "online":
            _add_bytes(
                archive,
                "000000000.video.npy",
                _npy_bytes(torch.arange(12, dtype=torch.float32).view(3, 1, 2, 2)),
            )
        else:
            latent = torch.tensor([1.5, -2.0], dtype=torch.bfloat16)
            _add_bytes(
                archive,
                "000000000.latent.npy",
                _npy_bytes(latent.view(torch.uint16)),
            )
    entries = _scan_shard(tmp_path, shard.name)
    (tmp_path / "index.jsonl").write_text(json.dumps(entries[0]) + "\n")
    (tmp_path / "manifest.json").write_text(
        json.dumps(
            {
                "format": FORMAT_VERSION,
                "complete": True,
                "mode": mode,
                "num_samples": 1,
                "index_file": "index.jsonl",
                "contexts": {
                    "task": {
                        "file": "contexts/task.npz",
                        "prompt": "Task: open the drawer",
                    }
                },
            }
        )
    )
    return RoboCasaWebDataset(tmp_path)


def test_indexed_webdataset_online_round_trip(tmp_path):
    sample = _build_webdataset(tmp_path, "online")[0]
    assert sample["idx"].item() == 0
    assert sample["source_idx"].item() == 17
    assert sample["video"].dtype == torch.float32
    assert sample["video"].shape == (3, 1, 2, 2)
    assert sample["context"].shape == (2, 5)
    assert sample["prompt"] == "Task: open the drawer"


def test_indexed_webdataset_offline_preserves_bfloat16_bits(tmp_path):
    sample = _build_webdataset(tmp_path, "offline")[0]
    assert sample["input_latents"].dtype == torch.bfloat16
    torch.testing.assert_close(
        sample["input_latents"],
        torch.tensor([1.5, -2.0], dtype=torch.bfloat16),
        rtol=0,
        atol=0,
    )
