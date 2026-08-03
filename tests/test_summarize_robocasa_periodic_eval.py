import importlib.util
import json
from pathlib import Path


SCRIPT = Path(__file__).parents[1] / "fast_wam" / "scripts" / "summarize_robocasa_periodic_eval.py"
SPEC = importlib.util.spec_from_file_location("summarize_periodic_eval", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


def _write_fixture(root: Path, *, omit_video: bool = False) -> None:
    for shard in range(4):
        shard_root = root / f"shard_{shard:02d}"
        videos = shard_root / "videos"
        videos.mkdir(parents=True)
        rows = []
        for episode in range(4):
            rows.append(
                {
                    "bucket": f"bucket_{shard}",
                    "task": f"task_{shard}_{episode // 2}",
                    "success": episode % 2 == 0,
                }
            )
            if not (omit_video and shard == 3 and episode == 3):
                (videos / f"episode_{episode}.mp4").write_bytes(b"video")
        (shard_root / "episode_results.jsonl").write_text(
            "".join(json.dumps(row) + "\n" for row in rows)
        )


def test_summarize_complete_periodic_eval(tmp_path: Path) -> None:
    _write_fixture(tmp_path)

    summary, rows, videos = MODULE.summarize(tmp_path, expected_episodes=16)

    assert len(rows) == 16
    assert len(videos) == 16
    assert summary["num_episodes"] == 16
    assert summary["num_errors"] == 0
    assert summary["num_videos"] == 16
    assert summary["num_successes"] == 8
    assert summary["success_rate"] == 0.5
    assert set(summary["by_bucket"]) == {f"bucket_{index}" for index in range(4)}


def test_summarize_detects_missing_video(tmp_path: Path) -> None:
    _write_fixture(tmp_path, omit_video=True)

    summary, _, _ = MODULE.summarize(tmp_path, expected_episodes=16)

    assert summary["num_episodes"] == 16
    assert summary["num_videos"] == 15
