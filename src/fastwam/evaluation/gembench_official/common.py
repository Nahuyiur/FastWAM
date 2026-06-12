from __future__ import annotations

import json
import os
import re
import subprocess
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable


PROJECT_ROOT = Path(__file__).resolve().parents[4]
OFFICIAL_CAMERA_NAMES = ("left_shoulder", "right_shoulder", "wrist", "front")
DEFAULT_GEMBENCH_ROOT = Path(os.environ.get("GEMBENCH_ROOT", "/mnt/yuhan/datasets/GEMBench"))


@dataclass(frozen=True)
class TaskVar:
    task: str
    variation: int

    @property
    def key(self) -> str:
        return f"{self.task}+{self.variation}"


@dataclass(frozen=True)
class TrialSpec:
    taskvar: str
    task: str
    variation: int
    demo_id: int
    seed: int
    split: str
    taskvar_file: str
    microstep_data_dir: str


def utc_now() -> str:
    return datetime.utcnow().isoformat(timespec="seconds") + "Z"


def safe_token(value: Any, fallback: str = "item") -> str:
    token = re.sub(r"[^A-Za-z0-9_.+-]+", "_", str(value)).strip("._")
    return token or fallback


def run_git(args: list[str]) -> str | None:
    try:
        return subprocess.check_output(
            ["git", *args],
            cwd=PROJECT_ROOT,
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except Exception:
        return None


def git_provenance() -> dict[str, Any]:
    status = run_git(["status", "--short"]) or ""
    return {
        "generated_at": utc_now(),
        "repo": str(PROJECT_ROOT),
        "git_head": run_git(["rev-parse", "HEAD"]),
        "git_branch": run_git(["branch", "--show-current"]),
        "git_status_short": status,
        "git_dirty": bool(status),
    }


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=True, indent=2) + "\n", encoding="utf-8")


def append_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=True) + "\n")


def dataclass_dict(value: Any) -> dict[str, Any]:
    return asdict(value)


def parse_taskvar(value: str) -> TaskVar:
    if "+" not in value:
        raise ValueError(f"Official GEMBench taskvar must be `task+variation`, got {value!r}")
    task, variation = value.rsplit("+", 1)
    return TaskVar(task=task, variation=int(variation))


def load_taskvars(path: Path) -> list[TaskVar]:
    data = read_json(path)
    if not isinstance(data, list):
        raise ValueError(f"Taskvar file must contain a JSON list: {path}")
    return [parse_taskvar(str(item)) for item in data]


def resolve_existing_path(path: str | Path, *, label: str, bases: Iterable[Path] = ()) -> Path:
    raw = Path(path).expanduser()
    candidates = [raw] if raw.is_absolute() else [base / raw for base in bases] + [PROJECT_ROOT / raw, raw]
    for candidate in candidates:
        resolved = candidate.resolve()
        if resolved.exists():
            return resolved
    tried = ", ".join(str(candidate) for candidate in candidates)
    raise FileNotFoundError(f"{label} does not exist. Tried: {tried}")


def resolve_optional_existing_path(path: str | Path | None, *, label: str, bases: Iterable[Path] = ()) -> Path | None:
    if path is None or str(path).strip() == "":
        return None
    return resolve_existing_path(path, label=label, bases=bases)


def latest_checkpoint(run_dir: Path) -> Path:
    weights_dir = run_dir / "checkpoints" / "weights"
    checkpoints = sorted(weights_dir.glob("step_*.pt"))
    if not checkpoints:
        raise FileNotFoundError(f"No step_*.pt checkpoint found under {weights_dir}")

    def step(path: Path) -> int:
        match = re.search(r"step_(\d+)\.pt$", path.name)
        return int(match.group(1)) if match else -1

    return max(checkpoints, key=step)


def resolve_checkpoint(run_dir: Path, checkpoint: str | None, *, dry_run: bool = False) -> Path | None:
    if checkpoint:
        return resolve_existing_path(checkpoint, label="checkpoint", bases=[run_dir])
    try:
        return latest_checkpoint(run_dir)
    except FileNotFoundError:
        if dry_run:
            return None
        raise


def resolve_assets_root(gembench_root: Path, assets_root: str | None, robot_3dlotus_root: str | None) -> Path:
    candidates: list[Path] = []
    if assets_root:
        candidates.append(Path(assets_root).expanduser())
    if os.environ.get("GEMBENCH_ASSETS_ROOT"):
        candidates.append(Path(os.environ["GEMBENCH_ASSETS_ROOT"]).expanduser())
    if robot_3dlotus_root:
        candidates.append(Path(robot_3dlotus_root).expanduser() / "assets")
    if os.environ.get("ROBOT_3DLOTUS_ROOT"):
        candidates.append(Path(os.environ["ROBOT_3DLOTUS_ROOT"]).expanduser() / "assets")
    candidates.extend([gembench_root / "assets", PROJECT_ROOT / "assets"])
    for candidate in candidates:
        if candidate.exists():
            return candidate.resolve()
    raise FileNotFoundError(
        "Could not resolve official GEMBench taskvar assets. "
        "Pass --assets-root or set GEMBENCH_ASSETS_ROOT/ROBOT_3DLOTUS_ROOT."
    )


def official_taskvar_file(assets_root: Path, split: str) -> Path:
    split = str(split)
    if split == "val":
        split = "train"
    path = assets_root / f"taskvars_{split}.json"
    if not path.exists():
        raise FileNotFoundError(f"Missing official taskvar file for split={split}: {path}")
    return path


def official_microstep_dir(gembench_root: Path, *, suite: str, seed: int) -> Path:
    if suite == "val":
        return gembench_root / "val_dataset" / "microsteps" / f"seed{seed}"
    if suite == "test":
        return gembench_root / "test_dataset" / "microsteps" / f"seed{seed}"
    raise ValueError(f"Unknown official suite: {suite!r}")


def taskvar_episode_dir(microstep_data_dir: Path, taskvar: TaskVar) -> Path:
    return microstep_data_dir / taskvar.task / f"variation{taskvar.variation}" / "episodes"


def available_demo_ids(microstep_data_dir: Path, taskvar: TaskVar) -> list[int]:
    episodes_dir = taskvar_episode_dir(microstep_data_dir, taskvar)
    if not episodes_dir.exists():
        return []
    episode_names = [path.name for path in episodes_dir.iterdir() if path.is_dir() and path.name.startswith("episode")]
    episode_names.sort(key=lambda name: int(name[len("episode") :]))
    return list(range(len(episode_names)))


def selected_trials(
    *,
    taskvars: list[TaskVar],
    microstep_data_dir: Path,
    seed: int,
    split: str,
    taskvar_file: Path,
    num_demos: int,
    num_trials: int | None = None,
) -> list[TrialSpec]:
    rows: list[TrialSpec] = []
    for taskvar in taskvars:
        demo_ids = available_demo_ids(microstep_data_dir, taskvar)
        for demo_id in demo_ids[: max(int(num_demos), 0)]:
            rows.append(
                TrialSpec(
                    taskvar=taskvar.key,
                    task=taskvar.task,
                    variation=taskvar.variation,
                    demo_id=int(demo_id),
                    seed=int(seed),
                    split=str(split),
                    taskvar_file=str(taskvar_file),
                    microstep_data_dir=str(microstep_data_dir),
                )
            )
            if num_trials is not None and len(rows) >= int(num_trials):
                return rows
    return rows
