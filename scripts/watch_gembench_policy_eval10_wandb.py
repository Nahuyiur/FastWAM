#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_EVAL10_CONFIG = PROJECT_ROOT / "configs" / "eval" / "gembench_official_eval10_periodic_indomain_ood_seed200.yaml"
DEFAULT_OUTPUT_ROOT = Path("/mnt/yuhan/gembench_watchdog_eval10")
DEFAULT_GEMBENCH_ROOT = Path("/mnt/yuhan/datasets/GEMBench")
DEFAULT_ROBOT_3DLOTUS_ROOT = Path("/mnt/yuhan/gembench_sim/robot-3dlotus")
DEFAULT_ASSETS_ROOT = DEFAULT_ROBOT_3DLOTUS_ROOT / "assets"
DEFAULT_COPPELIASIM_ROOT = Path("/mnt/yuhan/gembench_sim/CoppeliaSim_Edu_V4_1_0_Ubuntu20_04")
DEFAULT_RLBENCH_ROOT = Path("/mnt/yuhan/gembench_sim/RLBench")
DEFAULT_PYREP_ROOT = Path("/mnt/yuhan/gembench_sim/PyRep")
DEFAULT_GLOBAL_LOCK_PATH = DEFAULT_OUTPUT_ROOT / "eval.lock"


def _project_path(value: str | Path) -> Path:
    path = Path(value).expanduser()
    if path.is_absolute():
        return path
    return PROJECT_ROOT / path


def _step_from_checkpoint(path: Path) -> int | None:
    match = re.fullmatch(r"step_(\d+)\.pt", path.name)
    return int(match.group(1)) if match else None


def _load_json(path: Path, default: Any) -> Any:
    if not path.exists():
        return default
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return default


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def _infer_model_name(value: str | None, run_dir: Path, run_name: str | None) -> str:
    if value:
        return re.sub(r"[^A-Za-z0-9_.+-]+", "_", value).strip("._") or "model"
    text = f"{run_name or ''} {run_dir}".lower()
    if "trace" in text:
        return "trace"
    if "fastwam" in text:
        return "fastwam"
    return "model"


def _infer_run_id(value: str | None, run_dir: Path, run_name: str | None) -> str:
    raw = value or run_name or run_dir.name
    return re.sub(r"[^A-Za-z0-9_.+-]+", "_", str(raw)).strip("._") or run_dir.name


def _metric_prefix(args: argparse.Namespace) -> str:
    if args.wandb_metric_prefix:
        return str(args.wandb_metric_prefix).strip().strip("/") or "eval10_periodic"
    if _normalize_eval_protocol(args.eval_protocol) == "chunk_replan":
        return "chunk_replan_eval10"
    return "eval10_periodic"


def _normalize_eval_protocol(value: str) -> str:
    protocol = str(value)
    if protocol in {"chunk_replan", "fastwam_chunk_replan", "trace_chunk_replan"}:
        return "chunk_replan"
    return protocol


def _prepend_env_path(env: dict[str, str], key: str, values: list[str]) -> None:
    existing = env.get(key, "")
    parts = [str(value) for value in values if value and Path(value).exists()]
    if existing:
        parts.extend([part for part in existing.split(":") if part])
    deduped: list[str] = []
    seen: set[str] = set()
    for part in parts:
        if part in seen:
            continue
        seen.add(part)
        deduped.append(part)
    if deduped:
        env[key] = ":".join(deduped)


def _python_conda_prefix(python_bin: str) -> Path | None:
    path = Path(python_bin).expanduser()
    try:
        if path.name.startswith("python") and path.parent.name == "bin":
            return path.parent.parent
    except IndexError:
        return None
    return None


def _prepare_sim_env(args: argparse.Namespace) -> dict[str, str]:
    env = os.environ.copy()
    robot_root = Path(args.robot_3dlotus_root).expanduser().resolve()
    coppelia_root = Path(args.coppeliasim_root).expanduser().resolve()
    rlbench_root = Path(args.rlbench_root).expanduser().resolve()
    pyrep_root = Path(args.pyrep_root).expanduser().resolve()
    env["ROBOT_3DLOTUS_ROOT"] = str(robot_root)
    env["GEMBENCH_ROOT"] = str(Path(args.gembench_root).expanduser().resolve())
    env["GEMBENCH_ASSETS_ROOT"] = str(Path(args.assets_root).expanduser().resolve())
    env["COPPELIASIM_ROOT"] = str(coppelia_root)
    env["RLBENCH_ROOT"] = str(rlbench_root)
    env["GEMBENCH_RLBENCH_ROOT"] = str(rlbench_root)
    env["GEMBENCH_PYREP_ROOT"] = str(pyrep_root)
    conda_prefix = _python_conda_prefix(args.python_bin)
    if conda_prefix is not None and conda_prefix.exists():
        env.setdefault("CONDA_PREFIX", str(conda_prefix))
    ld_paths = [str(coppelia_root), str(coppelia_root / "platforms")]
    if conda_prefix is not None:
        ld_paths.insert(0, str(conda_prefix / "lib"))
    _prepend_env_path(env, "LD_LIBRARY_PATH", ld_paths)
    _prepend_env_path(env, "PYTHONPATH", [str(robot_root), str(rlbench_root), str(pyrep_root)])
    env.setdefault("QT_PLUGIN_PATH", str(coppelia_root))
    env.setdefault("QT_QPA_PLATFORM_PLUGIN_PATH", str(coppelia_root / "platforms"))
    env.setdefault("QT_XCB_GL_INTEGRATION", "xcb_glx")
    env.setdefault("LIBGL_ALWAYS_SOFTWARE", "1")
    env.setdefault("TOKENIZERS_PARALLELISM", "false")
    return env


def _maybe_wrap_xvfb(cmd: list[str], *, enabled: bool) -> list[str]:
    if not enabled:
        return cmd
    xvfb = shutil.which("xvfb-run")
    if not xvfb:
        return cmd
    return [xvfb, "-a", "-s", "-screen 0 1280x1024x24 +extension GLX +render -noreset", *cmd]


def _checkpoint_is_stable(path: Path, *, stable_seconds: float) -> bool:
    try:
        stat = path.stat()
    except FileNotFoundError:
        return False
    if stat.st_size <= 0:
        return False
    return (time.time() - float(stat.st_mtime)) >= float(stable_seconds)


def _candidate_checkpoints(
    run_dir: Path,
    *,
    interval_steps: int,
    min_step: int,
    checkpoint_stable_seconds: float,
) -> list[tuple[int, Path]]:
    weights_dir = run_dir / "checkpoints" / "weights"
    out: list[tuple[int, Path]] = []
    for path in sorted(weights_dir.glob("step_*.pt")):
        step = _step_from_checkpoint(path)
        if step is None:
            continue
        if step < min_step:
            continue
        if interval_steps > 0 and step % interval_steps != 0:
            continue
        if not _checkpoint_is_stable(path, stable_seconds=checkpoint_stable_seconds):
            continue
        out.append((step, path))
    return out


def _load_trial_rows(output_root: Path) -> list[dict[str, Any]]:
    rows = []
    path = output_root / "trials" / "results.jsonl"
    if not path.exists():
        path = output_root / "eval10_trials.jsonl"
    if not path.exists():
        return rows
    for line in path.read_text(encoding="utf-8", errors="ignore").splitlines():
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(row, dict):
            rows.append(row)
    return rows


def _row_video_frames(row: dict[str, Any]) -> int:
    counts = row.get("video_frame_counts") or {}
    if isinstance(counts, dict) and counts:
        values = []
        for value in counts.values():
            try:
                values.append(int(value))
            except (TypeError, ValueError):
                pass
        if values:
            return max(values)
    try:
        return int(row.get("video_max_frame_count") or 0)
    except (TypeError, ValueError):
        return 0


def _selected_videos(output_root: Path, *, max_videos: int, min_video_frames: int) -> list[tuple[Path, dict[str, Any]]]:
    selected: list[tuple[Path, dict[str, Any]]] = []
    for row in _load_trial_rows(output_root):
        if int(min_video_frames) > 0 and _row_video_frames(row) < int(min_video_frames):
            continue
        candidates = [Path(p) for p in row.get("video_mp4_paths") or []]
        video_path = row.get("video_path")
        if video_path:
            candidates.append(Path(video_path))
        for path in candidates:
            if path.suffix.lower() != ".mp4":
                continue
            if not path.exists():
                continue
            selected.append((path, row))
            break
        if len(selected) >= max_videos:
            break
    if selected:
        return selected
    return selected


def _selected_predicted_videos(output_root: Path, *, max_videos: int) -> list[tuple[Path, dict[str, Any], str]]:
    selected: list[tuple[Path, dict[str, Any], str]] = []
    for row in _load_trial_rows(output_root):
        candidates: list[tuple[str, str]] = []
        for key in ("predicted_prefix_timeline_path", "predicted_full_timeline_path"):
            value = row.get(key)
            if value:
                candidates.append((key, str(value)))
        for key in ("predicted_prefix_video_paths", "predicted_full_video_paths"):
            for value in row.get(key) or []:
                candidates.append((key, str(value)))
        for kind, path_text in candidates:
            path = Path(path_text)
            if path.suffix.lower() != ".mp4" or not path.exists():
                continue
            selected.append((path, row, kind))
            if len(selected) >= max_videos:
                return selected
    return selected


def _run_eval10(args: argparse.Namespace, *, step: int, checkpoint: Path) -> tuple[int, Path]:
    output_root = Path(args.output_root) / f"step_{step:06d}"
    output_root.mkdir(parents=True, exist_ok=True)
    cmd = [
        args.python_bin,
        str(PROJECT_ROOT / "scripts" / "eval_gembench_official_eval10_visual.py"),
        "--run-dir",
        str(Path(args.run_dir).resolve()),
        "--checkpoint",
        str(checkpoint.resolve()),
        "--output-root",
        str(output_root),
        "--eval10-config",
        str(_project_path(args.eval10_config).resolve()),
        "--gembench-root",
        str(Path(args.gembench_root).expanduser().resolve()),
        "--assets-root",
        str(Path(args.assets_root).expanduser().resolve()),
        "--robot-3dlotus-root",
        str(Path(args.robot_3dlotus_root).expanduser().resolve()),
        "--num-trials",
        str(args.num_trials),
        "--num-inference-steps",
        str(args.num_inference_steps),
        "--min-video-frames",
        str(args.min_video_frames),
        "--relation-mode",
        args.relation_mode,
        "--eval-protocol",
        _normalize_eval_protocol(args.eval_protocol),
        "--chunk-replan-steps",
        str(args.chunk_replan_steps),
        "--device",
        args.device,
        "--no-write-official-preds",
    ]
    if args.chunk_predict_video:
        cmd.append("--chunk-predict-video")
    if args.dry_run_eval:
        cmd.extend(["--dry-run", "--json-output", str(output_root / "dry_run.json")])
    if args.tiled:
        cmd.append("--tiled")
    if args.extra_eval_arg:
        cmd.extend(args.extra_eval_arg)
    env = _prepare_sim_env(args)
    if args.cuda_visible_devices is not None:
        env["CUDA_VISIBLE_DEVICES"] = args.cuda_visible_devices
    cmd = _maybe_wrap_xvfb(cmd, enabled=bool(args.xvfb_run))
    log_path = output_root / "eval10.log"
    print(f"[eval10-wandb] step={step} checkpoint={checkpoint} output={output_root}", flush=True)
    with log_path.open("w", encoding="utf-8") as log_file:
        if args.global_lock_path:
            import fcntl

            lock_path = Path(args.global_lock_path).expanduser()
            lock_path.parent.mkdir(parents=True, exist_ok=True)
            with lock_path.open("w", encoding="utf-8") as lock_file:
                print(f"[eval10-wandb] waiting for global eval lock: {lock_path}", flush=True)
                fcntl.flock(lock_file, fcntl.LOCK_EX)
                print(f"[eval10-wandb] acquired global eval lock: {lock_path}", flush=True)
                proc = subprocess.run(
                    cmd,
                    cwd=str(PROJECT_ROOT),
                    env=env,
                    stdout=log_file,
                    stderr=subprocess.STDOUT,
                    text=True,
                )
                fcntl.flock(lock_file, fcntl.LOCK_UN)
        else:
            proc = subprocess.run(
                cmd,
                cwd=str(PROJECT_ROOT),
                env=env,
                stdout=log_file,
                stderr=subprocess.STDOUT,
                text=True,
            )
    return int(proc.returncode), output_root


def _wandb_init(args: argparse.Namespace):
    if args.no_wandb:
        return None, None
    import wandb

    run_id = args.wandb_run_id
    if not run_id and args.wandb_mode == "same-run":
        run_id = os.environ.get("WANDB_RUN_ID")
    if not run_id and args.wandb_mode == "sidecar":
        base = os.environ.get("WANDB_RUN_ID") or args.run_name or Path(args.run_dir).name
        run_id = f"{base}-eval10-watchdog"
    name = args.wandb_name
    if not name:
        if args.wandb_mode == "sidecar":
            name = f"{args.run_name or Path(args.run_dir).name}-eval10-watchdog"
        else:
            name = args.run_name or Path(args.run_dir).name

    metric_prefix = _metric_prefix(args)
    tags = ["gembench", "policy", "eval10-visual", args.wandb_mode]
    eval_protocol = _normalize_eval_protocol(args.eval_protocol)
    if eval_protocol == "chunk_replan":
        tags.extend(["chunk-replan", "not-official-score"])
    init_kwargs = {
        "entity": args.wandb_entity or None,
        "project": args.wandb_project,
        "name": name,
        "id": run_id or None,
        "resume": "allow" if run_id else None,
        "group": args.wandb_group or None,
        "job_type": (
            "eval10-chunk-replan-visual"
            if args.wandb_mode == "sidecar" and eval_protocol == "chunk_replan"
            else "eval10-visual"
            if args.wandb_mode == "sidecar"
            else "train"
        ),
        "tags": tags,
        "config": {
            "source_run_dir": str(Path(args.run_dir).resolve()),
            "model_name": str(args.model_name),
            "run_id": str(args.run_id),
            "output_root": str(Path(args.output_root).resolve()),
            "eval10_config": str(_project_path(args.eval10_config).resolve()),
            "interval_steps": int(args.interval_steps),
            "num_trials": int(args.num_trials),
            "max_videos": int(args.max_videos),
            "min_video_frames": int(args.min_video_frames),
            "relation_mode": args.relation_mode,
            "eval_protocol": eval_protocol,
            "chunk_replan_steps": int(args.chunk_replan_steps),
            "chunk_predict_video": bool(args.chunk_predict_video),
            "wandb_metric_prefix": metric_prefix,
            "wandb_log_mode": args.wandb_mode,
            "official_full_score": False,
            "write_official_preds": False,
        },
    }
    try:
        run = wandb.init(**init_kwargs)
    except Exception as exc:
        if not args.wandb_fallback_offline:
            raise
        print(
            f"[eval10-wandb] online wandb init failed; falling back to offline mode: {type(exc).__name__}: {exc}",
            flush=True,
        )
        init_kwargs["mode"] = "offline"
        run = wandb.init(**init_kwargs)
    return wandb, run


def _wandb_log_eval10(wandb: Any, args: argparse.Namespace, *, step: int, output_root: Path, returncode: int) -> None:
    summary = _load_json(output_root / "summary.json", {})
    metric_prefix = _metric_prefix(args)
    payload: dict[str, Any] = {
        "eval10_periodic/step": int(step),
        "eval10_periodic/model_name": str(args.model_name),
        "eval10_periodic/run_id": str(args.run_id),
        "eval10_periodic/returncode": int(returncode),
        "eval10_periodic/output_root": str(output_root),
        "eval10_periodic/official_full_score": bool(summary.get("official_full_score", False)),
        "eval10_periodic/write_official_preds": bool(summary.get("write_official_preds", False)),
        "eval10_periodic/eval_protocol": str(summary.get("eval_protocol") or args.eval_protocol),
        "eval10_periodic/chunk_replan_steps": int(summary.get("chunk_replan_steps") or args.chunk_replan_steps),
        "eval10_periodic/chunk_predict_video": bool(summary.get("chunk_predict_video", args.chunk_predict_video)),
        "eval10_periodic/video_mode": str(summary.get("video_mode") or ""),
        "eval10_periodic/relation_mode": str(summary.get("relation_mode") or args.relation_mode),
        "eval10_periodic/action_horizon": int(summary.get("action_horizon") or 0),
        "eval10_periodic/training_action_horizon": int(summary.get("training_action_horizon") or 0),
        "eval10_periodic/executed_action_index": int(summary.get("executed_action_index") or 0),
        "eval10_periodic/trials": int(summary.get("trials") or 0),
        "eval10_periodic/successes": int(summary.get("successes") or 0),
        "eval10_periodic/success_rate": float(summary.get("success_rate") or 0.0),
        "eval10_periodic/valid_successes": int(summary.get("valid_successes") or 0),
        "eval10_periodic/valid_trials": int(summary.get("valid_trials") or 0),
        "eval10_periodic/valid_success_rate": float(summary.get("valid_success_rate") or 0.0),
        "eval10_periodic/min_video_frames": int(summary.get("min_video_frames") or args.min_video_frames),
        "eval10_periodic/visual_rollout_valid_trials": int(summary.get("visual_rollout_valid_trials") or 0),
        "eval10_periodic/visual_rollout_too_short_trials": int(summary.get("visual_rollout_too_short_trials") or 0),
        "eval10_periodic/visual_rollout_valid_fraction": float(summary.get("visual_rollout_valid_fraction") or 0.0),
    }
    payload["eval10_periodic/trial_split_counts_json"] = json.dumps(summary.get("trial_split_counts") or {}, sort_keys=True)
    payload["eval10_periodic/trial_taskvar_counts_json"] = json.dumps(summary.get("trial_taskvar_counts") or {}, sort_keys=True)
    videos = _selected_videos(
        output_root,
        max_videos=int(args.max_videos),
        min_video_frames=int(args.min_video_frames),
    )
    payload["eval10_periodic/video_count"] = len(videos)
    payload["eval10_periodic/video_paths_json"] = json.dumps([str(path) for path, _ in videos])
    for idx, (path, row) in enumerate(videos):
        taskvar = row.get("taskvar") or path.parent.name
        demo_id = row.get("demo_id", "unknown")
        success = row.get("success", "unknown")
        frames = _row_video_frames(row)
        caption = f"step={step} taskvar={taskvar} demo={demo_id} success={success} frames={frames}"
        payload[f"eval10_periodic/video_{idx:02d}"] = wandb.Video(
            str(path),
            fps=int(args.video_fps),
            format="mp4",
            caption=caption,
        )
    predicted_videos = _selected_predicted_videos(output_root, max_videos=int(args.max_videos))
    payload["eval10_periodic/predicted_video_count"] = len(predicted_videos)
    payload["eval10_periodic/predicted_video_paths_json"] = json.dumps(
        [str(path) for path, _, _ in predicted_videos]
    )
    for idx, (path, row, kind) in enumerate(predicted_videos):
        taskvar = row.get("taskvar") or path.parent.name
        demo_id = row.get("demo_id", "unknown")
        caption = f"step={step} kind={kind} taskvar={taskvar} demo={demo_id}"
        payload[f"eval10_periodic/predicted_video_{idx:02d}"] = wandb.Video(
            str(path),
            fps=int(args.video_fps),
            format="mp4",
            caption=caption,
        )
    if metric_prefix != "eval10_periodic":
        payload = {
            key.replace("eval10_periodic/", f"{metric_prefix}/", 1) if key.startswith("eval10_periodic/") else key: value
            for key, value in payload.items()
        }
    wandb.log(payload, step=int(step))
    print(
        f"[eval10-wandb] logged step={step} prefix={metric_prefix} videos={len(videos)} "
        f"success_rate={float(summary.get('success_rate') or 0.0):.4f}",
        flush=True,
    )


def _process_checkpoint(args: argparse.Namespace, *, step: int, checkpoint: Path, wandb: Any | None) -> dict[str, Any]:
    returncode, output_root = _run_eval10(args, step=step, checkpoint=checkpoint)
    if wandb is not None:
        _wandb_log_eval10(wandb, args, step=step, output_root=output_root, returncode=returncode)
    return {
        "step": int(step),
        "checkpoint": str(checkpoint),
        "output_root": str(output_root),
        "returncode": int(returncode),
        "time": time.time(),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Watch GEMBench policy checkpoints and log official-style eval10 videos to W&B.")
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--model-name", default=None, help="Stable output namespace, e.g. fastwam or trace.")
    parser.add_argument("--run-id", default=None, help="Stable output namespace under model-name. Defaults to run name or run-dir basename.")
    parser.add_argument("--output-root", default=None)
    parser.add_argument("--eval10-config", default=str(DEFAULT_EVAL10_CONFIG))
    parser.add_argument("--gembench-root", default=str(DEFAULT_GEMBENCH_ROOT))
    parser.add_argument("--assets-root", default=str(DEFAULT_ASSETS_ROOT))
    parser.add_argument("--robot-3dlotus-root", default=str(DEFAULT_ROBOT_3DLOTUS_ROOT))
    parser.add_argument("--coppeliasim-root", default=str(DEFAULT_COPPELIASIM_ROOT))
    parser.add_argument("--rlbench-root", default=str(DEFAULT_RLBENCH_ROOT))
    parser.add_argument("--pyrep-root", default=str(DEFAULT_PYREP_ROOT))
    parser.add_argument("--xvfb-run", dest="xvfb_run", action="store_true", default=True)
    parser.add_argument("--no-xvfb-run", dest="xvfb_run", action="store_false")
    parser.add_argument("--interval-steps", type=int, default=2000)
    parser.add_argument("--num-trials", type=int, default=10)
    parser.add_argument("--max-videos", type=int, default=10)
    parser.add_argument("--min-video-frames", type=int, default=60)
    parser.add_argument("--min-step", type=int, default=1)
    parser.add_argument("--stop-after-step", type=int, default=0)
    parser.add_argument("--checkpoint-stable-seconds", type=float, default=30.0)
    parser.add_argument("--max-retries-per-step", type=int, default=3)
    parser.add_argument("--poll-seconds", type=float, default=60.0)
    parser.add_argument("--once", action="store_true")
    parser.add_argument("--state-path", default=None)
    parser.add_argument("--global-lock-path", default=str(DEFAULT_GLOBAL_LOCK_PATH))
    parser.add_argument("--dry-run-eval", action="store_true", help="Run eval10 selection dry-run instead of loading simulator/model.")
    parser.add_argument("--python-bin", default=sys.executable)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--cuda-visible-devices", default=os.environ.get("GEMBENCH_POLICY_EVAL10_CUDA_VISIBLE_DEVICES"))
    parser.add_argument("--relation-mode", choices=["auto", "none", "noop_smoke", "online_current"], default="auto")
    parser.add_argument(
        "--eval-protocol",
        choices=["official_one_step", "chunk_replan", "fastwam_chunk_replan", "trace_chunk_replan"],
        default="chunk_replan",
    )
    parser.add_argument("--chunk-replan-steps", type=int, default=1)
    parser.add_argument("--chunk-predict-video", action="store_true")
    parser.add_argument("--num-inference-steps", type=int, default=10)
    parser.add_argument("--video-fps", type=int, default=30)
    parser.add_argument("--tiled", action="store_true")
    parser.add_argument("--extra-eval-arg", action="append", default=[])
    parser.add_argument("--no-wandb", action="store_true")
    parser.add_argument("--wandb-fallback-offline", dest="wandb_fallback_offline", action="store_true", default=True)
    parser.add_argument("--no-wandb-fallback-offline", dest="wandb_fallback_offline", action="store_false")
    parser.add_argument("--wandb-mode", choices=["sidecar", "same-run"], default=os.environ.get("GEMBENCH_POLICY_EVAL10_WANDB_MODE", "sidecar"))
    parser.add_argument("--wandb-project", default=os.environ.get("WANDB_PROJECT", "fastwam-gembench"))
    parser.add_argument("--wandb-entity", default=os.environ.get("WANDB_ENTITY") or os.environ.get("WANDB_WORKSPACE"))
    parser.add_argument("--wandb-group", default=os.environ.get("WANDB_GROUP"))
    parser.add_argument("--wandb-name", default=None)
    parser.add_argument("--wandb-run-id", default=None)
    parser.add_argument("--wandb-metric-prefix", default=None)
    parser.add_argument("--run-name", default=os.environ.get("WANDB_RUN_NAME") or os.environ.get("RUN_ID"))
    args = parser.parse_args()

    run_dir = Path(args.run_dir).expanduser().resolve()
    args.model_name = _infer_model_name(args.model_name, run_dir, args.run_name)
    args.run_id = _infer_run_id(args.run_id, run_dir, args.run_name)
    if args.output_root is None:
        args.output_root = str(DEFAULT_OUTPUT_ROOT / args.model_name / args.run_id)
    state_path = Path(args.state_path) if args.state_path else Path(args.output_root) / "watch_state.json"
    state = _load_json(state_path, {"processed_steps": []})
    processed_steps = {int(step) for step in state.get("processed_steps", [])}
    failed_steps = {int(step) for step in state.get("failed_steps", [])}
    attempts = {int(key): int(value) for key, value in dict(state.get("attempts", {})).items()}

    wandb, run = _wandb_init(args)
    try:
        while True:
            processed_this_loop = False
            candidates = _candidate_checkpoints(
                run_dir,
                interval_steps=int(args.interval_steps),
                min_step=int(args.min_step),
                checkpoint_stable_seconds=float(args.checkpoint_stable_seconds),
            )
            state["pending_steps"] = [
                int(step) for step, _ in candidates if int(step) not in processed_steps and int(step) not in failed_steps
            ]
            state["model_name"] = str(args.model_name)
            state["run_id"] = str(args.run_id)
            state["run_dir"] = str(run_dir)
            state["output_root"] = str(Path(args.output_root).resolve())
            state["interval_steps"] = int(args.interval_steps)
            state["eval_protocol"] = _normalize_eval_protocol(args.eval_protocol)
            state["chunk_replan_steps"] = int(args.chunk_replan_steps)
            state["chunk_predict_video"] = bool(args.chunk_predict_video)
            _write_json(state_path, state)
            for step, checkpoint in candidates:
                if step in processed_steps or step in failed_steps:
                    continue
                if attempts.get(step, 0) >= int(args.max_retries_per_step):
                    failed_steps.add(step)
                    state["failed_steps"] = sorted(failed_steps)
                    _write_json(state_path, state)
                    continue
                result = _process_checkpoint(args, step=step, checkpoint=checkpoint, wandb=wandb)
                attempts[step] = attempts.get(step, 0) + 1
                state.setdefault("history", []).append(result)
                state["last_step"] = int(step)
                state["attempts"] = {str(key): int(value) for key, value in sorted(attempts.items())}
                if int(result["returncode"]) == 0:
                    processed_steps.add(step)
                    state["processed_steps"] = sorted(processed_steps)
                elif attempts[step] >= int(args.max_retries_per_step):
                    failed_steps.add(step)
                    state["failed_steps"] = sorted(failed_steps)
                state["pending_steps"] = [
                    int(candidate_step)
                    for candidate_step in state.get("pending_steps", [])
                    if int(candidate_step) not in processed_steps and int(candidate_step) not in failed_steps
                ]
                _write_json(state_path, state)
                processed_this_loop = True
                if args.once:
                    return int(result["returncode"])
                if int(args.stop_after_step) > 0 and step >= int(args.stop_after_step):
                    print(f"[eval10-wandb] stop-after-step reached: {step}", flush=True)
                    return int(result["returncode"])
            if args.once:
                if not processed_this_loop:
                    print(f"[eval10-wandb] no matching checkpoints yet under {run_dir / 'checkpoints' / 'weights'}", flush=True)
                return 0
            if int(args.stop_after_step) > 0 and processed_steps and max(processed_steps) >= int(args.stop_after_step):
                print(f"[eval10-wandb] stop-after-step already processed: {max(processed_steps)}", flush=True)
                return 0
            time.sleep(float(args.poll_seconds))
    finally:
        if run is not None:
            run.finish()


if __name__ == "__main__":
    raise SystemExit(main())
