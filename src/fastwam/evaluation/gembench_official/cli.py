from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

from .common import (
    DEFAULT_GEMBENCH_ROOT,
    OFFICIAL_CAMERA_NAMES,
    PROJECT_ROOT,
    available_demo_ids,
    git_provenance,
    load_taskvars,
    official_microstep_dir,
    official_taskvar_file,
    resolve_assets_root,
    resolve_checkpoint,
    resolve_existing_path,
    safe_token,
    selected_trials,
    write_json,
)


DEFAULT_EVAL10_CONFIG = PROJECT_ROOT / "configs" / "eval" / "gembench_official_eval10_periodic_indomain_ood_seed200.yaml"
CHUNK_REPLAN_PROTOCOL_ALIASES = {"chunk_replan", "fastwam_chunk_replan", "trace_chunk_replan"}


def _timestamp() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def _parse_csv(value: str | None) -> list[str]:
    if value is None or str(value).strip() == "":
        return []
    return [item.strip() for item in str(value).split(",") if item.strip()]


def _parse_int_csv(value: str | None) -> list[int]:
    return [int(item) for item in _parse_csv(value)]


def _parse_pair(value: str) -> tuple[int, int]:
    parts = _parse_int_csv(value)
    if len(parts) != 2:
        raise argparse.ArgumentTypeError(f"Expected two comma-separated integers, got {value!r}")
    return int(parts[0]), int(parts[1])


def _normalize_eval_protocol(value: Any) -> str:
    protocol = str(value)
    if protocol in CHUNK_REPLAN_PROTOCOL_ALIASES:
        return "chunk_replan"
    return protocol


def _load_yaml_config(path: Path) -> dict[str, Any]:
    try:
        import yaml
    except ModuleNotFoundError:
        try:
            from omegaconf import OmegaConf
        except ModuleNotFoundError as exc:
            try:
                return _load_flat_yaml_config(path)
            except Exception as fallback_exc:
                raise RuntimeError(
                    f"Reading YAML config requires PyYAML/OmegaConf, or a flat YAML mapping supported "
                    f"by the stdlib fallback: {path}"
                ) from fallback_exc
        payload = OmegaConf.to_container(OmegaConf.load(path), resolve=True) or {}
    else:
        payload = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    if not isinstance(payload, dict):
        raise ValueError(f"YAML config must contain a mapping: {path}")
    return payload


def _parse_flat_yaml_scalar(value: str) -> Any:
    value = value.strip()
    if value == "":
        return ""
    if (value.startswith('"') and value.endswith('"')) or (value.startswith("'") and value.endswith("'")):
        return value[1:-1]
    lower = value.lower()
    if lower == "true":
        return True
    if lower == "false":
        return False
    if lower in {"null", "none"}:
        return None
    try:
        return int(value)
    except ValueError:
        pass
    try:
        return float(value)
    except ValueError:
        return value


def _load_flat_yaml_config(path: Path) -> dict[str, Any]:
    """Parse the simple top-level eval10 YAML used when PyYAML is unavailable."""
    payload: dict[str, Any] = {}
    current_list_key: str | None = None
    for line_no, raw_line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        line = raw_line.rstrip()
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        if line[:1].isspace():
            if current_list_key is None or not stripped.startswith("- "):
                raise ValueError(f"Unsupported nested YAML at {path}:{line_no}: {raw_line}")
            payload[current_list_key].append(_parse_flat_yaml_scalar(stripped[2:]))
            continue
        if ":" not in line:
            raise ValueError(f"Expected key/value YAML at {path}:{line_no}: {raw_line}")
        key, value = line.split(":", 1)
        key = key.strip()
        value = value.strip()
        if not key:
            raise ValueError(f"Empty YAML key at {path}:{line_no}")
        if value == "":
            payload[key] = []
            current_list_key = key
        else:
            payload[key] = _parse_flat_yaml_scalar(value)
            current_list_key = None
    return payload


def _csv_from_config(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    if isinstance(value, (list, tuple)):
        return ",".join(str(item) for item in value)
    raise ValueError(f"Expected string or list for config list value, got {type(value).__name__}")


def _parse_cam_ids(value: str | None) -> tuple[int, ...]:
    cam_ids = tuple(_parse_int_csv(value) or [0, 1, 2, 3])
    invalid = [idx for idx in cam_ids if idx < 0 or idx >= len(OFFICIAL_CAMERA_NAMES)]
    if invalid:
        raise ValueError(f"--cam-ids contains invalid official camera ids {invalid}; valid ids are 0..3.")
    return cam_ids


def _explicit_cli_dests(parser: argparse.ArgumentParser, argv: list[str]) -> set[str]:
    option_to_dest: dict[str, str] = {}
    for action in parser._actions:
        for option in action.option_strings:
            option_to_dest[option] = action.dest
    explicit: set[str] = set()
    for token in argv:
        option = token.split("=", 1)[0]
        dest = option_to_dest.get(option)
        if dest:
            explicit.add(dest)
    return explicit


def _load_cfg_and_stats(run_dir: Path) -> tuple[Any, Path]:
    cfg_path = run_dir / "config.yaml"
    if not cfg_path.exists():
        raise FileNotFoundError(f"Missing run config: {cfg_path}")
    try:
        from omegaconf import OmegaConf

        cfg = OmegaConf.load(cfg_path)
        stats_value = cfg.data.train.pretrained_norm_stats
    except ModuleNotFoundError:
        try:
            import yaml

            cfg = yaml.safe_load(cfg_path.read_text(encoding="utf-8"))
            stats_value = cfg["data"]["train"]["pretrained_norm_stats"]
        except ModuleNotFoundError:
            cfg = None
            stats_value = None
            for raw_line in cfg_path.read_text(encoding="utf-8").splitlines():
                stripped = raw_line.strip()
                if stripped.startswith("pretrained_norm_stats:"):
                    stats_value = _parse_flat_yaml_scalar(stripped.split(":", 1)[1])
                    break
            if stats_value is None:
                raise RuntimeError(
                    f"Reading run config without OmegaConf/PyYAML requires a pretrained_norm_stats line: {cfg_path}"
                )
    stats_path = resolve_existing_path(
        stats_value,
        label="pretrained_norm_stats",
        bases=[run_dir, PROJECT_ROOT],
    )
    return cfg, stats_path


def _nested_get(obj: Any, path: tuple[str, ...], default: Any = None) -> Any:
    cur = obj
    for key in path:
        if cur is None:
            return default
        try:
            if isinstance(cur, dict):
                cur = cur[key]
            else:
                cur = getattr(cur, key)
            continue
        except (AttributeError, KeyError, TypeError):
            pass
        try:
            cur = cur.get(key)
        except Exception:
            return default
    return cur


def _effective_chunk_action_horizon(args: argparse.Namespace, cfg: Any) -> int:
    policy_horizon = _nested_get(cfg, ("policy_contract", "action_horizon"), None)
    train_horizon = _nested_get(cfg, ("data", "train", "action_horizon"), 1)
    aux_horizon = _nested_get(cfg, ("policy_contract", "policy_vgm_auxiliary_action_horizon"), None)
    policy_horizon = int(policy_horizon) if policy_horizon is not None else int(train_horizon)
    if str(args.eval_protocol) != "chunk_replan":
        return int(policy_horizon)
    if int(args.chunk_action_horizon) > 0:
        return int(args.chunk_action_horizon)
    if int(args.chunk_replan_steps) <= int(policy_horizon):
        return int(policy_horizon)
    candidates = [int(policy_horizon), int(train_horizon), int(args.chunk_replan_steps)]
    if aux_horizon is not None:
        candidates.append(int(aux_horizon))
    return int(max(candidates))


def _apply_eval10_config(args: argparse.Namespace) -> None:
    config_arg = getattr(args, "eval10_config", None)
    args.eval10_config_path = None
    args.eval10_config_payload = {}
    if not config_arg:
        return
    explicit_args = set(getattr(args, "_explicit_args", set()))
    config_path = resolve_existing_path(config_arg, label="eval10_config", bases=[PROJECT_ROOT])
    config = _load_yaml_config(config_path)
    args.eval10_config_path = str(config_path)
    args.eval10_config_payload = config

    csv_keys = {"seeds", "taskvar_splits"}
    for key in (
        "suite",
        "seed",
        "seeds",
        "taskvar_split",
        "taskvar_splits",
        "taskvar_file",
        "microstep_data_dir",
        "limit_taskvars",
        "num_inference_steps",
        "max_steps",
        "max_tries",
        "record_video",
        "video_mode",
        "video_fps",
        "video_stride",
        "video_resolution",
        "video_include_robot_cameras",
        "video_rotate_cam",
        "video_recorder_required",
        "video_initial_snap",
        "min_video_frames",
        "initial_success_policy",
        "write_official_preds",
        "eval_protocol",
        "chunk_replan_steps",
        "chunk_action_horizon",
        "chunk_predict_video",
    ):
        if key in config and key not in explicit_args:
            setattr(args, key, _csv_from_config(config[key]) if key in csv_keys else config[key])

    if "taskvars" in config and "taskvars" not in explicit_args:
        args.taskvars = _csv_from_config(config["taskvars"])

    demos_per_taskvar = (
        config.get("demos_per_taskvar")
        if "demos_per_taskvar" in config
        else config.get("num_demos_per_taskvar", config.get("num_demos"))
    )
    if demos_per_taskvar is not None and "num_demos" not in explicit_args:
        args.num_demos = int(demos_per_taskvar)

    if "num_trials" in config and "num_trials" not in explicit_args:
        args.num_trials = int(config["num_trials"])
    elif "num_trials" not in explicit_args and demos_per_taskvar is not None and args.taskvars:
        args.num_trials = len(_parse_csv(args.taskvars)) * int(args.num_demos)


def _trial_counts(trials: list[Any]) -> dict[str, dict[str, int]]:
    split_counts: dict[str, int] = {}
    taskvar_counts: dict[str, int] = {}
    for trial in trials:
        split = str(getattr(trial, "split"))
        taskvar = str(getattr(trial, "taskvar"))
        split_counts[split] = split_counts.get(split, 0) + 1
        taskvar_counts[taskvar] = taskvar_counts.get(taskvar, 0) + 1
    return {
        "trial_split_counts": dict(sorted(split_counts.items())),
        "trial_taskvar_counts": dict(sorted(taskvar_counts.items())),
    }


def _official_contract_payload(args: argparse.Namespace, *, eval10: bool) -> dict[str, Any]:
    expected_cam_ids = tuple(range(len(OFFICIAL_CAMERA_NAMES)))
    checks = {
        "not_eval10": not eval10,
        "max_steps_25": int(args.max_steps) == 25,
        "max_tries_10": int(args.max_tries) == 10,
        "image_size_256": tuple(int(v) for v in args.image_size) == (256, 256),
        "official_four_cameras": _parse_cam_ids(args.cam_ids) == expected_cam_ids,
        "initial_success_record_only": str(args.initial_success_policy) == "record_only",
    }
    return {
        "official_contract_ok": all(checks.values()),
        "official_contract_checks": checks,
        "official_contract_expected": {
            "max_steps": 25,
            "max_tries": 10,
            "image_size": [256, 256],
            "camera_names": list(OFFICIAL_CAMERA_NAMES),
            "initial_success_policy": "record_only",
        },
    }


def _official_scope_payload(args: argparse.Namespace, *, eval10: bool) -> dict[str, Any]:
    test_seeds = _parse_int_csv(args.seeds) or [200, 300, 400, 500, 600]
    test_splits = _parse_csv(args.taskvar_splits) or ["train", "test_l2", "test_l3", "test_l4"]
    val_seed = int(args.seed if args.seed is not None else 100)
    contract_payload = _official_contract_payload(args, eval10=eval10)
    eval_protocol = _normalize_eval_protocol(getattr(args, "eval_protocol", "official_one_step"))
    official_full = (
        not eval10
        and eval_protocol == "official_one_step"
        and args.suite == "official_full"
        and val_seed == 100
        and test_seeds == [200, 300, 400, 500, 600]
        and test_splits == ["train", "test_l2", "test_l3", "test_l4"]
        and int(args.num_demos) == 20
        and args.limit_taskvars is None
        and not _parse_csv(args.taskvars)
    )
    official_test = (
        not eval10
        and eval_protocol == "official_one_step"
        and args.suite == "test"
        and test_seeds == [200, 300, 400, 500, 600]
        and test_splits == ["train", "test_l2", "test_l3", "test_l4"]
        and int(args.num_demos) == 20
        and args.limit_taskvars is None
        and not _parse_csv(args.taskvars)
    )
    if eval_protocol == "chunk_replan":
        scope = "chunk_replan_visual_diagnostic_not_official_score"
    elif eval10:
        scope = "eval10_visual_diagnostic_not_official_score"
    elif official_full and not contract_payload["official_contract_ok"]:
        scope = "official_full_matrix_with_nonofficial_parameters"
    elif official_full:
        scope = "official_full_val_seed100_plus_test_seeds_200_600_all_splits"
    elif official_test:
        scope = "official_test_matrix_only"
    else:
        scope = "partial_official_style_run"
    return {
        "official_full_scope_requested": official_full,
        "official_test_matrix_requested": official_test,
        "official_full_score": False,
        "official_test_matrix": official_test,
        "official_eval_scope": scope,
        "eval_protocol": eval_protocol,
        **contract_payload,
    }


def _official_completion_payload(
    summary: dict[str, Any],
    scope_payload: dict[str, Any],
    *,
    requested_num_demos: int,
) -> dict[str, Any]:
    rows = list(summary.get("official_results") or [])
    demo_load_failures = sum(len(row.get("demo_load_failures") or []) for row in rows)
    taskvar_failures = len(summary.get("skipped_taskvars") or [])
    rows_with_partial_demos = [
        row.get("taskvar")
        for row in rows
        if int(row.get("num_demos", 0)) != int(requested_num_demos)
        or int(row.get("requested_num_demos", requested_num_demos)) != int(requested_num_demos)
    ]
    completed_trials = sum(int(row.get("num_demos", 0)) for row in rows)
    expected_trials = sum(int(row.get("requested_num_demos", requested_num_demos)) for row in rows)
    official_rows_complete = (
        bool(rows)
        and demo_load_failures == 0
        and taskvar_failures == 0
        and not rows_with_partial_demos
        and int(summary.get("trials", 0)) == completed_trials
        and completed_trials == expected_trials
    )
    official_full_score = bool(
        scope_payload.get("official_full_scope_requested")
        and scope_payload.get("official_contract_ok")
        and official_rows_complete
        and bool(summary.get("write_official_preds"))
        and str(summary.get("relation_mode")) != "noop_smoke"
    )
    return {
        "official_full_score": official_full_score,
        "official_rows_complete": official_rows_complete,
        "official_completion": {
            "completed_trials": int(completed_trials),
            "expected_trials": int(expected_trials),
            "official_result_rows": len(rows),
            "demo_load_failures": int(demo_load_failures),
            "skipped_taskvars": int(taskvar_failures),
            "rows_with_partial_demos": rows_with_partial_demos,
        },
    }


def _resolve_output_root(args: argparse.Namespace, *, mode: str, checkpoint: Path | None) -> Path:
    if args.output_root:
        return Path(args.output_root).expanduser().resolve()
    ckpt_tag = checkpoint.stem if checkpoint is not None else "dry_run_no_checkpoint"
    return Path(args.run_dir).resolve() / mode / f"{ckpt_tag}_{_timestamp()}"


def _build_jobs(args: argparse.Namespace) -> list[dict[str, Any]]:
    gembench_root = Path(args.gembench_root).expanduser().resolve()
    assets_root = resolve_assets_root(gembench_root, args.assets_root, args.robot_3dlotus_root)
    jobs: list[dict[str, Any]] = []
    if args.suite == "val":
        seed = int(args.seed if args.seed is not None else 100)
        jobs.append(
            {
                "suite": "val",
                "seed": seed,
                "split": "train",
                "microstep_data_dir": official_microstep_dir(gembench_root, suite="val", seed=seed),
                "taskvar_file": official_taskvar_file(assets_root, "train"),
            }
        )
    elif args.suite == "test":
        seeds = _parse_int_csv(args.seeds) or [200, 300, 400, 500, 600]
        splits = _parse_csv(args.taskvar_splits) or ["train", "test_l2", "test_l3", "test_l4"]
        for seed in seeds:
            for split in splits:
                jobs.append(
                    {
                        "suite": "test",
                        "seed": int(seed),
                        "split": split,
                        "microstep_data_dir": official_microstep_dir(gembench_root, suite="test", seed=int(seed)),
                        "taskvar_file": official_taskvar_file(assets_root, split),
                    }
                )
    elif args.suite == "official_full":
        val_seed = int(args.seed if args.seed is not None else 100)
        jobs.append(
            {
                "suite": "val",
                "seed": val_seed,
                "split": "train",
                "microstep_data_dir": official_microstep_dir(gembench_root, suite="val", seed=val_seed),
                "taskvar_file": official_taskvar_file(assets_root, "train"),
            }
        )
        seeds = _parse_int_csv(args.seeds) or [200, 300, 400, 500, 600]
        splits = _parse_csv(args.taskvar_splits) or ["train", "test_l2", "test_l3", "test_l4"]
        for seed in seeds:
            for split in splits:
                jobs.append(
                    {
                        "suite": "test",
                        "seed": int(seed),
                        "split": split,
                        "microstep_data_dir": official_microstep_dir(gembench_root, suite="test", seed=int(seed)),
                        "taskvar_file": official_taskvar_file(assets_root, split),
                    }
                )
    elif args.suite == "custom":
        if args.microstep_data_dir is None or args.taskvar_file is None or args.seed is None:
            raise ValueError("--suite custom requires --microstep-data-dir, --taskvar-file, and --seed.")
        jobs.append(
            {
                "suite": "custom",
                "seed": int(args.seed),
                "split": str(args.taskvar_split),
                "microstep_data_dir": resolve_existing_path(args.microstep_data_dir, label="microstep_data_dir"),
                "taskvar_file": resolve_existing_path(args.taskvar_file, label="taskvar_file"),
            }
        )
    else:
        raise ValueError(f"Unsupported suite: {args.suite}")
    return jobs


def _select_trials(args: argparse.Namespace, *, eval10: bool) -> tuple[list[Any], list[str], list[dict[str, Any]]]:
    trials = []
    skipped: list[str] = []
    job_payloads: list[dict[str, Any]] = []
    taskvar_filter = _parse_csv(args.taskvars)
    taskvar_filter_set = set(taskvar_filter)
    taskvar_order = {item: idx for idx, item in enumerate(taskvar_filter)}
    seen_filter_items: set[str] = set()
    remaining = int(args.num_trials) if eval10 else None
    for job in _build_jobs(args):
        microstep_data_dir = Path(job["microstep_data_dir"]).resolve()
        taskvar_file = Path(job["taskvar_file"]).resolve()
        taskvars = load_taskvars(taskvar_file)
        if taskvar_filter_set:
            taskvars = [
                taskvar for taskvar in taskvars if taskvar.key in taskvar_filter_set or taskvar.task in taskvar_filter_set
            ]
            taskvars.sort(key=lambda taskvar: min(taskvar_order.get(taskvar.key, 10**9), taskvar_order.get(taskvar.task, 10**9)))
            for taskvar in taskvars:
                seen_filter_items.add(taskvar.key)
                seen_filter_items.add(taskvar.task)
        if args.limit_taskvars is not None:
            taskvars = taskvars[: int(args.limit_taskvars)]
        missing = [taskvar.key for taskvar in taskvars if not available_demo_ids(microstep_data_dir, taskvar)]
        skipped.extend(missing)
        if eval10:
            selected = _selected_eval10_round_robin(
                taskvars=taskvars,
                microstep_data_dir=microstep_data_dir,
                seed=int(job["seed"]),
                split=str(job["split"]),
                taskvar_file=taskvar_file,
                num_demos=int(args.num_demos),
                num_trials=int(remaining or 0),
            )
        else:
            selected = selected_trials(
                taskvars=taskvars,
                microstep_data_dir=microstep_data_dir,
                seed=int(job["seed"]),
                split=str(job["split"]),
                taskvar_file=taskvar_file,
                num_demos=int(args.num_demos),
                num_trials=None,
            )
        trials.extend(selected)
        if eval10 and remaining is not None:
            remaining -= len(selected)
            if remaining <= 0:
                remaining = 0
        job_payloads.append(
            {
                **job,
                "microstep_data_dir": str(microstep_data_dir),
                "taskvar_file": str(taskvar_file),
                "taskvars": len(taskvars),
                "taskvar_keys": [taskvar.key for taskvar in taskvars],
                "selected_trials": len(selected),
                "skipped_taskvars": missing,
            }
        )
        if eval10 and remaining == 0:
            break
    if taskvar_filter_set:
        missing_filter_items = [item for item in taskvar_filter if item not in seen_filter_items]
        skipped.extend(f"missing_taskvar_in_selected_files:{item}" for item in missing_filter_items)
    if eval10 and len(trials) > int(args.num_trials):
        trials = trials[: int(args.num_trials)]
    if not trials:
        raise RuntimeError("No official GEMBench trials selected. Check microsteps/taskvar files and filters.")
    return trials, skipped, job_payloads


def _selected_eval10_round_robin(
    *,
    taskvars: list[Any],
    microstep_data_dir: Path,
    seed: int,
    split: str,
    taskvar_file: Path,
    num_demos: int,
    num_trials: int,
) -> list[Any]:
    from .common import TrialSpec

    if num_trials <= 0:
        return []
    demo_ids_by_taskvar = {
        taskvar: available_demo_ids(microstep_data_dir, taskvar)[: max(int(num_demos), 0)]
        for taskvar in taskvars
    }
    out = []
    for demo_offset in range(max(int(num_demos), 0)):
        for taskvar in taskvars:
            demo_ids = demo_ids_by_taskvar.get(taskvar) or []
            if demo_offset >= len(demo_ids):
                continue
            out.append(
                TrialSpec(
                    taskvar=taskvar.key,
                    task=taskvar.task,
                    variation=taskvar.variation,
                    demo_id=int(demo_ids[demo_offset]),
                    seed=int(seed),
                    split=str(split),
                    taskvar_file=str(taskvar_file),
                    microstep_data_dir=str(microstep_data_dir),
                )
            )
            if len(out) >= int(num_trials):
                return out
    return out


def _dry_run_payload(
    args: argparse.Namespace,
    *,
    mode: str,
    checkpoint: Path | None,
    cfg: Any,
    stats_path: Path,
    output_root: Path,
    trials: list[Any],
    skipped: list[str],
    jobs: list[dict[str, Any]],
) -> dict[str, Any]:
    payload = {
        "evidence_type": f"gembench_official_{mode}_dry_run",
        "status": "dry_run",
        "provenance": git_provenance(),
        "run_dir": str(Path(args.run_dir).resolve()),
        "checkpoint": str(checkpoint) if checkpoint is not None else None,
        "dataset_stats": str(stats_path),
        "output_root": str(output_root),
        "suite": args.suite,
        "relation_mode": str(args.relation_mode),
        "num_inference_steps": int(args.num_inference_steps),
        "model_seed": int(args.model_seed),
        "max_steps": int(args.max_steps),
        "max_tries": int(args.max_tries),
        "num_demos": int(args.num_demos),
        "num_trials": len(trials),
        "configured_num_trials": int(args.num_trials),
        "taskvars": _parse_csv(args.taskvars),
        "eval10_config": getattr(args, "eval10_config_path", None),
        "eval10_config_payload": getattr(args, "eval10_config_payload", {}),
        "camera_names": [OFFICIAL_CAMERA_NAMES[idx] for idx in _parse_cam_ids(args.cam_ids)],
        "record_video": bool(args.record_video),
        "video_mode": str(args.video_mode),
        "video_fps": int(args.video_fps),
        "video_stride": int(args.video_stride),
        "video_resolution": int(args.video_resolution),
        "video_include_robot_cameras": bool(args.video_include_robot_cameras),
        "video_rotate_cam": bool(args.video_rotate_cam),
        "video_recorder_required": bool(args.video_recorder_required),
        "video_initial_snap": bool(args.video_initial_snap),
        "min_video_frames": int(args.min_video_frames),
        "initial_success_policy": str(args.initial_success_policy),
        "write_official_preds": bool(args.write_official_preds),
        "eval_protocol": str(args.eval_protocol),
        "chunk_replan_steps": int(args.chunk_replan_steps),
        "chunk_action_horizon": int(args.chunk_action_horizon),
        "effective_chunk_action_horizon": int(_effective_chunk_action_horizon(args, cfg)),
        "chunk_predict_video": bool(args.chunk_predict_video),
        **_official_scope_payload(args, eval10=(mode == "eval10")),
        "jobs": jobs,
        "trials": [trial.__dict__ for trial in trials],
        **_trial_counts(trials),
        "skipped_taskvars": skipped,
        "environment": {
            "COPPELIASIM_ROOT": os.environ.get("COPPELIASIM_ROOT"),
            "DISPLAY": os.environ.get("DISPLAY"),
            "xvfb_run": shutil.which("xvfb-run"),
            "ROBOT_3DLOTUS_ROOT": args.robot_3dlotus_root or os.environ.get("ROBOT_3DLOTUS_ROOT"),
            "text_cache_required": False,
            "text_encoder_required": True,
        },
    }
    if mode == "eval10":
        payload.update(
            {
                "official_full_score": False,
                "note": "Dry-run for configured closed-loop simulator visual smoke with max_steps=25 by default, not full official 20 demos/taskvar score.",
            }
        )
    return payload


def _completed_trial_keys(output_root: Path, checkpoint: Path | None, *, min_num_demos: int) -> set[tuple[int, str]]:
    if checkpoint is None:
        return set()
    checkpoint_text = str(checkpoint)
    checkpoint_resolved = str(Path(checkpoint).expanduser().resolve())
    completed: set[tuple[int, str]] = set()
    for results_path in sorted((output_root / "preds").glob("seed*/results.jsonl")):
        seed_name = results_path.parent.name
        if not seed_name.startswith("seed"):
            continue
        try:
            seed = int(seed_name[len("seed") :])
        except ValueError:
            continue
        for line in results_path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            row_checkpoint = str(row.get("checkpoint"))
            row_checkpoint_resolved = str(Path(row_checkpoint).expanduser().resolve()) if row_checkpoint else ""
            if row_checkpoint not in {checkpoint_text, checkpoint_resolved} and row_checkpoint_resolved != checkpoint_resolved:
                continue
            task = row.get("task")
            variation = row.get("variation")
            if task is None or variation is None:
                continue
            try:
                row_num_demos = int(row.get("num_demos"))
            except (TypeError, ValueError):
                continue
            if row_num_demos < int(min_num_demos):
                continue
            completed.add((seed, f"{task}+{int(variation)}"))
    return completed


def _filter_completed_trials(
    trials: list[Any],
    *,
    output_root: Path,
    checkpoint: Path | None,
    min_num_demos: int,
    skip_completed: bool = True,
) -> tuple[list[Any], list[str]]:
    if not skip_completed:
        return trials, []
    completed = _completed_trial_keys(output_root, checkpoint, min_num_demos=int(min_num_demos))
    if not completed:
        return trials, []
    filtered = []
    skipped = []
    for trial in trials:
        key = (int(trial.seed), str(trial.taskvar))
        if key in completed:
            skipped.append(f"already_completed:seed{trial.seed}:{trial.taskvar}:demo{trial.demo_id}")
        else:
            filtered.append(trial)
    return filtered, skipped


def _write_eval10_markdown(output_root: Path, manifest: dict[str, Any]) -> None:
    rows = []
    results_path = output_root / "trials" / "results.jsonl"
    if results_path.exists():
        for line in results_path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                rows.append(json.loads(line))
    success_rate = manifest.get("success_rate")
    valid_success_rate = manifest.get("valid_success_rate")
    success_text = "n/a" if success_rate is None else f"{float(success_rate):.4f}"
    valid_success_text = "n/a" if valid_success_rate is None else f"{float(valid_success_rate):.4f}"
    success_count = manifest.get("successes")
    trial_count = manifest.get("trials")
    valid_success_count = manifest.get("valid_successes")
    valid_trial_count = manifest.get("valid_trials")
    count_text = "n/a" if success_count is None or trial_count is None else f"{success_count}/{trial_count}"
    valid_count_text = (
        "n/a" if valid_success_count is None or valid_trial_count is None else f"{valid_success_count}/{valid_trial_count}"
    )
    lines = [
        "# GEMBench Official-Style Eval10",
        "",
        "This is a configured closed-loop simulator visual smoke, not the full official 20 demos/taskvar score.",
        "",
        f"Run dir: `{manifest['run_dir']}`",
        f"Checkpoint: `{manifest['checkpoint']}`",
        f"Success rate: `{success_text}` ({count_text})",
        f"Valid diagnostic success rate: `{valid_success_text}` ({valid_count_text})",
        f"Output dir: `{manifest['output_root']}`",
        f"Config: `{manifest.get('eval10_config') or ''}`",
        f"Video mode: `{manifest.get('video_mode', 'none')}`",
        f"Eval protocol: `{manifest.get('eval_protocol', 'official_one_step')}`",
        f"Chunk replan steps: `{manifest.get('chunk_replan_steps', 1)}`",
        f"Chunk action horizon: `{manifest.get('chunk_action_horizon', manifest.get('action_horizon', 1))}`",
        f"Effective chunk action horizon: `{manifest.get('effective_chunk_action_horizon', manifest.get('chunk_action_horizon', manifest.get('action_horizon', 1)))}`",
        f"Chunk predicted video: `{manifest.get('chunk_predict_video', False)}`",
        f"Max simulator steps: `{manifest.get('max_steps')}`",
        f"Minimum visual rollout frames: `{manifest.get('min_video_frames', 0)}`",
        f"Visual-valid rollouts: `{manifest.get('visual_rollout_valid_trials', 0)}/{manifest.get('trials', 0)}`",
        f"Initial-success policy: `{manifest.get('initial_success_policy')}`",
        "",
        "| trial | taskvar | demo | success | initial_success | valid | steps | frames | visual_valid | video | mp4s | actions | error | video_error |",
        "|---:|---|---:|---|---|---|---:|---:|---|---|---:|---|---|---|",
    ]
    for idx, row in enumerate(rows):
        mp4s = row.get("video_mp4_paths") or []
        lines.append(
            f"| {idx} | `{row.get('taskvar')}` | {row.get('demo_id')} | {row.get('success')} | "
            f"{row.get('initial_success')} | {row.get('valid_for_model_success')} | "
            f"{row.get('steps')} | {row.get('video_max_frame_count', '')} | {row.get('visual_rollout_valid', '')} | "
            f"`{row.get('video_path') or ''}` | {len(mp4s)} | `{row.get('actions_path') or ''}` | "
            f"{row.get('error') or ''} | {row.get('video_error') or ''} |"
        )
    lines.append("")
    (output_root / "eval10.md").write_text("\n".join(lines), encoding="utf-8")


def build_parser(*, eval10: bool) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Run official-style GEMBench closed-loop eval10 visualization."
            if eval10
            else "Run official-style GEMBench closed-loop success-rate evaluation."
        )
    )
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--output-root", default=None)
    parser.add_argument("--gembench-root", default=str(DEFAULT_GEMBENCH_ROOT))
    parser.add_argument("--assets-root", default=None)
    parser.add_argument("--robot-3dlotus-root", default=None)
    parser.add_argument("--suite", choices=["val", "test", "official_full", "custom"], default="val")
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--seeds", default="200,300,400,500,600")
    parser.add_argument("--taskvar-split", default="custom")
    parser.add_argument("--taskvar-splits", default="train,test_l2,test_l3,test_l4")
    parser.add_argument("--taskvar-file", default=None)
    parser.add_argument("--microstep-data-dir", default=None)
    parser.add_argument("--taskvars", default=None, help="Comma-separated taskvar keys or task names.")
    parser.add_argument("--limit-taskvars", type=int, default=None)
    parser.add_argument("--num-demos", type=int, default=10 if eval10 else 20)
    parser.add_argument("--num-trials", type=int, default=10)
    if eval10:
        parser.add_argument("--eval10-config", default=str(DEFAULT_EVAL10_CONFIG), help="YAML config for eval10 taskvars and demos per taskvar.")
    parser.add_argument("--max-steps", type=int, default=25)
    parser.add_argument("--max-tries", type=int, default=10)
    parser.add_argument("--image-size", type=_parse_pair, default=(256, 256))
    parser.add_argument("--cam-ids", default="0,1,2,3")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--mixed-precision", choices=["no", "fp16", "bf16"], default=None)
    parser.add_argument("--num-inference-steps", type=int, default=0)
    parser.add_argument("--rand-device", default="cpu")
    parser.add_argument("--model-seed", type=int, default=-1)
    parser.add_argument("--tiled", action="store_true")
    parser.add_argument("--relation-mode", choices=["auto", "none", "noop_smoke", "online_current"], default="auto")
    parser.add_argument(
        "--eval-protocol",
        choices=["official_one_step", "chunk_replan", "fastwam_chunk_replan", "trace_chunk_replan"],
        default="official_one_step",
        help=(
            "official_one_step preserves the GEMBench one-action-per-observation contract; "
            "chunk_replan executes the first K actions from each predicted chunk as a visual diagnostic."
        ),
    )
    parser.add_argument("--chunk-replan-steps", type=int, default=1)
    parser.add_argument(
        "--chunk-action-horizon",
        type=int,
        default=0,
        help=(
            "Diagnostic-only action horizon requested from the model in chunk_replan mode. "
            "0 means auto: official_one_step preserves the checkpoint policy horizon, while "
            "chunk_replan requests at least K actions and prefers the checkpoint train/aux horizon."
        ),
    )
    parser.add_argument("--chunk-predict-video", action="store_true", default=False)
    parser.add_argument("--no-chunk-predict-video", dest="chunk_predict_video", action="store_false")
    parser.add_argument("--record-video", action="store_true", default=True)
    parser.add_argument("--no-record-video", dest="record_video", action="store_false")
    parser.add_argument("--video-mode", choices=["observation", "official_recorder"], default="official_recorder")
    parser.add_argument("--video-fps", type=int, default=30)
    parser.add_argument("--video-stride", type=int, default=1)
    parser.add_argument("--video-resolution", type=int, default=480)
    parser.add_argument("--video-include-robot-cameras", dest="video_include_robot_cameras", action="store_true", default=True)
    parser.add_argument("--no-video-include-robot-cameras", dest="video_include_robot_cameras", action="store_false")
    parser.add_argument("--video-rotate-cam", action="store_true")
    parser.add_argument("--video-recorder-required", dest="video_recorder_required", action="store_true", default=eval10)
    parser.add_argument("--video-recorder-optional", dest="video_recorder_required", action="store_false")
    parser.add_argument("--video-initial-snap", dest="video_initial_snap", action="store_true", default=True)
    parser.add_argument("--no-video-initial-snap", dest="video_initial_snap", action="store_false")
    parser.add_argument(
        "--min-video-frames",
        type=int,
        default=60 if eval10 else 0,
        help="Minimum recorded frames for an eval10 video to count as a visual-valid long rollout.",
    )
    parser.add_argument(
        "--initial-success-policy",
        choices=["record_only", "mark_invalid", "fail"],
        default="mark_invalid" if eval10 else "record_only",
    )
    parser.add_argument("--write-official-preds", dest="write_official_preds", action="store_true", default=not eval10)
    parser.add_argument("--no-write-official-preds", dest="write_official_preds", action="store_false")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--json-output", default=None)
    return parser


def run_cli(*, eval10: bool) -> int:
    parser = build_parser(eval10=eval10)
    explicit_args = _explicit_cli_dests(parser, sys.argv[1:])
    args = parser.parse_args()
    args._explicit_args = explicit_args
    if eval10:
        _apply_eval10_config(args)
    args.eval_protocol = _normalize_eval_protocol(args.eval_protocol)
    if str(args.eval_protocol) != "chunk_replan":
        if "chunk_replan_steps" in explicit_args and int(args.chunk_replan_steps) != 1:
            raise ValueError("--chunk-replan-steps > 1 is diagnostic-only and requires --eval-protocol chunk_replan.")
        args.chunk_replan_steps = 1
    if eval10 and bool(args.write_official_preds):
        raise ValueError(
            "eval10 is a visual diagnostic and may not write official preds. "
            "Use the official success entrypoint for preds/seed*/results.jsonl."
        )
    if str(args.relation_mode) == "noop_smoke" and bool(args.write_official_preds):
        raise ValueError(
            "--relation-mode noop_smoke is an interface smoke mode and may not write official preds. "
            "Use --no-write-official-preds, or use --relation-mode online_current for TRaCE scoring."
        )
    if str(args.eval_protocol) != "official_one_step" and bool(args.write_official_preds):
        raise ValueError(
            f"--eval-protocol {args.eval_protocol} is a diagnostic protocol and may not write official preds. "
            "Use --no-write-official-preds."
        )
    if int(args.chunk_action_horizon) > 0:
        if str(args.eval_protocol) != "chunk_replan":
            raise ValueError("--chunk-action-horizon is diagnostic-only and requires --eval-protocol chunk_replan.")
        if bool(args.write_official_preds):
            raise ValueError("--chunk-action-horizon may not be used with --write-official-preds.")
        if int(args.chunk_action_horizon) < int(args.chunk_replan_steps):
            raise ValueError("--chunk-action-horizon must be >= --chunk-replan-steps.")
    if str(args.eval_protocol) == "chunk_replan" and int(args.chunk_replan_steps) < 1:
        raise ValueError("--chunk-replan-steps must be >= 1.")
    run_dir = Path(args.run_dir).expanduser().resolve()
    cfg, stats_path = _load_cfg_and_stats(run_dir)
    checkpoint = resolve_checkpoint(run_dir, args.checkpoint, dry_run=args.dry_run)
    output_root = _resolve_output_root(args, mode=("gembench_official_eval10" if eval10 else "gembench_official_success"), checkpoint=checkpoint)
    cam_ids = _parse_cam_ids(args.cam_ids)
    observation_camera_names = tuple(OFFICIAL_CAMERA_NAMES[idx] for idx in cam_ids)
    trials, skipped, jobs = _select_trials(args, eval10=eval10)
    trials, completed_skipped = _filter_completed_trials(
        trials,
        output_root=output_root,
        checkpoint=checkpoint,
        min_num_demos=int(args.num_demos),
        skip_completed=not (eval10 and bool(args.record_video)),
    )
    skipped.extend(completed_skipped)

    if args.dry_run:
        payload = _dry_run_payload(
            args,
            mode=("eval10" if eval10 else "success"),
            checkpoint=checkpoint,
            cfg=cfg,
            stats_path=stats_path,
            output_root=output_root,
            trials=trials,
            skipped=skipped,
            jobs=jobs,
        )
        json_output = Path(args.json_output) if args.json_output else output_root / "dry_run.json"
        write_json(json_output, payload)
        print(json.dumps({"status": "dry_run", "trials": len(trials), "output": str(json_output)}, indent=2))
        return 0

    if not trials:
        summary = {
            "evidence_type": "gembench_official_style_eval10_visual" if eval10 else "gembench_official_success_rate",
            "status": "already_completed",
            "provenance": git_provenance(),
            "run_dir": str(run_dir),
            "checkpoint": str(checkpoint) if checkpoint is not None else None,
            "output_root": str(output_root),
            **_official_scope_payload(args, eval10=eval10),
            "already_completed_no_new_trials": True,
            "relation_mode": str(args.relation_mode),
            "num_inference_steps": int(args.num_inference_steps),
            "model_seed": int(args.model_seed),
            "camera_names": list(observation_camera_names),
            "max_steps": int(args.max_steps),
            "num_demos": int(args.num_demos),
            "configured_num_trials": int(args.num_trials),
            "taskvars": _parse_csv(args.taskvars),
            "eval10_config": getattr(args, "eval10_config_path", None),
            "eval10_config_payload": getattr(args, "eval10_config_payload", {}),
            "record_video": bool(args.record_video),
            "video_mode": str(args.video_mode),
            "video_fps": int(args.video_fps),
            "video_stride": int(args.video_stride),
            "video_resolution": int(args.video_resolution),
            "video_include_robot_cameras": bool(args.video_include_robot_cameras),
            "video_rotate_cam": bool(args.video_rotate_cam),
            "video_recorder_required": bool(args.video_recorder_required),
            "video_initial_snap": bool(args.video_initial_snap),
            "min_video_frames": int(args.min_video_frames),
            "initial_success_policy": str(args.initial_success_policy),
            "write_official_preds": bool(args.write_official_preds),
            "eval_protocol": str(args.eval_protocol),
            "chunk_replan_steps": int(args.chunk_replan_steps),
            "chunk_action_horizon": int(args.chunk_action_horizon),
            "effective_chunk_action_horizon": int(_effective_chunk_action_horizon(args, cfg)),
            "chunk_predict_video": bool(args.chunk_predict_video),
            "trials": 0,
            "trial_split_counts": {},
            "trial_taskvar_counts": {},
            "successes": None,
            "success_rate": None,
            "skipped_taskvars": skipped,
            "note": "No new trials were run because matching official preds rows already exist for this checkpoint/output root.",
        }
        write_json(output_root / "summary.json", summary)
        if eval10:
            write_json(output_root / "eval10_manifest.json", summary)
            _write_eval10_markdown(output_root, summary)
        print(json.dumps({"status": "already_completed", "trials": 0, "output_root": str(output_root)}, indent=2))
        return 0

    if checkpoint is None:
        raise FileNotFoundError("A checkpoint is required when not running --dry-run.")
    if str(args.device).lower() == "auto":
        import torch

        args.device = "cuda:0" if torch.cuda.is_available() else "cpu"
    from .policy import GEMBenchOfficialActioner
    from .runner import GEMBenchOfficialRunner

    actioner = GEMBenchOfficialActioner.from_run_dir(
        run_dir=run_dir,
        checkpoint=checkpoint,
        device=str(args.device),
        mixed_precision=args.mixed_precision,
        num_inference_steps=int(args.num_inference_steps),
        relation_mode=str(args.relation_mode),
        observation_camera_names=observation_camera_names,
        rand_device=str(args.rand_device),
        model_seed=int(args.model_seed),
        tiled=bool(args.tiled),
        chunk_action_horizon=int(args.chunk_action_horizon) if int(args.chunk_action_horizon) > 0 else None,
        min_chunk_action_horizon=(
            int(args.chunk_replan_steps) if str(args.eval_protocol) == "chunk_replan" else None
        ),
    )
    runner = GEMBenchOfficialRunner(
        actioner=actioner,
        output_root=output_root,
        robot_3dlotus_root=args.robot_3dlotus_root,
        image_size=tuple(args.image_size),
        cam_ids=cam_ids,
        max_steps=int(args.max_steps),
        max_tries=int(args.max_tries),
        record_video=bool(args.record_video),
        video_fps=int(args.video_fps),
        video_stride=int(args.video_stride),
        video_mode=str(args.video_mode),
        video_resolution=int(args.video_resolution),
        video_include_robot_cameras=bool(args.video_include_robot_cameras),
        video_rotate_cam=bool(args.video_rotate_cam),
        video_recorder_required=bool(args.video_recorder_required),
        video_initial_snap=bool(args.video_initial_snap),
        min_video_frames=int(args.min_video_frames),
        initial_success_policy=str(args.initial_success_policy),
        write_official_preds=bool(args.write_official_preds),
        eval_protocol=str(args.eval_protocol),
        chunk_replan_steps=int(args.chunk_replan_steps),
        chunk_predict_video=bool(args.chunk_predict_video),
    )
    summary = runner.run(trials=trials, skipped_taskvars=skipped)
    summary["effective_chunk_action_horizon"] = int(getattr(actioner, "chunk_action_horizon", actioner.action_horizon))
    scope_payload = _official_scope_payload(args, eval10=eval10)
    summary.update(scope_payload)
    summary.update(
        _official_completion_payload(
            summary,
            scope_payload,
            requested_num_demos=int(args.num_demos),
        )
    )
    write_json(output_root / "summary.json", summary)
    if eval10:
        trial_src = output_root / "trials" / "results.jsonl"
        if trial_src.exists():
            (output_root / "eval10_trials.jsonl").write_text(trial_src.read_text(encoding="utf-8"), encoding="utf-8")
        manifest = {
            **summary,
            "evidence_type": "gembench_official_style_eval10_visual",
            "official_full_score": False,
            "configured_num_trials": int(args.num_trials),
            "taskvars": _parse_csv(args.taskvars),
            "eval10_config": getattr(args, "eval10_config_path", None),
            "eval10_config_payload": getattr(args, "eval10_config_payload", {}),
            **_trial_counts(trials),
            "note": "Configured closed-loop simulator visual smoke with max_steps=25 by default, not full official 20 demos/taskvar score.",
        }
        write_json(output_root / "summary.json", manifest)
        write_json(output_root / "eval10_manifest.json", manifest)
        _write_eval10_markdown(output_root, manifest)
    print(json.dumps({"status": "completed", "trials": summary["trials"], "success_rate": summary["success_rate"], "output_root": str(output_root)}, indent=2))
    return 0
