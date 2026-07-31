#!/usr/bin/env python3
"""Distributed rollout-only LIBERO evaluation for Megatron Fast-WAM."""

from __future__ import annotations

import argparse
import json
import os
import time
from collections import defaultdict
from pathlib import Path

import torch
import torch.distributed as dist

from fast_wam.checkpoint import load_lerobot_checkpoint, load_megatron_dcp
from fast_wam.components import WanFrozenComponents
from fast_wam.config import FastWAMConfig
from fast_wam.distributed import initialize, transformer_config
from fast_wam.libero import rollout
from fast_wam.model import FastWAMModel
from fast_wam.policy import FastWAMPolicy, MinMaxStats


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--assets", required=True)
    parser.add_argument("--tokenizer", required=True)
    parser.add_argument(
        "--manifest",
        default=str(Path(__file__).with_name("manifest_libero_spatial_5trials.json")),
    )
    parser.add_argument("--output", required=True)
    parser.add_argument("--tp", type=int, default=1)
    parser.add_argument("--dtype", choices=("float32", "bfloat16"), default="bfloat16")
    parser.add_argument("--dcp")
    parser.add_argument("--n-action-steps", type=int, default=10)
    parser.add_argument("--target-success-rate", type=float, default=0.94)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--result-wait-timeout", type=float, default=7200.0)
    return parser.parse_args()


def _expand_cases(manifest):
    if "cases" in manifest:
        return manifest["cases"]
    cases = []
    for grid in manifest["grids"]:
        for task_id in grid["task_ids"]:
            for init_state_id in range(grid["init_state_start"], grid["init_state_stop"]):
                cases.append(
                    {
                        "id": f"{grid['id_prefix']}-t{task_id}-i{init_state_id}",
                        "suite": grid["suite"],
                        "task_id": task_id,
                        "init_state_id": init_state_id,
                        "seed": grid["seed"],
                        "episode_length": grid["episode_length"],
                        "num_steps_wait": grid["num_steps_wait"],
                    }
                )
    if len({case["id"] for case in cases}) != len(cases):
        raise ValueError("Manifest expands to duplicate case IDs")
    return cases


def _per_task(results):
    grouped = defaultdict(list)
    for item in results:
        grouped[(item["suite"], item["task_id"])].append(item)
    return [
        {
            "suite": suite,
            "task_id": task_id,
            "successes": sum(item["success"] for item in items),
            "episodes": len(items),
            "success_rate": sum(item["success"] for item in items) / len(items),
        }
        for (suite, task_id), items in sorted(grouped.items())
    ]


def _per_suite(results, targets):
    grouped = defaultdict(list)
    for item in results:
        grouped[item["suite"]].append(item)
    summaries = []
    for suite, items in sorted(grouped.items()):
        successes = sum(item["success"] for item in items)
        success_rate = successes / len(items)
        target = targets.get(suite)
        summaries.append(
            {
                "suite": suite,
                "successes": successes,
                "episodes": len(items),
                "success_rate": success_rate,
                "paper_target_success_rate": target,
                "meets_paper_target": target is None or success_rate >= target,
            }
        )
    return summaries


def main():
    args = parse_args()
    info = initialize(args.tp)
    device = torch.device("cuda", int(os.environ.get("LOCAL_RANK", "0")))
    dtype = torch.float32 if args.dtype == "float32" else torch.bfloat16
    cfg = FastWAMConfig.from_pretrained(args.checkpoint)
    model = FastWAMModel(cfg, transformer_config(cfg, args.tp, dtype)).to(
        device=device, dtype=dtype
    )
    if args.dcp:
        load_megatron_dcp(model, args.dcp)
    else:
        load_lerobot_checkpoint(model, args.checkpoint, strict=True)
    stats = MinMaxStats.from_pretrained(args.checkpoint)
    components = (
        WanFrozenComponents.from_pretrained(args.assets, args.tokenizer, device=device, dtype=dtype)
        if info.tp_rank == 0
        else None
    )
    policy = FastWAMPolicy(model, cfg, stats, components)
    manifest = json.loads(Path(args.manifest).read_text())
    cases = _expand_cases(manifest)
    output = Path(args.output)
    case_dir = output / "cases"
    if info.tp_rank == 0:
        case_dir.mkdir(parents=True, exist_ok=True)
    local_results = []
    for index, case in enumerate(cases):
        if index % info.dp_size != info.dp_rank:
            continue
        case_path = case_dir / f"{case['id']}.json"
        if args.resume and case_path.is_file():
            if info.tp_rank == 0:
                local_results.append(json.loads(case_path.read_text()))
            continue
        success, steps = rollout(policy, case, args.n_action_steps)
        if info.tp_rank == 0:
            result = {**case, "success": success, "steps": steps}
            local_results.append(result)
            temporary = case_path.with_suffix(".json.tmp")
            temporary.write_text(json.dumps(result, indent=2) + "\n")
            temporary.replace(case_path)
            (output / f"dp_{info.dp_rank}.json").write_text(
                json.dumps(local_results, indent=2) + "\n"
            )
            print(json.dumps(result, ensure_ascii=False), flush=True)

    passed = True
    if info.global_rank == 0:
        deadline = time.monotonic() + args.result_wait_timeout
        while True:
            missing = [
                case["id"]
                for case in cases
                if not (case_dir / f"{case['id']}.json").is_file()
            ]
            if not missing:
                break
            if time.monotonic() >= deadline:
                raise RuntimeError(
                    f"Timed out waiting for {len(missing)} case results, "
                    f"first={missing[:8]}"
                )
            time.sleep(1.0)
        results = [
            json.loads((case_dir / f"{case['id']}.json").read_text())
            for case in cases
        ]
        order = {case["id"]: index for index, case in enumerate(cases)}
        results.sort(key=lambda item: order[item["id"]])
        successes = sum(item["success"] for item in results)
        success_rate = successes / len(results)
        targets = manifest.get("paper_target_success_rates", {})
        summary = {
            "engine": "megatron",
            "source_checkpoint": str(Path(args.checkpoint).expanduser().resolve()),
            "dcp": (
                str(Path(args.dcp).expanduser().resolve())
                if args.dcp
                else None
            ),
            "manifest": str(Path(args.manifest).expanduser().resolve()),
            "tp": args.tp,
            "dp": info.dp_size,
            "precision": args.dtype,
            "protocol": manifest.get("protocol", {}),
            "successes": successes,
            "episodes": len(results),
            "success_rate": success_rate,
            "target_success_rate": args.target_success_rate,
            "meets_target": success_rate >= args.target_success_rate,
            "per_suite": _per_suite(results, targets),
            "per_task": _per_task(results),
            "results": results,
        }
        summary["passed"] = summary["meets_target"]
        passed = summary["passed"]
        summary_path = output / "summary.json"
        temporary = summary_path.with_suffix(".json.tmp")
        temporary.write_text(json.dumps(summary, indent=2) + "\n")
        temporary.replace(summary_path)
        console_summary = {key: value for key, value in summary.items() if key != "results"}
        print(json.dumps(console_summary, ensure_ascii=False), flush=True)
    dist.destroy_process_group()
    if info.global_rank == 0 and not passed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
