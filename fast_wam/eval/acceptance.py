#!/usr/bin/env python3
"""Replay and closed-loop LIBERO acceptance for Megatron Fast-WAM."""

from __future__ import annotations

import argparse
import json
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
    parser.add_argument("--reference", required=True)
    parser.add_argument("--manifest", default=str(Path(__file__).with_name("manifest.json")))
    parser.add_argument("--output", required=True)
    parser.add_argument("--tp", type=int, default=1)
    parser.add_argument("--dtype", choices=("float32", "bfloat16"), default="float32")
    parser.add_argument("--dcp")
    parser.add_argument("--n-action-steps", type=int, default=10)
    return parser.parse_args()


def _images_from_record(record):
    return {key: value.float().div(255.0) for key, value in record["images"].items()}


def _replay(policy, archive, tolerance):
    max_abs = 0.0
    gripper_equal = True
    for record in archive["records"]:
        predicted = policy.predict_action_chunk(
            _images_from_record(record) if policy.is_tp_leader else None,
            record["state"] if policy.is_tp_leader else None,
            record["task"] if policy.is_tp_leader else None,
        )
        if policy.is_tp_leader:
            reference = record["action"].float()
            max_abs = max(max_abs, float((predicted[..., :-1] - reference[..., :-1]).abs().max()))
            gripper_equal &= torch.equal(torch.sign(predicted[..., -1]), torch.sign(reference[..., -1]))
    return max_abs, gripper_equal, max_abs <= tolerance and gripper_equal


def main():
    args = parse_args()
    info = initialize(args.tp)
    device = torch.device("cuda", int(__import__("os").environ.get("LOCAL_RANK", "0")))
    dtype = torch.float32 if args.dtype == "float32" else torch.bfloat16
    cfg = FastWAMConfig.from_pretrained(args.checkpoint)
    model = FastWAMModel(cfg, transformer_config(cfg, args.tp, dtype)).to(device=device, dtype=dtype)
    if args.dcp:
        load_megatron_dcp(model, args.dcp)
    else:
        load_lerobot_checkpoint(model, args.checkpoint, strict=True)
    stats = MinMaxStats.from_pretrained(args.checkpoint)
    components = (
        WanFrozenComponents.from_pretrained(
            args.assets, args.tokenizer, device=device, dtype=dtype
        )
        if info.tp_rank == 0
        else None
    )
    policy = FastWAMPolicy(model, cfg, stats, components)
    manifest = json.loads(Path(args.manifest).read_text())
    reference_dir = Path(args.reference)
    reference_summary = json.loads((reference_dir / "summary.json").read_text())
    tolerance = 1.0e-3
    local_results = []
    for index, case in enumerate(manifest["cases"]):
        if index % info.dp_size != info.dp_rank:
            continue
        archive = torch.load(reference_dir / f"{case['id']}.pt", map_location="cpu", weights_only=False)
        max_abs, gripper_equal, replay_ok = _replay(policy, archive, tolerance)
        success, steps = rollout(policy, case, args.n_action_steps)
        if info.tp_rank == 0:
            result = {
                **case,
                "replay_max_abs": max_abs,
                "gripper_equal": gripper_equal,
                "replay_ok": replay_ok,
                "success": success,
                "reference_success": bool(archive["success"]),
                "steps": steps,
            }
            local_results.append(result)
            print(json.dumps(result, ensure_ascii=False), flush=True)

    output = Path(args.output)
    if info.tp_rank == 0:
        output.mkdir(parents=True, exist_ok=True)
        (output / f"dp_{info.dp_rank}.json").write_text(json.dumps(local_results, indent=2) + "\n")
    dist.barrier(device_ids=[device.index])
    passed = True
    if info.global_rank == 0:
        results = []
        for rank in range(info.dp_size):
            results.extend(json.loads((output / f"dp_{rank}.json").read_text()))
        order = {case["id"]: index for index, case in enumerate(manifest["cases"])}
        results.sort(key=lambda item: order[item["id"]])
        success_vector = [item["success"] for item in results]
        reference_success_vector = reference_summary["success_vector"]
        success_count = sum(success_vector)
        reference_success_count = sum(reference_success_vector)
        summary = {
            "engine": "megatron",
            "tp": args.tp,
            "dp": info.dp_size,
            "precision": args.dtype,
            "tolerance": tolerance,
            "replay_pass": all(item["replay_ok"] for item in results),
            "success_vector": success_vector,
            "reference_success_vector": reference_success_vector,
            "exact_success_vector": success_vector == reference_success_vector,
            "outcome_mismatches": [
                item["id"]
                for item in results
                if item["success"] != item["reference_success"]
            ],
            "success_count": success_count,
            "reference_success_count": reference_success_count,
            "closed_loop_non_regression_pass": success_count >= reference_success_count,
            "results": results,
        }
        summary["passed"] = summary["replay_pass"] and summary["closed_loop_non_regression_pass"]
        passed = summary["passed"]
        (output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
        print(json.dumps(summary, ensure_ascii=False), flush=True)
    passed_tensor = torch.tensor([passed], dtype=torch.int64, device=device)
    dist.broadcast(passed_tensor, src=0)
    dist.destroy_process_group()
    if not bool(passed_tensor.item()):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
