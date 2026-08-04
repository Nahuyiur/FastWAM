#!/usr/bin/env python3
"""Verify every tensor loaded from the formal RoboCasa initial DCP."""

from __future__ import annotations

import argparse
import gc
import json
from dataclasses import replace
from pathlib import Path

import torch
import torch.distributed as dist

from fast_wam.checkpoint import load_megatron_dcp
from fast_wam.config import FastWAMConfig
from fast_wam.distributed import initialize, transformer_config
from fast_wam.model import FastWAMModel
from fast_wam.train.initialization import ActionExpert, _SourceIndex, _source_name


def _compare(actual: torch.Tensor, expected: torch.Tensor) -> float:
    return float((actual.detach().cpu().float() - expected.cpu().float()).abs().max())


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--initial-dcp", type=Path, required=True)
    parser.add_argument("--wan-checkpoint", type=Path, required=True)
    parser.add_argument("--action-checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    parallel = initialize(1)
    try:
        device = torch.device("cuda", parallel.global_rank)
        base = FastWAMConfig()
        config = replace(
            base,
            action=replace(base.action, action_dim=12),
            proprio_dim=16,
            joint_action_video_attention=True,
        )
        model = FastWAMModel(
            config,
            transformer_config(config, 1, torch.bfloat16),
        ).to(device=device, dtype=torch.bfloat16)
        load_megatron_dcp(model, args.initial_dcp)

        mismatches: list[dict[str, object]] = []
        counts = {"video": 0, "action_backbone": 0, "random_io": 0}
        source = _SourceIndex(args.wan_checkpoint)
        for name, parameter in model.video_expert.named_parameters():
            expected = source.get(_source_name(name)).to(dtype=parameter.dtype)
            error = _compare(parameter, expected)
            counts["video"] += 1
            if error != 0.0:
                mismatches.append({"group": "video", "name": name, "error": error})

        official_payload = torch.load(
            args.action_checkpoint,
            map_location="cpu",
            weights_only=True,
            mmap=True,
        )
        official = official_payload["backbone_state_dict"]
        for name, parameter in model.action_expert.named_parameters():
            if name.startswith(("action_encoder.", "head.")):
                continue
            expected = official[_source_name(name)].to(dtype=parameter.dtype)
            error = _compare(parameter, expected)
            counts["action_backbone"] += 1
            if error != 0.0:
                mismatches.append(
                    {"group": "action_backbone", "name": name, "error": error}
                )

        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(42)
            reference_action = ActionExpert(config.action, megatron_config=None)
            reference_proprio = torch.nn.Linear(config.proprio_dim, config.video.text_dim)
        reference_action_parameters = dict(reference_action.named_parameters())
        for name, parameter in model.action_expert.named_parameters():
            if not name.startswith(("action_encoder.", "head.")):
                continue
            expected = reference_action_parameters[name].to(dtype=parameter.dtype)
            error = _compare(parameter, expected)
            counts["random_io"] += 1
            if error != 0.0:
                mismatches.append({"group": "random_io", "name": name, "error": error})
        reference_proprio_parameters = dict(reference_proprio.named_parameters())
        for name, parameter in model.proprio_encoder.named_parameters():
            reference_name = name.removeprefix("linear.")
            expected = reference_proprio_parameters[reference_name].to(dtype=parameter.dtype)
            error = _compare(parameter, expected)
            counts["random_io"] += 1
            if error != 0.0:
                mismatches.append(
                    {"group": "random_io", "name": f"proprio_encoder.{name}", "error": error}
                )

        passed = counts == {"video": 825, "action_backbone": 820, "random_io": 6} and not mismatches
        payload = {
            "status": "PASS" if passed else "FAIL",
            "initial_dcp": str(args.initial_dcp.resolve()),
            "dtype": "bfloat16",
            "counts": counts,
            "num_mismatches": len(mismatches),
            "max_absolute_error": max(
                (float(item["error"]) for item in mismatches),
                default=0.0,
            ),
            "mismatches": mismatches[:32],
        }
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(payload, indent=2) + "\n")
        print(json.dumps(payload, indent=2), flush=True)
        del reference_action, reference_proprio, model
        gc.collect()
        torch.cuda.empty_cache()
        if not passed:
            raise SystemExit("Formal RoboCasa DCP initialization parity failed")
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()


if __name__ == "__main__":
    main()
