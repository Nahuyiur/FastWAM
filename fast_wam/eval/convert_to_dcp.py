#!/usr/bin/env python3
"""Convert a LeRobot Fast-WAM safetensors checkpoint to reshardable Megatron DCP."""

from __future__ import annotations

import argparse
import os

import torch
import torch.distributed as dist

from fast_wam.checkpoint import load_lerobot_checkpoint, save_megatron_dcp
from fast_wam.config import FastWAMConfig
from fast_wam.distributed import initialize, transformer_config
from fast_wam.model import FastWAMModel


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--tp", type=int, required=True)
    parser.add_argument("--dtype", choices=("float32", "bfloat16"), default="float32")
    args = parser.parse_args()
    info = initialize(args.tp)
    if info.dp_size != 1:
        raise ValueError("DCP conversion requires world_size == TP; use DP=1")
    dtype = torch.float32 if args.dtype == "float32" else torch.bfloat16
    device = torch.device("cuda", int(os.environ.get("LOCAL_RANK", "0")))
    cfg = FastWAMConfig.from_pretrained(args.checkpoint)
    model = FastWAMModel(cfg, transformer_config(cfg, args.tp, dtype)).to(device=device, dtype=dtype)
    load_lerobot_checkpoint(model, args.checkpoint, strict=True)
    save_megatron_dcp(model, args.output)
    dist.barrier(device_ids=[device.index])
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
