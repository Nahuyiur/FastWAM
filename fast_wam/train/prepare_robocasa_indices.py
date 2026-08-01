"""Create one deterministic RoboCasa window index shared by benchmark inputs."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import torch

from .robocasa_data import build_robocasa_datasets


def _write_json_atomic(path: Path, payload: dict) -> None:
    temporary = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, path)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", required=True)
    parser.add_argument("--task-config", default="robocasa_acg_v1_fastwam_8gpu")
    parser.add_argument("--split", choices=("train", "valid"), default="train")
    parser.add_argument("--num-samples", type=int, required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    train_dataset, valid_dataset, _ = build_robocasa_datasets(
        args.repo_root,
        args.task_config,
    )
    dataset = train_dataset if args.split == "train" else valid_dataset
    count = min(int(args.num_samples), len(dataset))
    if count <= 0:
        raise ValueError("num_samples must be positive")
    generator = torch.Generator(device="cpu").manual_seed(int(args.seed))
    source_indices = torch.randperm(len(dataset), generator=generator)[:count].tolist()
    payload = {
        "dataset_size": len(dataset),
        "selection": "random",
        "seed": int(args.seed),
        "source_indices": source_indices,
    }
    output = Path(args.output).expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.is_file():
        previous = json.loads(output.read_text(encoding="utf-8"))
        if previous != payload:
            raise ValueError(f"Refusing to replace a different index: {output}")
    else:
        _write_json_atomic(output, payload)
    print(
        f"completed RoboCasa benchmark index samples={count} "
        f"dataset_size={len(dataset)} output={output}",
        flush=True,
    )


if __name__ == "__main__":
    main()
