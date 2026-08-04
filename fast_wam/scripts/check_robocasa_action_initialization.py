#!/usr/bin/env python3
"""Compare synthesized ActionDiT backbone tensors with the official checkpoint."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import torch

from fast_wam.train.initialization import _SourceIndex, resize_action_backbone_tensor


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--wan-checkpoint", type=Path, required=True)
    parser.add_argument("--action-checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    payload = torch.load(
        args.action_checkpoint,
        map_location="cpu",
        weights_only=True,
        mmap=True,
    )
    official = payload.get("backbone_state_dict")
    if not isinstance(official, dict):
        raise TypeError("Official ActionDiT checkpoint has no backbone_state_dict")

    source = _SourceIndex(args.wan_checkpoint)
    mismatches: list[dict[str, object]] = []
    exact = 0
    interpolated = 0
    max_absolute_error = 0.0
    for name, expected in official.items():
        source_tensor = source.get(name)
        if tuple(source_tensor.shape) == tuple(expected.shape):
            actual = source_tensor.to(dtype=expected.dtype)
            exact += 1
        else:
            actual = resize_action_backbone_tensor(
                source_tensor.to(dtype=expected.dtype),
                tuple(expected.shape),
            )
            interpolated += 1
        actual = actual.to(dtype=expected.dtype)
        difference = (actual.float() - expected.float()).abs()
        error = float(difference.max())
        max_absolute_error = max(max_absolute_error, error)
        if not torch.equal(actual, expected):
            mismatches.append(
                {
                    "name": name,
                    "source_shape": list(source_tensor.shape),
                    "target_shape": list(expected.shape),
                    "max_absolute_error": error,
                }
            )

    passed = len(official) == 820 and not mismatches
    result = {
        "status": "PASS" if passed else "FAIL",
        "wan_checkpoint": str(args.wan_checkpoint.resolve()),
        "action_checkpoint": str(args.action_checkpoint.resolve()),
        "action_checkpoint_sha256": _sha256(args.action_checkpoint),
        "num_tensors": len(official),
        "exact_shape_tensors": exact,
        "interpolated_tensors": interpolated,
        "num_mismatches": len(mismatches),
        "max_absolute_error": max_absolute_error,
        "mismatches": mismatches[:32],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2), flush=True)
    if not passed:
        raise SystemExit("Official ActionDiT initialization parity failed")


if __name__ == "__main__":
    main()
