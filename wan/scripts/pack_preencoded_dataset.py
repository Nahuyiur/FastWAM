#!/usr/bin/env python3
"""Pack per-sample Wan pre-encoded tensors into training shards.

Input JSONL rows can use the same schema as `WanJsonlDataset`, for example:

    {"sample_path": "/path/sample.pt"}
    {"latents": "/path/latents.safetensors", "context": "/path/context.safetensors"}

Output is a shard manifest whose rows point to `shard_path` plus row-local
`index`. The training dataloader can slice safetensors shards without loading
the full shard into memory.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import torch

from wan.data.dataset import _load_tensor, _normalize_sample


def _read_rows(path: str):
    with open(path, "r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            yield line_no, row


def _load_sample(row: dict[str, Any]) -> dict[str, Any]:
    sample_path = row.get("sample_path")
    if isinstance(sample_path, str):
        obj = torch.load(sample_path, map_location="cpu", weights_only=False)
        if not isinstance(obj, dict):
            raise ValueError(f"sample_path must contain a dict: {sample_path}")
        sample = dict(obj)
    else:
        latents_path = row.get("latents_path", row.get("input_latents_path", row.get("latents")))
        context_path = row.get("context_path", row.get("context"))
        if not isinstance(latents_path, str) or not isinstance(context_path, str):
            raise ValueError("row must contain sample_path or latents/context paths")
        sample = {
            "input_latents": _load_tensor(latents_path, key=row.get("latents_key", "input_latents")),
            "context": _load_tensor(context_path, key=row.get("context_key", "context")),
        }
        first_frame_path = row.get("first_frame_latents_path")
        if isinstance(first_frame_path, str):
            sample["first_frame_latents"] = _load_tensor(
                first_frame_path,
                key=row.get("first_frame_latents_key", "first_frame_latents"),
            )
        if row.get("fuse_vae_embedding_in_latents", False):
            sample["fuse_vae_embedding_in_latents"] = True
    sample.setdefault("prompt", row.get("prompt", ""))
    sample.setdefault("video_path", row.get("video_path", ""))
    if row.get("fuse_vae_embedding_in_latents", False):
        sample["fuse_vae_embedding_in_latents"] = True
    return _normalize_sample(sample)


def _flush_shard(
    samples: list[dict[str, Any]],
    shard_id: int,
    output_dir: Path,
    manifest_fh,
    fmt: str,
) -> None:
    if not samples:
        return
    latents = torch.stack([s["input_latents"].to(torch.bfloat16) for s in samples], dim=0).contiguous()
    context = torch.stack([s["context"].to(torch.bfloat16) for s in samples], dim=0).contiguous()
    shard_path = output_dir / f"shard_{shard_id:06d}.{fmt}"
    tensors = {"input_latents": latents, "context": context}
    has_first_frame = all("first_frame_latents" in s for s in samples)
    has_fuse = all(bool(s.get("fuse_vae_embedding_in_latents", False)) for s in samples)
    if has_first_frame:
        tensors["first_frame_latents"] = torch.stack(
            [s["first_frame_latents"].to(torch.bfloat16) for s in samples],
            dim=0,
        ).contiguous()
    if fmt == "safetensors":
        try:
            from safetensors.torch import save_file
        except Exception as exc:
            raise RuntimeError("safetensors output requires safetensors") from exc
        save_file(tensors, str(shard_path))
    elif fmt == "pt":
        torch.save(tensors, shard_path)
    else:
        raise ValueError(f"unsupported format: {fmt}")

    for index, sample in enumerate(samples):
        manifest_fh.write(
            json.dumps(
                {
                    "shard_path": str(shard_path),
                    "index": index,
                    "latents_key": "input_latents",
                    "context_key": "context",
                    "prompt": sample.get("prompt", ""),
                    "video_path": sample.get("video_path", ""),
                    "latents_shape": list(sample["input_latents"].shape),
                    "context_shape": list(sample["context"].shape),
                    **({"first_frame_latents_key": "first_frame_latents"} if has_first_frame else {}),
                    **({"fuse_vae_embedding_in_latents": True} if has_fuse else {}),
                },
                ensure_ascii=True,
            )
            + "\n"
        )
    print(
        f"wrote {shard_path} rows={len(samples)} latents={tuple(latents.shape)} "
        f"context={tuple(context.shape)} first_frame={has_first_frame}"
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-jsonl", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--output-jsonl", required=True)
    parser.add_argument("--samples-per-shard", type=int, default=32)
    parser.add_argument("--format", choices=["safetensors", "pt"], default="safetensors")
    parser.add_argument(
        "--allow-shape-boundary",
        action="store_true",
        help="Start a new shard when sample shape changes. Use with bucketed manifests.",
    )
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_jsonl = Path(args.output_jsonl)
    output_jsonl.parent.mkdir(parents=True, exist_ok=True)

    shard_id = 0
    pending: list[dict[str, Any]] = []
    current_shape = None
    with open(output_jsonl, "w", encoding="utf-8") as out_f:
        for line_no, row in _read_rows(args.input_jsonl):
            sample = _load_sample(row)
            shape = (
                tuple(sample["input_latents"].shape),
                tuple(sample["context"].shape),
                tuple(sample["first_frame_latents"].shape) if "first_frame_latents" in sample else None,
                bool(sample.get("fuse_vae_embedding_in_latents", False)),
            )
            if current_shape is None:
                current_shape = shape
            if shape != current_shape:
                if not args.allow_shape_boundary:
                    raise ValueError(
                        f"shape changed at input line {line_no}: {shape} != {current_shape}. "
                        "Bucket by resolution/frame count first or pass --allow-shape-boundary."
                    )
                _flush_shard(pending, shard_id, output_dir, out_f, args.format)
                shard_id += 1
                pending = []
                current_shape = shape
            pending.append(sample)
            if len(pending) >= args.samples_per_shard:
                _flush_shard(pending, shard_id, output_dir, out_f, args.format)
                shard_id += 1
                pending = []
                current_shape = None
        _flush_shard(pending, shard_id, output_dir, out_f, args.format)
    print(f"wrote manifest {output_jsonl}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
