"""Precompute the 40 official LIBERO prompt embeddings with frozen UMT5."""

from __future__ import annotations

import argparse
import hashlib
import html
import os
import re
import uuid
from pathlib import Path

import torch
import torch.distributed as dist

from ..config import FastWAMConfig
from .data import _jsonl, official_dataset_dirs
from .text_encoder import WanT5Encoder


def _clean_whitespace(text: str) -> str:
    """The effective `clean="whitespace"` path used by Wan's tokenizer."""

    try:
        import ftfy
    except ImportError:  # All official English LIBERO prompts are already clean.
        fixed = text
    else:
        fixed = ftfy.fix_text(text)
    fixed = html.unescape(html.unescape(fixed))
    return re.sub(r"\s+", " ", fixed).strip()


def _prompts(root: str | Path, config: FastWAMConfig) -> list[str]:
    result: list[str] = []
    seen: set[str] = set()
    for dataset_dir in official_dataset_dirs(root):
        for row in _jsonl(dataset_dir / "meta" / "tasks.jsonl"):
            prompt = config.prompt_template.format(task=str(row["task"]))
            prompt = _clean_whitespace(prompt)
            if prompt not in seen:
                seen.add(prompt)
                result.append(prompt)
    if len(result) != 40:
        raise ValueError(f"Official four-suite LIBERO release must contain 40 prompts, got {len(result)}")
    return result


def _atomic_save(payload: dict[str, torch.Tensor], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp.{uuid.uuid4().hex}")
    torch.save(payload, temporary)
    os.replace(temporary, path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-root", required=True)
    parser.add_argument(
        "--text-checkpoint",
        required=True,
        help="Official models_t5_umt5-xxl-enc-bf16.pth",
    )
    parser.add_argument("--tokenizer", required=True, help="Local google/umt5-xxl directory")
    parser.add_argument("--output", required=True)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    world = int(os.environ.get("WORLD_SIZE", "1"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    if world > 1 and not dist.is_initialized():
        if torch.cuda.is_available():
            torch.cuda.set_device(local_rank)
        dist.init_process_group(backend="nccl" if torch.cuda.is_available() else "gloo")
    rank = dist.get_rank() if dist.is_initialized() else 0
    world = dist.get_world_size() if dist.is_initialized() else 1
    device = torch.device(f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu")
    dtype = torch.bfloat16
    config = FastWAMConfig()
    prompts = _prompts(args.dataset_root, config)[rank::world]
    output = Path(args.output).expanduser().resolve()
    try:
        from transformers import AutoTokenizer
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError("Fast-WAM text preparation requires transformers") from exc
    tokenizer = AutoTokenizer.from_pretrained(
        args.tokenizer,
        local_files_only=True,
    )
    encoder = WanT5Encoder.from_pretrained(
        args.text_checkpoint,
        device=device,
        dtype=dtype,
    )

    written = 0
    skipped = 0
    for start in range(0, len(prompts), args.batch_size):
        batch = prompts[start : start + args.batch_size]
        tokens = tokenizer(
            batch,
            padding="max_length",
            truncation=True,
            max_length=config.context_len,
            add_special_tokens=True,
            return_tensors="pt",
        )
        mask = tokens.attention_mask.to(device=device, dtype=torch.bool)
        context = encoder(
            tokens.input_ids.to(device),
            mask,
        )
        for item, item_context, item_mask in zip(batch, context, mask, strict=True):
            digest = hashlib.sha256(item.encode("utf-8")).hexdigest()
            path = output / f"{digest}.t5_len{config.context_len}.wan22ti2v5b.pt"
            if path.exists() and not args.overwrite:
                skipped += 1
                continue
            _atomic_save(
                {
                    "context": item_context.detach().to("cpu", dtype=torch.bfloat16).contiguous(),
                    "mask": item_mask.detach().to("cpu", dtype=torch.bool).contiguous(),
                },
                path,
            )
            written += 1
    print(f"rank={rank}: wrote={written}, skipped={skipped}, output={output}", flush=True)
    if dist.is_initialized():
        dist.barrier()
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
