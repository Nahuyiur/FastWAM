#!/usr/bin/env python
import argparse
import csv
import hashlib
import json
import os
import re
import sys
import uuid
from pathlib import Path

import torch
from tqdm import tqdm

REPO_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_DIR / "src"))

from fastwam.datasets.lerobot.robot_video_dataset import DEFAULT_PROMPT
from fastwam.models.wan22.helpers.loader import _load_registered_model, _resolve_configs
from fastwam.models.wan22.wan_video_text_encoder import HuggingfaceTokenizer
from fastwam.utils.logging_config import setup_logging


DEFAULT_MODEL_ID = "Wan-AI/Wan2.2-TI2V-5B"
DEFAULT_TOKENIZER_MODEL_ID = "Wan-AI/Wan2.1-T2V-1.3B"


def model_id_to_enc_id(model_id: str) -> str:
    base = str(model_id).split("/")[-1]
    enc_id = re.sub(r"[^a-z0-9]+", "", base.lower())
    return enc_id or "textenc"


def atomic_torch_save(payload: dict[str, torch.Tensor], output_path: Path):
    output_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = output_path.parent / f".{output_path.name}.tmp.{uuid.uuid4().hex}"
    torch.save(payload, str(tmp_path))
    os.replace(tmp_path, output_path)


def read_prompts(manifest: Path, splits: set[str], repos: set[str]) -> list[str]:
    prompts = []
    seen = set()
    with manifest.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        required = {"split", "repo", "task_text"}
        missing = required - set(reader.fieldnames or [])
        if missing:
            raise KeyError(f"{manifest} missing required columns: {sorted(missing)}")
        for row in reader:
            if row["split"] not in splits:
                continue
            if repos and row["repo"] not in repos:
                continue
            prompt = DEFAULT_PROMPT.format(task=str(row["task_text"]))
            if prompt in seen:
                continue
            seen.add(prompt)
            prompts.append(prompt)
    if not prompts:
        raise ValueError(f"No prompts selected from {manifest} splits={sorted(splits)} repos={sorted(repos)}")
    return prompts


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", default="/mnt/pub_dataset/RoboCasa365/splits/robocasa_acg_v1_episode_manifest.csv")
    parser.add_argument("--splits", default="train_id,val_id")
    parser.add_argument("--repos", default="robocasa365-pretrain-atomic")
    parser.add_argument("--cache-dir", default="/mnt/yuhan/experiments/robocasa_acg_v1/fastwam/cache/text_embeds/robocasa_acg_v1")
    parser.add_argument("--context-len", type=int, default=128)
    parser.add_argument("--model-id", default=DEFAULT_MODEL_ID)
    parser.add_argument("--tokenizer-model-id", default=DEFAULT_TOKENIZER_MODEL_ID)
    parser.add_argument("--redirect-common-files", action="store_true", default=False)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--summary-json", default=None)
    args = parser.parse_args()

    setup_logging()
    manifest = Path(args.manifest)
    splits = {s.strip() for s in args.splits.split(",") if s.strip()}
    repos = {s.strip() for s in args.repos.split(",") if s.strip()}
    prompts = read_prompts(manifest, splits=splits, repos=repos)
    cache_dir = Path(args.cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch_dtype = torch.bfloat16
    enc_id = model_id_to_enc_id(args.model_id)

    to_encode = []
    skipped = 0
    for prompt in prompts:
        hashed = hashlib.sha256(prompt.encode("utf-8")).hexdigest()
        cache_path = cache_dir / f"{hashed}.t5_len{args.context_len}.{enc_id}.pt"
        if cache_path.exists() and not args.overwrite:
            skipped += 1
            continue
        to_encode.append(prompt)

    print(json.dumps({"prompts": len(prompts), "to_encode": len(to_encode), "skipped": skipped, "cache_dir": str(cache_dir)}, indent=2))
    if not to_encode:
        return

    _, text_config, _, tokenizer_config = _resolve_configs(
        model_id=args.model_id,
        tokenizer_model_id=args.tokenizer_model_id,
        redirect_common_files=bool(args.redirect_common_files),
    )
    text_config.download_if_necessary()
    tokenizer_config.download_if_necessary()
    text_encoder = _load_registered_model(
        text_config.path,
        "wan_video_text_encoder",
        torch_dtype=torch_dtype,
        device=device,
    ).eval()
    tokenizer = HuggingfaceTokenizer(
        name=tokenizer_config.path,
        seq_len=int(args.context_len),
        clean="whitespace",
    )

    new = 0
    overwritten = 0
    with torch.no_grad():
        for start in tqdm(range(0, len(to_encode), int(args.batch_size)), desc="Encoding RoboCasa prompts"):
            batch_prompts = to_encode[start : start + int(args.batch_size)]
            ids, mask = tokenizer(batch_prompts, return_mask=True, add_special_tokens=True)
            ids = ids.to(device)
            mask = mask.to(device=device, dtype=torch.bool)
            context = text_encoder(ids, mask)
            for i, prompt in enumerate(batch_prompts):
                hashed = hashlib.sha256(prompt.encode("utf-8")).hexdigest()
                cache_path = cache_dir / f"{hashed}.t5_len{args.context_len}.{enc_id}.pt"
                payload = {
                    "context": context[i].detach().to(device="cpu", dtype=torch.bfloat16).contiguous(),
                    "mask": mask[i].detach().to(device="cpu", dtype=torch.bool).contiguous(),
                }
                if cache_path.exists():
                    overwritten += 1
                else:
                    new += 1
                atomic_torch_save(payload, cache_path)

    summary = {
        "manifest": str(manifest),
        "splits": sorted(splits),
        "repos": sorted(repos),
        "cache_dir": str(cache_dir),
        "prompts": len(prompts),
        "new": new,
        "overwritten": overwritten,
        "skipped": skipped,
        "context_len": int(args.context_len),
        "encoder_id": enc_id,
    }
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    if args.summary_json:
        out = Path(args.summary_json)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")


if __name__ == "__main__":
    main()
