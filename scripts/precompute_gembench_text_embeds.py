import argparse
import hashlib
import os
import re
from pathlib import Path

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
import sys
if str(PROJECT_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT / "src"))

from fastwam.datasets.gembench.instructions import iter_instructions, load_instruction_map, resolve_taskvars
from fastwam.datasets.gembench.lmdb_reader import LMDBEpisodeStore
from fastwam.datasets.lerobot.robot_video_dataset import DEFAULT_PROMPT
from fastwam.models.wan22.helpers.loader import _load_registered_model, _resolve_configs
from fastwam.models.wan22.wan_video_text_encoder import HuggingfaceTokenizer


def _model_id_to_enc_id(model_id: str) -> str:
    base = str(model_id).split("/")[-1]
    enc_id = re.sub(r"[^a-z0-9]+", "", base.lower())
    return enc_id or "textenc"


def _cache_path(cache_dir: Path, prompt: str, context_len: int, enc_id: str) -> Path:
    hashed = hashlib.sha256(prompt.encode("utf-8")).hexdigest()
    return cache_dir / f"{hashed}.t5_len{context_len}.{enc_id}.pt"


def _discover_taskvars(args) -> list[str]:
    requested = resolve_taskvars(args.taskvars)
    if requested is not None:
        return requested
    data_dir = Path(args.root).expanduser() / f"{args.split}_dataset" / args.subset / args.seed
    return LMDBEpisodeStore(data_dir).list_taskvars()


def main():
    parser = argparse.ArgumentParser(description="Precompute Wan/T5 text embeddings for GEMBench task instructions.")
    parser.add_argument("--root", default=os.environ.get("GEMBENCH_ROOT", "/mnt/yuhan/datasets/GEMBench"))
    parser.add_argument("--split", default="train")
    parser.add_argument("--subset", default="keysteps_bbox")
    parser.add_argument("--seed", default="seed0")
    parser.add_argument("--taskvars", default=None, help="Comma-separated taskvars or 'official_train'. Defaults to complete local taskvars.")
    parser.add_argument("--instruction-json-path", default=None)
    parser.add_argument("--cache-dir", default="data/text_embeds_cache/gembench_keysteps_bbox")
    parser.add_argument("--context-len", type=int, default=128)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--model-id", default="Wan-AI/Wan2.2-TI2V-5B")
    parser.add_argument("--tokenizer-model-id", default="Wan-AI/Wan2.1-T2V-1.3B")
    parser.add_argument("--redirect-common-files", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    cache_dir = Path(args.cache_dir).expanduser()
    cache_dir.mkdir(parents=True, exist_ok=True)
    enc_id = _model_id_to_enc_id(args.model_id)

    taskvars = _discover_taskvars(args)
    instruction_map = load_instruction_map(args.instruction_json_path)
    prompts = [DEFAULT_PROMPT.format(task=instruction) for instruction in iter_instructions(taskvars, instruction_map)]
    prompts = list(dict.fromkeys(prompts))
    pending = [p for p in prompts if args.overwrite or not _cache_path(cache_dir, p, args.context_len, enc_id).exists()]

    print(f"[gembench-text] taskvars={len(taskvars)} prompts={len(prompts)} pending={len(pending)} cache_dir={cache_dir}")
    if not pending:
        return

    torch_dtype = torch.bfloat16 if str(args.device).startswith("cuda") else torch.float32
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
        device=args.device,
    ).eval()
    tokenizer = HuggingfaceTokenizer(name=tokenizer_config.path, seq_len=args.context_len, clean="whitespace")

    for start in range(0, len(pending), args.batch_size):
        batch_prompts = pending[start : start + args.batch_size]
        with torch.no_grad():
            ids, mask = tokenizer(batch_prompts, return_mask=True, add_special_tokens=True)
            ids = ids.to(args.device)
            mask = mask.to(args.device, dtype=torch.bool)
            context = text_encoder(ids, mask)
            seq_lens = mask.gt(0).sum(dim=1).long()
            for i, seq_len in enumerate(seq_lens):
                context[i, seq_len:] = 0
        for prompt, ctx, m in zip(batch_prompts, context, mask):
            path = _cache_path(cache_dir, prompt, args.context_len, enc_id)
            tmp = path.with_suffix(path.suffix + ".tmp")
            torch.save({"context": ctx.detach().cpu(), "mask": m.detach().cpu()}, tmp)
            os.replace(tmp, path)
        print(f"[gembench-text] encoded {min(start + args.batch_size, len(pending))}/{len(pending)}")


if __name__ == "__main__":
    main()
