#!/bin/bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$ROOT_DIR"

RUN_NAME="${RUN_NAME:-robocasa_megatron_offline50k_4gpu}"
RUN_ROOT="${RUN_ROOT:-$ROOT_DIR/outputs/$RUN_NAME}"
CACHE_ROOT="${CACHE_ROOT:-$ROOT_DIR/outputs/robocasa_megatron_assets/latents_train_id_full}"
INITIAL_DCP="${INITIAL_DCP:-/mnt/yuhan/FastWAM_megatron_robocasa/outputs/robocasa_megatron_assets/initial_dcp_bf16}"
PYTHON_BIN="${PYTHON_BIN:-/mnt/yuhan/envs/motus-rebuilt-v2_10/bin/python}"
EXPECTED_SAMPLES="${EXPECTED_SAMPLES:-286101}"
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3}"
STOP_AFTER_CACHE="${STOP_AFTER_CACHE:-0}"

mkdir -p "$RUN_ROOT" "$CACHE_ROOT"
exec 9>"$RUN_ROOT/pipeline.lock"
if ! flock -n 9; then
  echo "Another cache/train pipeline already holds $RUN_ROOT/pipeline.lock" >&2
  exit 1
fi
echo $$ >"$RUN_ROOT/pipeline.pid"

on_exit() {
  local code=$?
  printf '%s\n' "$code" >"$RUN_ROOT/pipeline.exit_code"
  date -Iseconds >"$RUN_ROOT/pipeline.finished_at"
}
trap on_exit EXIT

if [ -f "$RUN_ROOT/DONE" ]; then
  echo "Training already completed: $RUN_ROOT/DONE"
  exit 0
fi
if [ ! -x "$PYTHON_BIN" ]; then
  echo "Missing Python runtime: $PYTHON_BIN" >&2
  exit 1
fi
if [ ! -f "$INITIAL_DCP/fast_wam_initialization.json" ]; then
  echo "Missing initial DCP: $INITIAL_DCP" >&2
  exit 1
fi

export PYTHONPATH="$ROOT_DIR:$ROOT_DIR/src:${PYTHONPATH:-}"
export GIT_PYTHON_REFRESH=quiet
export CUDA_VISIBLE_DEVICES

cache_complete() {
  "$PYTHON_BIN" - "$CACHE_ROOT/manifest.json" "$EXPECTED_SAMPLES" <<'PY'
import json
import sys
from pathlib import Path

path = Path(sys.argv[1])
expected = int(sys.argv[2])
if not path.is_file():
    raise SystemExit(1)
manifest = json.loads(path.read_text())
valid = (
    manifest.get("complete") is True
    and int(manifest.get("num_samples", -1)) == expected
    and manifest.get("dtype") == "bfloat16"
    and manifest.get("sample_shape") == [48, 3, 14, 28]
    and manifest.get("split") == "train"
)
raise SystemExit(0 if valid else 1)
PY
}

date -Iseconds >"$RUN_ROOT/pipeline.started_at"
if cache_complete; then
  echo "Reusing complete BF16 mmap cache: $CACHE_ROOT"
else
  echo "Building/resuming complete BF16 mmap cache: $CACHE_ROOT"
  OUTPUT="$CACHE_ROOT" \
  SPLIT=train \
  GPUS_PER_NODE=4 \
  BATCH_SIZE=2 \
  NUM_WORKERS=4 \
  SAMPLES_PER_SHARD=1024 \
  MASTER_PORT=29612 \
  PYTHON_BIN="$PYTHON_BIN" \
  bash fast_wam/scripts/prepare_robocasa_latents.sh \
    2>&1 | tee "$RUN_ROOT/cache.log"
fi

if ! cache_complete; then
  echo "Full latent cache failed validation: $CACHE_ROOT" >&2
  exit 1
fi
date -Iseconds >"$RUN_ROOT/CACHE_DONE"
if [ "$STOP_AFTER_CACHE" = "1" ]; then
  echo "Cache validation complete; STOP_AFTER_CACHE=1"
  exit 0
fi

if pgrep -af "[f]ast_wam.pretrain_robocasa" >/dev/null; then
  echo "A FastWAM RoboCasa training process is already running; refusing duplicate launch" >&2
  exit 1
fi

date -Iseconds >"$RUN_ROOT/TRAIN_STARTED"
INITIAL_DCP="$INITIAL_DCP" \
TRAIN_LATENT_CACHE="$CACHE_ROOT" \
SAVE_DIR="$RUN_ROOT" \
LOG_FILE="$RUN_ROOT/train.log" \
PYTHON_BIN="$PYTHON_BIN" \
GPUS_PER_NODE=4 \
TP_SIZE=1 \
MICRO_BATCH_SIZE=1 \
GLOBAL_BATCH_SIZE=32 \
NUM_WORKERS=2 \
TRAIN_ITERS=50000 \
LEARNING_RATE=5e-5 \
MIN_LR=5e-7 \
LR_WARMUP_INIT=2e-8 \
LR_WARMUP_FRACTION=0.05 \
WEIGHT_DECAY=0.01 \
ATTENTION_BACKEND=structured_sdpa \
KERNEL_MODE=optimized \
SAVE_INTERVAL=5000 \
SAVE_CHECKPOINTS=1 \
EVAL_INTERVAL=0 \
LOG_INTERVAL=20 \
OVERLAP_PARAM_GATHER=1 \
MASTER_PORT=29613 \
bash fast_wam/scripts/train_robocasa_megatron.sh

date -Iseconds >"$RUN_ROOT/DONE"
