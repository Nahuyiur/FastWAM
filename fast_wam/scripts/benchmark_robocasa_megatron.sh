#!/bin/bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "$0")/../.." && pwd)"
STAMP="$(date +%Y%m%d_%H%M%S)"
BENCH_ROOT="${BENCH_ROOT:-$ROOT_DIR/outputs/benchmark_robocasa_megatron_$STAMP}"

export SAVE_DIR="$BENCH_ROOT"
export LOG_FILE="$BENCH_ROOT/train.log"
export TRAIN_ITERS="${TRAIN_ITERS:-160}"
export LOG_INTERVAL="${LOG_INTERVAL:-1}"
export EVAL_INTERVAL="${EVAL_INTERVAL:-10000}"
export SAVE_CHECKPOINTS=0
export GPUS_PER_NODE="${GPUS_PER_NODE:-4}"
export TP_SIZE="${TP_SIZE:-1}"
export MICRO_BATCH_SIZE="${MICRO_BATCH_SIZE:-1}"
export GLOBAL_BATCH_SIZE="${GLOBAL_BATCH_SIZE:-32}"

mkdir -p "$BENCH_ROOT"
bash "$ROOT_DIR/fast_wam/scripts/train_robocasa_megatron.sh"
python "$ROOT_DIR/fast_wam/scripts/summarize_megatron_benchmark.py" \
  "$LOG_FILE" \
  --warmup-iters "${WARMUP_ITERS:-40}" \
  --output "$BENCH_ROOT/summary.json"
