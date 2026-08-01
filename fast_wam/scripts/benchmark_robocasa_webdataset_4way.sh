#!/bin/bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "$0")/../.." && pwd)"
STAMP="$(date +%Y%m%d_%H%M%S)"
BENCH_ROOT="${BENCH_ROOT:-$ROOT_DIR/outputs/benchmark_robocasa_webdataset_4way_$STAMP}"
ASSET_ROOT="${ASSET_ROOT:?Set ASSET_ROOT to the prepared four-way benchmark assets}"
INDEX_FILE="${INDEX_FILE:-$ASSET_ROOT/webdataset_online/source_indices.json}"
ORDINARY_LATENT_CACHE="${ORDINARY_LATENT_CACHE:-$ASSET_ROOT/ordinary_latents}"
WDS_ONLINE="${WDS_ONLINE:-$ASSET_ROOT/webdataset_online}"
WDS_OFFLINE="${WDS_OFFLINE:-$ASSET_ROOT/webdataset_offline}"
REPEATS="${REPEATS:-3}"
TRAIN_ITERS="${TRAIN_ITERS:-160}"
WARMUP_ITERS="${WARMUP_ITERS:-40}"

for required in \
  "$INDEX_FILE" \
  "$ORDINARY_LATENT_CACHE/manifest.json" \
  "$WDS_ONLINE/manifest.json" \
  "$WDS_OFFLINE/manifest.json"; do
  if [ ! -f "$required" ]; then
    echo "Missing benchmark asset: $required" >&2
    exit 1
  fi
done

mkdir -p "$BENCH_ROOT"
run_mode() {
  local repeat="$1"
  local mode="$2"
  local run_root="$BENCH_ROOT/repeat_${repeat}/${mode}"
  mkdir -p "$run_root"
  export SAVE_DIR="$run_root"
  export LOG_FILE="$run_root/train.log"
  export TRAIN_ITERS
  export LOG_INTERVAL=1
  export EVAL_INTERVAL=10000
  export SAVE_CHECKPOINTS=0
  export GPUS_PER_NODE="${GPUS_PER_NODE:-4}"
  export TP_SIZE="${TP_SIZE:-1}"
  export MICRO_BATCH_SIZE="${MICRO_BATCH_SIZE:-1}"
  export GLOBAL_BATCH_SIZE="${GLOBAL_BATCH_SIZE:-32}"
  export NUM_WORKERS="${NUM_WORKERS:-2}"
  export MASTER_PORT="$((29700 + repeat * 10))"
  export TRAIN_INDEX_FILE=""
  export TRAIN_LATENT_CACHE=""
  export TRAIN_WEBDATASET=""
  case "$mode" in
    ordinary_online)
      export TRAIN_INDEX_FILE="$INDEX_FILE"
      ;;
    webdataset_online)
      export TRAIN_WEBDATASET="$WDS_ONLINE"
      ;;
    ordinary_offline)
      export TRAIN_INDEX_FILE="$INDEX_FILE"
      export TRAIN_LATENT_CACHE="$ORDINARY_LATENT_CACHE"
      ;;
    webdataset_offline)
      export TRAIN_WEBDATASET="$WDS_OFFLINE"
      ;;
    *)
      echo "Unknown benchmark mode: $mode" >&2
      exit 1
      ;;
  esac
  bash "$ROOT_DIR/fast_wam/scripts/train_robocasa_megatron.sh"
  python "$ROOT_DIR/fast_wam/scripts/summarize_megatron_benchmark.py" \
    "$LOG_FILE" \
    --warmup-iters "$WARMUP_ITERS" \
    --output "$run_root/summary.json"
}

forward=(ordinary_online webdataset_online ordinary_offline webdataset_offline)
reverse=(webdataset_offline ordinary_offline webdataset_online ordinary_online)
for repeat in $(seq 1 "$REPEATS"); do
  if [ $((repeat % 2)) -eq 1 ]; then
    order=("${forward[@]}")
  else
    order=("${reverse[@]}")
  fi
  for mode in "${order[@]}"; do
    run_mode "$repeat" "$mode"
  done
done

python "$ROOT_DIR/fast_wam/scripts/summarize_robocasa_webdataset_4way.py" \
  "$BENCH_ROOT" \
  --output "$BENCH_ROOT/summary.json"
