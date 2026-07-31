#!/bin/bash
# Resume-safe optimized Fast-WAM official-length training followed by 2k eval.
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "$0")/../.." && pwd)"
SAVE_DIR="${SAVE_DIR:-$ROOT_DIR/outputs/fast_wam_libero_training_21700_optimized_20260725}"
EVAL_OUTPUT_DIR="${EVAL_OUTPUT_DIR:-$ROOT_DIR/outputs/fast_wam_libero_training_21700_optimized_20260725_eval_2k}"
LATENT_CACHE="${LATENT_CACHE:-/mnt/world_foundational_model/ruibin/data/Fast-WAM/cache/libero_mujoco3.3.2_wan22_bf16}"
TRAIN_ITERS="${TRAIN_ITERS:-21700}"
SAVE_INTERVAL="${SAVE_INTERVAL:-2170}"
PPU_METRICS_INTERVAL="${PPU_METRICS_INTERVAL:-30}"
PPU_SMI="${PPU_SMI:-/usr/local/PPU_SDK/ppu-smi/bin/ppu-smi}"
TRACKER="$SAVE_DIR/latest_checkpointed_iteration.txt"

mkdir -p "$SAVE_DIR"

LOAD_DIR="${LOAD_DIR:-}"
LOAD_STEP="${LOAD_STEP:-}"
if [ -f "$TRACKER" ]; then
    LATEST_STEP="$(< "$TRACKER")"
    if [[ ! "$LATEST_STEP" =~ ^[0-9]+$ ]]; then
        echo "Invalid checkpoint tracker: $TRACKER" >&2
        exit 1
    fi
    if [ "$LATEST_STEP" -gt "$TRAIN_ITERS" ]; then
        echo "Checkpoint step $LATEST_STEP exceeds TRAIN_ITERS=$TRAIN_ITERS" >&2
        exit 1
    fi
    LOAD_DIR="$SAVE_DIR"
    LOAD_STEP="$LATEST_STEP"
    echo "Resuming optimized Fast-WAM training from step $LATEST_STEP in $SAVE_DIR"
fi

monitor_pid=""
if [ "$PPU_METRICS_INTERVAL" -gt 0 ] && [ -x "$PPU_SMI" ]; then
    (
        while true; do
            "$PPU_SMI"
            sleep "$PPU_METRICS_INTERVAL"
        done
    ) >> "$SAVE_DIR/ppu_metrics.log" 2>&1 &
    monitor_pid="$!"
fi

cleanup() {
    if [ -n "$monitor_pid" ]; then
        kill "$monitor_pid" 2>/dev/null || true
        wait "$monitor_pid" 2>/dev/null || true
    fi
}
trap cleanup EXIT

SAVE_DIR="$SAVE_DIR" \
LOAD_DIR="$LOAD_DIR" \
LOAD_STEP="$LOAD_STEP" \
TRAIN_ITERS="$TRAIN_ITERS" \
SAVE_INTERVAL="$SAVE_INTERVAL" \
SAVE_OPTIM=1 \
LATENT_CACHE="$LATENT_CACHE" \
LOG_FILE="${LOG_FILE:-$SAVE_DIR/train.log}" \
bash "$ROOT_DIR/fast_wam/scripts/train_libero_optimized.sh"

TRAIN_DCP="$SAVE_DIR" \
OUTPUT_DIR="$EVAL_OUTPUT_DIR" \
bash "$ROOT_DIR/fast_wam/scripts/eval_libero_trained_2k.sh"
