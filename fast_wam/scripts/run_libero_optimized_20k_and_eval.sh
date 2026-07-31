#!/bin/bash
# Resume-safe optimized Fast-WAM LIBERO 20k training followed by the 2k gate.
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "$0")/../.." && pwd)"
SAVE_DIR="${SAVE_DIR:-$ROOT_DIR/outputs/fast_wam_libero_training_20k_optimized_20260725}"
EVAL_OUTPUT_DIR="${EVAL_OUTPUT_DIR:-$ROOT_DIR/outputs/fast_wam_libero_training_20k_optimized_20260725_eval_2k}"
LATENT_CACHE="${LATENT_CACHE:-/mnt/world_foundational_model/ruibin/data/Fast-WAM/cache/libero_mujoco3.3.2_wan22_bf16}"
TRAIN_ITERS="${TRAIN_ITERS:-20000}"
SAVE_INTERVAL="${SAVE_INTERVAL:-2000}"
PPU_METRICS_INTERVAL="${PPU_METRICS_INTERVAL:-30}"
PPU_SMI="${PPU_SMI:-/usr/local/PPU_SDK/ppu-smi/bin/ppu-smi}"

mkdir -p "$SAVE_DIR"

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
EVAL_OUTPUT_DIR="$EVAL_OUTPUT_DIR" \
TRAIN_ITERS="$TRAIN_ITERS" \
TRAIN_SCRIPT="$ROOT_DIR/fast_wam/scripts/train_libero_optimized.sh" \
LATENT_CACHE="$LATENT_CACHE" \
SAVE_INTERVAL="$SAVE_INTERVAL" \
LOG_FILE="${LOG_FILE:-$SAVE_DIR/train.log}" \
bash "$ROOT_DIR/fast_wam/scripts/run_libero_official_20k_and_eval.sh"
