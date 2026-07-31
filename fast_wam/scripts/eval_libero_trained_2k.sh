#!/bin/bash
# Evaluate a trained Megatron checkpoint with the existing fixed 2k protocol.
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$ROOT_DIR"

TRAIN_DCP="${TRAIN_DCP:-$ROOT_DIR/outputs/fast_wam_libero_training_20k}"
OUTPUT_DIR="${OUTPUT_DIR:-$ROOT_DIR/outputs/fast_wam_libero_training_20k_eval_2k}"
DCP="$TRAIN_DCP" \
OUTPUT="$OUTPUT_DIR" \
bash fast_wam/scripts/run_libero_full_2k_bf16.sh
