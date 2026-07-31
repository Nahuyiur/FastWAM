#!/bin/bash
# Resume-safe official Fast-WAM LIBERO 20k training followed by the 2k rollout gate.
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$ROOT_DIR"

SAVE_DIR="${SAVE_DIR:-$ROOT_DIR/outputs/fast_wam_libero_training_20k}"
EVAL_OUTPUT_DIR="${EVAL_OUTPUT_DIR:-$ROOT_DIR/outputs/fast_wam_libero_training_20k_eval_2k}"
TRAIN_ITERS="${TRAIN_ITERS:-20000}"
TRAIN_SCRIPT="${TRAIN_SCRIPT:-$ROOT_DIR/fast_wam/scripts/train_libero_official.sh}"
TRACKER="$SAVE_DIR/latest_checkpointed_iteration.txt"

if [ ! -f "$TRAIN_SCRIPT" ]; then
    echo "Missing training script: $TRAIN_SCRIPT" >&2
    exit 1
fi

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
    echo "Resuming Fast-WAM training from step $LATEST_STEP in $SAVE_DIR"
fi

SAVE_DIR="$SAVE_DIR" \
LOAD_DIR="$LOAD_DIR" \
LOAD_STEP="$LOAD_STEP" \
TRAIN_ITERS="$TRAIN_ITERS" \
SAVE_OPTIM=1 \
bash "$TRAIN_SCRIPT"

TRAIN_DCP="$SAVE_DIR" \
OUTPUT_DIR="$EVAL_OUTPUT_DIR" \
bash fast_wam/scripts/eval_libero_trained_2k.sh
