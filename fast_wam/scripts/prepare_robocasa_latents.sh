#!/bin/bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$ROOT_DIR"
export PYTHONPATH="$ROOT_DIR:$ROOT_DIR/src:${PYTHONPATH:-}"
export GIT_PYTHON_REFRESH=quiet

VAE_CHECKPOINT="${VAE_CHECKPOINT:-/mnt/yuhan/FastWAM/checkpoints/Wan-AI/Wan2.2-TI2V-5B/Wan2.2_VAE.pth}"
OUTPUT="${OUTPUT:-$ROOT_DIR/outputs/robocasa_megatron_assets/latents_train_id}"
TASK_CONFIG="${TASK_CONFIG:-robocasa_acg_v1_fastwam_8gpu}"
SPLIT="${SPLIT:-train}"
GPUS_PER_NODE="${GPUS_PER_NODE:-4}"
BATCH_SIZE="${BATCH_SIZE:-2}"
NUM_WORKERS="${NUM_WORKERS:-4}"
SAMPLES_PER_SHARD="${SAMPLES_PER_SHARD:-1024}"
MAX_SAMPLES="${MAX_SAMPLES:-}"
MASTER_PORT="${MASTER_PORT:-29602}"
PYTHON_BIN="${PYTHON_BIN:-python}"
PYTHON_BIN="$(command -v "$PYTHON_BIN")"
export PATH="$(dirname "$PYTHON_BIN"):$PATH"

MAX_ARGS=()
if [ -n "$MAX_SAMPLES" ]; then
  MAX_ARGS+=(--max-samples "$MAX_SAMPLES")
fi

"$PYTHON_BIN" -m torch.distributed.run --standalone --nproc_per_node "$GPUS_PER_NODE" --master_port "$MASTER_PORT" \
  -m fast_wam.train.prepare_robocasa_latents \
  --repo-root "$ROOT_DIR" \
  --task-config "$TASK_CONFIG" \
  --split "$SPLIT" \
  --vae-checkpoint "$VAE_CHECKPOINT" \
  --output "$OUTPUT" \
  --batch-size "$BATCH_SIZE" \
  --num-workers "$NUM_WORKERS" \
  --samples-per-shard "$SAMPLES_PER_SHARD" \
  "${MAX_ARGS[@]}"
