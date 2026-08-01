#!/bin/bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "$0")/../.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python}"
PYTHON_BIN="$(command -v "$PYTHON_BIN")"
ASSET_ROOT="${ASSET_ROOT:-$ROOT_DIR/outputs/robocasa_webdataset_benchmark_assets}"
BENCH_SAMPLES="${BENCH_SAMPLES:-1024}"
SAMPLES_PER_SHARD="${SAMPLES_PER_SHARD:-128}"
GPUS_PER_NODE="${GPUS_PER_NODE:-4}"
TASK_CONFIG="${TASK_CONFIG:-robocasa_acg_v1_fastwam_8gpu}"
VAE_CHECKPOINT="${VAE_CHECKPOINT:-/mnt/yuhan/FastWAM/checkpoints/Wan-AI/Wan2.2-TI2V-5B/Wan2.2_VAE.pth}"

export PYTHONPATH="$ROOT_DIR:$ROOT_DIR/src:${PYTHONPATH:-}"
export GIT_PYTHON_REFRESH=quiet
mkdir -p "$ASSET_ROOT"

"$PYTHON_BIN" -m fast_wam.train.prepare_robocasa_webdataset \
  --repo-root "$ROOT_DIR" \
  --task-config "$TASK_CONFIG" \
  --split train \
  --mode online \
  --output "$ASSET_ROOT/webdataset_online" \
  --max-samples "$BENCH_SAMPLES" \
  --selection random \
  --seed 42 \
  --samples-per-shard "$SAMPLES_PER_SHARD"

INDEX_FILE="$ASSET_ROOT/webdataset_online/source_indices.json"
"$PYTHON_BIN" -m torch.distributed.run \
  --standalone \
  --nproc_per_node "$GPUS_PER_NODE" \
  -m fast_wam.train.prepare_robocasa_latents \
  --repo-root "$ROOT_DIR" \
  --task-config "$TASK_CONFIG" \
  --split train \
  --vae-checkpoint "$VAE_CHECKPOINT" \
  --output "$ASSET_ROOT/ordinary_latents" \
  --batch-size 2 \
  --num-workers 2 \
  --samples-per-shard "$SAMPLES_PER_SHARD" \
  --source-index-file "$INDEX_FILE"

"$PYTHON_BIN" -m fast_wam.train.prepare_robocasa_webdataset \
  --repo-root "$ROOT_DIR" \
  --task-config "$TASK_CONFIG" \
  --split train \
  --mode offline \
  --latent-cache "$ASSET_ROOT/ordinary_latents" \
  --output "$ASSET_ROOT/webdataset_offline" \
  --source-index-file "$INDEX_FILE" \
  --samples-per-shard "$SAMPLES_PER_SHARD"

echo "Prepared four-way benchmark assets under $ASSET_ROOT"
