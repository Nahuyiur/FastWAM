#!/bin/bash
# Build/resume the official LIBERO BF16 latent cache on all local accelerators.
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$ROOT_DIR"
export PYTHONPATH="$ROOT_DIR:${PYTHONPATH:-}"

unset HTTP_PROXY HTTPS_PROXY ALL_PROXY http_proxy https_proxy all_proxy
export NO_PROXY='*'
export no_proxy='*'

DATASET_ROOT="${DATASET_ROOT:-/mnt/world_foundational_model/ruibin/data/Fast-WAM/official/libero_mujoco3.3.2}"
RELEASE_ROOT="${RELEASE_ROOT:-/mnt/world_foundational_model/ruibin/checkpoints/Fast-WAM/release}"
STATS_PATH="${STATS_PATH:-$RELEASE_ROOT/libero_uncond_2cam224_dataset_stats.json}"
VAE_CHECKPOINT="${VAE_CHECKPOINT:-/mnt/world_foundational_model/ruibin/checkpoints/Fast-WAM/base/Wan-AI/Wan2.2-TI2V-5B/Wan2.2_VAE.pth}"
TEXT_CACHE="${TEXT_CACHE:-$ROOT_DIR/outputs/fast_wam_libero_training_assets/text_embeds}"
LATENT_CACHE="${LATENT_CACHE:-/mnt/world_foundational_model/ruibin/data/Fast-WAM/cache/libero_mujoco3.3.2_wan22_bf16}"
GPUS="${GPUS:-8}"
BATCH_SIZE="${BATCH_SIZE:-16}"
SAMPLES_PER_SHARD="${SAMPLES_PER_SHARD:-1024}"
MASTER_PORT="${MASTER_PORT:-29561}"

torchrun \
    --standalone \
    --nproc_per_node "$GPUS" \
    --master_port "$MASTER_PORT" \
    -m fast_wam.train.prepare_latents \
    --dataset-root "$DATASET_ROOT" \
    --stats-path "$STATS_PATH" \
    --text-cache "$TEXT_CACHE" \
    --vae-checkpoint "$VAE_CHECKPOINT" \
    --output "$LATENT_CACHE" \
    --batch-size "$BATCH_SIZE" \
    --samples-per-shard "$SAMPLES_PER_SHARD"
