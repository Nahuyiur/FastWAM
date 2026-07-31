#!/bin/bash
# Accepted TP1+DP8 performance profile; requires the completed offline cache.
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "$0")/../.." && pwd)"
LATENT_CACHE="${LATENT_CACHE:-/mnt/world_foundational_model/ruibin/data/Fast-WAM/cache/libero_mujoco3.3.2_wan22_bf16}"

if [ ! -f "$LATENT_CACHE/manifest.json" ]; then
    echo "Missing latent cache: $LATENT_CACHE/manifest.json" >&2
    echo "Run fast_wam/scripts/prepare_libero_latents.sh first." >&2
    exit 1
fi

LATENT_CACHE="$LATENT_CACHE" \
ATTENTION_BACKEND="${ATTENTION_BACKEND:-structured_sdpa}" \
KERNEL_MODE="${KERNEL_MODE:-optimized}" \
OVERLAP_PARAM_GATHER="${OVERLAP_PARAM_GATHER:-1}" \
PARQUET_CACHE_SIZE="${PARQUET_CACHE_SIZE:-32}" \
bash "$ROOT_DIR/fast_wam/scripts/train_libero_official.sh"
