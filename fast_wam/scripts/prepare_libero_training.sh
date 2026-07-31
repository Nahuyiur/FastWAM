#!/bin/bash
# Prepare cached text embeddings and the deterministic initial Fast-WAM DCP.
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$ROOT_DIR"
export PYTHONPATH="$ROOT_DIR:${PYTHONPATH:-}"

unset HTTP_PROXY HTTPS_PROXY ALL_PROXY http_proxy https_proxy all_proxy
export NO_PROXY='*'
export no_proxy='*'

DATASET_ROOT="${DATASET_ROOT:-/mnt/world_foundational_model/ruibin/data/Fast-WAM/official/libero_mujoco3.3.2}"
FASTWAM_BASE="${FASTWAM_BASE:-/mnt/world_foundational_model/ruibin/checkpoints/Fast-WAM/base}"
WAN_CHECKPOINT="${WAN_CHECKPOINT:-$FASTWAM_BASE/Wan-AI/Wan2.2-TI2V-5B}"
TEXT_CHECKPOINT="${TEXT_CHECKPOINT:-$WAN_CHECKPOINT/models_t5_umt5-xxl-enc-bf16.pth}"
TOKENIZER="${TOKENIZER:-$FASTWAM_BASE/Wan-AI/Wan2.1-T2V-1.3B/google/umt5-xxl}"
ASSET_ROOT="${ASSET_ROOT:-$ROOT_DIR/outputs/fast_wam_libero_training_assets}"
TEXT_CACHE="${TEXT_CACHE:-$ASSET_ROOT/text_embeds}"
INITIAL_DCP="${INITIAL_DCP:-$ASSET_ROOT/initial_dcp_bf16}"
TEXT_GPUS="${TEXT_GPUS:-1}"
TEXT_OVERWRITE="${TEXT_OVERWRITE:-0}"
INIT_TP="${INIT_TP:-2}"
MASTER_PORT="${MASTER_PORT:-29541}"

if [ ! -d "$DATASET_ROOT" ]; then
    echo "Missing official LIBERO training release: $DATASET_ROOT" >&2
    exit 1
fi
if [ ! -f "$TEXT_CHECKPOINT" ]; then
    echo "Missing official Wan text encoder: $TEXT_CHECKPOINT" >&2
    exit 1
fi
if [ ! -f "$WAN_CHECKPOINT/diffusion_pytorch_model.safetensors.index.json" ]; then
    echo "Missing Wan2.2 VideoDiT checkpoint: $WAN_CHECKPOINT" >&2
    exit 1
fi

mkdir -p "$ASSET_ROOT"

TEXT_ARGS=()
if [ "$TEXT_OVERWRITE" = "1" ]; then
    TEXT_ARGS+=(--overwrite)
fi

torchrun \
    --standalone \
    --nproc_per_node "$TEXT_GPUS" \
    --master_port "$MASTER_PORT" \
    -m fast_wam.train.prepare_text \
    --dataset-root "$DATASET_ROOT" \
    --text-checkpoint "$TEXT_CHECKPOINT" \
    --tokenizer "$TOKENIZER" \
    --output "$TEXT_CACHE" \
    "${TEXT_ARGS[@]}"

if [ ! -f "$INITIAL_DCP/fast_wam_initialization.json" ]; then
    torchrun \
        --standalone \
        --nproc_per_node "$INIT_TP" \
        --master_port "$((MASTER_PORT + 1))" \
        -m fast_wam.train.initialization \
        --wan-checkpoint "$WAN_CHECKPOINT" \
        --output "$INITIAL_DCP" \
        --tp "$INIT_TP" \
        --seed 42 \
        --dtype bfloat16
else
    echo "Initial DCP already exists: $INITIAL_DCP"
fi

echo "Prepared Fast-WAM training assets:"
echo "  text cache: $TEXT_CACHE"
echo "  initial DCP: $INITIAL_DCP"
