#!/bin/bash
# Official current-code Fast-WAM LIBERO recipe on Megatron Core.
set -euo pipefail

export CUDA_DEVICE_MAX_CONNECTIONS="${CUDA_DEVICE_MAX_CONNECTIONS:-1}"
export NCCL_SOCKET_IFNAME="${NCCL_SOCKET_IFNAME:-eth0}"
export NCCL_IB_HCA="${NCCL_IB_HCA:-mlx5}"
export NCCL_ASYNC_ERROR_HANDLING="${NCCL_ASYNC_ERROR_HANDLING:-1}"
export NCCL_SOCKET_TIMEOUT="${NCCL_SOCKET_TIMEOUT:-3600}"
export NCCL_IB_TIMEOUT="${NCCL_IB_TIMEOUT:-3600}"
export NCCL_DEBUG="${NCCL_DEBUG:-WARN}"
export GLOO_SOCKET_IFNAME="${GLOO_SOCKET_IFNAME:-eth0}"

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
LATENT_CACHE="${LATENT_CACHE:-}"
ASSET_ROOT="${ASSET_ROOT:-$ROOT_DIR/outputs/fast_wam_libero_training_assets}"
TEXT_CACHE="${TEXT_CACHE:-$ASSET_ROOT/text_embeds}"
INITIAL_DCP="${INITIAL_DCP:-$ASSET_ROOT/initial_dcp_bf16}"
SAVE_DIR="${SAVE_DIR:-$ROOT_DIR/outputs/fast_wam_libero_training_20k}"
LOAD_DIR="${LOAD_DIR:-}"
LOAD_OPTIM="${LOAD_OPTIM:-1}"
LOAD_STEP="${LOAD_STEP:-}"

GPUS_PER_NODE="${GPUS_PER_NODE:-8}"
TP_SIZE="${TP_SIZE:-1}"
MICRO_BATCH_SIZE="${MICRO_BATCH_SIZE:-16}"
GLOBAL_BATCH_SIZE="${GLOBAL_BATCH_SIZE:-128}"
TRAIN_ITERS="${TRAIN_ITERS:-20000}"
NUM_WORKERS="${NUM_WORKERS:-8}"
ATTENTION_BACKEND="${ATTENTION_BACKEND:-sdpa}"
KERNEL_MODE="${KERNEL_MODE:-reference}"
PARQUET_CACHE_SIZE="${PARQUET_CACHE_SIZE:-32}"
SAVE_INTERVAL="${SAVE_INTERVAL:-2000}"
SAVE_CHECKPOINTS="${SAVE_CHECKPOINTS:-1}"
SAVE_OPTIM="${SAVE_OPTIM:-1}"
EXIT_INTERVAL="${EXIT_INTERVAL:-}"
LOG_INTERVAL="${LOG_INTERVAL:-10}"
LOG_FILE="${LOG_FILE:-$SAVE_DIR/train.log}"
APPEND_LOG="${APPEND_LOG:-0}"
LOG_BATCH_INDICES="${LOG_BATCH_INDICES:-0}"
LOG_FULL_MODEL_DIGEST="${LOG_FULL_MODEL_DIGEST:-0}"
DEBUG_REPEAT_FORWARD="${DEBUG_REPEAT_FORWARD:-0}"
CHECK_WEIGHT_HASH_INTERVAL="${CHECK_WEIGHT_HASH_INTERVAL:-}"
DDP_BUCKET_SIZE="${DDP_BUCKET_SIZE:-200000000}"
OVERLAP_PARAM_GATHER="${OVERLAP_PARAM_GATHER:-0}"
MASTER_PORT="${MASTER_PORT:-29551}"

if [ $((GPUS_PER_NODE % TP_SIZE)) -ne 0 ]; then
    echo "GPUS_PER_NODE must be divisible by TP_SIZE" >&2
    exit 1
fi
DP_SIZE=$((GPUS_PER_NODE / TP_SIZE))
if [ $((GLOBAL_BATCH_SIZE % (MICRO_BATCH_SIZE * DP_SIZE))) -ne 0 ]; then
    echo "GLOBAL_BATCH_SIZE must be divisible by MICRO_BATCH_SIZE*DP_SIZE" >&2
    exit 1
fi
if [ ! -f "$INITIAL_DCP/fast_wam_initialization.json" ]; then
    echo "Missing initial DCP; run fast_wam/scripts/prepare_libero_training.sh first" >&2
    exit 1
fi
if [ ! -f "$STATS_PATH" ]; then
    echo "Missing official dataset stats: $STATS_PATH" >&2
    exit 1
fi

mkdir -p "$SAVE_DIR"

LOAD_ARGS=()
if [ -n "$LOAD_DIR" ]; then
    LOAD_ARGS+=(--load "$LOAD_DIR")
    APPEND_LOG=1
    if [ -n "$LOAD_STEP" ]; then
        LOAD_ARGS+=(--ckpt-step "$LOAD_STEP")
    fi
    if [ "$LOAD_OPTIM" = "0" ]; then
        LOAD_ARGS+=(--no-load-optim --no-load-rng)
    fi
fi
EXIT_ARGS=()
if [ -n "$EXIT_INTERVAL" ]; then
    EXIT_ARGS+=(--exit-interval "$EXIT_INTERVAL")
fi
DEBUG_ARGS=()
if [ "$LOG_BATCH_INDICES" = "1" ]; then
    DEBUG_ARGS+=(--fast-wam-log-batch-indices)
fi
if [ "$LOG_FULL_MODEL_DIGEST" = "1" ]; then
    DEBUG_ARGS+=(--fast-wam-log-full-model-digest)
fi
if [ "$DEBUG_REPEAT_FORWARD" = "1" ]; then
    DEBUG_ARGS+=(--fast-wam-debug-repeat-forward)
fi
if [ -n "$CHECK_WEIGHT_HASH_INTERVAL" ]; then
    DEBUG_ARGS+=(
        --check-weight-hash-across-dp-replicas-interval
        "$CHECK_WEIGHT_HASH_INTERVAL"
    )
fi
SAVE_ARGS=()
if [ "$SAVE_CHECKPOINTS" = "1" ]; then
    SAVE_ARGS+=(--save "$SAVE_DIR" --save-interval "$SAVE_INTERVAL")
    if [ "$SAVE_OPTIM" = "0" ]; then
        SAVE_ARGS+=(--no-save-optim)
    fi
fi
TEE_ARGS=()
if [ "$APPEND_LOG" = "1" ]; then
    TEE_ARGS+=(-a)
fi
LATENT_ARGS=()
if [ -n "$LATENT_CACHE" ]; then
    if [ ! -f "$LATENT_CACHE/manifest.json" ]; then
        echo "Missing complete latent cache manifest: $LATENT_CACHE/manifest.json" >&2
        exit 1
    fi
    LATENT_ARGS+=(--fast-wam-latent-cache "$LATENT_CACHE")
else
    LATENT_ARGS+=(--fast-wam-vae-checkpoint "$VAE_CHECKPOINT")
fi
PARAM_GATHER_ARGS=()
if [ "$OVERLAP_PARAM_GATHER" = "1" ]; then
    PARAM_GATHER_ARGS+=(--overlap-param-gather)
fi

torchrun \
    --standalone \
    --nproc_per_node "$GPUS_PER_NODE" \
    --master_port "$MASTER_PORT" \
    -m fast_wam.pretrain \
    --fast-wam-initial-dcp "$INITIAL_DCP" \
    --fast-wam-dataset-root "$DATASET_ROOT" \
    --fast-wam-stats-path "$STATS_PATH" \
    --fast-wam-text-cache "$TEXT_CACHE" \
    --fast-wam-attention-backend "$ATTENTION_BACKEND" \
    --fast-wam-kernel-mode "$KERNEL_MODE" \
    --fast-wam-parquet-cache-size "$PARQUET_CACHE_SIZE" \
    --num-layers 30 \
    --hidden-size 3072 \
    --ffn-hidden-size 14336 \
    --num-attention-heads 24 \
    --seq-length 326 \
    --max-position-embeddings 326 \
    --vocab-size 1 \
    --make-vocab-size-divisible-by 1 \
    --tokenizer-type NullTokenizer \
    --split 100,0,0 \
    --num-workers "$NUM_WORKERS" \
    --dataloader-type cyclic \
    --use-mcore-models \
    --seed 42 \
    --optimizer adam \
    --adam-beta1 0.9 \
    --adam-beta2 0.95 \
    --adam-eps 1e-8 \
    --lr 1e-4 \
    --min-lr 1e-6 \
    --lr-warmup-init 1e-7 \
    --lr-warmup-fraction 0.05 \
    --lr-decay-style cosine \
    --lr-decay-iters "$TRAIN_ITERS" \
    --weight-decay 0.01 \
    --clip-grad 1.0 \
    --train-iters "$TRAIN_ITERS" \
    --micro-batch-size "$MICRO_BATCH_SIZE" \
    --global-batch-size "$GLOBAL_BATCH_SIZE" \
    --eval-micro-batch-size 1 \
    --eval-global-batch-size "$DP_SIZE" \
    --bf16 \
    --grad-reduce-in-bf16 \
    --use-distributed-optimizer \
    --overlap-grad-reduce \
    --ddp-bucket-size "$DDP_BUCKET_SIZE" \
    --no-create-attention-mask-in-dataloader \
    --tensor-model-parallel-size "$TP_SIZE" \
    --pipeline-model-parallel-size 1 \
    --context-parallel-size 1 \
    --attention-dropout 0.0 \
    --hidden-dropout 0.0 \
    --log-interval "$LOG_INTERVAL" \
    --ckpt-format torch_dist \
    --eval-interval 200 \
    --eval-iters 1 \
    --distributed-timeout-minutes 60 \
    "${LOAD_ARGS[@]}" \
    "${EXIT_ARGS[@]}" \
    "${DEBUG_ARGS[@]}" \
    "${LATENT_ARGS[@]}" \
    "${PARAM_GATHER_ARGS[@]}" \
    "${SAVE_ARGS[@]}" \
    2>&1 | tee "${TEE_ARGS[@]}" "$LOG_FILE"
