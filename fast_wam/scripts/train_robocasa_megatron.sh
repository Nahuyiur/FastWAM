#!/bin/bash
set -euo pipefail

export CUDA_DEVICE_MAX_CONNECTIONS="${CUDA_DEVICE_MAX_CONNECTIONS:-1}"
export NCCL_ASYNC_ERROR_HANDLING="${NCCL_ASYNC_ERROR_HANDLING:-1}"
export NCCL_DEBUG="${NCCL_DEBUG:-WARN}"

ROOT_DIR="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$ROOT_DIR"
export PYTHONPATH="$ROOT_DIR:$ROOT_DIR/src:${PYTHONPATH:-}"
export GIT_PYTHON_REFRESH=quiet

DATASET_ROOT="${DATASET_ROOT:-/mnt/pub_dataset/RoboCasa365/repos}"
STATS_PATH="${STATS_PATH:-/mnt/yuhan/experiments/robocasa_acg_v1/fastwam/cache/norm_stats/robocasa_acg_v1_train_id_dataset_stats.json}"
TEXT_CACHE="${TEXT_CACHE:-/mnt/yuhan/experiments/robocasa_acg_v1/fastwam/cache/text_embeds/robocasa_acg_v1}"
VAE_CHECKPOINT="${VAE_CHECKPOINT:-/mnt/yuhan/FastWAM/checkpoints/Wan-AI/Wan2.2-TI2V-5B/Wan2.2_VAE.pth}"
INITIAL_DCP="${INITIAL_DCP:-$ROOT_DIR/outputs/robocasa_megatron_assets/initial_dcp_bf16}"
TRAIN_LATENT_CACHE="${TRAIN_LATENT_CACHE:-}"
VALID_LATENT_CACHE="${VALID_LATENT_CACHE:-}"
TRAIN_WEBDATASET="${TRAIN_WEBDATASET:-}"
VALID_WEBDATASET="${VALID_WEBDATASET:-}"
TRAIN_INDEX_FILE="${TRAIN_INDEX_FILE:-}"
VALID_INDEX_FILE="${VALID_INDEX_FILE:-}"
TASK_CONFIG="${TASK_CONFIG:-robocasa_acg_v1_fastwam_8gpu}"
SAVE_DIR="${SAVE_DIR:-$ROOT_DIR/outputs/robocasa_megatron_training}"
LOAD_DIR="${LOAD_DIR:-}"

NNODES="${NNODES:-1}"
NODE_RANK="${NODE_RANK:-0}"
GPUS_PER_NODE="${GPUS_PER_NODE:-4}"
TP_SIZE="${TP_SIZE:-1}"
MICRO_BATCH_SIZE="${MICRO_BATCH_SIZE:-1}"
GLOBAL_BATCH_SIZE="${GLOBAL_BATCH_SIZE:-32}"
TRAIN_ITERS="${TRAIN_ITERS:-50000}"
NUM_WORKERS="${NUM_WORKERS:-2}"
ATTENTION_BACKEND="${ATTENTION_BACKEND:-structured_sdpa}"
KERNEL_MODE="${KERNEL_MODE:-optimized}"
LEARNING_RATE="${LEARNING_RATE:-5e-5}"
MIN_LR="${MIN_LR:-5e-7}"
LR_WARMUP_INIT="${LR_WARMUP_INIT:-2e-8}"
LR_WARMUP_FRACTION="${LR_WARMUP_FRACTION:-0.05}"
WEIGHT_DECAY="${WEIGHT_DECAY:-0.01}"
SAVE_INTERVAL="${SAVE_INTERVAL:-5000}"
SAVE_CHECKPOINTS="${SAVE_CHECKPOINTS:-1}"
EVAL_INTERVAL="${EVAL_INTERVAL:-0}"
EVAL_ITERS="${EVAL_ITERS:-1}"
LOG_INTERVAL="${LOG_INTERVAL:-20}"
WANDB_ENABLED="${WANDB_ENABLED:-0}"
WANDB_PROJECT="${WANDB_PROJECT:-robocasa-acg-fastwam}"
WANDB_EXP_NAME="${WANDB_EXP_NAME:-$(basename "$SAVE_DIR")}"
WANDB_ENTITY="${WANDB_ENTITY:-}"
WANDB_SAVE_DIR="${WANDB_SAVE_DIR:-$SAVE_DIR/wandb}"
WANDB_RUN_ID="${WANDB_RUN_ID:-$WANDB_EXP_NAME}"
WANDB_RESUME="${WANDB_RESUME:-allow}"
WANDB_MODE="${WANDB_MODE:-online}"
EXIT_INTERVAL="${EXIT_INTERVAL:-}"
OVERLAP_PARAM_GATHER="${OVERLAP_PARAM_GATHER:-1}"
MASTER_ADDR="${MASTER_ADDR:-127.0.0.1}"
MASTER_PORT="${MASTER_PORT:-29603}"
LOG_FILE="${LOG_FILE:-$SAVE_DIR/train.log}"
PYTHON_BIN="${PYTHON_BIN:-python}"
PYTHON_BIN="$(command -v "$PYTHON_BIN")"
export PATH="$(dirname "$PYTHON_BIN"):$PATH"

WORLD_SIZE=$((NNODES * GPUS_PER_NODE))
if [ $((WORLD_SIZE % TP_SIZE)) -ne 0 ]; then
  echo "WORLD_SIZE must be divisible by TP_SIZE" >&2
  exit 1
fi
DP_SIZE=$((WORLD_SIZE / TP_SIZE))
if [ $((GLOBAL_BATCH_SIZE % (MICRO_BATCH_SIZE * DP_SIZE))) -ne 0 ]; then
  echo "GLOBAL_BATCH_SIZE must be divisible by MICRO_BATCH_SIZE*DP_SIZE" >&2
  exit 1
fi
if [ ! -f "$INITIAL_DCP/fast_wam_initialization.json" ]; then
  echo "Missing initial DCP: $INITIAL_DCP" >&2
  exit 1
fi
if [ ! -f "$STATS_PATH" ]; then
  echo "Missing RoboCasa normalization stats: $STATS_PATH" >&2
  exit 1
fi

mkdir -p "$SAVE_DIR"
LOAD_ARGS=()
if [ -n "$LOAD_DIR" ]; then
  LOAD_ARGS+=(--load "$LOAD_DIR")
fi
SAVE_ARGS=()
if [ "$SAVE_CHECKPOINTS" = "1" ]; then
  SAVE_ARGS+=(--save "$SAVE_DIR" --save-interval "$SAVE_INTERVAL")
fi
EXIT_ARGS=()
if [ -n "$EXIT_INTERVAL" ]; then
  EXIT_ARGS+=(--exit-interval "$EXIT_INTERVAL")
fi
EVAL_ARGS=()
if [ "$EVAL_INTERVAL" -gt 0 ]; then
  EVAL_ARGS+=(--eval-interval "$EVAL_INTERVAL" --eval-iters "$EVAL_ITERS")
fi
WANDB_ARGS=()
if [ "$WANDB_ENABLED" = "1" ]; then
  if [ -z "$WANDB_PROJECT" ] || [ -z "$WANDB_EXP_NAME" ]; then
    echo "WANDB_PROJECT and WANDB_EXP_NAME must be non-empty when WANDB_ENABLED=1" >&2
    exit 1
  fi
  mkdir -p "$WANDB_SAVE_DIR"
  export WANDB_RUN_ID WANDB_RESUME WANDB_MODE
  WANDB_ARGS+=(
    --wandb-project "$WANDB_PROJECT"
    --wandb-exp-name "$WANDB_EXP_NAME"
    --wandb-save-dir "$WANDB_SAVE_DIR"
  )
  if [ -n "$WANDB_ENTITY" ]; then
    WANDB_ARGS+=(--wandb-entity "$WANDB_ENTITY")
  fi
fi
LATENT_ARGS=(--fast-wam-vae-checkpoint "$VAE_CHECKPOINT")
if [ -n "$TRAIN_WEBDATASET" ] && { [ -n "$TRAIN_LATENT_CACHE" ] || [ -n "$TRAIN_INDEX_FILE" ]; }; then
  echo "TRAIN_WEBDATASET is mutually exclusive with TRAIN_LATENT_CACHE/TRAIN_INDEX_FILE" >&2
  exit 1
fi
if [ -n "$VALID_WEBDATASET" ] && { [ -n "$VALID_LATENT_CACHE" ] || [ -n "$VALID_INDEX_FILE" ]; }; then
  echo "VALID_WEBDATASET is mutually exclusive with VALID_LATENT_CACHE/VALID_INDEX_FILE" >&2
  exit 1
fi
if [ -n "$TRAIN_LATENT_CACHE" ]; then
  LATENT_ARGS+=(--fast-wam-robocasa-train-latent-cache "$TRAIN_LATENT_CACHE")
fi
if [ -n "$VALID_LATENT_CACHE" ]; then
  LATENT_ARGS+=(--fast-wam-robocasa-valid-latent-cache "$VALID_LATENT_CACHE")
fi
if [ -n "$TRAIN_WEBDATASET" ]; then
  LATENT_ARGS+=(--fast-wam-robocasa-train-webdataset "$TRAIN_WEBDATASET")
fi
if [ -n "$VALID_WEBDATASET" ]; then
  LATENT_ARGS+=(--fast-wam-robocasa-valid-webdataset "$VALID_WEBDATASET")
fi
if [ -n "$TRAIN_INDEX_FILE" ]; then
  LATENT_ARGS+=(--fast-wam-robocasa-train-index-file "$TRAIN_INDEX_FILE")
fi
if [ -n "$VALID_INDEX_FILE" ]; then
  LATENT_ARGS+=(--fast-wam-robocasa-valid-index-file "$VALID_INDEX_FILE")
fi
OVERLAP_ARGS=()
if [ "$OVERLAP_PARAM_GATHER" = "1" ]; then
  OVERLAP_ARGS+=(--overlap-param-gather)
fi
if [ "$NNODES" = "1" ]; then
  TORCHRUN_ARGS=(--standalone --nproc_per_node "$GPUS_PER_NODE" --master_port "$MASTER_PORT")
else
  TORCHRUN_ARGS=(
    --nnodes "$NNODES"
    --node_rank "$NODE_RANK"
    --nproc_per_node "$GPUS_PER_NODE"
    --master_addr "$MASTER_ADDR"
    --master_port "$MASTER_PORT"
  )
fi

"$PYTHON_BIN" -m torch.distributed.run "${TORCHRUN_ARGS[@]}" -m fast_wam.pretrain_robocasa \
  --fast-wam-initial-dcp "$INITIAL_DCP" \
  --fast-wam-dataset-root "$DATASET_ROOT" \
  --fast-wam-stats-path "$STATS_PATH" \
  --fast-wam-text-cache "$TEXT_CACHE" \
  --fast-wam-robocasa-repo-root "$ROOT_DIR" \
  --fast-wam-robocasa-task-config "$TASK_CONFIG" \
  --fast-wam-action-dim 12 \
  --fast-wam-proprio-dim 16 \
  --fast-wam-attention-backend "$ATTENTION_BACKEND" \
  --fast-wam-kernel-mode "$KERNEL_MODE" \
  --fast-wam-joint-action-video-attention \
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
  --lr "$LEARNING_RATE" \
  --min-lr "$MIN_LR" \
  --lr-warmup-init "$LR_WARMUP_INIT" \
  --lr-warmup-fraction "$LR_WARMUP_FRACTION" \
  --lr-decay-style cosine \
  --lr-decay-iters "$TRAIN_ITERS" \
  --weight-decay "$WEIGHT_DECAY" \
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
  --no-gradient-accumulation-fusion \
  --no-create-attention-mask-in-dataloader \
  --tensor-model-parallel-size "$TP_SIZE" \
  --pipeline-model-parallel-size 1 \
  --context-parallel-size 1 \
  --attention-dropout 0.0 \
  --hidden-dropout 0.0 \
  --log-interval "$LOG_INTERVAL" \
  --ckpt-format torch_dist \
  --distributed-timeout-minutes 60 \
  "${LOAD_ARGS[@]}" \
  "${SAVE_ARGS[@]}" \
  "${EXIT_ARGS[@]}" \
  "${EVAL_ARGS[@]}" \
  "${WANDB_ARGS[@]}" \
  "${LATENT_ARGS[@]}" \
  "${OVERLAP_ARGS[@]}" \
  2>&1 | tee "$LOG_FILE"
