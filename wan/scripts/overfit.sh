#!/bin/bash
# 1-sample Wan FlowMatch overfit.
#
# Usage:
#   python wan/scripts/prepare_overfit_sample.py --output /tmp/wan_sample.pt
#   SAMPLE_PATH=/tmp/wan_sample.pt bash wan/scripts/overfit.sh
set -eo pipefail

export CUDA_DEVICE_MAX_CONNECTIONS=${CUDA_DEVICE_MAX_CONNECTIONS:-1}
export NCCL_SOCKET_IFNAME=${NCCL_SOCKET_IFNAME:-ibp}
export NCCL_IB_HCA=${NCCL_IB_HCA:-mlx5}
export NCCL_ASYNC_ERROR_HANDLING=${NCCL_ASYNC_ERROR_HANDLING:-1}
export NCCL_SOCKET_TIMEOUT=${NCCL_SOCKET_TIMEOUT:-3600}
export NCCL_IB_TIMEOUT=${NCCL_IB_TIMEOUT:-3600}
export NCCL_DEBUG=${NCCL_DEBUG:-WARN}
export GLOO_SOCKET_IFNAME=${GLOO_SOCKET_IFNAME:-bond0}

ROOT_DIR="$(cd "$(dirname "$0")/../.." && pwd)"
export PYTHONPATH="$ROOT_DIR:${PYTHONPATH:-}"
cd "$ROOT_DIR"

WAN_PRESET="${WAN_PRESET:-tiny}"
SAVE_DIR="${SAVE_DIR:-/workspace/checkpoints/wan_overfit_${WAN_PRESET}}"
SAMPLE_PATH="${SAMPLE_PATH:-$ROOT_DIR/wan/scripts/data/overfit_sample.pt}"
LOAD_DIR="${LOAD_DIR:-}"
WAN_LOAD_OFFICIAL_CKPT="${WAN_LOAD_OFFICIAL_CKPT:-}"
TRAIN_ITERS="${TRAIN_ITERS:-1000}"
LR="${LR:-1e-4}"
MIN_LR="${MIN_LR:-1e-5}"
GPUS_PER_NODE="${GPUS_PER_NODE:-1}"
TP_SIZE="${TP_SIZE:-1}"
PP_SIZE="${PP_SIZE:-1}"
CP_SIZE="${CP_SIZE:-1}"
BATCH_SIZE="${BATCH_SIZE:-1}"
GLOBAL_BATCH_SIZE="${GLOBAL_BATCH_SIZE:-1}"
WAN_ATTENTION_BACKEND="${WAN_ATTENTION_BACKEND:-te}"
WAN_LOCAL_QKV="${WAN_LOCAL_QKV:-}"
if [ "$WAN_ATTENTION_BACKEND" = "te" ]; then
    export NVTE_FLASH_ATTN="${NVTE_FLASH_ATTN:-1}"
    export NVTE_FUSED_ATTN="${NVTE_FUSED_ATTN:-0}"
    export NVTE_UNFUSED_ATTN="${NVTE_UNFUSED_ATTN:-0}"
fi

if [ ! -f "$SAMPLE_PATH" ]; then
    echo "Sample not found: $SAMPLE_PATH"
    echo "Create one with: python wan/scripts/prepare_overfit_sample.py --output $SAMPLE_PATH"
    exit 1
fi

SP_ENABLED=0
if [ -n "${SEQUENCE_PARALLEL:-}" ]; then
    SP_ENABLED=1
fi
WAN_SHAPE_INFO="$(python -c 'import sys, torch; from wan.model.config import PRESETS; preset, sample_path, tp_s, cp_s, sp_s = sys.argv[1:6]; tp, cp, sp = int(tp_s), int(cp_s), bool(int(sp_s)); obj = torch.load(sample_path, map_location="cpu", weights_only=False); lat = obj.get("input_latents", obj.get("latents")); ctx = obj["context"]; lat = lat[0] if lat.ndim == 5 else lat; cfg = PRESETS[preset]; pt, ph, pw = cfg.patch_size; video_seq = (lat.shape[1] // pt) * (lat.shape[2] // ph) * (lat.shape[3] // pw); ctx_seq = ctx.shape[-2]; fused = obj.get("first_frame_latents") is not None and cfg.seperated_timestep; packed_seq = video_seq + ctx_seq + (video_seq * 7 if fused else 7); local_video_seq = video_seq; local_video_seq //= cp; local_video_seq //= (tp if sp else 1); local_packed_seq = local_video_seq + ctx_seq + (local_video_seq * 7 if fused else 7); print(packed_seq, video_seq, local_packed_seq, cfg.dim, cfg.num_heads, cfg.num_layers)' "$WAN_PRESET" "$SAMPLE_PATH" "$TP_SIZE" "$CP_SIZE" "$SP_ENABLED")"
read -r AUTO_PACKED_SEQ_LENGTH AUTO_VIDEO_SEQ_LENGTH AUTO_LOCAL_PACKED_SEQ_LENGTH AUTO_HIDDEN_SIZE AUTO_NUM_HEADS AUTO_NUM_LAYERS <<< "$WAN_SHAPE_INFO"
if [ "$WAN_ATTENTION_BACKEND" = "te" ] || [ "$PP_SIZE" != "1" ] || [ "$CP_SIZE" != "1" ] || [ -n "${SEQUENCE_PARALLEL:-}" ]; then
    DUMMY_HIDDEN_SIZE="${DUMMY_HIDDEN_SIZE:-$AUTO_HIDDEN_SIZE}"
    DUMMY_NUM_HEADS="${DUMMY_NUM_HEADS:-$AUTO_NUM_HEADS}"
    DUMMY_NUM_LAYERS="${DUMMY_NUM_LAYERS:-$AUTO_NUM_LAYERS}"
    if [ "$PP_SIZE" != "1" ]; then
        if [ "$CP_SIZE" != "1" ] || [ -n "${SEQUENCE_PARALLEL:-}" ]; then
            PP_SEQ_MULTIPLIER="$CP_SIZE"
            if [ -n "${SEQUENCE_PARALLEL:-}" ]; then
                PP_SEQ_MULTIPLIER=$((PP_SEQ_MULTIPLIER * TP_SIZE))
            fi
            SEQ_LENGTH="${SEQ_LENGTH:-$((AUTO_LOCAL_PACKED_SEQ_LENGTH * PP_SEQ_MULTIPLIER))}"
        else
            SEQ_LENGTH="${SEQ_LENGTH:-$AUTO_PACKED_SEQ_LENGTH}"
        fi
    else
        SEQ_LENGTH="${SEQ_LENGTH:-$AUTO_VIDEO_SEQ_LENGTH}"
    fi
else
    DUMMY_HIDDEN_SIZE="${DUMMY_HIDDEN_SIZE:-$TP_SIZE}"
    DUMMY_NUM_HEADS="${DUMMY_NUM_HEADS:-$TP_SIZE}"
    DUMMY_NUM_LAYERS="${DUMMY_NUM_LAYERS:-1}"
    SEQ_LENGTH="${SEQ_LENGTH:-16}"
fi
MAX_POSITION_EMBEDDINGS="${MAX_POSITION_EMBEDDINGS:-$SEQ_LENGTH}"

mkdir -p "$SAVE_DIR"

WAN_CKPT_ARGS=()
if [ -n "$WAN_LOAD_OFFICIAL_CKPT" ]; then
    WAN_CKPT_ARGS+=(--wan-load-official-ckpt "$WAN_LOAD_OFFICIAL_CKPT")
fi
if [ -n "${WAN_STRICT_LOAD:-}" ]; then
    WAN_CKPT_ARGS+=(--wan-strict-load)
fi

MEGATRON_LOAD_ARGS=()
if [ -n "$LOAD_DIR" ]; then
    MEGATRON_LOAD_ARGS+=(--load "$LOAD_DIR")
fi
if [ -n "${OVERRIDE_OPT_PARAM_SCHEDULER:-}" ]; then
    MEGATRON_LOAD_ARGS+=(--override-opt-param-scheduler)
fi
if [ -n "${USE_CHECKPOINT_OPT_PARAM_SCHEDULER:-}" ]; then
    MEGATRON_LOAD_ARGS+=(--use-checkpoint-opt-param-scheduler)
fi
if [ -n "${NO_LOAD_OPTIM:-}" ]; then
    MEGATRON_LOAD_ARGS+=(--no-load-optim)
fi
if [ -n "${NO_LOAD_RNG:-}" ]; then
    MEGATRON_LOAD_ARGS+=(--no-load-rng)
fi

MEGATRON_SAVE_ARGS=()
if [ -z "${NO_SAVE:-}" ]; then
    MEGATRON_SAVE_ARGS+=(--save-interval "${SAVE_INTERVAL:-100000}" --save "$SAVE_DIR")
fi

MEGATRON_OPT_ARGS=()
if [ -n "${USE_DISTRIBUTED_OPTIMIZER:-}" ]; then
    MEGATRON_OPT_ARGS+=(--use-distributed-optimizer)
fi
if [ -n "${SEQUENCE_PARALLEL:-}" ]; then
    MEGATRON_OPT_ARGS+=(--sequence-parallel)
fi

MEGATRON_RECOMPUTE_ARGS=()
if [ -n "${RECOMPUTE_GRANULARITY:-}" ]; then
    MEGATRON_RECOMPUTE_ARGS+=(--recompute-granularity "$RECOMPUTE_GRANULARITY")
fi
if [ -n "${RECOMPUTE_METHOD:-}" ]; then
    MEGATRON_RECOMPUTE_ARGS+=(--recompute-method "$RECOMPUTE_METHOD")
fi
if [ -n "${RECOMPUTE_NUM_LAYERS:-}" ]; then
    MEGATRON_RECOMPUTE_ARGS+=(--recompute-num-layers "$RECOMPUTE_NUM_LAYERS")
fi

DISTRIBUTED_ARGS=(
    --nproc_per_node "$GPUS_PER_NODE"
    --nnodes 1
    --node_rank 0
    --master_addr 127.0.0.1
    --master_port "${MASTER_PORT:-25300}"
)

set -x
torchrun "${DISTRIBUTED_ARGS[@]}" "$ROOT_DIR/wan/pretrain.py" \
    --wan-preset "$WAN_PRESET" \
    --wan-attention-backend "$WAN_ATTENTION_BACKEND" \
    ${WAN_LOCAL_QKV:+--wan-local-qkv} \
    --wan-sample-path "$SAMPLE_PATH" \
    --wan-train-timesteps "${WAN_TRAIN_TIMESTEPS:-1000}" \
    --wan-sigma-shift "${WAN_SIGMA_SHIFT:-5.0}" \
    --wan-noise-scale "${WAN_NOISE_SCALE:-1.0}" \
    --wan-min-timestep-boundary "${WAN_MIN_TIMESTEP_BOUNDARY:-0.0}" \
    --wan-max-timestep-boundary "${WAN_MAX_TIMESTEP_BOUNDARY:-1.0}" \
    ${WAN_GRADIENT_CHECKPOINTING:+--wan-gradient-checkpointing} \
    "${WAN_CKPT_ARGS[@]}" \
    \
    --num-layers "$DUMMY_NUM_LAYERS" \
    --hidden-size "$DUMMY_HIDDEN_SIZE" \
    --num-attention-heads "$DUMMY_NUM_HEADS" \
    --seq-length "$SEQ_LENGTH" \
    --max-position-embeddings "$MAX_POSITION_EMBEDDINGS" \
    --vocab-size 1 \
    --make-vocab-size-divisible-by 1 \
    --tokenizer-type NullTokenizer \
    --split 100,0,0 \
    --num-workers 0 \
    --use-mcore-models \
    --seed 1234 \
    --adam-beta1 0.9 --adam-beta2 0.999 \
    --lr-decay-style cosine \
    --train-iters "$TRAIN_ITERS" \
    --micro-batch-size "$BATCH_SIZE" --global-batch-size "$GLOBAL_BATCH_SIZE" \
    --lr "$LR" --min-lr "$MIN_LR" \
    --weight-decay 0.0 --clip-grad 1.0 \
    --bf16 \
    "${MEGATRON_OPT_ARGS[@]}" \
    "${MEGATRON_RECOMPUTE_ARGS[@]}" \
    --no-create-attention-mask-in-dataloader \
    --tensor-model-parallel-size "$TP_SIZE" \
    --pipeline-model-parallel-size "$PP_SIZE" \
    --context-parallel-size "$CP_SIZE" \
    --log-interval "${LOG_INTERVAL:-1}" \
    "${MEGATRON_SAVE_ARGS[@]}" \
    "${MEGATRON_LOAD_ARGS[@]}" \
    --eval-interval 100000 --eval-iters 0 \
    --distributed-timeout-minutes 30 \
    2>&1 | tee "$SAVE_DIR/overfit.log"
