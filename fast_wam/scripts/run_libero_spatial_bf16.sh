#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd -- "$SCRIPT_DIR/../.." && pwd)"
cd "$PROJECT_ROOT"

CHECKPOINT="${CHECKPOINT:-/mnt/world_foundational_model/ruibin/checkpoints/Fast-WAM/lerobot/fastwam_libero_uncond_2cam224}"
ASSETS="${ASSETS:-/mnt/world_foundational_model/ruibin/checkpoints/Fast-WAM/lerobot/hf_home/hub/models--Wan-AI--Wan2.2-TI2V-5B-Diffusers/snapshots/b8fff7315c768468a5333511427288870b2e9635}"
TOKENIZER="${TOKENIZER:-$CHECKPOINT/google/umt5-xxl}"
DCP="${DCP:-$PROJECT_ROOT/outputs/fast_wam_dcp_bf16_tp2_20260723}"
OUTPUT="${OUTPUT:-$PROJECT_ROOT/outputs/fast_wam_megatron_dcp_bf16_spatial_5trials_repro}"
MANIFEST="${MANIFEST:-$PROJECT_ROOT/fast_wam/eval/manifest_libero_spatial_5trials.json}"
EVAL_DEVICES="${EVAL_DEVICES:-0,1,2,3,4,5,6,7}"
DCP_DEVICES="${DCP_DEVICES:-0,1}"
NPROC_PER_NODE="${NPROC_PER_NODE:-8}"
TP="${TP:-2}"

unset HTTP_PROXY HTTPS_PROXY ALL_PROXY http_proxy https_proxy all_proxy
export NO_PROXY='*'
export no_proxy='*'
export LIBERO_CONFIG_PATH="${LIBERO_CONFIG_PATH:-/mnt/world_foundational_model/ruibin/code/LIBERO/.libero}"
export MUJOCO_GL="${MUJOCO_GL:-osmesa}"
export PYOPENGL_PLATFORM="${PYOPENGL_PLATFORM:-osmesa}"
export NUMBA_DISABLE_JIT="${NUMBA_DISABLE_JIT:-1}"
export HF_HOME="${HF_HOME:-/mnt/world_foundational_model/ruibin/checkpoints/Fast-WAM/lerobot/hf_home}"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export CUDA_DEVICE_MAX_CONNECTIONS=1
export NCCL_DEBUG="${NCCL_DEBUG:-WARN}"
export PYTHONPATH="/mnt/world_foundational_model/ruibin/code/lerobot/.eval_deps:/mnt/world_foundational_model/ruibin/code/lerobot/src:/mnt/world_foundational_model/ruibin/code/LIBERO:$PROJECT_ROOT:/opt/ac2/lib/python3.12/site-packages:/mnt/world_foundational_model/ruibin/code/Fast-WAM/code/.eval_deps:/mnt/world_foundational_model/gkz/MUKA0/.venv/lib/python3.12/site-packages${PYTHONPATH:+:$PYTHONPATH}"

if [[ ! -f "$DCP/.metadata" ]]; then
  CUDA_VISIBLE_DEVICES="$DCP_DEVICES" torchrun \
    --standalone --nproc_per_node=2 \
    -m fast_wam.eval.convert_to_dcp \
    --checkpoint "$CHECKPOINT" \
    --output "$DCP" \
    --tp 2 \
    --dtype bfloat16
fi

CUDA_VISIBLE_DEVICES="$EVAL_DEVICES" torchrun \
  --standalone --nproc_per_node="$NPROC_PER_NODE" \
  -m fast_wam.eval.evaluate_libero \
  --checkpoint "$CHECKPOINT" \
  --dcp "$DCP" \
  --assets "$ASSETS" \
  --tokenizer "$TOKENIZER" \
  --manifest "$MANIFEST" \
  --output "$OUTPUT" \
  --tp "$TP" \
  --dtype bfloat16 \
  --n-action-steps 10 \
  --target-success-rate 0.94
