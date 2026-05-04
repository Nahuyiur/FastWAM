#!/usr/bin/env bash
# Shared path setup for runs on baidu_ryh_4gpu. Keep downloads, caches, and temp files under /mnt/world_foundational_model/yuhan.
YUHAN_ROOT="${YUHAN_ROOT:-/mnt/world_foundational_model/yuhan}"
FASTWAM_CACHE_ROOT="${FASTWAM_CACHE_ROOT:-${YUHAN_ROOT}/cache/FastWAM}"
FASTWAM_PRETRAIN_ROOT="${FASTWAM_PRETRAIN_ROOT:-${YUHAN_ROOT}/pretrained_weights/FastWAM}"

export YUHAN_ROOT
export FASTWAM_CACHE_ROOT
export FASTWAM_PRETRAIN_ROOT
export DIFFSYNTH_MODEL_BASE_PATH="${DIFFSYNTH_MODEL_BASE_PATH:-${FASTWAM_PRETRAIN_ROOT}}"

export HF_HOME="${HF_HOME:-${FASTWAM_CACHE_ROOT}/huggingface}"
export HF_HUB_CACHE="${HF_HUB_CACHE:-${HF_HOME}/hub}"
export HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-${HF_HOME}/datasets}"
export TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-${HF_HOME}/transformers}"
export TORCH_HOME="${TORCH_HOME:-${FASTWAM_CACHE_ROOT}/torch}"
export TORCH_EXTENSIONS_DIR="${TORCH_EXTENSIONS_DIR:-${FASTWAM_CACHE_ROOT}/torch_extensions}"
export XDG_CACHE_HOME="${XDG_CACHE_HOME:-${FASTWAM_CACHE_ROOT}/xdg}"
export XDG_RUNTIME_DIR="${XDG_RUNTIME_DIR:-${FASTWAM_CACHE_ROOT}/xdg_runtime}"
export TMPDIR="${TMPDIR:-${FASTWAM_CACHE_ROOT}/tmp}"
export TMP="${TMP:-${TMPDIR}}"
export TEMP="${TEMP:-${TMPDIR}}"
export WANDB_DIR="${WANDB_DIR:-${FASTWAM_CACHE_ROOT}/wandb}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"
export PYTHONUNBUFFERED=1

mkdir -p \
  "${DIFFSYNTH_MODEL_BASE_PATH}" \
  "${HF_HOME}" \
  "${HF_HUB_CACHE}" \
  "${HF_DATASETS_CACHE}" \
  "${TRANSFORMERS_CACHE}" \
  "${TORCH_HOME}" \
  "${TORCH_EXTENSIONS_DIR}" \
  "${XDG_CACHE_HOME}" \
  "${XDG_RUNTIME_DIR}" \
  "${TMPDIR}" \
  "${WANDB_DIR}"

chmod 700 "${XDG_RUNTIME_DIR}" 2>/dev/null || true
