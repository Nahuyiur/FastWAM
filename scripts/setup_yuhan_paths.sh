#!/usr/bin/env bash
# Shared path setup for FastWAM runs on jinshan_pub.
#
# Defaults match the current server layout:
#   FastWAM repo: /mnt/yuhan/FastWAM
#   conda root:   /mnt/miniconda3
#   checkpoints:  /mnt/yuhan/FastWAM/checkpoints
#
# Every value can still be overridden by exporting the variable before sourcing
# this file.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
FASTWAM_ROOT="${FASTWAM_ROOT:-$(cd "${SCRIPT_DIR}/.." && pwd)}"
if [[ -z "${YUHAN_ROOT:-}" ]]; then
  if [[ -d /mnt/yuhan ]]; then
    YUHAN_ROOT="/mnt/yuhan"
  else
    YUHAN_ROOT="$(cd "${FASTWAM_ROOT}/.." && pwd)"
  fi
fi

CONDA_ROOT="${CONDA_ROOT:-/mnt/miniconda3}"
FASTWAM_CONDA_ENV="${FASTWAM_CONDA_ENV:-${CONDA_ROOT}/envs/fastwam}"
GEMBENCH_ROOT="${GEMBENCH_ROOT:-${YUHAN_ROOT}/datasets/GEMBench}"
FASTWAM_CACHE_ROOT="${FASTWAM_CACHE_ROOT:-${YUHAN_ROOT}/cache/FastWAM}"
FASTWAM_PRETRAIN_ROOT="${FASTWAM_PRETRAIN_ROOT:-${FASTWAM_ROOT}/checkpoints}"
FASTWAM_RUNS_ROOT="${FASTWAM_RUNS_ROOT:-${FASTWAM_ROOT}/runs}"
RLBENCH_PICK_LIFT_ROOT="${RLBENCH_PICK_LIFT_ROOT:-${YUHAN_ROOT}/data/rlbench_pick_lift_color_shape}"
RLBENCH_LEROBOT_TRAIN_DIR="${RLBENCH_LEROBOT_TRAIN_DIR:-${RLBENCH_PICK_LIFT_ROOT}/lerobot/train}"
RLBENCH_LEROBOT_TEST_DIR="${RLBENCH_LEROBOT_TEST_DIR:-${RLBENCH_PICK_LIFT_ROOT}/lerobot/test}"
GEMBENCH_SIM_ROOT="${GEMBENCH_SIM_ROOT:-${YUHAN_ROOT}/gembench_sim}"
RLBENCH_ROOT="${RLBENCH_ROOT:-${GEMBENCH_SIM_ROOT}/RLBench}"
COPPELIASIM_ROOT="${COPPELIASIM_ROOT:-${GEMBENCH_SIM_ROOT}/CoppeliaSim_Edu_V4_1_0_Ubuntu20_04}"
RLBENCH_STUB_ROOT="${RLBENCH_STUB_ROOT:-${YUHAN_ROOT}/rlbench_lerobot_tools/stubs}"
RLBENCH_PYREP_SITE="${RLBENCH_PYREP_SITE:-${FASTWAM_CONDA_ENV}/lib/python3.10/site-packages}"

export FASTWAM_ROOT
export YUHAN_ROOT
export CONDA_ROOT
export FASTWAM_CONDA_ENV
export GEMBENCH_ROOT
export FASTWAM_CACHE_ROOT
export FASTWAM_PRETRAIN_ROOT
export FASTWAM_RUNS_ROOT
export RLBENCH_PICK_LIFT_ROOT
export RLBENCH_LEROBOT_TRAIN_DIR
export RLBENCH_LEROBOT_TEST_DIR
export GEMBENCH_SIM_ROOT
export RLBENCH_ROOT
export COPPELIASIM_ROOT
export RLBENCH_STUB_ROOT
export RLBENCH_PYREP_SITE
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

if [[ -d "${COPPELIASIM_ROOT}" ]]; then
  case ":${LD_LIBRARY_PATH:-}:" in
    *":${COPPELIASIM_ROOT}:"*) ;;
    *) export LD_LIBRARY_PATH="${LD_LIBRARY_PATH:-}:${COPPELIASIM_ROOT}" ;;
  esac
  export QT_QPA_PLATFORM_PLUGIN_PATH="${QT_QPA_PLATFORM_PLUGIN_PATH:-${COPPELIASIM_ROOT}}"
fi

case ":${PYTHONPATH:-}:" in
  *":${FASTWAM_ROOT}/src:"*) ;;
  *) export PYTHONPATH="${FASTWAM_ROOT}/src:${PYTHONPATH:-}" ;;
esac
if [[ -d "${RLBENCH_ROOT}" ]]; then
  case ":${PYTHONPATH:-}:" in
    *":${RLBENCH_ROOT}:"*) ;;
    *) export PYTHONPATH="${RLBENCH_ROOT}:${PYTHONPATH:-}" ;;
  esac
fi

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
