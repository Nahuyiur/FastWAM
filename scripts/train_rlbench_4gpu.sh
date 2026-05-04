#!/usr/bin/env bash
set -euo pipefail

if [[ $# -gt 1 ]]; then
  echo "Usage: bash scripts/train_rlbench_4gpu.sh [task_name]" >&2
  exit 1
fi

TASK_NAME="${1:-rlbench_uncond_3cam224_1e-4}"
NPROC_PER_NODE=4

FASTWAM_ROOT="/mnt/world_foundational_model/yuhan/FastWAM"
CONDA_ROOT="/mnt/world_foundational_model/yuhan/miniconda3"
CONDA_ENV_NAME="fastwam"
ACTION_DIT_CKPT="checkpoints/ActionDiT_linear_interp_Wan22_alphascale_1024hdim.pt"
ACTION_DIT_CKPT_NAME="ActionDiT_linear_interp_Wan22_alphascale_1024hdim.pt"
MODEL_CONFIG="configs/model/fastwam.yaml"

cd "${FASTWAM_ROOT}"

if [[ ! -f "${CONDA_ROOT}/etc/profile.d/conda.sh" ]]; then
  echo "Missing conda activation script: ${CONDA_ROOT}/etc/profile.d/conda.sh" >&2
  exit 1
fi

# Some conda package activation hooks assume nounset is disabled.
set +u
# shellcheck disable=SC1091
source "${CONDA_ROOT}/etc/profile.d/conda.sh"
conda activate "${CONDA_ENV_NAME}"
set -u

# shellcheck disable=SC1091
source "${FASTWAM_ROOT}/scripts/setup_yuhan_paths.sh"
ACTION_DIT_CKPT="${FASTWAM_PRETRAIN_ROOT}/${ACTION_DIT_CKPT_NAME}"
export ACCELERATE_CONFIG_FILE="${ACCELERATE_CONFIG_FILE:-scripts/accelerate_configs/accelerate_zero2_ds.yaml}"

if [[ ! -f "${ACTION_DIT_CKPT}" ]]; then
  echo "[setup] Missing ${ACTION_DIT_CKPT}; preprocessing ActionDiT backbone."
  python scripts/preprocess_action_dit_backbone.py \
    --model-config "${MODEL_CONFIG}" \
    --output "${ACTION_DIT_CKPT}" \
    --device cuda \
    --dtype bfloat16
else
  echo "[setup] Found ${ACTION_DIT_CKPT}; skipping ActionDiT preprocessing."
fi

echo "[precompute] Encoding text prompts for task=${TASK_NAME} on ${NPROC_PER_NODE} GPUs."
torchrun \
  --standalone \
  --nproc_per_node="${NPROC_PER_NODE}" \
  scripts/precompute_text_embeds.py \
  "task=${TASK_NAME}" \
  +overwrite=false

echo "[train] Launching FastWAM RLBench training task=${TASK_NAME} on ${NPROC_PER_NODE} GPUs."
bash scripts/train_zero1.sh \
  "${NPROC_PER_NODE}" \
  "task=${TASK_NAME}" \
  "model.action_dit_pretrained_path=${ACTION_DIT_CKPT}"
