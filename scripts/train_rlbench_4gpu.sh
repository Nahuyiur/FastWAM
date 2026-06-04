#!/usr/bin/env bash
set -euo pipefail

if [[ $# -gt 1 ]]; then
  echo "Usage: bash scripts/train_rlbench_4gpu.sh [task_name]" >&2
  echo "  category tasks: rlbench_original_3cam224_1e-4 | rlbench_color_3cam224_1e-4 | rlbench_shape_3cam224_1e-4 | rlbench_color_shape_3cam224_1e-4" >&2
  exit 1
fi

TASK_NAME="${1:-rlbench_original_3cam224_1e-4}"
NPROC_PER_NODE=4

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
FASTWAM_ROOT="${FASTWAM_ROOT:-$(cd "${SCRIPT_DIR}/.." && pwd)}"
ACTION_DIT_CKPT_NAME="ActionDiT_linear_interp_Wan22_alphascale_1024hdim.pt"
MODEL_CONFIG="configs/model/fastwam.yaml"

cd "${FASTWAM_ROOT}"

# shellcheck disable=SC1091
source "${FASTWAM_ROOT}/scripts/setup_yuhan_paths.sh"

if [[ ! -f "${CONDA_ROOT}/bin/activate" ]]; then
  echo "Missing conda activation script: ${CONDA_ROOT}/bin/activate" >&2
  exit 1
fi

if [[ ! -d "${FASTWAM_CONDA_ENV}" ]]; then
  echo "Missing FastWAM conda env: ${FASTWAM_CONDA_ENV}" >&2
  exit 1
fi

# Some conda package activation hooks assume nounset is disabled.
set +u
# shellcheck disable=SC1091
source "${CONDA_ROOT}/bin/activate" "${FASTWAM_CONDA_ENV}"
set -u

ACTION_DIT_CKPT="${FASTWAM_PRETRAIN_ROOT}/${ACTION_DIT_CKPT_NAME}"
export ACCELERATE_CONFIG_FILE="${ACCELERATE_CONFIG_FILE:-scripts/accelerate_configs/accelerate_zero2_ds.yaml}"

bash scripts/check_jinshan_fastwam_ready.sh --train

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
