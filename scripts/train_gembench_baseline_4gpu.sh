#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
FASTWAM_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${FASTWAM_ROOT}"

# Baseline GEMBench training entrypoint.
#
# This wrapper intentionally keeps the actual hyperparameters in
# configs/task/gembench_keysteps_bbox_3cam224_1e-4.yaml. That config mirrors the
# existing FastWAM RLBench 3-camera setup as closely as possible:
# lr=1e-4, cosine schedule, bf16, batch_size=2/GPU, grad_accum=2, max_steps=50000,
# eval/save every 2000 steps, and ZeRO-2 launch.

export NPROC_PER_NODE="${NPROC_PER_NODE:-4}"
export STRICT_GEMBENCH_COMPLETE="${STRICT_GEMBENCH_COMPLETE:-1}"
export PRECOMPUTE_GEMBENCH_TEXT="${PRECOMPUTE_GEMBENCH_TEXT:-1}"
export VERIFY_GEMBENCH_CONTRACT="${VERIFY_GEMBENCH_CONTRACT:-1}"
export GEMBENCH_PREP_ONLY=0
export GEMBENCH_VERIFY_SAMPLES="${GEMBENCH_VERIFY_SAMPLES:-31}"
export GEMBENCH_VERIFY_WORKERS="${GEMBENCH_VERIFY_WORKERS:-2}"
export RUN_ID="${RUN_ID:-baseline_$(date +%Y%m%d_%H%M%S)}"

WANDB_ENABLED="${WANDB_ENABLED:-true}"
export WANDB_SUBPROJECT="${WANDB_SUBPROJECT:-fastwam-gembench-baseline}"
export WANDB_GROUP="${WANDB_GROUP:-${WANDB_SUBPROJECT}}"
export WANDB_RUN_NAME="${WANDB_RUN_NAME:-${RUN_ID}}"

echo "[gembench-baseline] run_id=${RUN_ID}"
echo "[gembench-baseline] nproc=${NPROC_PER_NODE}"
echo "[gembench-baseline] config=task/gembench_keysteps_bbox_3cam224_1e-4"
echo "[gembench-baseline] output=./runs/gembench_keysteps_bbox_3cam224_1e-4/${RUN_ID}"

exec bash scripts/train_gembench_4gpu.sh \
  "$@"
