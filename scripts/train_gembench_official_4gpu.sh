#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
FASTWAM_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${FASTWAM_ROOT}"

# Official-style GEMBench baseline entrypoint.
#
# The actual training hyperparameters live in:
#   configs/task/gembench_keysteps_bbox_3cam224_1e-4.yaml
#
# That config mirrors the FastWAM RLBench 3-camera 1e-4 setup:
# batch_size=2/GPU, grad_accum=2, bf16, cosine lr=1e-4, max_steps=50000,
# eval/save every 2000 steps, and ZeRO-2 launch. This wrapper only fixes
# server/runtime prerequisites and does not change training hyperparameters.

export NPROC_PER_NODE="${NPROC_PER_NODE:-4}"
export MASTER_PORT="${MASTER_PORT:-29610}"
export STRICT_GEMBENCH_COMPLETE="${STRICT_GEMBENCH_COMPLETE:-1}"
export PRECOMPUTE_GEMBENCH_TEXT="${PRECOMPUTE_GEMBENCH_TEXT:-1}"
export VERIFY_GEMBENCH_CONTRACT="${VERIFY_GEMBENCH_CONTRACT:-1}"
export GEMBENCH_PREP_ONLY=0
export GEMBENCH_VERIFY_SAMPLES="${GEMBENCH_VERIFY_SAMPLES:-31}"
export GEMBENCH_VERIFY_WORKERS="${GEMBENCH_VERIFY_WORKERS:-2}"
export GEMBENCH_NORM_STATS="${GEMBENCH_NORM_STATS:-data/gembench_keysteps_bbox_dataset_stats.json}"
export GEMBENCH_NORM_MODE="${GEMBENCH_NORM_MODE:-z-score}"
export RUN_ID="${RUN_ID:-official_$(date +%Y%m%d_%H%M%S)}"

WANDB_ENABLED="${WANDB_ENABLED:-true}"
export WANDB_SUBPROJECT="${WANDB_SUBPROJECT:-fastwam-gembench-official}"
export WANDB_GROUP="${WANDB_GROUP:-${WANDB_SUBPROJECT}}"
export WANDB_RUN_NAME="${WANDB_RUN_NAME:-${RUN_ID}}"

echo "[gembench-official] run_id=${RUN_ID}"
echo "[gembench-official] nproc=${NPROC_PER_NODE}"
echo "[gembench-official] master_port=${MASTER_PORT}"
echo "[gembench-official] config=task/gembench_keysteps_bbox_3cam224_1e-4"
echo "[gembench-official] norm_stats=${GEMBENCH_NORM_STATS}"
echo "[gembench-official] output=./runs/gembench_keysteps_bbox_3cam224_1e-4/${RUN_ID}"

exec bash scripts/train_gembench_4gpu.sh \
  "$@"
