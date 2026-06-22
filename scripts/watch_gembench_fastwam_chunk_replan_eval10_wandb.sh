#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

RUN_DIR="${FASTWAM_RUN_DIR:-}"
if [[ -z "${RUN_DIR}" && "$#" -gt 0 ]]; then
  RUN_DIR="$1"
  shift
fi
if [[ -z "${RUN_DIR}" ]]; then
  echo "Usage: FASTWAM_RUN_DIR=/path/to/run $0" >&2
  echo "   or: $0 /path/to/run" >&2
  exit 2
fi

RUN_ID="${FASTWAM_RUN_ID:-$(basename "${RUN_DIR}")}"
INTERVAL_STEPS="${INTERVAL_STEPS:-2000}"
CHUNK_REPLAN_STEPS="${CHUNK_REPLAN_STEPS:-8}"
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
EXTRA_CHUNK_ARGS=()
if [[ -n "${CHUNK_ACTION_HORIZON:-}" ]]; then
  EXTRA_CHUNK_ARGS=(--chunk-action-horizon "${CHUNK_ACTION_HORIZON}")
fi

export CUDA_VISIBLE_DEVICES
export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"
export WANDB_PROJECT="${WANDB_PROJECT:-fastwam-gembench-9v32}"
export WANDB_GROUP="${WANDB_GROUP:-gembench-wam-9v32-chunk-replan-eval10-watchdog}"

python scripts/watch_gembench_policy_eval10_wandb.py \
  --run-dir "${RUN_DIR}" \
  --model-name fastwam \
  --run-id "${RUN_ID}" \
  --eval10-config configs/eval/gembench_official_eval10_fastwam_chunk_replan_seed200.yaml \
  --interval-steps "${INTERVAL_STEPS}" \
  --num-trials 10 \
  --num-inference-steps 10 \
  --max-videos "${MAX_VIDEOS:-10}" \
  --min-video-frames "${MIN_VIDEO_FRAMES:-60}" \
  --relation-mode none \
  --eval-protocol chunk_replan \
  --chunk-replan-steps "${CHUNK_REPLAN_STEPS}" \
  "${EXTRA_CHUNK_ARGS[@]}" \
  --chunk-predict-video \
  --device "${DEVICE:-cuda:0}" \
  --cuda-visible-devices "${CUDA_VISIBLE_DEVICES}" \
  --wandb-metric-prefix chunk_replan_eval10 \
  --output-root "${OUTPUT_ROOT:-/mnt/yuhan/gembench_watchdog_eval10/fastwam/${RUN_ID}}" \
  "$@"
