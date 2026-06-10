#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
FASTWAM_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${FASTWAM_ROOT}"
source scripts/setup_yuhan_paths.sh

PYTHON_BIN="${FASTWAM_CONDA_ENV}/bin/python"
export PATH="${FASTWAM_CONDA_ENV}/bin:${PATH}"

NUM_SHARDS="${NUM_SHARDS:-24}"
SHARD_ID="${SHARD_ID:-0}"
SHARD_ID_PAD="$(printf "%03d" "${SHARD_ID}")"
NUM_SHARDS_PAD="$(printf "%03d" "${NUM_SHARDS}")"
SHARD_DIR="${GEMBENCH_9V32_4CAM_SHARD_DIR:-${GEMBENCH_ROOT}/fastwam_cache/microsteps_9v32_4cam224_shards_${NUM_SHARDS}}"
MANIFEST="${GEMBENCH_9V32_4CAM_SHARD_MANIFEST:-${SHARD_DIR}/manifest_shard_${SHARD_ID_PAD}_of_${NUM_SHARDS_PAD}.json}"
RGB_CACHE_DIR="${GEMBENCH_9V32_4CAM_RGB_CACHE_DIR:-${GEMBENCH_ROOT}/fastwam_cache/microsteps_9v32_4cam224_rgb}"
ROBOT_3DLOTUS_ROOT="${ROBOT_3DLOTUS_ROOT:-/mnt/yuhan/gembench_sim/robot-3dlotus}"
RUN_ID="${RUN_ID:-gembench_9v32_4cam224_render_s${SHARD_ID_PAD}_$(date +%Y%m%d_%H%M%S)}"
LOG_DIR="${LOG_DIR:-${GEMBENCH_ROOT}/fastwam_cache/render_logs/${RUN_ID}}"
OUTPUT_JSON="${OUTPUT_JSON:-${LOG_DIR}/render.json}"
AUDIT_AFTER="${AUDIT_AFTER:-1}"
MIN_PRESENT_FRACTION="${MIN_PRESENT_FRACTION:-1.0}"

if [[ ! -x "${PYTHON_BIN}" ]]; then
  echo "[gembench-9v32-4cam-render] missing python: ${PYTHON_BIN}" >&2
  exit 1
fi
if [[ ! -f "${MANIFEST}" ]]; then
  echo "[gembench-9v32-4cam-render] missing shard manifest: ${MANIFEST}" >&2
  exit 1
fi

MAX_DEMOS_ARGS=()
if [[ -n "${MAX_DEMOS:-}" ]]; then
  MAX_DEMOS_ARGS=(--max-demos "${MAX_DEMOS}")
fi
TASKVARS_ARGS=()
if [[ -n "${TASKVARS:-}" ]]; then
  TASKVARS_ARGS=(--taskvars "${TASKVARS}")
fi
OVERWRITE_ARGS=()
if [[ "${OVERWRITE:-0}" == "1" ]]; then
  OVERWRITE_ARGS=(--overwrite)
fi
DRY_RUN_ARGS=()
if [[ "${DRY_RUN:-0}" == "1" ]]; then
  DRY_RUN_ARGS=(--dry-run)
fi

mkdir -p "${LOG_DIR}" "${RGB_CACHE_DIR}"

echo "[gembench-9v32-4cam-render] run_id=${RUN_ID}"
echo "[gembench-9v32-4cam-render] manifest=${MANIFEST}"
echo "[gembench-9v32-4cam-render] rgb_cache_dir=${RGB_CACHE_DIR}"
echo "[gembench-9v32-4cam-render] output_json=${OUTPUT_JSON}"

xvfb-run -a -s "-screen 0 1280x1024x24" \
  "${PYTHON_BIN}" scripts/render_gembench_microsteps_9v32_cache.py \
  --manifest "${MANIFEST}" \
  --rgb-cache-dir "${RGB_CACHE_DIR}" \
  --robot-3dlotus-root "${ROBOT_3DLOTUS_ROOT}" \
  --cache-camera-order left_shoulder,right_shoulder,wrist,front \
  --image-size 224 \
  "${MAX_DEMOS_ARGS[@]}" \
  "${TASKVARS_ARGS[@]}" \
  "${OVERWRITE_ARGS[@]}" \
  "${DRY_RUN_ARGS[@]}" \
  --extract-if-missing \
  --keep-going \
  --output-json "${OUTPUT_JSON}" \
  "${@}"

if [[ "${DRY_RUN:-0}" != "1" && "${AUDIT_AFTER}" == "1" ]]; then
  PYTHONPATH=src "${PYTHON_BIN}" scripts/audit_gembench_microsteps_9v32_cache.py \
    --manifest "${MANIFEST}" \
    --rgb-cache-dir "${RGB_CACHE_DIR}" \
    --min-present-fraction "${MIN_PRESENT_FRACTION}" \
    --output-json "${LOG_DIR}/cache_audit.json" \
    --output-md "${LOG_DIR}/cache_audit.md"
fi

echo "[gembench-9v32-4cam-render] log_dir=${LOG_DIR}"
