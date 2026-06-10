#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
FASTWAM_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${FASTWAM_ROOT}"
source scripts/setup_yuhan_paths.sh

PYTHON_BIN="${FASTWAM_CONDA_ENV}/bin/python"
export PATH="${FASTWAM_CONDA_ENV}/bin:${PATH}"

NUM_SHARDS="${NUM_SHARDS:-16}"
OUTPUT_DIR="${GEMBENCH_9V32_4CAM_SHARD_DIR:-${GEMBENCH_ROOT}/fastwam_cache/microsteps_9v32_4cam224_shards_${NUM_SHARDS}}"
RGB_CACHE_DIR="${GEMBENCH_9V32_4CAM_RGB_CACHE_DIR:-${GEMBENCH_ROOT}/fastwam_cache/microsteps_9v32_4cam224_rgb}"

PYTHONPATH=src "${PYTHON_BIN}" scripts/build_gembench_microsteps_9v32_shards.py \
  --root "${GEMBENCH_ROOT}" \
  --rgb-cache-dir "${RGB_CACHE_DIR}" \
  --output-dir "${OUTPUT_DIR}" \
  --num-shards "${NUM_SHARDS}" \
  --official-camera-order \
  --official-cache-camera-order \
  --image-size 224 \
  "${@}"

echo "[gembench-9v32-4cam-shards] output_dir=${OUTPUT_DIR}"
echo "[gembench-9v32-4cam-shards] rgb_cache_dir=${RGB_CACHE_DIR}"
