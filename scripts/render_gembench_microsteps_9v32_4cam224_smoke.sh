#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
FASTWAM_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${FASTWAM_ROOT}"
source scripts/setup_yuhan_paths.sh

PYTHON_BIN="${FASTWAM_CONDA_ENV}/bin/python"
export PATH="${FASTWAM_CONDA_ENV}/bin:${PATH}"

RUN_ID="${RUN_ID:-gembench_9v32_4cam224_smoke_$(date +%Y%m%d_%H%M%S)}"
SMOKE_ROOT="${SMOKE_ROOT:-${GEMBENCH_ROOT}/fastwam_cache/smoke/${RUN_ID}}"
MANIFEST="${GEMBENCH_9V32_4CAM_SMOKE_MANIFEST:-${SMOKE_ROOT}/manifest_2demos.json}"
RGB_CACHE_DIR="${GEMBENCH_9V32_4CAM_SMOKE_RGB_CACHE_DIR:-${SMOKE_ROOT}/rgb}"
AUDIT_DIR="${GEMBENCH_9V32_4CAM_SMOKE_AUDIT_DIR:-${SMOKE_ROOT}/audit}"
ROBOT_3DLOTUS_ROOT="${ROBOT_3DLOTUS_ROOT:-/mnt/yuhan/gembench_sim/robot-3dlotus}"

mkdir -p "${SMOKE_ROOT}" "${AUDIT_DIR}"

PYTHONPATH=src "${PYTHON_BIN}" scripts/audit_gembench_microsteps_9v32_contract.py \
  --root "${GEMBENCH_ROOT}" \
  --rgb-cache-dir "${RGB_CACHE_DIR}" \
  --length-source key_frameids \
  --max-demos "${MAX_DEMOS:-2}" \
  --official-camera-order \
  --official-cache-camera-order \
  --image-size 224 \
  --output-json "${MANIFEST}" \
  --output-md "${AUDIT_DIR}/contract.md"

xvfb-run -a -s "-screen 0 1280x1024x24" \
  "${PYTHON_BIN}" scripts/render_gembench_microsteps_9v32_cache.py \
  --manifest "${MANIFEST}" \
  --rgb-cache-dir "${RGB_CACHE_DIR}" \
  --robot-3dlotus-root "${ROBOT_3DLOTUS_ROOT}" \
  --cache-camera-order left_shoulder,right_shoulder,wrist,front \
  --image-size 224 \
  --max-demos "${MAX_DEMOS:-2}" \
  --extract-if-missing \
  --keep-going \
  --output-json "${AUDIT_DIR}/render.json"

PYTHONPATH=src "${PYTHON_BIN}" scripts/audit_gembench_microsteps_9v32_cache.py \
  --manifest "${MANIFEST}" \
  --rgb-cache-dir "${RGB_CACHE_DIR}" \
  --output-json "${AUDIT_DIR}/cache.json" \
  --output-md "${AUDIT_DIR}/cache.md"

echo "[gembench-9v32-4cam-smoke] manifest=${MANIFEST}"
echo "[gembench-9v32-4cam-smoke] rgb_cache_dir=${RGB_CACHE_DIR}"
echo "[gembench-9v32-4cam-smoke] audit_dir=${AUDIT_DIR}"
