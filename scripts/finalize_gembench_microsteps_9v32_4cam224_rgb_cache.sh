#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
FASTWAM_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${FASTWAM_ROOT}"
source scripts/setup_yuhan_paths.sh

PYTHON_BIN="${FASTWAM_CONDA_ENV}/bin/python"
export PATH="${FASTWAM_CONDA_ENV}/bin:${PATH}"

NUM_SHARDS="${NUM_SHARDS:-24}"
SHARD_DIR="${GEMBENCH_9V32_4CAM_SHARD_DIR:-${GEMBENCH_ROOT}/fastwam_cache/microsteps_9v32_4cam224_shards_${NUM_SHARDS}}"
SOURCE_MANIFEST="${GEMBENCH_9V32_4CAM_FULL_MANIFEST:-${SHARD_DIR}/manifest_full.json}"
RGB_CACHE_DIR="${GEMBENCH_9V32_4CAM_RGB_CACHE_DIR:-${GEMBENCH_ROOT}/fastwam_cache/microsteps_9v32_4cam224_rgb}"
FILTERED_MANIFEST="${GEMBENCH_9V32_4CAM_MANIFEST:-${GEMBENCH_ROOT}/fastwam_cache/microsteps_9v32_4cam224_manifest.json}"
RUN_ID="${RUN_ID:-gembench_9v32_4cam224_rgb_finalize_$(date +%Y%m%d_%H%M%S)}"
AUDIT_DIR="${AUDIT_DIR:-runs/gembench_microsteps_9v32_4cam224_audits/${RUN_ID}}"
MIN_PRESENT_FRACTION="${MIN_PRESENT_FRACTION:-1.0}"
export FILTERED_MANIFEST

if [[ ! -x "${PYTHON_BIN}" ]]; then
  echo "[gembench-9v32-4cam-finalize] missing python: ${PYTHON_BIN}" >&2
  exit 1
fi
if [[ ! -f "${SOURCE_MANIFEST}" ]]; then
  echo "[gembench-9v32-4cam-finalize] missing source manifest: ${SOURCE_MANIFEST}" >&2
  exit 1
fi

mkdir -p "${AUDIT_DIR}" "$(dirname "${FILTERED_MANIFEST}")"

echo "[gembench-9v32-4cam-finalize] source_manifest=${SOURCE_MANIFEST}"
echo "[gembench-9v32-4cam-finalize] rgb_cache_dir=${RGB_CACHE_DIR}"
echo "[gembench-9v32-4cam-finalize] filtered_manifest=${FILTERED_MANIFEST}"
echo "[gembench-9v32-4cam-finalize] min_present_fraction=${MIN_PRESENT_FRACTION}"

PYTHONPATH=src "${PYTHON_BIN}" scripts/filter_gembench_microsteps_9v32_manifest_by_cache.py \
  --manifest "${SOURCE_MANIFEST}" \
  --rgb-cache-dir "${RGB_CACHE_DIR}" \
  --output-json "${FILTERED_MANIFEST}" \
  --output-md "${AUDIT_DIR}/filtered_manifest.md" \
  --min-present-fraction "${MIN_PRESENT_FRACTION}"

PYTHONPATH=src "${PYTHON_BIN}" scripts/audit_gembench_microsteps_9v32_cache.py \
  --manifest "${FILTERED_MANIFEST}" \
  --rgb-cache-dir "${RGB_CACHE_DIR}" \
  --min-present-fraction 1.0 \
  --output-json "${AUDIT_DIR}/cache_audit.json" \
  --output-md "${AUDIT_DIR}/cache_audit.md"

"${PYTHON_BIN}" - <<'PY'
import json
import os
from pathlib import Path

manifest = Path(os.environ["FILTERED_MANIFEST"])
payload = json.loads(manifest.read_text())
print(json.dumps({
    "status": payload.get("status"),
    "filtered_manifest": str(manifest),
    "filtered_demos": payload.get("summary", {}).get("filtered_demos"),
    "eligible_windows": payload.get("summary", {}).get("eligible_32_start_count"),
    "rgb_cache_dir": payload.get("rgb_cache_dir"),
}, indent=2, sort_keys=True))
PY
