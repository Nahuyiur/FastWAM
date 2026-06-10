#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
FASTWAM_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${FASTWAM_ROOT}"

source scripts/setup_yuhan_paths.sh
export PATH="${FASTWAM_CONDA_ENV}/bin:${PATH}"

PYTHON_BIN="${FASTWAM_CONDA_ENV}/bin/python"
NUM_SHARDS="${NUM_SHARDS:-4}"
CUDA_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3}"
RUN_ID="${RUN_ID:-gembench_9v32_4cam224_vae_cache_$(date +%Y%m%d_%H%M%S)}"
LOG_DIR="${LOG_DIR:-logs/${RUN_ID}}"

export GEMBENCH_9V32_4CAM_MANIFEST="${GEMBENCH_9V32_4CAM_MANIFEST:-${GEMBENCH_ROOT}/fastwam_cache/microsteps_9v32_4cam224_manifest.json}"
export GEMBENCH_9V32_4CAM_RGB_CACHE_DIR="${GEMBENCH_9V32_4CAM_RGB_CACHE_DIR:-${GEMBENCH_ROOT}/fastwam_cache/microsteps_9v32_4cam224_rgb}"
export GEMBENCH_9V32_4CAM_VAE_CACHE_DIR="${GEMBENCH_9V32_4CAM_VAE_CACHE_DIR:-${GEMBENCH_ROOT}/fastwam_cache/vae_latents/microsteps_9v32_seed0_4cam224x896_t9_a32_v1}"

IFS=',' read -r -a GPU_IDS <<< "${CUDA_DEVICES}"
if (( ${#GPU_IDS[@]} < NUM_SHARDS )); then
  echo "[9v32-4cam-vae-cache] CUDA_VISIBLE_DEVICES has ${#GPU_IDS[@]} ids but NUM_SHARDS=${NUM_SHARDS}: ${CUDA_DEVICES}" >&2
  exit 1
fi

mkdir -p "${LOG_DIR}" "${GEMBENCH_9V32_4CAM_VAE_CACHE_DIR}"

COMMON_ARGS=(
  --manifest "${GEMBENCH_9V32_4CAM_MANIFEST}"
  --rgb-cache-dir "${GEMBENCH_9V32_4CAM_RGB_CACHE_DIR}"
  --cache-dir "${GEMBENCH_9V32_4CAM_VAE_CACHE_DIR}"
  --video-size 224 896
  --camera-order left_shoulder right_shoulder wrist front
  --cache-camera-order left_shoulder right_shoulder wrist front
  --num-shards "${NUM_SHARDS}"
)

echo "[9v32-4cam-vae-cache] init cache=${GEMBENCH_9V32_4CAM_VAE_CACHE_DIR} run_id=${RUN_ID}"
"${PYTHON_BIN}" scripts/precompute_gembench_microsteps_9v32_vae_latents.py "${COMMON_ARGS[@]}" --init-only \
  2>&1 | tee "${LOG_DIR}/init.log"

pids=()
for shard in $(seq 0 $((NUM_SHARDS - 1))); do
  gpu="${GPU_IDS[$shard]}"
  log="${LOG_DIR}/shard_${shard}.log"
  echo "[9v32-4cam-vae-cache] launching shard=${shard}/${NUM_SHARDS} gpu=${gpu} log=${log}"
  CUDA_VISIBLE_DEVICES="${gpu}" "${PYTHON_BIN}" scripts/precompute_gembench_microsteps_9v32_vae_latents.py \
    "${COMMON_ARGS[@]}" \
    --shard-id "${shard}" \
    2>&1 | tee "${log}" &
  pids+=("$!")
done

failed=0
for pid in "${pids[@]}"; do
  if ! wait "${pid}"; then
    failed=1
  fi
done

if (( failed != 0 )); then
  echo "[9v32-4cam-vae-cache] one or more shards failed" >&2
  exit 1
fi

echo "[9v32-4cam-vae-cache] all shards finished"
"${PYTHON_BIN}" - <<'PY'
import json
import os
from pathlib import Path

import numpy as np

cache_dir = Path(os.environ["GEMBENCH_9V32_4CAM_VAE_CACHE_DIR"])
manifest = cache_dir / "manifest.json"
payload = json.loads(manifest.read_text())
if not payload.get("complete"):
    completed = np.load(cache_dir / "completed_windows.bool.npy", mmap_mode="r")
    if bool(np.all(completed)):
        payload["complete"] = True
        tmp = manifest.with_name(f".{manifest.name}.tmp")
        tmp.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
        tmp.replace(manifest)
print(json.dumps({
    "cache_dir": str(cache_dir),
    "complete": payload.get("complete"),
    "num_windows": payload.get("num_windows"),
    "latent_shape": payload.get("latent_shape"),
    "latents": str(cache_dir / "video_latents.float32.npy"),
}, indent=2))
if not payload.get("complete"):
    raise SystemExit("cache manifest is not complete")
PY
