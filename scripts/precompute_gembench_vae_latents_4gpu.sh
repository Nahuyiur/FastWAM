#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
FASTWAM_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${FASTWAM_ROOT}"
source scripts/setup_yuhan_paths.sh

PYTHON_BIN="${PYTHON_BIN:-${FASTWAM_CONDA_ENV}/bin/python}"
CACHE_DIR="${GEMBENCH_VAE_CACHE_DIR:-/mnt/yuhan/datasets/GEMBench/fastwam_cache/vae_latents/keysteps_bbox_seed0_3cam224x672_t9_v1}"
LOG_DIR="${GEMBENCH_VAE_CACHE_LOG_DIR:-runs/gembench_vae_cache_precompute}"
NUM_SHARDS="${GEMBENCH_VAE_CACHE_NUM_SHARDS:-4}"
REBUILD="${GEMBENCH_VAE_CACHE_REBUILD:-0}"
mkdir -p "${LOG_DIR}"

if [[ "${REBUILD}" == "1" && -d "${CACHE_DIR}" ]]; then
  BACKUP_DIR="${CACHE_DIR}.no_autocast_backup_$(date +%Y%m%d_%H%M%S)"
  echo "[vae-cache-4gpu] moving existing cache to ${BACKUP_DIR}"
  mv "${CACHE_DIR}" "${BACKUP_DIR}"
fi

if [[ -f "${CACHE_DIR}/manifest.json" && "${REBUILD}" != "1" ]]; then
  echo "[vae-cache-4gpu] cache already has manifest: ${CACHE_DIR}/manifest.json" >&2
  echo "[vae-cache-4gpu] set GEMBENCH_VAE_CACHE_REBUILD=1 to rebuild intentionally." >&2
  exit 1
fi

COMMON_ARGS=(
  --root "${GEMBENCH_ROOT}"
  --cache-dir "${CACHE_DIR}"
  --num-shards "${NUM_SHARDS}"
  --log-every "${GEMBENCH_VAE_CACHE_LOG_EVERY:-25}"
)

pids=()
for shard in $(seq 0 $((NUM_SHARDS - 1))); do
  gpu="${shard}"
  log_file="${LOG_DIR}/precompute_autocast_shard${shard}.log"
  extra_args=()
  if [[ "${shard}" == "0" ]]; then
    extra_args+=(--no-resume)
  fi
  echo "[vae-cache-4gpu] launch shard=${shard}/${NUM_SHARDS} gpu=${gpu} log=${log_file}"
  (
    export CUDA_VISIBLE_DEVICES="${gpu}"
    "${PYTHON_BIN}" scripts/precompute_gembench_vae_latents.py \
      "${COMMON_ARGS[@]}" \
      --shard-id "${shard}" \
      "${extra_args[@]}"
  ) >"${log_file}" 2>&1 &
  pids+=("$!")
  if [[ "${shard}" == "0" ]]; then
    for _ in $(seq 1 120); do
      [[ -f "${CACHE_DIR}/manifest.json" && -f "${CACHE_DIR}/completed_rows.bool.npy" ]] && break
      sleep 1
    done
    if [[ ! -f "${CACHE_DIR}/manifest.json" || ! -f "${CACHE_DIR}/completed_rows.bool.npy" ]]; then
      echo "[vae-cache-4gpu] shard0 did not initialize cache files in time; see ${log_file}" >&2
      exit 1
    fi
  fi
  sleep 1
done

status=0
for pid in "${pids[@]}"; do
  if ! wait "${pid}"; then
    status=1
  fi
done
if [[ "${status}" != "0" ]]; then
  echo "[vae-cache-4gpu] at least one shard failed; inspect ${LOG_DIR}/precompute_autocast_shard*.log" >&2
  exit "${status}"
fi

"${PYTHON_BIN}" scripts/verify_gembench_vae_cache.py \
  --root "${GEMBENCH_ROOT}" \
  --vae-cache-dir "${CACHE_DIR}" \
  --samples "${GEMBENCH_VAE_CACHE_VERIFY_SAMPLES:-4}" \
  --latent-atol "${GEMBENCH_VAE_CACHE_LATENT_ATOL:-1e-3}" \
  --json-output "runs/gembench_verification/vae_cache_autocast_full.json" \
  --markdown-output "runs/gembench_verification/vae_cache_autocast_full.md"

echo "[vae-cache-4gpu] complete and verified: ${CACHE_DIR}"
