#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
FASTWAM_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${FASTWAM_ROOT}"
source scripts/setup_yuhan_paths.sh

export GEMBENCH_VAE_CACHE_DIR="${GEMBENCH_VAE_CACHE_DIR:-/mnt/yuhan/datasets/GEMBench/fastwam_cache/vae_latents/keysteps_bbox_seed0_3cam224x672_t9_v1}"
export TASK_NAME="${TASK_NAME:-gembench_keysteps_bbox_3cam224_vaecache_1e-4}"
export WANDB_SUBPROJECT="${WANDB_SUBPROJECT:-fastwam-gembench-vae-cache}"

if [[ ! -f "${GEMBENCH_VAE_CACHE_DIR}/manifest.json" || ! -f "${GEMBENCH_VAE_CACHE_DIR}/video_latents.float32.npy" ]]; then
  echo "[gembench-vae-cache] missing VAE cache under ${GEMBENCH_VAE_CACHE_DIR}" >&2
  echo "[gembench-vae-cache] run scripts/precompute_gembench_vae_latents.py first." >&2
  exit 1
fi

if [[ "${VERIFY_GEMBENCH_VAE_CACHE:-1}" == "1" ]]; then
  PYTHON_BIN="${PYTHON_BIN:-${FASTWAM_CONDA_ENV}/bin/python}"
  mkdir -p runs/gembench_verification
  echo "[gembench-vae-cache] verifying latent cache before launch: ${GEMBENCH_VAE_CACHE_DIR}"
  "${PYTHON_BIN}" scripts/verify_gembench_vae_cache.py \
    --root "${GEMBENCH_ROOT}" \
    --vae-cache-dir "${GEMBENCH_VAE_CACHE_DIR}" \
    --samples "${GEMBENCH_VAE_CACHE_PREFLIGHT_SAMPLES:-0}" \
    --skip-latent-encode \
    --json-output "runs/gembench_verification/vae_cache_preflight_${TASK_NAME}.json" \
    --markdown-output "runs/gembench_verification/vae_cache_preflight_${TASK_NAME}.md"
fi

exec bash scripts/train_gembench_4gpu.sh "$@"
