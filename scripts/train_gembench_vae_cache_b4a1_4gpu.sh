#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
FASTWAM_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${FASTWAM_ROOT}"

export TASK_NAME="${TASK_NAME:-gembench_keysteps_bbox_3cam224_vaecache_b4a1_1e-4}"
export WANDB_SUBPROJECT="${WANDB_SUBPROJECT:-fastwam-gembench-vae-cache-b4a1}"

exec bash scripts/train_gembench_vae_cache_4gpu.sh "$@"
