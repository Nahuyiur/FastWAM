#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
FASTWAM_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${FASTWAM_ROOT}"
source scripts/setup_yuhan_paths.sh

CANDIDATE="${1:-vae_b4a1_zero2}"
shift || true

export NPROC_PER_NODE="${NPROC_PER_NODE:-4}"
export VERIFY_GEMBENCH_CONTRACT="${VERIFY_GEMBENCH_CONTRACT:-0}"
export PRECOMPUTE_GEMBENCH_TEXT="${PRECOMPUTE_GEMBENCH_TEXT:-0}"
export VERIFY_GEMBENCH_VAE_CACHE="${VERIFY_GEMBENCH_VAE_CACHE:-0}"
export WANDB_ENABLED="${WANDB_ENABLED:-false}"
if [[ "${WANDB_ENABLED}" == "false" ]]; then
  export DISABLE_WANDB="${DISABLE_WANDB:-1}"
fi

case "${CANDIDATE}" in
  vae_zero2)
    export TASK_NAME="gembench_keysteps_bbox_3cam224_vaecache_1e-4"
    ;;
  vae_b4a1_zero2)
    export TASK_NAME="gembench_keysteps_bbox_3cam224_vaecache_b4a1_1e-4"
    ;;
  vae_zero2_tuned)
    export TASK_NAME="gembench_keysteps_bbox_3cam224_vaecache_1e-4"
    export ACCELERATE_CONFIG_FILE=scripts/accelerate_configs/accelerate_zero2_tuned_ds.yaml
    ;;
  vae_b4a1_zero2_tuned)
    export TASK_NAME="gembench_keysteps_bbox_3cam224_vaecache_b4a1_1e-4"
    export ACCELERATE_CONFIG_FILE=scripts/accelerate_configs/accelerate_zero2_tuned_ds.yaml
    ;;
  vae_zero1_tuned)
    export TASK_NAME="gembench_keysteps_bbox_3cam224_vaecache_1e-4"
    export ACCELERATE_CONFIG_FILE=scripts/accelerate_configs/accelerate_zero1_tuned_ds.yaml
    ;;
  vae_b4a1_zero1_tuned)
    export TASK_NAME="gembench_keysteps_bbox_3cam224_vaecache_b4a1_1e-4"
    export ACCELERATE_CONFIG_FILE=scripts/accelerate_configs/accelerate_zero1_tuned_ds.yaml
    ;;
  *)
    echo "usage: $0 {vae_zero2|vae_b4a1_zero2|vae_zero2_tuned|vae_b4a1_zero2_tuned|vae_zero1_tuned|vae_b4a1_zero1_tuned} [hydra_overrides...]" >&2
    exit 2
    ;;
esac

export RUN_ID="profile_${CANDIDATE}_$(date +%Y%m%d_%H%M%S)"
PROFILE_STEPS="${GEMBENCH_TORCH_PROFILE_STEPS:-8}"
PROFILE_WARMUP_STEPS="${GEMBENCH_PROFILE_WARMUP_STEPS:-0}"
TORCH_WAIT="${GEMBENCH_TORCH_PROFILE_WAIT:-1}"
TORCH_WARMUP="${GEMBENCH_TORCH_PROFILE_WARMUP:-1}"
TORCH_ACTIVE="${GEMBENCH_TORCH_PROFILE_ACTIVE:-2}"
TORCH_REPEAT="${GEMBENCH_TORCH_PROFILE_REPEAT:-1}"

echo "[profile] candidate=${CANDIDATE} task=${TASK_NAME} run_id=${RUN_ID}"
bash scripts/train_gembench_vae_cache_4gpu.sh \
  max_steps="${PROFILE_STEPS}" \
  save_every=0 \
  save_final_checkpoint=false \
  eval_every=0 \
  log_every=1 \
  profile.enabled=true \
  profile.warmup_steps="${PROFILE_WARMUP_STEPS}" \
  profile.torch_profiler.enabled=true \
  profile.torch_profiler.wait="${TORCH_WAIT}" \
  profile.torch_profiler.warmup="${TORCH_WARMUP}" \
  profile.torch_profiler.active="${TORCH_ACTIVE}" \
  profile.torch_profiler.repeat="${TORCH_REPEAT}" \
  profile.torch_profiler.record_shapes="${GEMBENCH_TORCH_PROFILE_RECORD_SHAPES:-false}" \
  profile.torch_profiler.profile_memory="${GEMBENCH_TORCH_PROFILE_MEMORY:-false}" \
  profile.torch_profiler.write_trace="${GEMBENCH_TORCH_PROFILE_WRITE_TRACE:-false}" \
  profile.torch_profiler.sort_by="${GEMBENCH_TORCH_PROFILE_SORT_BY:-self_cuda_time_total}" \
  profile.torch_profiler.row_limit="${GEMBENCH_TORCH_PROFILE_ROW_LIMIT:-80}" \
  "$@"

echo "[profile] step profile: runs/${TASK_NAME}/${RUN_ID}/profile/step_times.jsonl"
echo "[profile] torch table: runs/${TASK_NAME}/${RUN_ID}/profile/torch_profiler_key_averages.txt"
echo "[profile] torch traces: runs/${TASK_NAME}/${RUN_ID}/profile/torch_profiler"
