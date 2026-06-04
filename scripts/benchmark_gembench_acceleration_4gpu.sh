#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
FASTWAM_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${FASTWAM_ROOT}"
source scripts/setup_yuhan_paths.sh

CANDIDATE="${1:-vae_zero2}"
shift || true
export NPROC_PER_NODE="${NPROC_PER_NODE:-4}"
export VERIFY_GEMBENCH_CONTRACT="${VERIFY_GEMBENCH_CONTRACT:-0}"
export PRECOMPUTE_GEMBENCH_TEXT="${PRECOMPUTE_GEMBENCH_TEXT:-0}"
export VERIFY_GEMBENCH_VAE_CACHE="${VERIFY_GEMBENCH_VAE_CACHE:-1}"
export GEMBENCH_PROFILE_STEPS="${GEMBENCH_PROFILE_STEPS:-50}"
export WANDB_ENABLED="${WANDB_ENABLED:-false}"
if [[ "${WANDB_ENABLED}" == "false" ]]; then
  export DISABLE_WANDB="${DISABLE_WANDB:-1}"
fi

case "${CANDIDATE}" in
  rgb_zero2)
    export RUN_ID="bench_rgb_zero2_$(date +%Y%m%d_%H%M%S)"
    exec bash scripts/train_gembench_4gpu.sh \
      task=gembench_keysteps_bbox_3cam224_1e-4 \
      max_steps="${GEMBENCH_PROFILE_STEPS}" save_every=0 save_final_checkpoint=false eval_every=0 log_every=1 \
      profile.enabled=true profile.warmup_steps=20 "$@"
    ;;
  vae_zero2)
    export RUN_ID="bench_vae_zero2_$(date +%Y%m%d_%H%M%S)"
    exec bash scripts/train_gembench_vae_cache_4gpu.sh \
      max_steps="${GEMBENCH_PROFILE_STEPS}" save_every=0 save_final_checkpoint=false eval_every=0 log_every=1 \
      profile.enabled=true profile.warmup_steps=20 "$@"
    ;;
  vae_b4a1_zero2)
    export RUN_ID="bench_vae_b4a1_zero2_$(date +%Y%m%d_%H%M%S)"
    export TASK_NAME=gembench_keysteps_bbox_3cam224_vaecache_b4a1_1e-4
    exec bash scripts/train_gembench_vae_cache_4gpu.sh \
      max_steps="${GEMBENCH_PROFILE_STEPS}" save_every=0 save_final_checkpoint=false eval_every=0 log_every=1 \
      profile.enabled=true profile.warmup_steps=20 "$@"
    ;;
  vae_b4a1_zero2_tuned)
    export ACCELERATE_CONFIG_FILE=scripts/accelerate_configs/accelerate_zero2_tuned_ds.yaml
    export RUN_ID="bench_vae_b4a1_zero2_tuned_$(date +%Y%m%d_%H%M%S)"
    export TASK_NAME=gembench_keysteps_bbox_3cam224_vaecache_b4a1_1e-4
    exec bash scripts/train_gembench_vae_cache_4gpu.sh \
      max_steps="${GEMBENCH_PROFILE_STEPS}" save_every=0 save_final_checkpoint=false eval_every=0 log_every=1 \
      profile.enabled=true profile.warmup_steps=20 "$@"
    ;;
  vae_zero2_tuned)
    export ACCELERATE_CONFIG_FILE=scripts/accelerate_configs/accelerate_zero2_tuned_ds.yaml
    export RUN_ID="bench_vae_zero2_tuned_$(date +%Y%m%d_%H%M%S)"
    exec bash scripts/train_gembench_vae_cache_4gpu.sh \
      max_steps="${GEMBENCH_PROFILE_STEPS}" save_every=0 save_final_checkpoint=false eval_every=0 log_every=1 \
      profile.enabled=true profile.warmup_steps=20 "$@"
    ;;
  vae_b4a1_zero1_tuned)
    export ACCELERATE_CONFIG_FILE=scripts/accelerate_configs/accelerate_zero1_tuned_ds.yaml
    export RUN_ID="bench_vae_b4a1_zero1_tuned_$(date +%Y%m%d_%H%M%S)"
    export TASK_NAME=gembench_keysteps_bbox_3cam224_vaecache_b4a1_1e-4
    exec bash scripts/train_gembench_vae_cache_4gpu.sh \
      max_steps="${GEMBENCH_PROFILE_STEPS}" save_every=0 save_final_checkpoint=false eval_every=0 log_every=1 \
      profile.enabled=true profile.warmup_steps=20 "$@"
    ;;
  vae_zero1_tuned)
    export ACCELERATE_CONFIG_FILE=scripts/accelerate_configs/accelerate_zero1_tuned_ds.yaml
    export RUN_ID="bench_vae_zero1_tuned_$(date +%Y%m%d_%H%M%S)"
    exec bash scripts/train_gembench_vae_cache_4gpu.sh \
      max_steps="${GEMBENCH_PROFILE_STEPS}" save_every=0 save_final_checkpoint=false eval_every=0 log_every=1 \
      profile.enabled=true profile.warmup_steps=20 "$@"
    ;;
  *)
    echo "usage: $0 {rgb_zero2|vae_zero2|vae_b4a1_zero2|vae_b4a1_zero2_tuned|vae_zero2_tuned|vae_b4a1_zero1_tuned|vae_zero1_tuned} [hydra_overrides...]" >&2
    exit 2
    ;;
esac
