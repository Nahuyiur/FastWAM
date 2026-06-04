#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
FASTWAM_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${FASTWAM_ROOT}"
source scripts/setup_yuhan_paths.sh

CANDIDATE="${1:-vae_zero2}"
shift || true
PYTHON_BIN="${PYTHON_BIN:-${FASTWAM_CONDA_ENV}/bin/python}"
CACHE_DIR="${GEMBENCH_VAE_CACHE_DIR:-/mnt/yuhan/datasets/GEMBench/fastwam_cache/vae_latents/keysteps_bbox_seed0_3cam224x672_t9_v1}"
VERIFY_SAMPLES="${GEMBENCH_GATE_VERIFY_SAMPLES:-4}"
PROFILE_STEPS="${GEMBENCH_PROFILE_STEPS:-50}"
WARMUP_STEPS="${GEMBENCH_PROFILE_WARMUP_STEPS:-20}"
RUN_TS="$(date +%Y%m%d_%H%M%S)"
GATE_DIR="runs/gembench_acceleration_gates/${CANDIDATE}_${RUN_TS}"
mkdir -p "${GATE_DIR}"
ACCUM_PARITY_REQUIRED=0

case "${CANDIDATE}" in
  rgb_zero2)
    TASK_NAME="gembench_keysteps_bbox_3cam224_1e-4"
    RUN_TASK_DIR="${TASK_NAME}"
    TRAIN_CMD=(bash scripts/train_gembench_4gpu.sh task="${TASK_NAME}")
    CACHE_REQUIRED=0
    ;;
  vae_zero2)
    TASK_NAME="gembench_keysteps_bbox_3cam224_vaecache_1e-4"
    RUN_TASK_DIR="${TASK_NAME}"
    TRAIN_CMD=(bash scripts/train_gembench_vae_cache_4gpu.sh)
    CACHE_REQUIRED=1
    ;;
  vae_b4a1_zero2)
    TASK_NAME="gembench_keysteps_bbox_3cam224_vaecache_b4a1_1e-4"
    RUN_TASK_DIR="${TASK_NAME}"
    TRAIN_CMD=(bash scripts/train_gembench_vae_cache_4gpu.sh)
    CACHE_REQUIRED=1
    ACCUM_PARITY_REQUIRED=1
    ;;
  vae_b4a1_zero2_tuned)
    TASK_NAME="gembench_keysteps_bbox_3cam224_vaecache_b4a1_1e-4"
    RUN_TASK_DIR="${TASK_NAME}"
    export ACCELERATE_CONFIG_FILE=scripts/accelerate_configs/accelerate_zero2_tuned_ds.yaml
    TRAIN_CMD=(bash scripts/train_gembench_vae_cache_4gpu.sh)
    CACHE_REQUIRED=1
    ACCUM_PARITY_REQUIRED=1
    ;;
  vae_zero2_tuned)
    TASK_NAME="gembench_keysteps_bbox_3cam224_vaecache_1e-4"
    RUN_TASK_DIR="${TASK_NAME}"
    export ACCELERATE_CONFIG_FILE=scripts/accelerate_configs/accelerate_zero2_tuned_ds.yaml
    TRAIN_CMD=(bash scripts/train_gembench_vae_cache_4gpu.sh)
    CACHE_REQUIRED=1
    ;;
  vae_b4a1_zero1_tuned)
    TASK_NAME="gembench_keysteps_bbox_3cam224_vaecache_b4a1_1e-4"
    RUN_TASK_DIR="${TASK_NAME}"
    export ACCELERATE_CONFIG_FILE=scripts/accelerate_configs/accelerate_zero1_tuned_ds.yaml
    TRAIN_CMD=(bash scripts/train_gembench_vae_cache_4gpu.sh)
    CACHE_REQUIRED=1
    ACCUM_PARITY_REQUIRED=1
    ;;
  vae_zero1_tuned)
    TASK_NAME="gembench_keysteps_bbox_3cam224_vaecache_1e-4"
    RUN_TASK_DIR="${TASK_NAME}"
    export ACCELERATE_CONFIG_FILE=scripts/accelerate_configs/accelerate_zero1_tuned_ds.yaml
    TRAIN_CMD=(bash scripts/train_gembench_vae_cache_4gpu.sh)
    CACHE_REQUIRED=1
    ;;
  *)
    echo "usage: $0 {rgb_zero2|vae_zero2|vae_b4a1_zero2|vae_b4a1_zero2_tuned|vae_zero2_tuned|vae_b4a1_zero1_tuned|vae_zero1_tuned} [hydra_overrides...]" >&2
    exit 2
    ;;
esac

if [[ "${CACHE_REQUIRED}" == "1" ]]; then
  echo "[gate] verifying VAE cache: ${CACHE_DIR}"
  "${PYTHON_BIN}" scripts/verify_gembench_vae_cache.py \
    --root "${GEMBENCH_ROOT}" \
    --vae-cache-dir "${CACHE_DIR}" \
    --samples "${VERIFY_SAMPLES}" \
    --latent-atol "${GEMBENCH_GATE_LATENT_ATOL:-1e-3}" \
    --json-output "${GATE_DIR}/vae_cache.json" \
    --markdown-output "${GATE_DIR}/vae_cache.md"

  echo "[gate] checking loss/grad parity before benchmark"
  "${PYTHON_BIN}" scripts/check_gembench_loss_grad_parity.py \
    --batch-size "${GEMBENCH_GATE_PARITY_BATCH_SIZE:-1}" \
    --seed "${GEMBENCH_GATE_PARITY_SEED:-1234}" \
    --backward \
    --json-output "${GATE_DIR}/loss_grad_parity.json" \
    --markdown-output "${GATE_DIR}/loss_grad_parity.md"
fi

if [[ "${ACCUM_PARITY_REQUIRED}" == "1" ]]; then
  echo "[gate] checking batch-regroup accumulation parity before benchmark"
  "${PYTHON_BIN}" scripts/check_gembench_accumulation_parity.py \
    --task "${TASK_NAME}" \
    --effective-batch-size "${GEMBENCH_GATE_ACCUM_EFFECTIVE_BATCH_SIZE:-4}" \
    --micro-batch-size "${GEMBENCH_GATE_ACCUM_MICRO_BATCH_SIZE:-2}" \
    --seed "${GEMBENCH_GATE_PARITY_SEED:-1234}" \
    --json-output "${GATE_DIR}/accumulation_parity.json" \
    --markdown-output "${GATE_DIR}/accumulation_parity.md"
fi

export TASK_NAME
export RUN_ID="gate_${CANDIDATE}_${RUN_TS}"
export GEMBENCH_PROFILE_STEPS="${PROFILE_STEPS}"
export VERIFY_GEMBENCH_CONTRACT="${VERIFY_GEMBENCH_CONTRACT:-0}"
export PRECOMPUTE_GEMBENCH_TEXT="${PRECOMPUTE_GEMBENCH_TEXT:-0}"
export VERIFY_GEMBENCH_VAE_CACHE="${VERIFY_GEMBENCH_VAE_CACHE:-0}"
export WANDB_ENABLED="${WANDB_ENABLED:-false}"
if [[ "${WANDB_ENABLED}" == "false" ]]; then
  export DISABLE_WANDB="${DISABLE_WANDB:-1}"
fi

PROFILE_OVERRIDES=(
  max_steps="${PROFILE_STEPS}"
  save_every=0
  save_final_checkpoint=false
  eval_every=0
  log_every=1
  profile.enabled=true
  profile.warmup_steps="${WARMUP_STEPS}"
)

echo "[gate] running candidate=${CANDIDATE} task=${TASK_NAME} run_id=${RUN_ID} profile_steps=${PROFILE_STEPS}"
"${TRAIN_CMD[@]}" "${PROFILE_OVERRIDES[@]}" "$@" 2>&1 | tee "${GATE_DIR}/train.log"

PROFILE_JSONL="runs/${RUN_TASK_DIR}/${RUN_ID}/profile/step_times.jsonl"
if [[ ! -f "${PROFILE_JSONL}" ]]; then
  echo "[gate] profile output missing: ${PROFILE_JSONL}" >&2
  exit 1
fi
"${PYTHON_BIN}" scripts/summarize_gembench_profile.py "${PROFILE_JSONL}" \
  --json-output "${GATE_DIR}/profile_summary.json" | tee "${GATE_DIR}/profile_summary.txt"

echo "[gate] complete candidate=${CANDIDATE} gate_dir=${GATE_DIR} profile=${PROFILE_JSONL}"
