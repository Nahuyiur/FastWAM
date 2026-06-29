#!/usr/bin/env bash
set -euo pipefail

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${REPO_DIR}"

export PATH="${CONDA_BIN:-/opt/conda/envs/motus/bin}:$PATH"
export PYTHON="${PYTHON:-/opt/conda/envs/motus/bin/python}"
export PYTHONPATH="${REPO_DIR}/src:${PYTHONPATH:-}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}"
export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"
export WANDB_PROJECT="${WANDB_PROJECT:-robocasa-acg-fastwam}"
export WANDB_MODE="${WANDB_MODE:-online}"
export DIFFSYNTH_DOWNLOAD_SOURCE="${DIFFSYNTH_DOWNLOAD_SOURCE:-huggingface}"
export HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"

DATA_ROOT="${DATA_ROOT:-/mnt/pub_dataset/RoboCasa365}"
MANIFEST="${MANIFEST:-${DATA_ROOT}/splits/robocasa_acg_v1_episode_manifest.csv}"
RUN_ROOT="${RUN_ROOT:-/mnt/yuhan/experiments/robocasa_acg_v1/fastwam/runs}"
CACHE_ROOT="${CACHE_ROOT:-/mnt/yuhan/experiments/robocasa_acg_v1/fastwam/cache}"
LOG_ROOT="${LOG_ROOT:-${REPO_DIR}/logs}"
NORM_STATS="${NORM_STATS:-${CACHE_ROOT}/norm_stats/robocasa_acg_v1_train_id_dataset_stats.json}"
TEXT_CACHE="${TEXT_CACHE:-${CACHE_ROOT}/text_embeds/robocasa_acg_v1}"
RUN_ID="${RUN_ID:-robocasa_acg_v1_fastwam_8gpu_$(date +%Y%m%d_%H%M%S)}"
NPROC="${NPROC:-8}"

mkdir -p "${RUN_ROOT}" "${CACHE_ROOT}/norm_stats" "${TEXT_CACHE}" "${LOG_ROOT}"

echo "[preflight] repo=${REPO_DIR}"
echo "[preflight] run_id=${RUN_ID}"
echo "[preflight] python=$("${PYTHON}" -c 'import sys; print(sys.executable)')"
echo "[preflight] cuda_visible_devices=${CUDA_VISIBLE_DEVICES}"
echo "[preflight] data_root=${DATA_ROOT}"

"${PYTHON}" scripts/audit_robocasa_acg_split.py \
  --root "${DATA_ROOT}" \
  --manifest "${MANIFEST}" \
  --strict-counts \
  --json-out "${LOG_ROOT}/${RUN_ID}.split_audit.json"

if [[ ! -s "${NORM_STATS}" ]]; then
  "${PYTHON}" scripts/precompute_robocasa_norm_stats.py \
    --data-root "${DATA_ROOT}" \
    --manifest "${MANIFEST}" \
    --split train_id \
    --output "${NORM_STATS}"
else
  echo "[preflight] norm stats already exists: ${NORM_STATS}"
fi

"${PYTHON}" scripts/precompute_robocasa_text_embeds.py \
  --manifest "${MANIFEST}" \
  --splits train_id,val_id \
  --repos robocasa365-pretrain-atomic \
  --cache-dir "${TEXT_CACHE}" \
  --summary-json "${LOG_ROOT}/${RUN_ID}.text_cache.json"

"${PYTHON}" scripts/smoke_robocasa_acg_dataset.py \
  --require-text-cache \
  data.train.pretrained_norm_stats="${NORM_STATS}" \
  data.val.pretrained_norm_stats="${NORM_STATS}" \
  data.train.text_embedding_cache_dir="${TEXT_CACHE}" \
  data.val.text_embedding_cache_dir="${TEXT_CACHE}" \
  > "${LOG_ROOT}/${RUN_ID}.dataset_smoke.json"

export RUN_ID
exec bash scripts/train_zero1.sh "${NPROC}" \
  task=robocasa_acg_v1_fastwam_8gpu \
  output_dir="${RUN_ROOT}/${RUN_ID}" \
  data.train.dataset_dirs="[${DATA_ROOT}/repos/robocasa365-pretrain-atomic]" \
  data.val.dataset_dirs="[${DATA_ROOT}/repos/robocasa365-pretrain-atomic]" \
  data.train.episode_manifest_path="${MANIFEST}" \
  data.val.episode_manifest_path="${MANIFEST}" \
  data.train.pretrained_norm_stats="${NORM_STATS}" \
  data.val.pretrained_norm_stats="${NORM_STATS}" \
  data.train.text_embedding_cache_dir="${TEXT_CACHE}" \
  data.val.text_embedding_cache_dir="${TEXT_CACHE}" \
  wandb.project="${WANDB_PROJECT}" \
  wandb.name="${RUN_ID}" \
  wandb.group="robocasa-acg-v1-fastwam" \
  wandb.mode="${WANDB_MODE}"
