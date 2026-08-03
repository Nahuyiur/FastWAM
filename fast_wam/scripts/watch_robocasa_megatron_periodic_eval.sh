#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT:-/mnt/yuhan/FastWAM_megatron_robocasa_webdataset}"
TRAIN_RUN="${TRAIN_RUN:-${ROOT}/outputs/robocasa_megatron_offline50k_4gpu_20260803}"
OUT_ROOT="${OUT_ROOT:-${ROOT}/outputs/robocasa_megatron_offline50k_4gpu_20260803_periodic_eval16}"
PLAN="${PLAN:-${ROOT}/fast_wam/configs/robocasa_periodic_eval_16.json}"
PYTHON="${PYTHON:-/mnt/yuhan/envs/motus-rebuilt-v2_10/bin/python}"
VAE="${VAE:-/mnt/yuhan/FastWAM/checkpoints/Wan-AI/Wan2.2-TI2V-5B/Wan2.2_VAE.pth}"
ROBOSUITE_REPO="${ROBOSUITE_REPO:-/mnt/yuhan/repos/robosuite}"
ROBOCASA_REPO="${ROBOCASA_REPO:-/mnt/yuhan/repos/robocasa}"
POLL_SECONDS="${POLL_SECONDS:-60}"
IDLE_CONFIRMATIONS="${IDLE_CONFIRMATIONS:-2}"
ALLOW_SHARED_GPUS="${ALLOW_SHARED_GPUS:-0}"
MIN_STEP="${MIN_STEP:-5000}"
MAX_STEP="${MAX_STEP:-50000}"
EXPECTED_EPISODES=16
WANDB_RUN_ID="${WANDB_RUN_ID:-robocasa_megatron_offline50k_4gpu_20260803_periodic_eval}"
RENDER_BACKEND="${RENDER_BACKEND:-egl}"
OSMESA_ROOT="${OSMESA_ROOT:-${ROOT}/.runtime/osmesa/root}"

mkdir -p "${OUT_ROOT}"
exec >> "${OUT_ROOT}/watcher.log" 2>&1
echo "[watcher] started $(date -Is) host=$(hostname) render_backend=${RENDER_BACKEND}"

export MUJOCO_GL="${RENDER_BACKEND}"
export PYOPENGL_PLATFORM="${RENDER_BACKEND}"
if [[ "${RENDER_BACKEND}" == "osmesa" ]]; then
  OSMESA_LIB="${OSMESA_ROOT}/usr/lib/x86_64-linux-gnu"
  [[ -s "${OSMESA_LIB}/libOSMesa.so.8" ]] || {
    echo "[watcher] missing user-space OSMesa library: ${OSMESA_LIB}/libOSMesa.so.8"
    exit 2
  }
  export LD_LIBRARY_PATH="${OSMESA_LIB}:${LD_LIBRARY_PATH:-}"
fi
export TOKENIZERS_PARALLELISM=false
export PYTHONUNBUFFERED=1
export PYTHONPATH="${ROOT}/src:${ROBOSUITE_REPO}:${ROBOCASA_REPO}:${PYTHONPATH:-}"

gpu_compute_busy() {
  nvidia-smi --query-compute-apps=pid --format=csv,noheader,nounits 2>/dev/null | grep -Eq '[0-9]'
}

wait_for_idle_gpus() {
  if [[ "${ALLOW_SHARED_GPUS}" == "1" ]]; then
    echo "[watcher] shared GPU mode enabled; existing compute processes are not stopped"
    nvidia-smi --query-gpu=index,memory.used,memory.free,utilization.gpu \
      --format=csv,noheader,nounits || true
    return 0
  fi
  local consecutive=0
  while (( consecutive < IDLE_CONFIRMATIONS )); do
    if gpu_compute_busy; then
      consecutive=0
      echo "[watcher] waiting for all four eval GPUs; time=$(date -Is)"
    else
      consecutive=$((consecutive + 1))
      echo "[watcher] idle confirmation ${consecutive}/${IDLE_CONFIRMATIONS}; time=$(date -Is)"
    fi
    if (( consecutive < IDLE_CONFIRMATIONS )); then
      sleep "${POLL_SECONDS}"
    fi
  done
}

checkpoint_ready() {
  local checkpoint="$1"
  [[ -s "${checkpoint}/common.pt" ]] || return 1
  [[ -s "${checkpoint}/.metadata" ]] || return 1
  [[ $(find "${checkpoint}" -maxdepth 1 -type f -name '__*_0.distcp' | wc -l) -eq 4 ]]
}

run_checkpoint() {
  local checkpoint="$1"
  local step="$2"
  local output="${OUT_ROOT}/step_$(printf '%06d' "${step}")"
  local pids=()
  local failed=0
  mkdir -p "${output}/logs"

  echo "[watcher] evaluating step=${step} checkpoint=${checkpoint} time=$(date -Is)"
  for shard in 0 1 2 3; do
    local shard_dir="${output}/shard_$(printf '%02d' "${shard}")"
    mkdir -p "${shard_dir}"
    (
      export CUDA_VISIBLE_DEVICES="${shard}"
      export MASTER_ADDR=127.0.0.1
      export MASTER_PORT=$((29720 + shard))
      "${PYTHON}" "${ROOT}/scripts/robocasa_acg_eval.py" \
        --policy-backend fastwam_megatron \
        --action-layout base_first \
        --plan "${PLAN}" \
        --bucket "periodic_shard_${shard}" \
        --output-dir "${shard_dir}" \
        --seed 7 \
        --replan-steps 32 \
        --render-every 2 \
        --video-policy all \
        --fastwam-repo "${ROOT}" \
        --fastwam-checkpoint "${checkpoint}" \
        --fastwam-vae-checkpoint "${VAE}" \
        --fastwam-device cuda:0 \
        --fastwam-mixed-precision bf16 \
        --fastwam-num-video-frames 9 \
        --fastwam-action-horizon 32 \
        --fastwam-num-inference-steps 20 \
        --fastwam-action-dim 12 \
        --fastwam-proprio-dim 16
    ) > "${output}/logs/shard_$(printf '%02d' "${shard}").log" 2>&1 &
    pids+=("$!")
  done

  for pid in "${pids[@]}"; do
    if ! wait "${pid}"; then
      failed=1
    fi
  done
  if [[ "${failed}" -ne 0 ]]; then
    date -Is > "${output}/WORKER_FAILED"
    return 1
  fi

  if ! "${PYTHON}" "${ROOT}/fast_wam/scripts/summarize_robocasa_periodic_eval.py" \
    --root "${output}" \
    --checkpoint-step "${step}" \
    --expected-episodes "${EXPECTED_EPISODES}" \
    --wandb-entity ruiyuhan0110-southern-california-edison \
    --wandb-project robocasa-acg-fastwam \
    --wandb-run-id "${WANDB_RUN_ID}" \
    > "${output}/summary.log" 2>&1; then
    date -Is > "${output}/VALIDATION_FAILED"
    return 1
  fi
  date -Is > "${output}/DONE"
  echo "[watcher] completed step=${step} time=$(date -Is)"
}

while true; do
  found=0
  while IFS= read -r checkpoint; do
    found=1
    name="$(basename "${checkpoint}")"
    step=$((10#${name#iter_}))
    if (( step < MIN_STEP || step > MAX_STEP )); then
      continue
    fi
    output="${OUT_ROOT}/step_$(printf '%06d' "${step}")"
    if [[ -f "${output}/DONE" ]]; then
      continue
    fi
    if ! checkpoint_ready "${checkpoint}"; then
      continue
    fi
    wait_for_idle_gpus
    run_checkpoint "${checkpoint}" "${step}"
  done < <(find "${TRAIN_RUN}" -maxdepth 1 -type d -name 'iter_[0-9]*' | sort -V)

  if [[ -f "${TRAIN_RUN}/DONE" && -f "${OUT_ROOT}/step_050000/DONE" ]]; then
    date -Is > "${OUT_ROOT}/DONE"
    exit 0
  fi
  if [[ "${found}" -eq 0 ]]; then
    echo "[watcher] waiting for checkpoints; time=$(date -Is)"
  fi
  sleep "${POLL_SECONDS}"
done
