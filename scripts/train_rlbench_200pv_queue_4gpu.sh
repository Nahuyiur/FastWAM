#!/usr/bin/env bash
set -euo pipefail

ROOT="/mnt/world_foundational_model/yuhan/FastWAM"
MEM_THRESHOLD_MIB="${MEM_THRESHOLD_MIB:-40000}"
POLL_SECONDS="${POLL_SECONDS:-300}"
RUN_ROOT="${RUN_ROOT:-${ROOT}/runs/rlbench_200pv_train_queue/$(date +%Y-%m-%d_%H-%M-%S)}"

TASKS=(
  "rlbench_original_3cam224_1e-4"
  "rlbench_color_3cam224_1e-4"
  "rlbench_shape_3cam224_1e-4"
  "rlbench_color_shape_3cam224_1e-4"
)

mkdir -p "${RUN_ROOT}"
cd "${ROOT}"

log() {
  printf '[%s] %s\n' "$(date '+%Y-%m-%d %H:%M:%S')" "$*" | tee -a "${RUN_ROOT}/queue.log"
}

gpu_mem_used_csv() {
  nvidia-smi --query-gpu=index,memory.used,memory.total,utilization.gpu --format=csv,noheader,nounits
}

gpus_ready() {
  while IFS=',' read -r idx used total util; do
    idx="${idx//[[:space:]]/}"
    used="${used//[[:space:]]/}"
    total="${total//[[:space:]]/}"
    util="${util//[[:space:]]/}"
    if (( used >= MEM_THRESHOLD_MIB )); then
      return 1
    fi
  done < <(gpu_mem_used_csv)
  return 0
}

wait_for_gpus() {
  local task="$1"
  while true; do
    log "GPU status before ${task}:"
    gpu_mem_used_csv | tee -a "${RUN_ROOT}/queue.log"
    if gpus_ready; then
      log "All GPUs are below ${MEM_THRESHOLD_MIB} MiB. Starting ${task}."
      return 0
    fi
    log "Waiting ${POLL_SECONDS}s because at least one GPU is above ${MEM_THRESHOLD_MIB} MiB."
    sleep "${POLL_SECONDS}"
  done
}

log "RLBench 200-per-variant 4-GPU queue started."
log "Run root: ${RUN_ROOT}"
log "Memory threshold: ${MEM_THRESHOLD_MIB} MiB per GPU; poll interval: ${POLL_SECONDS}s."
log "Tasks: ${TASKS[*]}"

for task in "${TASKS[@]}"; do
  wait_for_gpus "${task}"
  task_log="${RUN_ROOT}/${task}.log"
  log "Launching ${task}; task log: ${task_log}"
  bash scripts/train_rlbench_4gpu.sh "${task}" 2>&1 | tee "${task_log}"
  log "Finished ${task}."
done

log "All RLBench 200-per-variant tasks finished."
