#!/usr/bin/env bash
set -euo pipefail

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
RUN_ID="${1:-}"
LOG_ROOT="${LOG_ROOT:-${REPO_DIR}/logs}"
RUN_ROOT="${RUN_ROOT:-/mnt/yuhan/experiments/robocasa_acg_v1/fastwam/runs}"
LATEST_RUN_FILE="${LOG_ROOT}/latest_robocasa_acg_fastwam_run_id.txt"

if [[ -z "${RUN_ID}" ]]; then
  if [[ -f "${LATEST_RUN_FILE}" ]]; then
    RUN_ID="$(tr -d '[:space:]' < "${LATEST_RUN_FILE}")"
  else
    RUN_ID="$(ls -1t "${LOG_ROOT}"/robocasa_acg_v1_fastwam_8gpu_*.launcher.log 2>/dev/null | head -1 | sed 's#.*/##; s#\.launcher\.log$##')"
  fi
fi

LOG_FILE="${LOG_ROOT}/${RUN_ID}.launcher.log"
OUT_DIR="${RUN_ROOT}/${RUN_ID}"

echo "[status] repo=${REPO_DIR}"
echo "[status] run_id=${RUN_ID:-none}"
echo "[status] time=$(date -Is)"
nvidia-smi --query-gpu=index,name,memory.used,memory.total,utilization.gpu --format=csv,noheader || true
if [[ -n "${RUN_ID}" ]]; then
  echo "[status] launcher_log=${LOG_FILE}"
  echo "[status] output_dir=${OUT_DIR}"
  echo "[status] accelerate_launchers=$(pgrep -af "accelerate launch" | grep -F "${RUN_ID}" | grep -v grep | wc -l)"
  echo "[status] train_ranks_direct=$(ps -eo ppid,cmd | grep -F "${RUN_ID}" | grep -F -- "-u scripts/train.py" | grep -v grep | awk '$1 ~ /^[0-9]+$/ {parents[$1]++; total++} END{for (p in parents) if (parents[p] == 8) direct=8; print direct+0}')"
  echo "[status] train_processes_all=$(ps -eo cmd | grep -F "${RUN_ID}" | grep -F -- "scripts/train.py" | grep -v grep | wc -l)"
  if [[ -f "${LOG_FILE}" ]]; then
    echo "[status] latest_steps"
    grep -n "step=" "${LOG_FILE}" | tail -8 || true
    echo "[status] latest_loss=$(tail -120 "${LOG_FILE}" | grep "loss=" | tail -1 | xargs || true)"
    echo "[status] recent_errors"
    grep -E "Traceback|RuntimeError|CUDA|NaN| nan|ERROR|Exception|Killed|OOM" "${LOG_FILE}" | tail -20 || true
  else
    echo "[status] missing launcher log"
  fi
  if [[ -d "${OUT_DIR}/checkpoints" ]]; then
    echo "[status] checkpoint_files=$(find "${OUT_DIR}/checkpoints" -maxdepth 4 -type f | wc -l)"
    find "${OUT_DIR}/checkpoints" -maxdepth 2 -type d | sort | tail -10
  else
    echo "[status] checkpoint_files=0"
  fi
fi
