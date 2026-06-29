#!/usr/bin/env bash
set -euo pipefail

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
RUN_ID="${1:-}"
LOG_ROOT="${LOG_ROOT:-${REPO_DIR}/logs}"
RUN_ROOT="${RUN_ROOT:-/mnt/yuhan/experiments/robocasa_acg_v1/fastwam/runs}"

if [[ -z "${RUN_ID}" ]]; then
  RUN_ID="$(ls -1t "${LOG_ROOT}"/robocasa_acg_v1_fastwam_8gpu_*.launcher.log 2>/dev/null | head -1 | sed 's#.*/##; s#\\.launcher\\.log$##')"
fi

echo "[status] repo=${REPO_DIR}"
echo "[status] run_id=${RUN_ID:-none}"
nvidia-smi --query-gpu=index,name,memory.used,memory.total,utilization.gpu --format=csv,noheader || true
if [[ -n "${RUN_ID}" ]]; then
  echo "[status] launcher_log=${LOG_ROOT}/${RUN_ID}.launcher.log"
  echo "[status] output_dir=${RUN_ROOT}/${RUN_ID}"
  pgrep -af "${RUN_ID}|scripts/train.py|accelerate launch" || true
  if [[ -f "${LOG_ROOT}/${RUN_ID}.launcher.log" ]]; then
    echo "[status] last_log_lines"
    tail -80 "${LOG_ROOT}/${RUN_ID}.launcher.log"
  fi
fi
