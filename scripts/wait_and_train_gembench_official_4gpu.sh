#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
FASTWAM_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${FASTWAM_ROOT}"

CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3}"
FREE_GPU_MAX_USED_MB="${FREE_GPU_MAX_USED_MB:-10000}"
FREE_GPU_POLL_SECONDS="${FREE_GPU_POLL_SECONDS:-60}"
SKIP_GPU_WAIT="${SKIP_GPU_WAIT:-0}"

max_gpu_used_mb() {
  nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits \
    | python -c 'import sys; vals=[int(line.strip().split()[0]) for line in sys.stdin if line.strip()]; print(max(vals) if vals else 0)'
}

print_train_processes() {
  ps -eo pid,etime,pcpu,pmem,args \
    | grep -E "scripts/train.py|train_gembench|accelerate" \
    | grep -v grep \
    | head -20 || true
}

echo "[gembench-official-launch] cwd=${FASTWAM_ROOT}"
echo "[gembench-official-launch] cuda_visible_devices=${CUDA_VISIBLE_DEVICES}"
echo "[gembench-official-launch] free_gpu_max_used_mb=${FREE_GPU_MAX_USED_MB}"
echo "[gembench-official-launch] run_id=${RUN_ID:-auto}"

if [[ "${SKIP_GPU_WAIT}" != "1" ]]; then
  while true; do
    used="$(max_gpu_used_mb)"
    now="$(date '+%Y-%m-%d %H:%M:%S')"
    echo "[gembench-official-launch] ${now} max_gpu_used_mb=${used}"
    if (( used < FREE_GPU_MAX_USED_MB )); then
      break
    fi
    print_train_processes
    sleep "${FREE_GPU_POLL_SECONDS}"
  done
fi

echo "[gembench-official-launch] GPUs available; starting official training."
exec bash scripts/train_gembench_official_4gpu.sh "$@"
