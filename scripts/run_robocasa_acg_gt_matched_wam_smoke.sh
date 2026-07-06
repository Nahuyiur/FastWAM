#!/usr/bin/env bash
set -euo pipefail

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
  cat <<'EOF'
Usage:
  RUN_ID=<run_id> [SPLIT=val] [NUM_SAMPLES=5] [DEVICE=cuda:0] \
    bash scripts/run_robocasa_acg_gt_matched_wam_smoke.sh

Optional filters:
  EPISODE_INDEX=1234[,5678]   restrict to RoboCasa365 episode_index values
  WINDOW_START=0[,32]         restrict to dataset window_start values

Output:
  /mnt/yuhan/experiments/robocasa_acg_v1/fastwam/gt_matched_wam/<RUN_ID>_<split>_gtmatched_<timestamp>

Protocol:
  dataset_window_gt_matched_wam. This is a GT-matched open-loop WAM diagnostic,
  not an online RoboCasa success-rate rollout.
EOF
  exit 0
fi

REPO_DIR="${FASTWAM_REPO:-/mnt/yuhan/FastWAM_robocasa_acg_8gpu}"
PYTHON="${PYTHON:-/opt/conda/envs/motus/bin/python}"
RUN_ID="${RUN_ID:-robocasa_acg_v1_fastwam_8gpu_20260629_214052_evalfix}"
RUN_DIR="${RUN_DIR:-/mnt/yuhan/experiments/robocasa_acg_v1/fastwam/runs/${RUN_ID}}"
SPLIT="${SPLIT:-val}"
NUM_SAMPLES="${NUM_SAMPLES:-5}"
NUM_INFERENCE_STEPS="${NUM_INFERENCE_STEPS:-10}"
VIDEO_FPS="${VIDEO_FPS:-8}"
SEED="${SEED:-47}"
DEVICE="${DEVICE:-cuda:0}"
ROOT="${ROBOCASA_GT_MATCHED_ROOT:-/mnt/yuhan/experiments/robocasa_acg_v1/fastwam/gt_matched_wam}"
TIMESTAMP="$(TZ=Asia/Shanghai date +%Y%m%d_%H%M%S)"
OUTPUT_DIR="${OUTPUT_DIR:-${ROOT}/${RUN_ID}_${SPLIT}_gtmatched_${TIMESTAMP}}"

cd "${REPO_DIR}"
mkdir -p "${ROOT}/logs" "${OUTPUT_DIR}"

if [[ -z "${CHECKPOINT:-}" ]]; then
  CHECKPOINT="$(ls -1 "${RUN_DIR}"/checkpoints/weights/step_*.pt | sort -V | tail -1)"
fi

args=(
  scripts/eval_robocasa_acg_open_loop_wam_smoke.py
  --run-dir "${RUN_DIR}"
  --checkpoint "${CHECKPOINT}"
  --output-dir "${OUTPUT_DIR}"
  --split "${SPLIT}"
  --num-samples "${NUM_SAMPLES}"
  --num-inference-steps "${NUM_INFERENCE_STEPS}"
  --video-fps "${VIDEO_FPS}"
  --seed "${SEED}"
  --device "${DEVICE}"
)

if [[ -n "${EPISODE_INDEX:-}" ]]; then
  IFS=',' read -r -a episode_indices <<< "${EPISODE_INDEX}"
  for episode_index in "${episode_indices[@]}"; do
    args+=(--episode-index "${episode_index}")
  done
fi

if [[ -n "${WINDOW_START:-}" ]]; then
  IFS=',' read -r -a window_starts <<< "${WINDOW_START}"
  for window_start in "${window_starts[@]}"; do
    args+=(--window-start "${window_start}")
  done
fi

echo "[gt-matched] run_dir=${RUN_DIR}"
echo "[gt-matched] checkpoint=${CHECKPOINT}"
echo "[gt-matched] output_dir=${OUTPUT_DIR}"
echo "[gt-matched] protocol=dataset_window_gt_matched_wam"

"${PYTHON}" "${args[@]}" 2>&1 | tee "${ROOT}/logs/$(basename "${OUTPUT_DIR}").log"
echo "${OUTPUT_DIR}" > "${ROOT}/logs/latest_gt_matched_output_dir.txt"
echo "[gt-matched] done ${OUTPUT_DIR}"
