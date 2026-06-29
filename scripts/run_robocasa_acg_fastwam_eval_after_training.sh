#!/usr/bin/env bash
set -euo pipefail

REPO_DIR="${FASTWAM_REPO:-/mnt/yuhan/FastWAM_robocasa_acg_8gpu}"
RUN_ID="${RUN_ID:-$(cat "${REPO_DIR}/logs/latest_robocasa_acg_fastwam_run_id.txt")}"
RUN_ROOT="${RUN_ROOT:-/mnt/yuhan/experiments/robocasa_acg_v1/fastwam/runs}"
EVAL_ROOT="${ROBOCASA_EVAL_ROOT:-/mnt/yuhan/experiments/robocasa_acg_v1/fastwam/eval}"
PYTHON="${PYTHON:-/opt/conda/envs/motus/bin/python}"
PLAN_PATH="${PLAN_PATH:-${REPO_DIR}/scripts/robocasa_acg_eval_plan_v1.json}"
EXPECTED_MIN_STEP="${EXPECTED_MIN_STEP:-50000}"
POLL_SECONDS="${POLL_SECONDS:-120}"
WAIT_FOR_TRAINING="${WAIT_FOR_TRAINING:-1}"
TRIALS_PER_TASK="${TRIALS_PER_TASK:-5}"
FASTWAM_DEVICE="${FASTWAM_DEVICE:-cuda:0}"
FASTWAM_INFER_STEPS="${FASTWAM_INFER_STEPS:-10}"
FASTWAM_REPLAN_STEPS="${FASTWAM_REPLAN_STEPS:-5}"
FASTWAM_RAND_DEVICE="${FASTWAM_RAND_DEVICE:-cpu}"
VIDEO_POLICY="${VIDEO_POLICY:-all}"
ROBOSUITE_REPO="${ROBOSUITE_REPO:-/mnt/yuhan/repos/robosuite}"
ROBOCASA_REPO="${ROBOCASA_REPO:-/mnt/yuhan/repos/robocasa}"

RUN_DIR="${RUN_ROOT}/${RUN_ID}"
LOG_DIR="${EVAL_ROOT}/logs"
mkdir -p "${LOG_DIR}"
WATCH_LOG="${LOG_DIR}/watch_${RUN_ID}.log"
exec > >(tee -a "${WATCH_LOG}") 2>&1

echo "[watch] started $(date -Is)"
echo "[watch] run_id=${RUN_ID}"
echo "[watch] run_dir=${RUN_DIR}"

latest_weights_checkpoint() {
  find "${RUN_DIR}/checkpoints/weights" -maxdepth 1 -type f -name 'step_*.pt' 2>/dev/null \
    | sort -V \
    | tail -1
}

checkpoint_step() {
  local path="$1"
  basename "${path}" | sed -E 's/^step_0*([0-9]+)\.pt$/\1/'
}

wait_for_training_exit() {
  if [[ "${WAIT_FOR_TRAINING}" != "1" ]]; then
    return
  fi
  while pgrep -af "scripts/train.py" | grep -F "${RUN_ID}" | grep -v grep >/dev/null; do
    ckpt="$(latest_weights_checkpoint || true)"
    step="none"
    if [[ -n "${ckpt}" ]]; then
      step="$(checkpoint_step "${ckpt}")"
    fi
    echo "[watch] training still running; latest_checkpoint_step=${step}; time=$(date -Is)"
    sleep "${POLL_SECONDS}"
  done
  echo "[watch] training process exited; time=$(date -Is)"
}

wait_for_training_exit

CKPT="$(latest_weights_checkpoint || true)"
if [[ -z "${CKPT}" ]]; then
  echo "[watch] no FastWAM weights checkpoint found under ${RUN_DIR}/checkpoints/weights" >&2
  exit 10
fi
STEP="$(checkpoint_step "${CKPT}")"
if (( STEP < EXPECTED_MIN_STEP )); then
  echo "[watch] latest checkpoint step ${STEP} < EXPECTED_MIN_STEP=${EXPECTED_MIN_STEP}; refusing eval" >&2
  exit 11
fi

bash "${REPO_DIR}/scripts/setup_robocasa_eval_env.sh"

export MUJOCO_GL="${MUJOCO_GL:-egl}"
export PYOPENGL_PLATFORM="${PYOPENGL_PLATFORM:-egl}"
export TOKENIZERS_PARALLELISM=false
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export PYTHONUNBUFFERED=1
export PYTHONPATH="${REPO_DIR}/src:${ROBOSUITE_REPO}:${ROBOCASA_REPO}:${PYTHONPATH:-}"

OUT_DIR="${EVAL_ROOT}/runs/${RUN_ID}_$(basename "${CKPT}" .pt)_stage1_$(date +%Y%m%d_%H%M%S)"
mkdir -p "${OUT_DIR}"
echo "${OUT_DIR}" > "${LOG_DIR}/latest_eval_output_dir.txt"

CMD=(
  "${PYTHON}" "${REPO_DIR}/scripts/robocasa_acg_eval.py"
  --policy-backend fastwam
  --plan "${PLAN_PATH}"
  --output-dir "${OUT_DIR}"
  --num-trials "${TRIALS_PER_TASK}"
  --fastwam-repo "${REPO_DIR}"
  --fastwam-checkpoint "${CKPT}"
  --fastwam-device "${FASTWAM_DEVICE}"
  --fastwam-num-inference-steps "${FASTWAM_INFER_STEPS}"
  --fastwam-rand-device "${FASTWAM_RAND_DEVICE}"
  --replan-steps "${FASTWAM_REPLAN_STEPS}"
  --video-policy "${VIDEO_POLICY}"
)

if [[ -n "${BUCKET:-}" ]]; then
  CMD+=(--bucket "${BUCKET}")
fi

echo "[eval] checkpoint=${CKPT}"
echo "[eval] output_dir=${OUT_DIR}"
"${CMD[@]}" 2>&1 | tee "${OUT_DIR}/eval_stdout.log"

find "${OUT_DIR}/videos" -type f -name "*.mp4" | sort > "${OUT_DIR}/video_files.txt" || true
if [[ -s "${OUT_DIR}/video_files.txt" ]]; then
  tar -czf "${OUT_DIR}/sample_videos_first20.tar.gz" -C "${OUT_DIR}" $(sed -n '1,20p' "${OUT_DIR}/video_files.txt" | sed "s#^${OUT_DIR}/##")
fi
date -Is > "${OUT_DIR}/DONE"
echo "[eval] done ${OUT_DIR}"
