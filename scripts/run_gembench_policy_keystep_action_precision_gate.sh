#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
FASTWAM_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${FASTWAM_ROOT}"
source scripts/setup_yuhan_paths.sh

RUN_DIR="${1:-${FASTWAM_POLICY_ACTION_DIAG_RUN_DIR:-}}"
CHECKPOINT="${2:-${FASTWAM_POLICY_ACTION_DIAG_CHECKPOINT:-}}"
if [[ -z "${RUN_DIR}" || -z "${CHECKPOINT}" ]]; then
  cat >&2 <<'EOF'
Usage:
  scripts/run_gembench_policy_keystep_action_precision_gate.sh <run_dir> <checkpoint.pt>

Or set:
  FASTWAM_POLICY_ACTION_DIAG_RUN_DIR=/path/to/run
  FASTWAM_POLICY_ACTION_DIAG_CHECKPOINT=/path/to/step_xxxxxx.pt

This is a checkpoint-promotion gate. It compares model-predicted actions with
GT next-key actions on the same GT key states and exits non-zero when the
precision gate fails. It is not an official GEMBench score.
EOF
  exit 2
fi

if [[ ! -d "${RUN_DIR}" ]]; then
  echo "[policy-action-gate] missing run_dir: ${RUN_DIR}" >&2
  exit 1
fi
if [[ ! -f "${CHECKPOINT}" ]]; then
  echo "[policy-action-gate] missing checkpoint: ${CHECKPOINT}" >&2
  exit 1
fi

PYTHON_BIN="${FASTWAM_CONDA_ENV}/bin/python"
if [[ ! -x "${PYTHON_BIN}" ]]; then
  echo "[policy-action-gate] missing python: ${PYTHON_BIN}" >&2
  exit 1
fi

export GEMBENCH_9V32_4CAM_MANIFEST="${GEMBENCH_9V32_4CAM_MANIFEST:-${GEMBENCH_ROOT}/fastwam_cache/microsteps_9v32_4cam224_manifest.json}"
export GEMBENCH_KEY_FRAMEIDS_CACHE="${GEMBENCH_KEY_FRAMEIDS_CACHE:-${GEMBENCH_ROOT}/fastwam_cache/microsteps_9v32_seed0_key_frameids.json}"

RUN_NAME="$(basename "${RUN_DIR}")"
CKPT_NAME="$(basename "${CHECKPOINT}" .pt)"
OUTPUT_ROOT="${FASTWAM_POLICY_ACTION_DIAG_OUTPUT_ROOT:-runs/gembench_policy_keystep_action_diagnostics/${RUN_NAME}_${CKPT_NAME}_$(date +%Y%m%d_%H%M%S)}"
mkdir -p "${OUTPUT_ROOT}" logs

DEVICE="${FASTWAM_POLICY_ACTION_DIAG_DEVICE:-cuda:0}"
MAX_TRIALS="${FASTWAM_POLICY_ACTION_DIAG_MAX_TRIALS:-4}"
MAX_KEY_TRANSITIONS="${FASTWAM_POLICY_ACTION_DIAG_MAX_KEY_TRANSITIONS:-8}"
NUM_INFERENCE_STEPS="${FASTWAM_POLICY_ACTION_DIAG_NUM_INFERENCE_STEPS:-10}"
MIXED_PRECISION="${FASTWAM_POLICY_ACTION_DIAG_MIXED_PRECISION:-bf16}"
OFFICIAL_LOCAL_NUM_POINTS="${FASTWAM_POLICY_ACTION_DIAG_OFFICIAL_LOCAL_NUM_POINTS:-${FASTWAM_GEMBENCH_POLICY_KEYSTEP_LOCAL_NUM_POINTS:-0}}"
OFFICIAL_LOCAL_VOXEL_SIZE="${FASTWAM_POLICY_ACTION_DIAG_OFFICIAL_LOCAL_VOXEL_SIZE:-${FASTWAM_GEMBENCH_POLICY_KEYSTEP_LOCAL_TRAIN_VOXEL_SIZE:-0.0}}"

CMD=(
  "${PYTHON_BIN}" scripts/diagnose_gembench_policy_keystep_action_error.py
  --run-dir "${RUN_DIR}"
  --checkpoint "${CHECKPOINT}"
  --manifest "${GEMBENCH_9V32_4CAM_MANIFEST}"
  --key-frameids-path "${GEMBENCH_KEY_FRAMEIDS_CACHE}"
  --gembench-root "${GEMBENCH_ROOT}"
  --robot-3dlotus-root "${ROBOT_3DLOTUS_ROOT}"
  --device "${DEVICE}"
  --mixed-precision "${MIXED_PRECISION}"
  --num-inference-steps "${NUM_INFERENCE_STEPS}"
  --max-trials "${MAX_TRIALS}"
  --max-key-transitions "${MAX_KEY_TRANSITIONS}"
  --official-local-num-points "${OFFICIAL_LOCAL_NUM_POINTS}"
  --official-local-voxel-size "${OFFICIAL_LOCAL_VOXEL_SIZE}"
  --fail-on-gate
  --output-root "${OUTPUT_ROOT}"
)

if [[ "${FASTWAM_POLICY_ACTION_DIAG_PROBE_MOVER:-1}" == "0" ]]; then
  CMD+=(--no-probe-predicted-mover)
fi

echo "[policy-action-gate] run_dir=${RUN_DIR}"
echo "[policy-action-gate] checkpoint=${CHECKPOINT}"
echo "[policy-action-gate] output_root=${OUTPUT_ROOT}"
echo "[policy-action-gate] device=${DEVICE} max_trials=${MAX_TRIALS} max_key_transitions=${MAX_KEY_TRANSITIONS}"

if [[ "${FASTWAM_POLICY_ACTION_DIAG_XVFB:-1}" == "1" ]]; then
  CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}" xvfb-run -a -s "-screen 0 1280x1024x24 +extension GLX +render -noreset" "${CMD[@]}"
else
  CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}" "${CMD[@]}"
fi

echo "[policy-action-gate] passed output_root=${OUTPUT_ROOT}"
