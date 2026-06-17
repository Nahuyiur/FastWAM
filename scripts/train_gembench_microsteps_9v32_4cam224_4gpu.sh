#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
FASTWAM_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${FASTWAM_ROOT}"
source scripts/setup_yuhan_paths.sh
if [[ -z "${__GEMBENCH_WANDB_PROJECT_USER_PROVIDED:-}" ]]; then
  if [[ -n "${WANDB_PROJECT:-}" ]]; then
    export __GEMBENCH_WANDB_PROJECT_USER_PROVIDED=1
  else
    export __GEMBENCH_WANDB_PROJECT_USER_PROVIDED=0
  fi
fi
export WANDB_PROJECT="${WANDB_PROJECT:-fastwam-gembench-9v32}"
source scripts/setup_gembench_wandb.sh

NPROC_PER_NODE="${NPROC_PER_NODE:-4}"
TASK_NAME="gembench_microsteps_9v32_4cam224_1e-4"
PYTHON_BIN="${FASTWAM_CONDA_ENV}/bin/python"
export PATH="${FASTWAM_CONDA_ENV}/bin:${PATH}"

RUN_ID="${RUN_ID:-fastwam_gembench_wam_9v32_4cam224_b4a1_$(date +%Y%m%d_%H%M%S)}"
export GEMBENCH_9V32_4CAM_MANIFEST="${GEMBENCH_9V32_4CAM_MANIFEST:-${GEMBENCH_ROOT}/fastwam_cache/microsteps_9v32_4cam224_manifest.json}"
export GEMBENCH_9V32_4CAM_RGB_CACHE_DIR="${GEMBENCH_9V32_4CAM_RGB_CACHE_DIR:-${GEMBENCH_ROOT}/fastwam_cache/microsteps_9v32_4cam224_rgb}"
export GEMBENCH_9V32_4CAM_VAE_CACHE_DIR="${GEMBENCH_9V32_4CAM_VAE_CACHE_DIR:-${GEMBENCH_ROOT}/fastwam_cache/vae_latents/microsteps_9v32_seed0_4cam224x896_t9_a32_v1}"
export GEMBENCH_9V32_MANIFEST="${GEMBENCH_9V32_4CAM_MANIFEST}"
export GEMBENCH_9V32_RGB_CACHE_DIR="${GEMBENCH_9V32_4CAM_RGB_CACHE_DIR}"
export GEMBENCH_9V32_VAE_CACHE_DIR="${GEMBENCH_9V32_4CAM_VAE_CACHE_DIR}"
AUDIT_DIR="${GEMBENCH_9V32_4CAM_AUDIT_DIR:-runs/gembench_microsteps_9v32_4cam224_audits/${RUN_ID}}"

if [[ ! -x "${PYTHON_BIN}" ]]; then
  echo "[gembench-9v32-4cam-train] missing python: ${PYTHON_BIN}" >&2
  exit 1
fi

if [[ ! -f "${GEMBENCH_ROOT}/train_dataset/microsteps.tar.gz" ]]; then
  echo "[gembench-9v32-4cam-train] missing train microsteps tar: ${GEMBENCH_ROOT}/train_dataset/microsteps.tar.gz" >&2
  exit 1
fi
if [[ ! -d "${GEMBENCH_ROOT}/train_dataset/keysteps_bbox/seed0" ]]; then
  echo "[gembench-9v32-4cam-train] missing train keysteps LMDB: ${GEMBENCH_ROOT}/train_dataset/keysteps_bbox/seed0" >&2
  exit 1
fi
for arg in "$@"; do
  case "${arg}" in
    task=*|+task=*|++task=*|task.*=*|+task.*=*|++task.*=*|data=*|+data=*|++data=*|data.*=*|+data.*=*|++data.*=*)
      echo "[gembench-9v32-4cam-train] refusing override that could bypass the 4cam 9V/32A dataset contract: ${arg}" >&2
      exit 1
      ;;
  esac
done

mkdir -p "${AUDIT_DIR}" logs

if [[ ! -f "${GEMBENCH_9V32_4CAM_MANIFEST}" ]]; then
  PYTHONPATH=src "${PYTHON_BIN}" scripts/audit_gembench_microsteps_9v32_contract.py \
    --root "${GEMBENCH_ROOT}" \
    --rgb-cache-dir "${GEMBENCH_9V32_4CAM_RGB_CACHE_DIR}" \
    --length-source key_frameids \
    --official-camera-order \
    --official-cache-camera-order \
    --image-size 224 \
    --output-json "${GEMBENCH_9V32_4CAM_MANIFEST}" \
    --output-md "${AUDIT_DIR}/microsteps_9v32_4cam224_contract.md"
fi

PYTHONPATH=src "${PYTHON_BIN}" scripts/audit_gembench_microsteps_9v32_cache.py \
  --manifest "${GEMBENCH_9V32_4CAM_MANIFEST}" \
  --rgb-cache-dir "${GEMBENCH_9V32_4CAM_RGB_CACHE_DIR}" \
  --output-json "${AUDIT_DIR}/microsteps_9v32_4cam224_cache.json" \
  --output-md "${AUDIT_DIR}/microsteps_9v32_4cam224_cache.md"

CACHE_DIR="${GEMBENCH_TEXT_EMBED_CACHE:-data/text_embeds_cache/gembench_microsteps_9v32}"
export DIFFSYNTH_DOWNLOAD_SOURCE="${DIFFSYNTH_DOWNLOAD_SOURCE:-huggingface}"
export HF_HUB_DISABLE_XET="${HF_HUB_DISABLE_XET:-1}"
if [[ -n "${FASTWAM_PROXY:-}" && -z "${http_proxy:-}${https_proxy:-}${HTTP_PROXY:-}${HTTPS_PROXY:-}" ]]; then
  export http_proxy="${FASTWAM_PROXY}"
  export https_proxy="${FASTWAM_PROXY}"
  export HTTP_PROXY="${FASTWAM_PROXY}"
  export HTTPS_PROXY="${FASTWAM_PROXY}"
fi
if [[ "${PRECOMPUTE_GEMBENCH_TEXT:-1}" == "1" ]]; then
  "${PYTHON_BIN}" scripts/precompute_gembench_text_embeds.py \
    --no-redirect-common-files \
    --cache-dir "${CACHE_DIR}" \
    --text-encoder-id umt5_xxl
fi

export ACCELERATE_CONFIG_FILE="${ACCELERATE_CONFIG_FILE:-scripts/accelerate_configs/accelerate_zero2_ds.yaml}"
WANDB_SUBPROJECT_DEFAULT="${WANDB_SUBPROJECT:-fastwam-gembench-wam-9v32-4cam224}"
WANDB_RUN_NAME_DEFAULT="${WANDB_RUN_NAME:-${RUN_ID}}"
WANDB_GROUP_DEFAULT="${WANDB_GROUP:-fastwam-gembench-wam-9v32-4cam224-b4a1}"
mapfile -t WANDB_OVERRIDES < <(gembench_wandb_hydra_overrides "${WANDB_SUBPROJECT_DEFAULT}" "${WANDB_RUN_NAME_DEFAULT}" "${WANDB_GROUP_DEFAULT}")
mapfile -t WANDB_OVERRIDES < <(gembench_filter_wandb_hydra_overrides "${WANDB_OVERRIDES[@]}" -- "$@")

TRAIN_OVERRIDES=()
add_hydra_override_from_env() {
  local env_name="$1"
  local hydra_key="$2"
  local value="${!env_name:-}"
  if [[ -n "${value}" ]]; then
    TRAIN_OVERRIDES+=("${hydra_key}=${value}")
  fi
}

add_hydra_override_from_env FASTWAM_RESUME_STATE checkpoint.resume_from
add_hydra_override_from_env FASTWAM_INIT_WEIGHTS checkpoint.init_from_weights
add_hydra_override_from_env FASTWAM_LOAD_STEP_FROM_WEIGHTS checkpoint.load_step_from_weights
add_hydra_override_from_env FASTWAM_INITIAL_STEP checkpoint.initial_step
add_hydra_override_from_env FASTWAM_ADVANCE_SCHEDULER_TO_STEP checkpoint.advance_scheduler_to_step
add_hydra_override_from_env FASTWAM_MAX_STEPS max_steps
add_hydra_override_from_env FASTWAM_LEARNING_RATE learning_rate
add_hydra_override_from_env FASTWAM_BATCH_SIZE batch_size
add_hydra_override_from_env FASTWAM_GRAD_ACCUM gradient_accumulation_steps
add_hydra_override_from_env FASTWAM_SAVE_EVERY save_every
add_hydra_override_from_env FASTWAM_EVAL_EVERY eval_every
add_hydra_override_from_env FASTWAM_OPEN_LOOP_WAM_EVAL_ENABLED open_loop_wam_eval.enabled
add_hydra_override_from_env FASTWAM_OPEN_LOOP_WAM_EVAL_EVERY open_loop_wam_eval.every
add_hydra_override_from_env FASTWAM_OPEN_LOOP_WAM_EVAL_NUM_SAMPLES open_loop_wam_eval.num_samples
add_hydra_override_from_env FASTWAM_OPEN_LOOP_WAM_EVAL_ROLLOUT_CHUNKS open_loop_wam_eval.rollout_chunks
add_hydra_override_from_env FASTWAM_OPEN_LOOP_WAM_EVAL_CHUNK_STRIDE open_loop_wam_eval.chunk_stride
add_hydra_override_from_env FASTWAM_OPEN_LOOP_WAM_EVAL_NUM_INFERENCE_STEPS open_loop_wam_eval.num_inference_steps
add_hydra_override_from_env FASTWAM_OPEN_LOOP_WAM_EVAL_SAVE_VIDEO open_loop_wam_eval.save_video
add_hydra_override_from_env FASTWAM_OPEN_LOOP_WAM_EVAL_MAX_WANDB_VIDEOS open_loop_wam_eval.max_wandb_videos
add_hydra_override_from_env FASTWAM_OUTPUT_DIR output_dir

echo "[gembench-9v32-4cam-train] run_id=${RUN_ID} task=${TASK_NAME} nproc=${NPROC_PER_NODE}"
echo "[gembench-9v32-4cam-train] manifest=${GEMBENCH_9V32_4CAM_MANIFEST}"
echo "[gembench-9v32-4cam-train] rgb_cache_dir=${GEMBENCH_9V32_4CAM_RGB_CACHE_DIR}"
if [[ -n "${GEMBENCH_9V32_4CAM_VAE_CACHE_DIR}" ]]; then
  if [[ ! -f "${GEMBENCH_9V32_4CAM_VAE_CACHE_DIR}/manifest.json" ]]; then
    echo "[gembench-9v32-4cam-train] missing VAE latent cache manifest: ${GEMBENCH_9V32_4CAM_VAE_CACHE_DIR}/manifest.json" >&2
    exit 1
  fi
  "${PYTHON_BIN}" - <<'PY'
import json
import os
from pathlib import Path

cache_dir = Path(os.environ["GEMBENCH_9V32_4CAM_VAE_CACHE_DIR"])
payload = json.loads((cache_dir / "manifest.json").read_text())
dataset = payload.get("dataset", {})
checks = {
    "cache_version": payload.get("cache_version") == "gembench_microsteps_9v32_vae_latents_v1",
    "complete": bool(payload.get("complete")),
    "action_horizon": dataset.get("action_horizon") == 32,
    "video_size": dataset.get("video_size") == [224, 896],
    "camera_order": dataset.get("camera_order") == ["left_shoulder", "right_shoulder", "wrist", "front"],
    "cache_camera_order": dataset.get("cache_camera_order") == ["left_shoulder", "right_shoulder", "wrist", "front"],
}
failed = {key: value for key, value in checks.items() if not value}
if failed:
    raise SystemExit(f"invalid GEMBench 9V32 4cam VAE cache {cache_dir}: failed={failed}")
PY
  echo "[gembench-9v32-4cam-train] vae_latent_cache_dir=${GEMBENCH_9V32_4CAM_VAE_CACHE_DIR}"
fi
bash scripts/train_zero2.sh "${NPROC_PER_NODE}" "task=${TASK_NAME}" "${TRAIN_OVERRIDES[@]}" "$@" "${WANDB_OVERRIDES[@]}"
