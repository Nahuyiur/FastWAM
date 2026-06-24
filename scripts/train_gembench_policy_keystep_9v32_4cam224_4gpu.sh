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
export WANDB_PROJECT="${WANDB_PROJECT:-fastwam-gembench-policy-keystep}"
source scripts/setup_gembench_wandb.sh

NPROC_PER_NODE="${NPROC_PER_NODE:-4}"
TASK_NAME="${FASTWAM_GEMBENCH_POLICY_KEYSTEP_TASK_NAME:-gembench_policy_keystep_9v32_4cam224_1e-4}"
PYTHON_BIN="${FASTWAM_CONDA_ENV}/bin/python"
export PATH="${FASTWAM_CONDA_ENV}/bin:${PATH}"

RUN_ID="${RUN_ID:-${FASTWAM_GEMBENCH_POLICY_KEYSTEP_RUN_PREFIX:-fastwam_gembench_policy_keystep_4cam224_wamaux9v32_b4a1}_$(date +%Y%m%d_%H%M%S)}"
export GEMBENCH_9V32_4CAM_MANIFEST="${GEMBENCH_9V32_4CAM_MANIFEST:-${GEMBENCH_ROOT}/fastwam_cache/microsteps_9v32_4cam224_manifest.json}"
export GEMBENCH_9V32_4CAM_RGB_CACHE_DIR="${GEMBENCH_9V32_4CAM_RGB_CACHE_DIR:-${GEMBENCH_ROOT}/fastwam_cache/microsteps_9v32_4cam224_rgb}"
export GEMBENCH_9V32_4CAM_VAE_CACHE_DIR="${GEMBENCH_9V32_4CAM_VAE_CACHE_DIR:-${GEMBENCH_ROOT}/fastwam_cache/vae_latents/microsteps_9v32_seed0_4cam224x896_t9_a32_v1}"
export GEMBENCH_KEYSTEPS_BBOX_DIR="${GEMBENCH_KEYSTEPS_BBOX_DIR:-${GEMBENCH_ROOT}/train_dataset/keysteps_bbox/seed0}"
export GEMBENCH_KEYSTEPS_BBOX_PCD_DIR="${GEMBENCH_KEYSTEPS_BBOX_PCD_DIR:-${GEMBENCH_ROOT}/train_dataset/keysteps_bbox_pcd/seed0/voxel1cm}"
export GEMBENCH_KEY_FRAMEIDS_CACHE="${GEMBENCH_KEY_FRAMEIDS_CACHE:-${GEMBENCH_ROOT}/fastwam_cache/microsteps_9v32_seed0_key_frameids.json}"
export GEMBENCH_POLICY_KEYSTEP_NORM_STATS="${GEMBENCH_POLICY_KEYSTEP_NORM_STATS:-${FASTWAM_ROOT}/data/gembench_policy_keystep_9v32_4cam224_stats.json}"
export GEMBENCH_9V32_MANIFEST="${GEMBENCH_9V32_4CAM_MANIFEST}"
export GEMBENCH_9V32_RGB_CACHE_DIR="${GEMBENCH_9V32_4CAM_RGB_CACHE_DIR}"
export GEMBENCH_9V32_VAE_CACHE_DIR="${GEMBENCH_9V32_4CAM_VAE_CACHE_DIR}"
AUDIT_DIR="${GEMBENCH_POLICY_KEYSTEP_9V32_4CAM_AUDIT_DIR:-runs/gembench_policy_keystep_9v32_4cam224_audits/${RUN_ID}}"

if [[ ! -x "${PYTHON_BIN}" ]]; then
  echo "[gembench-policy-keystep-train] missing python: ${PYTHON_BIN}" >&2
  exit 1
fi

if [[ ! -f "${GEMBENCH_ROOT}/train_dataset/microsteps.tar.gz" ]]; then
  echo "[gembench-policy-keystep-train] missing train microsteps tar: ${GEMBENCH_ROOT}/train_dataset/microsteps.tar.gz" >&2
  exit 1
fi
if [[ ! -d "${GEMBENCH_KEYSTEPS_BBOX_DIR}" ]]; then
  echo "[gembench-policy-keystep-train] missing train keysteps LMDB: ${GEMBENCH_KEYSTEPS_BBOX_DIR}" >&2
  exit 1
fi
if [[ "${FASTWAM_GEMBENCH_POLICY_KEYSTEP_RUN_PCD_AUDIT:-0}" == "1" && ! -d "${GEMBENCH_KEYSTEPS_BBOX_PCD_DIR}" ]]; then
  echo "[gembench-policy-keystep-train] missing train keysteps PCD LMDB: ${GEMBENCH_KEYSTEPS_BBOX_PCD_DIR}" >&2
  exit 1
fi
for arg in "$@"; do
  case "${arg}" in
    task=*|+task=*|++task=*|task.*=*|+task.*=*|++task.*=*|data=*|+data=*|++data=*|data.*=*|+data.*=*|++data.*=*)
      echo "[gembench-policy-keystep-train] refusing override that could bypass the policy-keystep 9V32 dataset contract: ${arg}" >&2
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

if [[ ! -f "${GEMBENCH_KEY_FRAMEIDS_CACHE}" ]]; then
  echo "[gembench-policy-keystep-train] building key-frame sidecar: ${GEMBENCH_KEY_FRAMEIDS_CACHE}"
  PYTHONPATH=src "${PYTHON_BIN}" scripts/build_gembench_microsteps_key_frameids.py \
    --manifest "${GEMBENCH_9V32_4CAM_MANIFEST}" \
    --microsteps-tar "${GEMBENCH_ROOT}/train_dataset/microsteps.tar.gz" \
    --seed seed0 \
    --output-json "${GEMBENCH_KEY_FRAMEIDS_CACHE}" \
    --fail-on-missing
fi
"${PYTHON_BIN}" - <<'PY'
import json
import os
from pathlib import Path

manifest_path = Path(os.environ["GEMBENCH_9V32_4CAM_MANIFEST"])
sidecar_path = Path(os.environ["GEMBENCH_KEY_FRAMEIDS_CACHE"])
manifest = json.loads(manifest_path.read_text())
sidecar = json.loads(sidecar_path.read_text())
demos = manifest.get("demos", [])
entries = sidecar.get("entries", {})
if not isinstance(demos, list) or not isinstance(entries, dict):
    raise SystemExit(f"invalid key-frame sidecar schema: {sidecar_path}")
if sidecar.get("num_errors", 0) or sidecar.get("num_missing_members", 0):
    raise SystemExit(
        f"incomplete key-frame sidecar {sidecar_path}: "
        f"errors={sidecar.get('num_errors')} missing={sidecar.get('num_missing_members')}"
    )
if len(entries) < len(demos):
    raise SystemExit(f"key-frame sidecar only covers {len(entries)}/{len(demos)} demos: {sidecar_path}")
PY

POLICY_STATS_ARGS=()
if [[ -n "${FASTWAM_GEMBENCH_POLICY_KEYSTEP_POLICY_TARGET_FRAME:-}" ]]; then
  POLICY_STATS_ARGS+=(--policy-target-frame "${FASTWAM_GEMBENCH_POLICY_KEYSTEP_POLICY_TARGET_FRAME}")
fi
if [[ -n "${GEMBENCH_POLICY_KEYSTEP_STATS_TASKVARS:-}" ]]; then
  POLICY_STATS_ARGS+=(--taskvars "${GEMBENCH_POLICY_KEYSTEP_STATS_TASKVARS}")
fi
if [[ -n "${GEMBENCH_POLICY_KEYSTEP_STATS_MAX_INDEX_DEMOS:-}" ]]; then
  POLICY_STATS_ARGS+=(--policy-max-index-demos "${GEMBENCH_POLICY_KEYSTEP_STATS_MAX_INDEX_DEMOS}")
fi
if [[ -n "${GEMBENCH_POLICY_KEYSTEP_STATS_MAX_SAMPLES:-}" ]]; then
  POLICY_STATS_ARGS+=(--max-samples "${GEMBENCH_POLICY_KEYSTEP_STATS_MAX_SAMPLES}")
fi
if [[ "${FASTWAM_GEMBENCH_POLICY_KEYSTEP_POLICY_TARGET_FRAME:-}" == "official_pcd_local" ]]; then
  POLICY_STATS_ARGS+=(
    --policy-pcd-data-dir "${GEMBENCH_KEYSTEPS_BBOX_PCD_DIR}"
    --policy-local-xyz-shift "${FASTWAM_GEMBENCH_POLICY_KEYSTEP_LOCAL_XYZ_SHIFT:-center}"
    --policy-local-rm-robot "${FASTWAM_GEMBENCH_POLICY_KEYSTEP_LOCAL_RM_ROBOT:-none}"
    --policy-local-num-points "${FASTWAM_GEMBENCH_POLICY_KEYSTEP_LOCAL_NUM_POINTS:-0}"
    --policy-local-train-voxel-size "${FASTWAM_GEMBENCH_POLICY_KEYSTEP_LOCAL_TRAIN_VOXEL_SIZE:-0.0}"
    --robot-3dlotus-root "${ROBOT_3DLOTUS_ROOT:-/mnt/yuhan/gembench_sim/robot-3dlotus}"
  )
fi
if [[ "${PRECOMPUTE_GEMBENCH_POLICY_KEYSTEP_STATS:-1}" == "1" && ! -f "${GEMBENCH_POLICY_KEYSTEP_NORM_STATS}" ]]; then
  echo "[gembench-policy-keystep-train] computing key-step policy norm stats: ${GEMBENCH_POLICY_KEYSTEP_NORM_STATS}"
  PYTHONPATH=src "${PYTHON_BIN}" scripts/precompute_gembench_policy_keystep_norm_stats.py \
    --manifest "${GEMBENCH_9V32_4CAM_MANIFEST}" \
    --rgb-cache-dir "${GEMBENCH_9V32_4CAM_RGB_CACHE_DIR}" \
    --keysteps-dir "${GEMBENCH_KEYSTEPS_BBOX_DIR}" \
    --key-frameids-path "${GEMBENCH_KEY_FRAMEIDS_CACHE}" \
    --output "${GEMBENCH_POLICY_KEYSTEP_NORM_STATS}" \
    "${POLICY_STATS_ARGS[@]}"
fi
if [[ ! -f "${GEMBENCH_POLICY_KEYSTEP_NORM_STATS}" ]]; then
  echo "[gembench-policy-keystep-train] missing key-step policy norm stats: ${GEMBENCH_POLICY_KEYSTEP_NORM_STATS}" >&2
  echo "[gembench-policy-keystep-train] run scripts/precompute_gembench_policy_keystep_norm_stats.py or set GEMBENCH_POLICY_KEYSTEP_NORM_STATS" >&2
  exit 1
fi

CACHE_DIR="${GEMBENCH_TEXT_EMBED_CACHE:-data/text_embeds_cache/gembench_microsteps_9v32}"
TEXT_ENCODER_ID="${GEMBENCH_TEXT_ENCODER_ID:-wan22ti2v5b}"
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
    --text-encoder-id "${TEXT_ENCODER_ID}"
fi

if [[ -n "${GEMBENCH_9V32_4CAM_VAE_CACHE_DIR}" ]]; then
  if [[ ! -f "${GEMBENCH_9V32_4CAM_VAE_CACHE_DIR}/manifest.json" ]]; then
    echo "[gembench-policy-keystep-train] missing VAE latent cache manifest: ${GEMBENCH_9V32_4CAM_VAE_CACHE_DIR}/manifest.json" >&2
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
  echo "[gembench-policy-keystep-train] vae_latent_cache_dir=${GEMBENCH_9V32_4CAM_VAE_CACHE_DIR}"
fi

POLICY_AUDIT_ARGS=()
if [[ -n "${GEMBENCH_POLICY_KEYSTEP_AUDIT_MAX_INDEX_DEMOS:-}" ]]; then
  POLICY_AUDIT_ARGS+=(--policy-max-index-demos "${GEMBENCH_POLICY_KEYSTEP_AUDIT_MAX_INDEX_DEMOS}")
fi
if [[ "${GEMBENCH_POLICY_KEYSTEP_AUDIT_ALLOW_MISSING_TEXT:-0}" == "1" ]]; then
  POLICY_AUDIT_ARGS+=(--allow-missing-text-embeds)
fi
if [[ -n "${FASTWAM_GEMBENCH_POLICY_KEYSTEP_POLICY_TARGET_FRAME:-}" ]]; then
  POLICY_AUDIT_ARGS+=(--policy-target-frame "${FASTWAM_GEMBENCH_POLICY_KEYSTEP_POLICY_TARGET_FRAME}")
fi
if [[ "${FASTWAM_GEMBENCH_POLICY_KEYSTEP_POLICY_TARGET_FRAME:-}" == "official_pcd_local" ]]; then
  POLICY_AUDIT_ARGS+=(
    --policy-pcd-data-dir "${GEMBENCH_KEYSTEPS_BBOX_PCD_DIR}"
    --policy-local-xyz-shift "${FASTWAM_GEMBENCH_POLICY_KEYSTEP_LOCAL_XYZ_SHIFT:-center}"
    --policy-local-rm-robot "${FASTWAM_GEMBENCH_POLICY_KEYSTEP_LOCAL_RM_ROBOT:-none}"
    --policy-local-num-points "${FASTWAM_GEMBENCH_POLICY_KEYSTEP_LOCAL_NUM_POINTS:-0}"
    --policy-local-train-voxel-size "${FASTWAM_GEMBENCH_POLICY_KEYSTEP_LOCAL_TRAIN_VOXEL_SIZE:-0.0}"
    --robot-3dlotus-root "${ROBOT_3DLOTUS_ROOT:-/mnt/yuhan/gembench_sim/robot-3dlotus}"
  )
fi

PYTHONPATH=src "${PYTHON_BIN}" scripts/audit_gembench_policy_keystep_9v32_contract.py \
  --manifest "${GEMBENCH_9V32_4CAM_MANIFEST}" \
  --rgb-cache-dir "${GEMBENCH_9V32_4CAM_RGB_CACHE_DIR}" \
  --keysteps-dir "${GEMBENCH_KEYSTEPS_BBOX_DIR}" \
  --key-frameids-path "${GEMBENCH_KEY_FRAMEIDS_CACHE}" \
  --vae-latent-cache-dir "${GEMBENCH_9V32_4CAM_VAE_CACHE_DIR}" \
  --text-embedding-cache-dir "${CACHE_DIR}" \
  --text-encoder-id "${TEXT_ENCODER_ID}" \
  --pretrained-norm-stats "${GEMBENCH_POLICY_KEYSTEP_NORM_STATS}" \
  --norm-default-mode min/max \
  --output-json "${AUDIT_DIR}/policy_keystep_9v32_4cam224_contract.json" \
  --output-md "${AUDIT_DIR}/policy_keystep_9v32_4cam224_contract.md" \
  "${POLICY_AUDIT_ARGS[@]}"

if [[ "${FASTWAM_GEMBENCH_POLICY_KEYSTEP_RUN_PCD_AUDIT:-0}" == "1" ]]; then
  PCD_AUDIT_ARGS=()
  if [[ "${FASTWAM_GEMBENCH_POLICY_KEYSTEP_STRICT_MANIFEST_COVERAGE:-0}" == "1" ]]; then
    PCD_AUDIT_ARGS+=(--require-manifest-covers-expected-taskvars)
  fi
  PYTHONPATH=src "${PYTHON_BIN}" scripts/audit_gembench_official_pcd_policy_contract.py \
    --pcd-data-dir "${GEMBENCH_KEYSTEPS_BBOX_PCD_DIR}" \
    --fastwam-9v32-manifest "${GEMBENCH_9V32_4CAM_MANIFEST}" \
    --rm-robot "${FASTWAM_GEMBENCH_POLICY_KEYSTEP_PCD_AUDIT_RM_ROBOT:-none}" \
    --episodes-per-taskvar "${FASTWAM_GEMBENCH_POLICY_KEYSTEP_PCD_AUDIT_EPISODES:-1}" \
    --max-steps-per-episode "${FASTWAM_GEMBENCH_POLICY_KEYSTEP_PCD_AUDIT_STEPS:-2}" \
    --output-json "${AUDIT_DIR}/official_pcd_policy_contract.json" \
    --output-md "${AUDIT_DIR}/official_pcd_policy_contract.md" \
    "${PCD_AUDIT_ARGS[@]}"
fi

export ACCELERATE_CONFIG_FILE="${ACCELERATE_CONFIG_FILE:-scripts/accelerate_configs/accelerate_zero2_ds.yaml}"
WANDB_SUBPROJECT_DEFAULT="${WANDB_SUBPROJECT:-${FASTWAM_GEMBENCH_POLICY_KEYSTEP_WANDB_SUBPROJECT:-fastwam-gembench-policy-keystep-4cam224}}"
WANDB_RUN_NAME_DEFAULT="${WANDB_RUN_NAME:-${RUN_ID}}"
WANDB_GROUP_DEFAULT="${WANDB_GROUP:-${FASTWAM_GEMBENCH_POLICY_KEYSTEP_WANDB_GROUP:-fastwam-gembench-policy-keystep-4cam224-wamaux9v32-b4a1}}"
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

echo "[gembench-policy-keystep-train] run_id=${RUN_ID} task=${TASK_NAME} nproc=${NPROC_PER_NODE}"
echo "[gembench-policy-keystep-train] manifest=${GEMBENCH_9V32_4CAM_MANIFEST}"
echo "[gembench-policy-keystep-train] rgb_cache_dir=${GEMBENCH_9V32_4CAM_RGB_CACHE_DIR}"
bash scripts/train_zero2.sh "${NPROC_PER_NODE}" "task=${TASK_NAME}" "${TRAIN_OVERRIDES[@]}" "$@" "${WANDB_OVERRIDES[@]}"
