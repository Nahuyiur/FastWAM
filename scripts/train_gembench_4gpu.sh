#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
FASTWAM_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${FASTWAM_ROOT}"
source scripts/setup_yuhan_paths.sh
source scripts/setup_gembench_wandb.sh

NPROC_PER_NODE="${NPROC_PER_NODE:-4}"
TASK_NAME="${TASK_NAME:-gembench_keysteps_bbox_3cam224_1e-4}"
PYTHON_BIN="${FASTWAM_CONDA_ENV}/bin/python"
export PATH="${FASTWAM_CONDA_ENV}/bin:${PATH}"

GEMBENCH_TMPDIR="${GEMBENCH_TMPDIR:-/tmp/fastwam-gembench-${USER:-user}}"
mkdir -p "${GEMBENCH_TMPDIR}"
export TMPDIR="${GEMBENCH_TMPDIR}"
export TMP="${GEMBENCH_TMPDIR}"
export TEMP="${GEMBENCH_TMPDIR}"

if [[ ! -x "${PYTHON_BIN}" ]]; then
  echo "[gembench-train] missing python: ${PYTHON_BIN}" >&2
  exit 1
fi

"${PYTHON_BIN}" - <<'PY'
import importlib
missing=[]
for name in ["fastwam", "lmdb", "msgpack", "msgpack_numpy"]:
    try:
        importlib.import_module(name)
    except Exception as exc:
        missing.append(f"{name}: {exc}")
if missing:
    raise SystemExit("Missing required modules:\n" + "\n".join(missing))
PY

GEMBENCH_TRAIN_DIR="${GEMBENCH_ROOT}/train_dataset/keysteps_bbox/seed0"
if [[ ! -d "${GEMBENCH_TRAIN_DIR}" ]]; then
  echo "[gembench-train] missing GEMBench train keysteps: ${GEMBENCH_TRAIN_DIR}" >&2
  exit 1
fi

RELEVANT_INCOMPLETE_COUNT="$({ find "${GEMBENCH_ROOT}/.cache/huggingface/download/train_dataset/keysteps_bbox/seed0" -name '*.incomplete' 2>/dev/null || true; } | wc -l | tr -d ' ')"
GLOBAL_INCOMPLETE_COUNT="$({ find "${GEMBENCH_ROOT}/.cache/huggingface/download" -name '*.incomplete' 2>/dev/null || true; } | wc -l | tr -d ' ')"
COMPLETE_TASKVAR_COUNT="$(find "${GEMBENCH_TRAIN_DIR}" -mindepth 1 -maxdepth 1 -type d 2>/dev/null | while read -r taskvar_dir; do [[ -f "${taskvar_dir}/data.mdb" && -f "${taskvar_dir}/results.json" ]] && basename "${taskvar_dir}"; done | wc -l | tr -d ' ')"
EXPECTED_TASKVAR_COUNT="${GEMBENCH_EXPECTED_TASKVARS:-31}"
if [[ "${STRICT_GEMBENCH_COMPLETE:-0}" == "1" && "${RELEVANT_INCOMPLETE_COUNT}" != "0" ]]; then
  echo "[gembench-train] GEMBench train keysteps still has ${RELEVANT_INCOMPLETE_COUNT} incomplete HF download files." >&2
  exit 1
fi
if [[ "${STRICT_GEMBENCH_COMPLETE:-0}" == "1" && "${COMPLETE_TASKVAR_COUNT}" -lt "${EXPECTED_TASKVAR_COUNT}" ]]; then
  echo "[gembench-train] expected at least ${EXPECTED_TASKVAR_COUNT} complete train taskvars, found ${COMPLETE_TASKVAR_COUNT}." >&2
  exit 1
fi
if [[ "${RELEVANT_INCOMPLETE_COUNT}" != "0" ]]; then
  echo "[gembench-train] warning: train keysteps has ${RELEVANT_INCOMPLETE_COUNT} incomplete HF files; dataset will skip incomplete taskvars." >&2
elif [[ "${COMPLETE_TASKVAR_COUNT}" -lt "${EXPECTED_TASKVAR_COUNT}" ]]; then
  echo "[gembench-train] warning: found ${COMPLETE_TASKVAR_COUNT}/${EXPECTED_TASKVAR_COUNT} complete train taskvars." >&2
fi
if [[ "${GLOBAL_INCOMPLETE_COUNT}" != "${RELEVANT_INCOMPLETE_COUNT}" ]]; then
  echo "[gembench-train] note: ${GLOBAL_INCOMPLETE_COUNT} total HF incomplete files remain, but only ${RELEVANT_INCOMPLETE_COUNT} are under train_dataset/keysteps_bbox/seed0." >&2
fi

CACHE_DIR="${GEMBENCH_TEXT_EMBED_CACHE:-data/text_embeds_cache/gembench_keysteps_bbox}"
NORM_STATS="${GEMBENCH_NORM_STATS:-data/gembench_keysteps_bbox_dataset_stats.json}"
NORM_MODE="${GEMBENCH_NORM_MODE:-z-score}"
export DIFFSYNTH_DOWNLOAD_SOURCE="${DIFFSYNTH_DOWNLOAD_SOURCE:-huggingface}"
export HF_HUB_DISABLE_XET="${HF_HUB_DISABLE_XET:-1}"
if [[ -n "${FASTWAM_PROXY:-}" && -z "${http_proxy:-}${https_proxy:-}${HTTP_PROXY:-}${HTTPS_PROXY:-}" ]]; then
  export http_proxy="${FASTWAM_PROXY}"
  export https_proxy="${FASTWAM_PROXY}"
  export HTTP_PROXY="${FASTWAM_PROXY}"
  export HTTPS_PROXY="${FASTWAM_PROXY}"
fi
if [[ "${PRECOMPUTE_GEMBENCH_TEXT:-1}" == "1" ]]; then
  "${PYTHON_BIN}" scripts/precompute_gembench_text_embeds.py --no-redirect-common-files --cache-dir "${CACHE_DIR}"
fi

if [[ "${VERIFY_GEMBENCH_CONTRACT:-1}" == "1" ]]; then
  "${PYTHON_BIN}" scripts/verify_gembench_dataset_contract.py \
    --cache-dir "${CACHE_DIR}" \
    --pretrained-norm-stats "${NORM_STATS}" \
    --norm-default-mode "${NORM_MODE}" \
    --num-samples "${GEMBENCH_VERIFY_SAMPLES:-8}" \
    --num-workers "${GEMBENCH_VERIFY_WORKERS:-2}"
fi

if [[ "${GEMBENCH_PREP_ONLY:-0}" == "1" ]]; then
  echo "[gembench-train] prep-only checks passed."
  exit 0
fi

export ACCELERATE_CONFIG_FILE="${ACCELERATE_CONFIG_FILE:-scripts/accelerate_configs/accelerate_zero2_ds.yaml}"
WANDB_SUBPROJECT_DEFAULT="${WANDB_SUBPROJECT:-fastwam-gembench-train}"
WANDB_RUN_NAME_DEFAULT="${WANDB_RUN_NAME:-${RUN_ID:-${TASK_NAME}}}"
WANDB_GROUP_DEFAULT="${WANDB_GROUP:-${WANDB_SUBPROJECT_DEFAULT}}"
mapfile -t WANDB_OVERRIDES < <(gembench_wandb_hydra_overrides "${WANDB_SUBPROJECT_DEFAULT}" "${WANDB_RUN_NAME_DEFAULT}" "${WANDB_GROUP_DEFAULT}")

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
add_hydra_override_from_env FASTWAM_SAVE_FULL_STATE checkpoint.save_full_state
add_hydra_override_from_env FASTWAM_REQUIRE_FULL_STATE checkpoint.require_full_state
add_hydra_override_from_env FASTWAM_WEIGHT_MIN_FREE_GB checkpoint.weight_min_free_gb
add_hydra_override_from_env FASTWAM_FULL_STATE_MIN_FREE_GB checkpoint.full_state_min_free_gb
add_hydra_override_from_env FASTWAM_KEEP_LAST_FULL_STATES checkpoint.keep_last_full_states
add_hydra_override_from_env FASTWAM_MAX_STEPS max_steps
add_hydra_override_from_env FASTWAM_LEARNING_RATE learning_rate
add_hydra_override_from_env FASTWAM_BATCH_SIZE batch_size
add_hydra_override_from_env FASTWAM_GRAD_ACCUM gradient_accumulation_steps
add_hydra_override_from_env FASTWAM_SAVE_EVERY save_every
add_hydra_override_from_env FASTWAM_EVAL_EVERY eval_every
add_hydra_override_from_env FASTWAM_OUTPUT_DIR output_dir

echo "[gembench-train] task=${TASK_NAME} nproc=${NPROC_PER_NODE} root=${GEMBENCH_ROOT}"
bash scripts/train_zero2.sh "${NPROC_PER_NODE}" "task=${TASK_NAME}" "${TRAIN_OVERRIDES[@]}" "$@" "${WANDB_OVERRIDES[@]}"
