#!/usr/bin/env bash
# Source this file from GEMBench training entrypoints. It uses /mnt/yuhan/basic_info.sh
# for credentials without printing secrets, and emits Hydra W&B overrides via
# gembench_wandb_hydra_overrides.

if [[ -f /mnt/yuhan/basic_info.sh ]]; then
  set +u
  # shellcheck source=/mnt/yuhan/basic_info.sh
  source /mnt/yuhan/basic_info.sh >/dev/null 2>&1 || true
  set -u
fi

if [[ -z "${__GEMBENCH_WANDB_PROJECT_USER_PROVIDED:-}" ]]; then
  if [[ -n "${WANDB_PROJECT:-}" ]]; then
    export __GEMBENCH_WANDB_PROJECT_USER_PROVIDED=1
  else
    export __GEMBENCH_WANDB_PROJECT_USER_PROVIDED=0
  fi
fi
if [[ -z "${__GEMBENCH_WANDB_RUN_ID_USER_PROVIDED:-}" ]]; then
  if [[ -n "${WANDB_RUN_ID:-}" ]]; then
    export __GEMBENCH_WANDB_RUN_ID_USER_PROVIDED=1
  else
    export __GEMBENCH_WANDB_RUN_ID_USER_PROVIDED=0
  fi
fi

if [[ -n "${WANDB_TOKEN:-}" && -z "${WANDB_API_KEY:-}" ]]; then
  export WANDB_API_KEY="${WANDB_TOKEN}"
fi

export WANDB_PROJECT="${WANDB_PROJECT:-trace-gembench}"
export WANDB_MODE="${WANDB_MODE:-online}"
export WANDB_SILENT="${WANDB_SILENT:-true}"
export WANDB_DIR="${WANDB_DIR:-${FASTWAM_CACHE_ROOT:-${PWD}/.cache}/wandb}"
mkdir -p "${WANDB_DIR}"

# This request supersedes older queued launch commands that set WANDB_ENABLED=false.
# Use DISABLE_WANDB=1 as the explicit opt-out for GEMBench runs.
if [[ "${DISABLE_WANDB:-0}" == "1" ]]; then
  export WANDB_ENABLED=false
else
  export WANDB_ENABLED=true
fi

gembench_wandb_hydra_overrides() {
  local subproject="${1:?subproject required}"
  local run_name="${2:?run_name required}"
  local group="${3:-${subproject}}"

  if [[ -n "${WANDB_RUN_ID:-}" && "${__GEMBENCH_WANDB_RUN_ID_USER_PROVIDED:-0}" == "1" && "${__GEMBENCH_WANDB_PROJECT_USER_PROVIDED:-0}" != "1" ]]; then
    echo "[gembench-wandb] WANDB_RUN_ID was provided explicitly; set WANDB_PROJECT to the original run project to avoid cross-project resume." >&2
    return 2
  fi

  printf "%s\\n" \
    "wandb.enabled=${WANDB_ENABLED}" \
    "wandb.project=${WANDB_PROJECT}" \
    "wandb.name=${run_name}" \
    "wandb.group=${group}" \
    "wandb.job_type=${subproject}" \
    "wandb.tags=[gembench,${subproject}]" \
    "wandb.subproject=${subproject}" \
    "wandb.mode=${WANDB_MODE}"

  if [[ -n "${WANDB_RUN_ID:-}" ]]; then
    printf "%s\\n" \
      "wandb.id=${WANDB_RUN_ID}" \
      "wandb.resume=${WANDB_RESUME:-allow}"
  fi

  if [[ -n "${WANDB_ENTITY:-}" ]]; then
    printf "%s\\n" "wandb.workspace=${WANDB_ENTITY}"
  elif [[ -n "${WANDB_WORKSPACE:-}" ]]; then
    printf "%s\\n" "wandb.workspace=${WANDB_WORKSPACE}"
  fi
}

gembench_filter_wandb_hydra_overrides() {
  local -a defaults=()
  local -a user_args=()
  local seen_separator=0
  local arg
  for arg in "$@"; do
    if [[ "${arg}" == "--" && "${seen_separator}" == "0" ]]; then
      seen_separator=1
    elif [[ "${seen_separator}" == "0" ]]; then
      defaults+=("${arg}")
    else
      user_args+=("${arg}")
    fi
  done

  local -a user_wandb_keys=()
  local key
  for arg in "${user_args[@]}"; do
    case "${arg}" in
      wandb.*=*)
        user_wandb_keys+=("${arg%%=*}")
        ;;
      +wandb.*=*|++wandb.*=*)
        key="${arg%%=*}"
        key="${key#++}"
        key="${key#+}"
        user_wandb_keys+=("${key}")
        ;;
    esac
  done

  local override skip user_key
  for override in "${defaults[@]}"; do
    key="${override%%=*}"
    skip=0
    for user_key in "${user_wandb_keys[@]}"; do
      if [[ "${key}" == "${user_key}" ]]; then
        skip=1
        break
      fi
    done
    if [[ "${skip}" == "0" ]]; then
      printf "%s\\n" "${override}"
    fi
  done
}
