#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
FASTWAM_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${FASTWAM_ROOT}"

export FASTWAM_GEMBENCH_POLICY_KEYSTEP_TASK_NAME="${FASTWAM_GEMBENCH_POLICY_KEYSTEP_TASK_NAME:-gembench_policy_keystep_9v32_4cam224_official_local_1e-4}"
export FASTWAM_GEMBENCH_POLICY_KEYSTEP_RUN_PREFIX="${FASTWAM_GEMBENCH_POLICY_KEYSTEP_RUN_PREFIX:-fastwam_gembench_policy_keystep_officiallocal_4cam224_wamaux9v32_b4a1}"
export FASTWAM_GEMBENCH_POLICY_KEYSTEP_RUN_PCD_AUDIT="${FASTWAM_GEMBENCH_POLICY_KEYSTEP_RUN_PCD_AUDIT:-1}"
export FASTWAM_GEMBENCH_POLICY_KEYSTEP_POLICY_TARGET_FRAME="${FASTWAM_GEMBENCH_POLICY_KEYSTEP_POLICY_TARGET_FRAME:-official_pcd_local}"
export FASTWAM_GEMBENCH_POLICY_KEYSTEP_WANDB_SUBPROJECT="${FASTWAM_GEMBENCH_POLICY_KEYSTEP_WANDB_SUBPROJECT:-fastwam-gembench-policy-keystep-official-local}"
export FASTWAM_GEMBENCH_POLICY_KEYSTEP_WANDB_GROUP="${FASTWAM_GEMBENCH_POLICY_KEYSTEP_WANDB_GROUP:-fastwam-gembench-policy-keystep-official-local-4cam224-wamaux9v32-b4a1}"

exec bash "${FASTWAM_ROOT}/scripts/train_gembench_policy_keystep_9v32_4cam224_4gpu.sh" "$@"
