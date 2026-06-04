#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Usage: bash scripts/check_jinshan_fastwam_ready.sh [--basic|--train|--eval|--gembench|--all]

Checks that FastWAM paths and assets resolve on jinshan_pub without launching
training or simulator evaluation.

Modes:
  --basic    repo, conda env, Wan2.2 base weights
  --train    basic checks plus RLBench LeRobot train/test data
  --eval     basic checks plus simulator/runtime/checkpoint requirements
  --gembench GEMBench dataset presence and incomplete-download markers
  --all      all checks (default)
EOF
}

MODE="${1:---all}"
if [[ "${MODE}" == "-h" || "${MODE}" == "--help" ]]; then
  usage
  exit 0
fi

case "${MODE}" in
  --basic|--train|--eval|--gembench|--all) ;;
  *)
    usage >&2
    exit 2
    ;;
esac

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck disable=SC1091
source "${SCRIPT_DIR}/setup_yuhan_paths.sh"

FAILED=0
WARNED=0

ok() {
  printf '[OK]   %s\n' "$1"
}

warn() {
  WARNED=1
  printf '[WARN] %s\n' "$1"
}

fail() {
  FAILED=1
  printf '[MISS] %s\n' "$1"
}

need_path() {
  local path="$1"
  local label="$2"
  if [[ -e "${path}" ]]; then
    ok "${label}: ${path}"
  else
    fail "${label}: ${path}"
  fi
}

warn_path() {
  local path="$1"
  local label="$2"
  if [[ -e "${path}" ]]; then
    ok "${label}: ${path}"
  else
    warn "${label}: ${path}"
  fi
}

check_basic() {
  echo "== basic =="
  need_path "${FASTWAM_ROOT}" "FastWAM repo"
  need_path "${CONDA_ROOT}/bin/activate" "conda activate"
  need_path "${FASTWAM_CONDA_ENV}" "fastwam conda env"
  need_path "${FASTWAM_CONDA_ENV}/bin/python" "fastwam python"
  need_path "${FASTWAM_PRETRAIN_ROOT}/Wan-AI/Wan2.2-TI2V-5B/diffusion_pytorch_model-00001-of-00003.safetensors" "Wan2.2 shard 1"
  need_path "${FASTWAM_PRETRAIN_ROOT}/Wan-AI/Wan2.2-TI2V-5B/diffusion_pytorch_model-00002-of-00003.safetensors" "Wan2.2 shard 2"
  need_path "${FASTWAM_PRETRAIN_ROOT}/Wan-AI/Wan2.2-TI2V-5B/diffusion_pytorch_model-00003-of-00003.safetensors" "Wan2.2 shard 3"
  need_path "${FASTWAM_PRETRAIN_ROOT}/Wan-AI/Wan2.2-TI2V-5B/Wan2.2_VAE.pth" "Wan2.2 VAE"
  warn_path "${FASTWAM_PRETRAIN_ROOT}/ActionDiT_linear_interp_Wan22_alphascale_1024hdim.pt" "ActionDiT init checkpoint"
  check_python_imports
  echo
}

check_python_imports() {
  local python_bin="${FASTWAM_CONDA_ENV}/bin/python"
  if [[ ! -x "${python_bin}" ]]; then
    fail "FastWAM python is not executable: ${python_bin}"
    return
  fi

  local output status
  set +e
  output="$(PYTHONPATH="${FASTWAM_ROOT}/src:${PYTHONPATH:-}" "${python_bin}" - <<'PY'
mods = [
    "torch",
    "hydra",
    "omegaconf",
    "accelerate",
    "transformers",
    "safetensors",
    "fastwam",
]
failed = []
for name in mods:
    try:
        mod = __import__(name)
        version = getattr(mod, "__version__", "ok")
        print(f"OK {name} {version}")
    except Exception as exc:
        failed.append(name)
        print(f"MISS {name} {type(exc).__name__}: {exc}")
raise SystemExit(1 if failed else 0)
PY
  )"
  status=$?
  set -e

  while IFS= read -r line; do
    if [[ "${line}" == OK* ]]; then
      ok "python import ${line#OK }"
    elif [[ "${line}" == MISS* ]]; then
      fail "python import ${line#MISS }"
    fi
  done <<< "${output}"

  if [[ "${status}" -ne 0 ]]; then
    fail "FastWAM Python dependency set is incomplete in ${FASTWAM_CONDA_ENV}"
  fi
}

check_train() {
  echo "== train =="
  need_path "${RLBENCH_LEROBOT_TRAIN_DIR}" "RLBench LeRobot train dir"
  need_path "${RLBENCH_LEROBOT_TEST_DIR}" "RLBench LeRobot test dir"
  echo
}

latest_checkpoint_for_task() {
  local task="$1"
  if [[ ! -d "${FASTWAM_RUNS_ROOT}/${task}" ]]; then
    return 0
  fi
  find "${FASTWAM_RUNS_ROOT}/${task}" -path '*/checkpoints/weights/step_*.pt' -type f 2>/dev/null \
    | sort -V \
    | tail -1
}

check_eval() {
  echo "== eval =="
  need_path "${RLBENCH_ROOT}" "RLBench checkout"
  need_path "${COPPELIASIM_ROOT}" "CoppeliaSim root"
  need_path "${RLBENCH_STUB_ROOT}" "RLBench lerobot stubs"
  need_path "${RLBENCH_PYREP_SITE}" "PyRep/gembench site-packages"

  local tasks=(
    rlbench_original_3cam224_1e-4
    rlbench_color_3cam224_1e-4
    rlbench_shape_3cam224_1e-4
    rlbench_color_shape_3cam224_1e-4
  )

  local task ckpt stats
  for task in "${tasks[@]}"; do
    ckpt="$(latest_checkpoint_for_task "${task}")"
    if [[ -n "${ckpt}" ]]; then
      ok "latest ${task} checkpoint: ${ckpt}"
      stats="$(dirname "$(dirname "$(dirname "${ckpt}")")")/dataset_stats.json"
      need_path "${stats}" "${task} dataset_stats.json"
    else
      fail "latest ${task} checkpoint under ${FASTWAM_RUNS_ROOT}/${task}"
    fi
  done
  echo
}

check_gembench() {
  echo "== gembench =="
  need_path "${GEMBENCH_ROOT}" "GEMBench root"
  need_path "${GEMBENCH_ROOT}/README.md" "GEMBench README"
  need_path "${GEMBENCH_ROOT}/train_dataset/keysteps_bbox/seed0" "GEMBench train keysteps seed0"
  need_path "${GEMBENCH_ROOT}/test_dataset/microsteps.tar.gz" "GEMBench test microsteps archive"
  need_path "${GEMBENCH_ROOT}/val_dataset" "GEMBench val_dataset"

  local incomplete_count lock_count train_task_count
  incomplete_count="$({ find "${GEMBENCH_ROOT}/.cache/huggingface/download" -name '*.incomplete' 2>/dev/null || true; } | wc -l | tr -d ' ')"
  lock_count="$({ find "${GEMBENCH_ROOT}/.cache/huggingface/download" -name '*.lock' 2>/dev/null || true; } | wc -l | tr -d ' ')"
  train_task_count="$({ find "${GEMBENCH_ROOT}/train_dataset/keysteps_bbox/seed0" -mindepth 1 -maxdepth 1 -type d 2>/dev/null || true; } | wc -l | tr -d ' ')"
  if [[ "${incomplete_count}" == "0" ]]; then
    ok "GEMBench incomplete cache files: 0"
  else
    fail "GEMBench incomplete cache files: ${incomplete_count}"
  fi
  if [[ "${lock_count}" == "0" ]]; then
    ok "GEMBench stale lock files: 0"
  else
    warn "GEMBench lock files in HF cache: ${lock_count}"
  fi
  ok "GEMBench train task directories under seed0: ${train_task_count}"
  echo
}

case "${MODE}" in
  --basic)
    check_basic
    ;;
  --train)
    check_basic
    check_train
    ;;
  --eval)
    check_basic
    check_eval
    ;;
  --gembench)
    check_gembench
    ;;
  --all)
    check_basic
    check_train
    check_eval
    check_gembench
    ;;
esac

if [[ "${FAILED}" -ne 0 ]]; then
  echo "Result: missing required assets or paths."
  exit 1
fi

if [[ "${WARNED}" -ne 0 ]]; then
  echo "Result: required checks passed, with warnings."
else
  echo "Result: ready."
fi
