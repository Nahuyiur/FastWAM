#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
FASTWAM_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${FASTWAM_ROOT}"
source scripts/setup_yuhan_paths.sh

PYTHON_BIN="${FASTWAM_CONDA_ENV}/bin/python"
export PATH="${FASTWAM_CONDA_ENV}/bin:${PATH}"
export PYTHONPATH="${FASTWAM_ROOT}/src:${PYTHONPATH:-}"

if [[ -n "${COPPELIASIM_ROOT:-}" ]]; then
  export LD_LIBRARY_PATH="${LD_LIBRARY_PATH:-}:${COPPELIASIM_ROOT}"
  export QT_QPA_PLATFORM_PLUGIN_PATH="${QT_QPA_PLATFORM_PLUGIN_PATH:-${COPPELIASIM_ROOT}}"
fi

if [[ ! -x "${PYTHON_BIN}" ]]; then
  echo "[gembench-success] missing python: ${PYTHON_BIN}" >&2
  exit 1
fi

RUN_WITH_XVFB="${RUN_WITH_XVFB:-1}"
if [[ "${RUN_WITH_XVFB}" == "1" ]]; then
  if ! command -v xvfb-run >/dev/null 2>&1; then
    echo "[gembench-success] xvfb-run not found. Install Xvfb or set RUN_WITH_XVFB=0 if a DISPLAY is already available." >&2
    exit 1
  fi
  exec xvfb-run -a "${PYTHON_BIN}" scripts/eval_gembench_success_rate.py "$@"
fi

exec "${PYTHON_BIN}" scripts/eval_gembench_success_rate.py "$@"
