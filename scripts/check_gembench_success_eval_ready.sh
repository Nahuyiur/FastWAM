#!/usr/bin/env bash
set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
FASTWAM_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${FASTWAM_ROOT}"
source scripts/setup_yuhan_paths.sh

PYTHON_BIN="${FASTWAM_CONDA_ENV}/bin/python"
STATUS=0

ok() {
  printf '[ok] %s\n' "$1"
}

fail() {
  printf '[missing] %s\n' "$1" >&2
  STATUS=1
}

warn() {
  printf '[warn] %s\n' "$1" >&2
}

[[ -x "${PYTHON_BIN}" ]] && ok "python ${PYTHON_BIN}" || fail "python ${PYTHON_BIN}"
[[ -d "${GEMBENCH_ROOT}" ]] && ok "GEMBENCH_ROOT ${GEMBENCH_ROOT}" || fail "GEMBENCH_ROOT ${GEMBENCH_ROOT}"

for rel in \
  val_dataset/microsteps/seed100 \
  test_dataset/microsteps/seed200 \
  test_dataset/microsteps/seed300 \
  test_dataset/microsteps/seed400 \
  test_dataset/microsteps/seed500 \
  test_dataset/microsteps/seed600
do
  [[ -d "${GEMBENCH_ROOT}/${rel}" ]] && ok "${rel}" || fail "${rel} (run scripts/extract_gembench_microsteps.sh)"
done

for rel in val_dataset/microsteps.tar.gz test_dataset/microsteps.tar.gz; do
  [[ -f "${GEMBENCH_ROOT}/${rel}" ]] && ok "${rel}" || warn "${rel} not found"
done

if [[ -d "${COPPELIASIM_ROOT}" ]]; then
  ok "COPPELIASIM_ROOT ${COPPELIASIM_ROOT}"
else
  fail "COPPELIASIM_ROOT ${COPPELIASIM_ROOT}"
fi

command -v xvfb-run >/dev/null 2>&1 && ok "xvfb-run" || fail "xvfb-run"

"${PYTHON_BIN}" - <<'PY'
import importlib

missing = []
for name in ["fastwam", "rlbench", "pyrep"]:
    try:
        mod = importlib.import_module(name)
        print(f"[ok] import {name}: {getattr(mod, '__file__', '<namespace>')}")
    except Exception as exc:
        print(f"[missing] import {name}: {type(exc).__name__}: {exc}")
        missing.append(name)

try:
    from rlbench.backend.utils import task_file_to_task_class
    task_file_to_task_class("push_button")
    print("[ok] modified RLBench task lookup")
except Exception as exc:
    print(f"[missing] modified RLBench task lookup: {type(exc).__name__}: {exc}")
    missing.append("rlbench_task_lookup")

raise SystemExit(1 if missing else 0)
PY
PY_STATUS=$?
if [[ "${PY_STATUS}" -ne 0 ]]; then
  STATUS=1
fi

if [[ "${STATUS}" -eq 0 ]]; then
  ok "GEMBench success-rate eval runtime is ready."
else
  cat >&2 <<'EOF'

GEMBench success-rate eval still needs the simulator runtime:
  - CoppeliaSim V4_1_0 Ubuntu20_04
  - PyRep from https://github.com/cshizhe/PyRep
  - modified RLBench from https://github.com/rjgpinel/RLBench
  - xvfb-run/Xvfb for headless launch

After installing them, export COPPELIASIM_ROOT and ensure LD_LIBRARY_PATH and
QT_QPA_PLATFORM_PLUGIN_PATH include that directory before running eval.
EOF
fi

exit "${STATUS}"
