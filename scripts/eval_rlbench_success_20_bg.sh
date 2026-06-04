#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="${FASTWAM_ROOT:-$(cd "${SCRIPT_DIR}/.." && pwd)}"
SESSION="${1:-fastwam_rlbench_success_20}"
OUT_ROOT="${2:-${ROOT}/runs/rlbench_success_eval_20/$(date +%Y-%m-%d_%H-%M-%S)}"

cd "${ROOT}"
# shellcheck disable=SC1091
source scripts/setup_yuhan_paths.sh
bash scripts/check_jinshan_fastwam_ready.sh --eval

set +u
# shellcheck disable=SC1091
source "${CONDA_ROOT}/bin/activate" "${FASTWAM_CONDA_ENV}"
set -u

export PYTHONPATH="${RLBENCH_STUB_ROOT}:${RLBENCH_ROOT}:${RLBENCH_PYREP_SITE}:${PYTHONPATH:-}"
export LD_LIBRARY_PATH="${CONDA_PREFIX}/lib:${COPPELIASIM_ROOT}:${COPPELIASIM_ROOT}/platforms:${LD_LIBRARY_PATH:-}"
export QT_PLUGIN_PATH="${COPPELIASIM_ROOT}"
export QT_QPA_PLATFORM_PLUGIN_PATH="${COPPELIASIM_ROOT}/platforms"
export QT_XCB_GL_INTEGRATION=xcb_glx
export LIBGL_ALWAYS_SOFTWARE=1
export TOKENIZERS_PARALLELISM=false
export XDG_RUNTIME_DIR="${FASTWAM_CACHE_ROOT}/xdg_runtime"
mkdir -p "${OUT_ROOT}/logs" "${XDG_RUNTIME_DIR}" "${TMPDIR}"
chmod 700 "${XDG_RUNTIME_DIR}"

TASKS=(
  rlbench_original_3cam224_1e-4
  rlbench_color_3cam224_1e-4
  rlbench_shape_3cam224_1e-4
  rlbench_color_shape_3cam224_1e-4
)
GPUS=(0 1 2 3)

if tmux has-session -t "${SESSION}" 2>/dev/null; then
  echo "tmux session already exists: ${SESSION}" >&2
  echo "Use: tmux kill-session -t ${SESSION}" >&2
  exit 1
fi

for i in "${!TASKS[@]}"; do
  task="${TASKS[$i]}"
  gpu="${GPUS[$i]}"
  log="${OUT_ROOT}/logs/${task}.log"
  cmd="cd ${ROOT}; source scripts/setup_yuhan_paths.sh; set +u; source ${CONDA_ROOT}/bin/activate ${FASTWAM_CONDA_ENV}; set -u; export RLBENCH_ROOT=${RLBENCH_ROOT}; export COPPELIASIM_ROOT=${COPPELIASIM_ROOT}; export RLBENCH_STUB_ROOT=${RLBENCH_STUB_ROOT}; export RLBENCH_PYREP_SITE=${RLBENCH_PYREP_SITE}; export PYTHONPATH=${PYTHONPATH}; export LD_LIBRARY_PATH=${LD_LIBRARY_PATH}; export QT_PLUGIN_PATH=${QT_PLUGIN_PATH}; export QT_QPA_PLATFORM_PLUGIN_PATH=${QT_QPA_PLATFORM_PLUGIN_PATH}; export QT_XCB_GL_INTEGRATION=xcb_glx; export LIBGL_ALWAYS_SOFTWARE=1; export TOKENIZERS_PARALLELISM=false; export XDG_RUNTIME_DIR=${XDG_RUNTIME_DIR}; export TMPDIR=${TMPDIR}; CUDA_VISIBLE_DEVICES=${gpu} xvfb-run -a -s '-screen 0 1280x1024x24 +extension GLX +render -noreset' python scripts/eval_rlbench_success_rate.py --task ${task} --trials 20 --output-root ${OUT_ROOT} --device cuda --renderer opengl --max-steps 240 --replan-steps 8 --num-inference-steps 10 2>&1 | tee ${log}"
  if [[ "${i}" == "0" ]]; then
    tmux new-session -d -s "${SESSION}" -n "${task}" "bash -lc ${cmd@Q}"
  else
    tmux new-window -t "${SESSION}" -n "${task}" "bash -lc ${cmd@Q}"
  fi
done

echo "Started tmux session: ${SESSION}"
echo "Output root: ${OUT_ROOT}"
echo "Logs:"
for task in "${TASKS[@]}"; do
  echo "  ${OUT_ROOT}/logs/${task}.log"
done
