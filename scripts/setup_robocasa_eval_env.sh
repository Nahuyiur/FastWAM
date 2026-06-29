#!/usr/bin/env bash
set -euo pipefail

REPO_DIR="${FASTWAM_REPO:-/mnt/yuhan/FastWAM_robocasa_acg_8gpu}"
ROOT="${ROBOCASA_EVAL_ROOT:-/mnt/yuhan/experiments/robocasa_acg_v1/fastwam/eval}"
PYTHON="${PYTHON:-/opt/conda/envs/motus/bin/python}"
ROBOSUITE_REPO="${ROBOSUITE_REPO:-/mnt/yuhan/repos/robosuite}"
ROBOCASA_REPO="${ROBOCASA_REPO:-/mnt/yuhan/repos/robocasa}"

mkdir -p "${ROOT}/logs" "$(dirname "${ROBOSUITE_REPO}")" "$(dirname "${ROBOCASA_REPO}")"

clone_if_missing() {
  local url="$1"
  local dst="$2"
  if [[ ! -d "${dst}/.git" ]]; then
    git clone --depth 1 "${url}" "${dst}"
  fi
}

ensure_private_macro() {
  local repo="$1"
  local package="$2"
  local src="${repo}/${package}/macros.py"
  local dst="${repo}/${package}/macros_private.py"
  if [[ -f "${src}" && ! -f "${dst}" ]]; then
    cp "${src}" "${dst}"
  fi
}

python_ok() {
  PYTHONPATH="${REPO_DIR}/src:${ROBOSUITE_REPO}:${ROBOCASA_REPO}:${PYTHONPATH:-}" "${PYTHON}" - <<'PY'
import gymnasium
import imageio
import mujoco
import robocasa
import robosuite
assert mujoco.__version__ == "3.3.1", mujoco.__version__
print("robocasa_eval_import_ok", robocasa.__path__[0], robosuite.__path__[0], mujoco.__version__)
PY
}

if ! python_ok; then
  echo "[setup] preparing RoboCasa eval env under ${ROOT}" | tee -a "${ROOT}/logs/setup_robocasa_eval_env.log"
  clone_if_missing "https://github.com/ARISE-Initiative/robosuite.git" "${ROBOSUITE_REPO}"
  clone_if_missing "https://github.com/robocasa/robocasa.git" "${ROBOCASA_REPO}"
  "${PYTHON}" -m pip install -U pip setuptools wheel >> "${ROOT}/logs/setup_robocasa_eval_env.log" 2>&1
  "${PYTHON}" -m pip install "mujoco==3.3.1" gymnasium imageio imageio-ffmpeg tqdm pillow >> "${ROOT}/logs/setup_robocasa_eval_env.log" 2>&1
  "${PYTHON}" -m pip install -e "${ROBOSUITE_REPO}" >> "${ROOT}/logs/setup_robocasa_eval_env.log" 2>&1
  "${PYTHON}" -m pip install -e "${ROBOCASA_REPO}" >> "${ROOT}/logs/setup_robocasa_eval_env.log" 2>&1
fi

ensure_private_macro "${ROBOSUITE_REPO}" "robosuite"
ensure_private_macro "${ROBOCASA_REPO}" "robocasa"

ASSET_SENTINEL="${ROBOCASA_REPO}/.robocasa_assets_all.done"
if [[ ! -f "${ASSET_SENTINEL}" ]]; then
  echo "[setup] downloading RoboCasa kitchen assets; this is about 10GB" | tee -a "${ROOT}/logs/setup_robocasa_eval_env.log"
  PYTHONPATH="${REPO_DIR}/src:${ROBOSUITE_REPO}:${ROBOCASA_REPO}:${PYTHONPATH:-}" \
    bash -lc "printf 'y\n' | '${PYTHON}' -m robocasa.scripts.download_kitchen_assets --type all" \
    >> "${ROOT}/logs/setup_robocasa_eval_env.log" 2>&1
  date -Is > "${ASSET_SENTINEL}"
fi

python_ok | tee -a "${ROOT}/logs/setup_robocasa_eval_env.log"
echo "[setup] done"
