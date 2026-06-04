#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
FASTWAM_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${FASTWAM_ROOT}"
source scripts/setup_yuhan_paths.sh

extract_one() {
  local split_dir="$1"
  local expected_seed="$2"
  local archive="${GEMBENCH_ROOT}/${split_dir}/microsteps.tar.gz"
  local seed_dir="${GEMBENCH_ROOT}/${split_dir}/microsteps/${expected_seed}"

  if [[ -d "${seed_dir}" ]]; then
    echo "[microsteps] exists: ${seed_dir}"
    return 0
  fi
  if [[ ! -f "${archive}" ]]; then
    echo "[microsteps] missing archive: ${archive}" >&2
    return 1
  fi
  echo "[microsteps] extracting ${archive} -> ${GEMBENCH_ROOT}/${split_dir}"
  tar -xzf "${archive}" -C "${GEMBENCH_ROOT}/${split_dir}"
  if [[ ! -d "${seed_dir}" ]]; then
    echo "[microsteps] expected ${seed_dir} after extraction, but it is missing." >&2
    return 1
  fi
}

extract_one "val_dataset" "seed100"
extract_one "test_dataset" "seed200"

for seed in seed300 seed400 seed500 seed600; do
  if [[ ! -d "${GEMBENCH_ROOT}/test_dataset/microsteps/${seed}" ]]; then
    echo "[microsteps] expected test ${seed}; test_dataset archive may be incomplete." >&2
    exit 1
  fi
done

echo "[microsteps] ready under ${GEMBENCH_ROOT}"
