#!/bin/bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$ROOT_DIR"
export PYTHONPATH="$ROOT_DIR:$ROOT_DIR/src:${PYTHONPATH:-}"
export GIT_PYTHON_REFRESH=quiet

WAN_ROOT="${WAN_ROOT:-/mnt/yuhan/FastWAM/checkpoints/Wan-AI/Wan2.2-TI2V-5B}"
WAN_CHECKPOINT="${WAN_CHECKPOINT:-$WAN_ROOT}"
OUTPUT="${OUTPUT:-$ROOT_DIR/outputs/robocasa_megatron_assets/initial_dcp_bf16}"
TP_SIZE="${TP_SIZE:-1}"
MASTER_PORT="${MASTER_PORT:-29601}"
PYTHON_BIN="${PYTHON_BIN:-python}"
PYTHON_BIN="$(command -v "$PYTHON_BIN")"
export PATH="$(dirname "$PYTHON_BIN"):$PATH"

mkdir -p "$(dirname "$OUTPUT")"
"$PYTHON_BIN" -m torch.distributed.run --standalone --nproc_per_node "$TP_SIZE" --master_port "$MASTER_PORT" \
  -m fast_wam.train.initialization \
  --wan-checkpoint "$WAN_CHECKPOINT" \
  --output "$OUTPUT" \
  --tp "$TP_SIZE" \
  --dtype bfloat16 \
  --seed 42 \
  --action-dim 12 \
  --proprio-dim 16
