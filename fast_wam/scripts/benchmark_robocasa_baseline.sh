#!/bin/bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$ROOT_DIR"

STAMP="$(date +%Y%m%d_%H%M%S)"
BENCH_ROOT="${BENCH_ROOT:-$ROOT_DIR/outputs/benchmark_robocasa_baseline_$STAMP}"
PYTHON_BIN="${PYTHON_BIN:-python}"
PYTHON_BIN="$(command -v "$PYTHON_BIN")"
GPUS_PER_NODE="${GPUS_PER_NODE:-4}"
MICRO_BATCH_SIZE="${MICRO_BATCH_SIZE:-1}"
GLOBAL_BATCH_SIZE="${GLOBAL_BATCH_SIZE:-32}"
TRAIN_ITERS="${TRAIN_ITERS:-60}"
WARMUP_ITERS="${WARMUP_ITERS:-20}"

REQUIRED_CHECKPOINTS=(
  "$ROOT_DIR/checkpoints/Wan-AI/Wan2.2-TI2V-5B/diffusion_pytorch_model-00001-of-00003.safetensors"
  "$ROOT_DIR/checkpoints/Wan-AI/Wan2.2-TI2V-5B/diffusion_pytorch_model-00002-of-00003.safetensors"
  "$ROOT_DIR/checkpoints/Wan-AI/Wan2.2-TI2V-5B/diffusion_pytorch_model-00003-of-00003.safetensors"
  "$ROOT_DIR/checkpoints/Wan-AI/Wan2.2-TI2V-5B/Wan2.2_VAE.pth"
  "$ROOT_DIR/checkpoints/ActionDiT_linear_interp_Wan22_alphascale_1024hdim.pt"
)
for checkpoint in "${REQUIRED_CHECKPOINTS[@]}"; do
  if [ ! -f "$checkpoint" ]; then
    echo "Missing required baseline checkpoint: $checkpoint" >&2
    echo "Restore the baseline checkpoints symlink before benchmarking; refusing network download." >&2
    exit 1
  fi
done

if [ $((GLOBAL_BATCH_SIZE % (MICRO_BATCH_SIZE * GPUS_PER_NODE))) -ne 0 ]; then
  echo "GLOBAL_BATCH_SIZE must be divisible by MICRO_BATCH_SIZE*GPUS_PER_NODE" >&2
  exit 1
fi
GRADIENT_ACCUMULATION_STEPS=$((
  GLOBAL_BATCH_SIZE / (MICRO_BATCH_SIZE * GPUS_PER_NODE)
))

export PATH="$(dirname "$PYTHON_BIN"):$PATH"
export PYTHON="$PYTHON_BIN"
export PYTHONPATH="$ROOT_DIR/src:${PYTHONPATH:-}"
export GIT_PYTHON_REFRESH=quiet
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3}"
export RUN_ID="baseline_benchmark_$STAMP"
export WANDB_MODE=disabled

mkdir -p "$BENCH_ROOT"
bash scripts/train_zero1.sh "$GPUS_PER_NODE" \
  task=robocasa_acg_v1_fastwam_8gpu \
  output_dir="$BENCH_ROOT" \
  batch_size="$MICRO_BATCH_SIZE" \
  gradient_accumulation_steps="$GRADIENT_ACCUMULATION_STEPS" \
  max_steps="$TRAIN_ITERS" \
  log_every=1 \
  save_every=0 \
  save_final_checkpoint=false \
  eval_every=0 \
  wandb.enabled=false \
  profile.enabled=true \
  profile.warmup_steps="$WARMUP_ITERS" \
  profile.sync_cuda=true

"$PYTHON_BIN" fast_wam/scripts/summarize_robocasa_baseline_benchmark.py \
  "$BENCH_ROOT/profile/step_times.jsonl" \
  --output "$BENCH_ROOT/summary.json"
