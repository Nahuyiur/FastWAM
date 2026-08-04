#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT:-/mnt/yuhan/FastWAM_megatron_robocasa_webdataset}"
CHECKPOINT="${CHECKPOINT:?CHECKPOINT is required}"
OUT_ROOT="${OUT_ROOT:?OUT_ROOT is required}"
CERT_OUTPUT="${CERT_OUTPUT:?CERT_OUTPUT is required}"
PLAN="${PLAN:-$ROOT/fast_wam/configs/robocasa_prelaunch_eval_4.json}"
PYTHON="${PYTHON:-/mnt/yuhan/envs/motus-rebuilt-v2_10/bin/python}"
VAE="${VAE:-/mnt/yuhan/FastWAM/checkpoints/Wan-AI/Wan2.2-TI2V-5B/Wan2.2_VAE.pth}"
ROBOSUITE_REPO="${ROBOSUITE_REPO:-/mnt/yuhan/repos/robosuite}"
ROBOCASA_REPO="${ROBOCASA_REPO:-/mnt/yuhan/repos/robocasa}"
NVIDIA_DRIVER_VERSION="${NVIDIA_DRIVER_VERSION:-$(nvidia-smi --query-gpu=driver_version --format=csv,noheader | head -1 | tr -d ' ')}"
NVIDIA_EGL_ROOT="${NVIDIA_EGL_ROOT:-$ROOT/.runtime/nvidia-egl-${NVIDIA_DRIVER_VERSION}/root/usr/lib/x86_64-linux-gnu}"
NVIDIA_EGL_VENDOR_JSON="${NVIDIA_EGL_VENDOR_JSON:-$ROOT/fast_wam/runtime/10_nvidia.json}"
EXPECTED_ATTENTION_BACKEND="${EXPECTED_ATTENTION_BACKEND:-sdpa}"
EXPECTED_KERNEL_MODE="${EXPECTED_KERNEL_MODE:-reference}"

case "$EXPECTED_ATTENTION_BACKEND" in
  sdpa|structured_sdpa) ;;
  *) echo "Unsupported attention backend: $EXPECTED_ATTENTION_BACKEND" >&2; exit 2 ;;
esac
[[ "$EXPECTED_KERNEL_MODE" == "reference" ]] || {
  echo "Only the baseline-reference kernel mode is accepted" >&2
  exit 2
}

[[ -s "$CHECKPOINT/common.pt" && -s "$CHECKPOINT/.metadata" ]] || {
  echo "Incomplete smoke checkpoint: $CHECKPOINT" >&2
  exit 2
}
[[ ! -e "$OUT_ROOT/DONE" ]] || {
  echo "Prelaunch eval already completed: $OUT_ROOT" >&2
  exit 2
}
[[ -s "$NVIDIA_EGL_ROOT/libEGL_nvidia.so.${NVIDIA_DRIVER_VERSION}" ]] || {
  echo "Missing NVIDIA EGL runtime: $NVIDIA_EGL_ROOT" >&2
  exit 2
}

mkdir -p "$OUT_ROOT/logs"
export MUJOCO_GL=egl
export PYOPENGL_PLATFORM=egl
export LD_LIBRARY_PATH="$NVIDIA_EGL_ROOT:${LD_LIBRARY_PATH:-}"
export __EGL_VENDOR_LIBRARY_FILENAMES="$NVIDIA_EGL_VENDOR_JSON"
export TOKENIZERS_PARALLELISM=false
export PYTHONUNBUFFERED=1
export PYTHONPATH="$ROOT:$ROOT/src:$ROBOSUITE_REPO:$ROBOCASA_REPO:${PYTHONPATH:-}"
"$PYTHON" -c 'import OpenGL.EGL; print("robocasa_prelaunch_egl_import_ok")'

pids=()
for shard in 0 1 2 3; do
  shard_dir="$OUT_ROOT/shard_$(printf '%02d' "$shard")"
  mkdir -p "$shard_dir"
  (
    export CUDA_VISIBLE_DEVICES="$shard"
    export MASTER_ADDR=127.0.0.1
    export MASTER_PORT=$((29820 + shard))
    "$PYTHON" "$ROOT/scripts/robocasa_acg_eval.py" \
      --policy-backend fastwam_megatron \
      --action-layout base_first \
      --plan "$PLAN" \
      --bucket "periodic_shard_${shard}" \
      --output-dir "$shard_dir" \
      --seed 7 \
      --replan-steps 32 \
      --render-every 2 \
      --video-policy all \
      --fastwam-repo "$ROOT" \
      --fastwam-checkpoint "$CHECKPOINT" \
      --fastwam-vae-checkpoint "$VAE" \
      --fastwam-device cuda:0 \
      --fastwam-mixed-precision bf16 \
      --fastwam-num-video-frames 9 \
      --fastwam-action-horizon 32 \
      --fastwam-num-inference-steps 20 \
      --fastwam-action-dim 12 \
      --fastwam-proprio-dim 16
  ) >"$OUT_ROOT/logs/shard_$(printf '%02d' "$shard").log" 2>&1 &
  pids+=("$!")
done

failed=0
for pid in "${pids[@]}"; do
  if ! wait "$pid"; then
    failed=1
  fi
done
if [[ "$failed" -ne 0 ]]; then
  date -Is >"$OUT_ROOT/WORKER_FAILED"
  exit 3
fi

"$PYTHON" "$ROOT/fast_wam/scripts/summarize_robocasa_periodic_eval.py" \
  --root "$OUT_ROOT" \
  --checkpoint-step 40 \
  --expected-episodes 4 \
  --expected-replan-steps 32 \
  --expected-inference-steps 20 \
  --protocol-tag fastwam_formal_baseline_v1 \
  --expected-attention-backend "$EXPECTED_ATTENTION_BACKEND" \
  --expected-kernel-mode "$EXPECTED_KERNEL_MODE" \
  --expected-render-backend egl \
  >"$OUT_ROOT/summary.log" 2>&1
date -Is >"$OUT_ROOT/DONE"
"$PYTHON" "$ROOT/fast_wam/scripts/certify_robocasa_egl_eval.py" \
  --eval-dir "$OUT_ROOT" \
  --expected-checkpoint "$CHECKPOINT" \
  --min-episodes 4 \
  --output "$CERT_OUTPUT"
