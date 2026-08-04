#!/bin/bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$ROOT_DIR"

RUN_NAME="${RUN_NAME:-robocasa_megatron_offline50k_4gpu}"
RUN_ROOT="${RUN_ROOT:-$ROOT_DIR/outputs/$RUN_NAME}"
CACHE_ROOT="${CACHE_ROOT:-$ROOT_DIR/outputs/robocasa_megatron_assets/latents_train_id_full_baseline_exact_bs1_20260804}"
INITIAL_DCP="${INITIAL_DCP:-$ROOT_DIR/outputs/robocasa_megatron_assets/initial_dcp_bf16_baseline_exact_20260804}"
PYTHON_BIN="${PYTHON_BIN:-/mnt/yuhan/envs/motus-rebuilt-v2_10/bin/python}"
EXPECTED_SAMPLES="${EXPECTED_SAMPLES:-286101}"
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3}"
STOP_AFTER_CACHE="${STOP_AFTER_CACHE:-0}"
WANDB_PROJECT="${WANDB_PROJECT:-robocasa-acg-fastwam}"
WANDB_ENTITY="${WANDB_ENTITY:-ruiyuhan0110-southern-california-edison}"
WANDB_RUN_ID="${WANDB_RUN_ID:-$RUN_NAME}"
WANDB_SIDECAR_POLL_SECONDS="${WANDB_SIDECAR_POLL_SECONDS:-30}"
BASELINE_REPO="${BASELINE_REPO:-/mnt/yuhan/FastWAM_robocasa_acg_8gpu}"
BASELINE_COMMIT="${BASELINE_COMMIT:-f86adf7b5b4d352ef615690493286a2b57288059}"
VAE_CHECKPOINT="${VAE_CHECKPOINT:-/mnt/yuhan/FastWAM/checkpoints/Wan-AI/Wan2.2-TI2V-5B/Wan2.2_VAE.pth}"
ACTION_CHECKPOINT="${ACTION_CHECKPOINT:-/mnt/yuhan/FastWAM/checkpoints/ActionDiT_linear_interp_Wan22_alphascale_1024hdim.pt}"
PARITY_GATE_DIR="${PARITY_GATE_DIR:-}"

mkdir -p "$RUN_ROOT" "$CACHE_ROOT"
exec 9>"$RUN_ROOT/pipeline.lock"
if ! flock -n 9; then
  echo "Another cache/train pipeline already holds $RUN_ROOT/pipeline.lock" >&2
  exit 1
fi
echo $$ >"$RUN_ROOT/pipeline.pid"

on_exit() {
  local code=$?
  printf '%s\n' "$code" >"$RUN_ROOT/pipeline.exit_code"
  date -Iseconds >"$RUN_ROOT/pipeline.finished_at"
}
trap on_exit EXIT

if [ -f "$RUN_ROOT/DONE" ]; then
  echo "Training already completed: $RUN_ROOT/DONE"
  exit 0
fi
if [ ! -x "$PYTHON_BIN" ]; then
  echo "Missing Python runtime: $PYTHON_BIN" >&2
  exit 1
fi
if [ ! -f "$INITIAL_DCP/fast_wam_initialization.json" ]; then
  echo "Missing initial DCP: $INITIAL_DCP" >&2
  exit 1
fi
if [ -z "$PARITY_GATE_DIR" ]; then
  echo "PARITY_GATE_DIR is required; refusing an uncertified formal launch" >&2
  exit 1
fi

export PYTHONPATH="$ROOT_DIR:$ROOT_DIR/src:${PYTHONPATH:-}"
export GIT_PYTHON_REFRESH=quiet
export CUDA_VISIBLE_DEVICES

cache_complete() {
  "$PYTHON_BIN" - "$CACHE_ROOT/manifest.json" "$EXPECTED_SAMPLES" <<'PY'
import json
import sys
from pathlib import Path

path = Path(sys.argv[1])
expected = int(sys.argv[2])
if not path.is_file():
    raise SystemExit(1)
manifest = json.loads(path.read_text())
valid = (
    manifest.get("complete") is True
    and int(manifest.get("num_samples", -1)) == expected
    and manifest.get("dtype") == "bfloat16"
    and int(manifest.get("encoding_batch_size", -1)) == 1
    and manifest.get("sample_shape") == [48, 3, 14, 28]
    and manifest.get("split") == "train"
)
raise SystemExit(0 if valid else 1)
PY
}

date -Iseconds >"$RUN_ROOT/pipeline.started_at"
if cache_complete; then
  echo "Reusing complete BF16 mmap cache: $CACHE_ROOT"
else
  echo "Building/resuming complete BF16 mmap cache: $CACHE_ROOT"
  OUTPUT="$CACHE_ROOT" \
  SPLIT=train \
  GPUS_PER_NODE=4 \
  BATCH_SIZE=1 \
  NUM_WORKERS=4 \
  SAMPLES_PER_SHARD=1024 \
  MASTER_PORT=29612 \
  PYTHON_BIN="$PYTHON_BIN" \
  bash fast_wam/scripts/prepare_robocasa_latents.sh \
    2>&1 | tee "$RUN_ROOT/cache.log"
fi

if ! cache_complete; then
  echo "Full latent cache failed validation: $CACHE_ROOT" >&2
  exit 1
fi
date -Iseconds >"$RUN_ROOT/CACHE_DONE"
if [ "$STOP_AFTER_CACHE" = "1" ]; then
  echo "Cache validation complete; STOP_AFTER_CACHE=1"
  exit 0
fi

mkdir -p "$PARITY_GATE_DIR"
"$PYTHON_BIN" fast_wam/scripts/audit_robocasa_training_contract.py \
  --repo-root "$ROOT_DIR" \
  --baseline-repo "$BASELINE_REPO" \
  --baseline-commit "$BASELINE_COMMIT" \
  --latent-cache "$CACHE_ROOT" \
  --initial-dcp "$INITIAL_DCP" \
  --vae-checkpoint "$VAE_CHECKPOINT" \
  --action-checkpoint "$ACTION_CHECKPOINT" \
  --world-size 4 \
  --tensor-parallel-size 1 \
  --micro-batch-size 1 \
  --global-batch-size 32 \
  --train-iters 50000 \
  --attention-backend structured_sdpa \
  --kernel-mode reference \
  --optimizer-weight-decay-policy all_trainable \
  --output "$PARITY_GATE_DIR/training_contract.json"

"$PYTHON_BIN" - "$PARITY_GATE_DIR" "$INITIAL_DCP" "$CACHE_ROOT" <<'PY'
import json
import sys
from pathlib import Path

root = Path(sys.argv[1])
initial_dcp = str(Path(sys.argv[2]).resolve())
latent_cache = str(Path(sys.argv[3]).resolve())
required = (
    "training_contract.json",
    "action_initialization_parity.json",
    "dcp_initialization_parity.json",
    "latent_online_parity.json",
    "full_model_parity.json",
    "training_smoke.json",
    "fresh_egl_video_gate.json",
)
for name in required:
    path = root / name
    if not path.is_file():
        raise SystemExit(f"Missing required parity certificate: {path}")
    payload = json.loads(path.read_text())
    if payload.get("status") != "PASS":
        raise SystemExit(f"Parity certificate is not PASS: {path}")

dcp = json.loads((root / "dcp_initialization_parity.json").read_text())
if dcp.get("initial_dcp") != initial_dcp:
    raise SystemExit("DCP parity certificate belongs to a different initialization")
latent = json.loads((root / "latent_online_parity.json").read_text())
if latent.get("latent_cache") != latent_cache:
    raise SystemExit("Latent parity certificate belongs to a different cache")
full = json.loads((root / "full_model_parity.json").read_text())
if full.get("initial_dcp") != initial_dcp:
    raise SystemExit("Full-model parity certificate belongs to a different initialization")
candidate = full.get("candidate") or {}
if candidate.get("backend") != "structured_sdpa" or candidate.get("kernel") != "reference":
    raise SystemExit("Full-model parity certificate has the wrong training backend/kernel")
smoke = json.loads((root / "training_smoke.json").read_text())
if smoke.get("initial_dcp") != initial_dcp or smoke.get("latent_cache") != latent_cache:
    raise SystemExit("Training smoke certificate belongs to different launch assets")
smoke_candidate = smoke.get("candidate") or {}
if smoke_candidate != {
    "attention_backend": "structured_sdpa",
    "kernel_mode": "reference",
    "optimizer": "AdamW",
    "weight_decay_policy": "all_trainable",
}:
    raise SystemExit("Training smoke certificate has the wrong optimizer/kernel contract")
fresh_egl = json.loads((root / "fresh_egl_video_gate.json").read_text())
if fresh_egl.get("source_checkpoint") != smoke.get("checkpoint"):
    raise SystemExit("Fresh EGL certificate is not tied to the training smoke checkpoint")
PY

if pgrep -af "[f]ast_wam.pretrain_robocasa" >/dev/null; then
  echo "A FastWAM RoboCasa training process is already running; refusing duplicate launch" >&2
  exit 1
fi

date -Iseconds >"$RUN_ROOT/TRAIN_STARTED"
SIDECAR_ROOT="$RUN_ROOT/wandb_sidecar"
mkdir -p "$SIDECAR_ROOT"
rm -f "$SIDECAR_ROOT/STOP"
"$PYTHON_BIN" fast_wam/scripts/sync_megatron_train_log_to_wandb.py \
  --log-path "$RUN_ROOT/train.log" \
  --state-path "$SIDECAR_ROOT/state.json" \
  --entity "$WANDB_ENTITY" \
  --project "$WANDB_PROJECT" \
  --run-id "$WANDB_RUN_ID" \
  --run-name "$RUN_NAME" \
  --wandb-dir "$SIDECAR_ROOT/wandb" \
  --poll-seconds "$WANDB_SIDECAR_POLL_SECONDS" \
  --stop-file "$SIDECAR_ROOT/STOP" \
  --follow \
  >"$SIDECAR_ROOT/sidecar.log" 2>&1 &
SIDECAR_PID=$!
printf '%s\n' "$SIDECAR_PID" >"$SIDECAR_ROOT/sidecar.pid"
sleep 2
if ! kill -0 "$SIDECAR_PID" 2>/dev/null; then
  echo "W&B sidecar exited during initialization; refusing an unobservable run" >&2
  exit 1
fi

stop_sidecar() {
  if kill -0 "$SIDECAR_PID" 2>/dev/null; then
    touch "$SIDECAR_ROOT/STOP"
    for _ in $(seq 1 40); do
      if ! kill -0 "$SIDECAR_PID" 2>/dev/null; then
        break
      fi
      sleep 1
    done
  fi
  if kill -0 "$SIDECAR_PID" 2>/dev/null; then
    kill "$SIDECAR_PID" 2>/dev/null || true
    wait "$SIDECAR_PID" 2>/dev/null || true
  fi
}
trap stop_sidecar EXIT

INITIAL_DCP="$INITIAL_DCP" \
TRAIN_LATENT_CACHE="$CACHE_ROOT" \
VAE_CHECKPOINT="$VAE_CHECKPOINT" \
SAVE_DIR="$RUN_ROOT" \
LOG_FILE="$RUN_ROOT/train.log" \
PYTHON_BIN="$PYTHON_BIN" \
GPUS_PER_NODE=4 \
TP_SIZE=1 \
MICRO_BATCH_SIZE=1 \
GLOBAL_BATCH_SIZE=32 \
NUM_WORKERS=2 \
TRAIN_ITERS=50000 \
LEARNING_RATE=5e-5 \
MIN_LR=5e-7 \
LR_WARMUP_INIT=2e-8 \
LR_WARMUP_FRACTION=0.05 \
WEIGHT_DECAY=0.01 \
ATTENTION_BACKEND=structured_sdpa \
KERNEL_MODE=reference \
SAVE_INTERVAL=5000 \
SAVE_CHECKPOINTS=1 \
EVAL_INTERVAL=0 \
LOG_INTERVAL=20 \
WANDB_ENABLED=1 \
WANDB_PROJECT="$WANDB_PROJECT" \
WANDB_ENTITY="$WANDB_ENTITY" \
WANDB_EXP_NAME="$RUN_NAME" \
WANDB_RUN_ID="$WANDB_RUN_ID" \
WANDB_RESUME=allow \
WANDB_MODE=online \
WANDB_SAVE_DIR="$RUN_ROOT/wandb" \
OVERLAP_PARAM_GATHER=1 \
MASTER_PORT=29613 \
bash fast_wam/scripts/train_robocasa_megatron.sh

stop_sidecar
trap - EXIT
date -Iseconds >"$RUN_ROOT/DONE"
