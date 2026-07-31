#!/bin/bash
# Exhaustive Wan2.2-TI2V-5B Megatron parallel smoke matrix for one 8-GPU node.
#
# Run inside the standard Megatron container from /workspace/megatron.
# The default matrix uses random-init 5B weights and the real pre-encoded
# overfit.mp4 sample; set WAN_MATRIX_LOAD_OFFICIAL=1 to load official DiT
# weights for every case.

set -euo pipefail

export CUDA_DEVICE_MAX_CONNECTIONS=${CUDA_DEVICE_MAX_CONNECTIONS:-1}
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-10}
export NCCL_SOCKET_IFNAME=${NCCL_SOCKET_IFNAME:-ibp}
export NCCL_IB_HCA=${NCCL_IB_HCA:-mlx5}
export NCCL_ASYNC_ERROR_HANDLING=${NCCL_ASYNC_ERROR_HANDLING:-1}
export NCCL_SOCKET_TIMEOUT=${NCCL_SOCKET_TIMEOUT:-3600}
export NCCL_IB_TIMEOUT=${NCCL_IB_TIMEOUT:-3600}
export NCCL_DEBUG=${NCCL_DEBUG:-WARN}
export GLOO_SOCKET_IFNAME=${GLOO_SOCKET_IFNAME:-bond0}
export PYTORCH_CUDA_ALLOC_CONF=${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}

cd "${ROOT_DIR:-/workspace/megatron}"
export PYTHONPATH="${PYTHONPATH:-/workspace/megatron}"

RUN_ID="${WAN_MATRIX_RUN_ID:-$(date +%Y%m%d_%H%M%S)}"
ROOT_SAVE="${WAN_MATRIX_ROOT_SAVE:-/workspace/checkpoints/wan22_5b_parallel_matrix_${RUN_ID}}"
SAMPLE="${WAN_MATRIX_SAMPLE:-/workspace/checkpoints/wan_tmp/overfit_mp4_real_preencoded_wan22_ti2v_512x384_49f_12fps.pt}"
OFFICIAL="${WAN_MATRIX_OFFICIAL:-/workspace/checkpoints/wan/Wan-AI/Wan2.2-TI2V-5B}"
LOAD_OFFICIAL="${WAN_MATRIX_LOAD_OFFICIAL:-0}"
MAX_GPUS="${WAN_MATRIX_MAX_GPUS:-8}"
mkdir -p "$ROOT_SAVE"
SUMMARY="$ROOT_SAVE/summary.tsv"
echo -e "case\tgpus\tstatus\tseconds" > "$SUMMARY"

run_case() {
  local name="$1"
  local gpus="$2"
  local cuda_visible="$3"
  local port="$4"
  shift 4
  if [ "$gpus" -gt "$MAX_GPUS" ]; then
    echo "===== SKIP ${name}: requires ${gpus} GPUs, WAN_MATRIX_MAX_GPUS=${MAX_GPUS} ====="
    return
  fi
  echo
  echo "===== WAN22 5B MATRIX CASE ${name} START $(date -Is) ====="
  local start_ts
  start_ts="$(date +%s)"
  set +e
  (
    set -euo pipefail
    export CUDA_VISIBLE_DEVICES="$cuda_visible"
    export WAN_PRESET=ti2v-5b
    export SAMPLE_PATH="$SAMPLE"
    if [ "$LOAD_OFFICIAL" = "1" ]; then
      export WAN_LOAD_OFFICIAL_CKPT="$OFFICIAL"
    else
      unset WAN_LOAD_OFFICIAL_CKPT
    fi
    export SAVE_DIR="${ROOT_SAVE}/${name}"
    export TRAIN_ITERS="${WAN_MATRIX_TRAIN_ITERS:-8}"
    export LOG_INTERVAL=1
    export NO_SAVE=1
    export GPUS_PER_NODE="$gpus"
    export LR="${WAN_MATRIX_LR:-1e-5}"
    export MIN_LR="${WAN_MATRIX_MIN_LR:-1e-6}"
    export WAN_GRADIENT_CHECKPOINTING=1
    export USE_DISTRIBUTED_OPTIMIZER=1
    export RECOMPUTE_GRANULARITY=full
    export RECOMPUTE_METHOD=uniform
    export RECOMPUTE_NUM_LAYERS=1
    export MASTER_PORT="$port"
    unset SEQUENCE_PARALLEL
    eval "$@"
    bash wan/scripts/overfit.sh
  )
  local status=$?
  set -e
  local end_ts
  end_ts="$(date +%s)"
  echo -e "${name}\t${gpus}\t${status}\t$((end_ts - start_ts))" >> "$SUMMARY"
  if [ "$status" -ne 0 ]; then
    echo "===== WAN22 5B MATRIX CASE ${name} FAILED status=${status} $(date -Is) ====="
    return 0
  fi
  echo "===== WAN22 5B MATRIX CASE ${name} DONE $(date -Is) ====="
}

run_case tp1                  1 0               25601 "export TP_SIZE=1; export CP_SIZE=1; export PP_SIZE=1"
run_case cp2                  2 0,1             25602 "export TP_SIZE=1; export CP_SIZE=2; export PP_SIZE=1"
run_case pp2                  2 0,1             25603 "export TP_SIZE=1; export CP_SIZE=1; export PP_SIZE=2"
run_case pp2_cp2              4 0,1,2,3         25604 "export TP_SIZE=1; export CP_SIZE=2; export PP_SIZE=2"

run_case tp2                  2 0,1             25605 "export TP_SIZE=2; export CP_SIZE=1; export PP_SIZE=1"
run_case tp2_sp               2 0,1             25606 "export TP_SIZE=2; export CP_SIZE=1; export PP_SIZE=1; export SEQUENCE_PARALLEL=1"
run_case tp2_cp2              4 0,1,2,3         25607 "export TP_SIZE=2; export CP_SIZE=2; export PP_SIZE=1"
run_case tp2_cp2_sp           4 0,1,2,3         25608 "export TP_SIZE=2; export CP_SIZE=2; export PP_SIZE=1; export SEQUENCE_PARALLEL=1"
run_case tp2_pp2              4 0,1,2,3         25609 "export TP_SIZE=2; export CP_SIZE=1; export PP_SIZE=2"
run_case tp2_pp2_sp           4 0,1,2,3         25610 "export TP_SIZE=2; export CP_SIZE=1; export PP_SIZE=2; export SEQUENCE_PARALLEL=1"
run_case tp2_pp2_cp2          8 0,1,2,3,4,5,6,7 25611 "export TP_SIZE=2; export CP_SIZE=2; export PP_SIZE=2"
run_case tp2_pp2_cp2_sp       8 0,1,2,3,4,5,6,7 25612 "export TP_SIZE=2; export CP_SIZE=2; export PP_SIZE=2; export SEQUENCE_PARALLEL=1"

run_case tp4                  4 0,1,2,3         25613 "export TP_SIZE=4; export CP_SIZE=1; export PP_SIZE=1"
run_case tp4_sp               4 0,1,2,3         25614 "export TP_SIZE=4; export CP_SIZE=1; export PP_SIZE=1; export SEQUENCE_PARALLEL=1"
run_case tp4_cp2              8 0,1,2,3,4,5,6,7 25615 "export TP_SIZE=4; export CP_SIZE=2; export PP_SIZE=1"
run_case tp4_cp2_sp           8 0,1,2,3,4,5,6,7 25616 "export TP_SIZE=4; export CP_SIZE=2; export PP_SIZE=1; export SEQUENCE_PARALLEL=1"
run_case tp4_pp2              8 0,1,2,3,4,5,6,7 25617 "export TP_SIZE=4; export CP_SIZE=1; export PP_SIZE=2"
run_case tp4_pp2_sp           8 0,1,2,3,4,5,6,7 25618 "export TP_SIZE=4; export CP_SIZE=1; export PP_SIZE=2; export SEQUENCE_PARALLEL=1"

echo
echo "===== WAN22 5B MATRIX ALL CASES COMPLETE $(date -Is) ====="
echo "root_save=$ROOT_SAVE"
find "$ROOT_SAVE" -maxdepth 2 -name overfit.log -print
