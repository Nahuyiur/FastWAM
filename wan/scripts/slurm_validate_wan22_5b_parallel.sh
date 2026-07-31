#!/bin/bash
# Validate Wan2.2-TI2V-5B Megatron parallel combinations on one debug node.
# Submit from repo root:
#   sbatch wan/scripts/slurm_validate_wan22_5b_parallel.sh

#SBATCH --job-name=wan22_5b_parallel
#SBATCH --partition=audio-debug
#SBATCH --account=xuewei
#SBATCH --qos=for_xuewei_debug
#SBATCH --nodes=1
#SBATCH --nodelist=dgx-043
#SBATCH --gres=gpu:8
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=96
#SBATCH --time=02:30:00
#SBATCH --output=/aifs4su/mmcode/codeclm/Megatron-Wan/wan/log/slurm/%x_%j.out

set -euo pipefail

unset HTTP_PROXY HTTPS_PROXY ALL_PROXY NO_PROXY
unset http_proxy https_proxy all_proxy no_proxy

CONTAINER=/aifs4su/mmcode/codeclm/containers/pytorch-25.09-py3.sqsh
MOUNTS="/aifs4su/mmcode/codeclm/Megatron-Wan:/workspace/megatron,\
/aifs4su/mmcode/codeclm/dataset:/workspace/dataset,\
/aifs4su/mmcode/codeclm/checkpoints:/workspace/checkpoints,\
/aifs4su/mmcode/codeclm:/aifs4su/mmcode/codeclm,\
/aifs4su/mmdata:/aifs4su/mmdata"

srun --container-image="$CONTAINER" \
     --container-mounts="$MOUNTS" \
     --container-workdir=/workspace/megatron \
     bash -lc '
set -euo pipefail
cd /workspace/megatron

export PYTHONPATH=/workspace/megatron
export CUDA_DEVICE_MAX_CONNECTIONS=1
export OMP_NUM_THREADS=10
export NCCL_SOCKET_IFNAME=ibp
export NCCL_IB_HCA=mlx5
export NCCL_ASYNC_ERROR_HANDLING=1
export NCCL_SOCKET_TIMEOUT=3600
export NCCL_IB_TIMEOUT=3600
export NCCL_DEBUG=WARN
export GLOO_SOCKET_IFNAME=bond0
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

RUN_ID="slurm_${SLURM_JOB_ID}"
ROOT_SAVE="/workspace/checkpoints/wan22_5b_parallel_${RUN_ID}"
mkdir -p "$ROOT_SAVE"
MAX_GPUS="${WAN_VALIDATE_MAX_GPUS:-8}"

SAMPLE=/workspace/checkpoints/wan_tmp/overfit_mp4_real_preencoded_wan22_ti2v_512x384_49f_12fps.pt
OFFICIAL=/workspace/checkpoints/wan/Wan-AI/Wan2.2-TI2V-5B
LOAD_OFFICIAL="${WAN_VALIDATE_LOAD_OFFICIAL:-1}"

run_case() {
  local name="$1"
  local gpus="$2"
  local cuda_visible="$3"
  local port="$4"
  shift 4
  echo
  echo "===== WAN22 5B CASE ${name} START $(date -Is) ====="
  (
    export CUDA_VISIBLE_DEVICES="$cuda_visible"
    export WAN_PRESET=ti2v-5b
    export SAMPLE_PATH="$SAMPLE"
    if [ "$LOAD_OFFICIAL" = "1" ]; then
      export WAN_LOAD_OFFICIAL_CKPT="$OFFICIAL"
    else
      unset WAN_LOAD_OFFICIAL_CKPT
    fi
    export SAVE_DIR="${ROOT_SAVE}/${name}"
    export TRAIN_ITERS=1
    export LOG_INTERVAL=1
    export NO_SAVE=1
    export GPUS_PER_NODE="$gpus"
    export LR=1e-5
    export MIN_LR=1e-6
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
  echo "===== WAN22 5B CASE ${name} DONE $(date -Is) ====="
}

run_case tp2 2 0,1 25501 "export TP_SIZE=2; export PP_SIZE=1; export CP_SIZE=1"
run_case cp2 2 0,1 25502 "export TP_SIZE=1; export PP_SIZE=1; export CP_SIZE=2"
run_case tp2_sp 2 0,1 25503 "export TP_SIZE=2; export PP_SIZE=1; export CP_SIZE=1; export SEQUENCE_PARALLEL=1"
run_case tp2_cp2 4 0,1,2,3 25504 "export TP_SIZE=2; export PP_SIZE=1; export CP_SIZE=2"
run_case tp2_cp2_sp 4 0,1,2,3 25505 "export TP_SIZE=2; export PP_SIZE=1; export CP_SIZE=2; export SEQUENCE_PARALLEL=1"
run_case pp2 2 0,1 25506 "export TP_SIZE=1; export PP_SIZE=2; export CP_SIZE=1"
run_case pp2_cp2 4 0,1,2,3 25507 "export TP_SIZE=1; export PP_SIZE=2; export CP_SIZE=2"
run_case pp2_tp2_sp 4 0,1,2,3 25508 "export TP_SIZE=2; export PP_SIZE=2; export CP_SIZE=1; export SEQUENCE_PARALLEL=1"
if [ "$MAX_GPUS" -ge 8 ]; then
  run_case pp2_tp2_cp2_sp 8 0,1,2,3,4,5,6,7 25509 "export TP_SIZE=2; export PP_SIZE=2; export CP_SIZE=2; export SEQUENCE_PARALLEL=1"
else
  echo "===== SKIP pp2_tp2_cp2_sp: WAN_VALIDATE_MAX_GPUS=${MAX_GPUS} < 8 ====="
fi

echo
echo "===== WAN22 5B ALL CASES COMPLETE $(date -Is) ====="
find "$ROOT_SAVE" -maxdepth 2 -name overfit.log -print
'
