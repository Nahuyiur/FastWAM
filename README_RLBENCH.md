# FastWAM RLBench Notes

This repository includes a small, non-invasive RLBench setup for the
`pick_lift_color_shape` LeRobot dataset used on `baidu_ryh_4gpu`.

## Environment

All scripts default to the shared machine layout:

- FastWAM root: `/mnt/world_foundational_model/yuhan/FastWAM`
- Conda env: `/mnt/world_foundational_model/yuhan/miniconda3/envs/fastwam`
- Cache/temp root: `/mnt/world_foundational_model/yuhan/cache/FastWAM`
- Pretrained weights: `/mnt/world_foundational_model/yuhan/pretrained_weights/FastWAM`
- RLBench checkout: `/mnt/world_foundational_model/yuhan/RLBench`

Run scripts source `scripts/setup_yuhan_paths.sh` so Hugging Face, Torch,
W&B, XDG, and temporary files stay under `/mnt/world_foundational_model/yuhan`
instead of `/tmp`.

## Training

Start 4-GPU training for one Hydra task:

```bash
bash scripts/train_rlbench_4gpu.sh [task_name]
```

The default task is `rlbench_uncond_3cam224_1e-4`.

Available task configs:

- `rlbench_uncond_3cam224_1e-4`
- `rlbench_original_3cam224_1e-4`
- `rlbench_color_3cam224_1e-4`
- `rlbench_shape_3cam224_1e-4`
- `rlbench_color_shape_3cam224_1e-4`

The RLBench task script internally uses ZeRO-2 through
`ACCELERATE_CONFIG_FILE`, while the generic `scripts/train_zero1.sh` keeps its
original ZeRO-1 default unless this environment variable is set.

## Offline Video Eval

Generate 129-frame long final-eval videos from the latest checkpoint of each
task:

```bash
source scripts/setup_yuhan_paths.sh
CUDA_VISIBLE_DEVICES=0 python scripts/eval_rlbench_final_long_video.py --num-frames 129
```

## True Success-Rate Eval

Run true RLBench simulator success-rate evaluation, 20 trials per task:

```bash
bash scripts/eval_rlbench_success_20_bg.sh
```

This starts one tmux window per task and writes:

- `summary.json`
- `results.jsonl`
- rollout videos under `videos/*.mp4`
- per-task logs under `logs/*.log`

The success eval reuses the runtime color/shape task wrapper from the dataset
generation workflow and executes predicted 8-D Panda joint-position + gripper
actions in RLBench.
