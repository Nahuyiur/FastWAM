# FastWAM GEMBench v1

This branch adds a direct LMDB adapter for training FastWAM on GEMBench
`train_dataset/keysteps_bbox/seed0`.

## Data Contract

- GEMBench root defaults to `/mnt/yuhan/datasets/GEMBench`.
- Complete taskvars are directories with both `data.mdb` and `results.json`.
- Incomplete taskvars are skipped by default. Set `STRICT_GEMBENCH_COMPLETE=1`
  to fail before training if Hugging Face still has `.incomplete` files.
- The v1 adapter reads official LMDB values:
  - `rgb`: `[T, N, H, W, 3]`
  - `action`: `[T, 8]`
- `key_frameids`: retained in LMDB but not needed for tensor construction
- Raw camera order is assumed to be
  `left_shoulder,right_shoulder,wrist,front`, matching the official
  `robot-3dlotus/preprocess/generate_dataset_keysteps.py` default camera list.
- Training camera order defaults to `front,wrist,left_shoulder` and is
  concatenated horizontally into video shape `[3, 9, 224, 672]`.
- `action` is the next sampled keyframe pose for each transition.
- `proprio` is the current sampled keyframe pose for each transition.
- Text context cache uses FastWAM's `DEFAULT_PROMPT` around GEMBench task
  instructions.

## Files

- `src/fastwam/datasets/gembench/`: dataset adapter, LMDB reader,
  instructions, and normalization shim.
- `configs/data/gembench_keysteps_bbox.yaml`: train/val dataset config.
- `configs/task/gembench_keysteps_bbox_3cam224_1e-4.yaml`: training config.
- `scripts/precompute_gembench_text_embeds.py`: Wan/T5 text cache builder.
- `scripts/precompute_gembench_norm_stats.py`: one-time action/proprio stats
  builder for `data/gembench_keysteps_bbox_dataset_stats.json`.
- `scripts/smoke_gembench_dataset.py`: simple shape/video smoke test.
- `scripts/verify_gembench_dataset_contract.py`: stricter contract test for
  shapes, split disjointness, text cache, normalization round-trip, eval sample
  batching, and multi-worker LMDB loading.
- `scripts/train_gembench_4gpu.sh`: main jinshan training entrypoint.

## Verification

```bash
cd /mnt/yuhan/FastWAM

/mnt/miniconda3/envs/fastwam/bin/python scripts/smoke_gembench_dataset.py --num-samples 3

/mnt/miniconda3/envs/fastwam/bin/python scripts/verify_gembench_dataset_contract.py --num-samples 8

/mnt/miniconda3/envs/fastwam/bin/python scripts/precompute_gembench_norm_stats.py

GEMBENCH_PREP_ONLY=1 bash scripts/train_gembench_4gpu.sh
```

The prep-only command runs dependency checks, text-cache precompute for complete
local taskvars, and the contract verifier without launching training. The
normalization precompute is a one-time full LMDB scan; the training config reads
the resulting stats with `norm_default_mode: z-score`.

## Training

```bash
cd /mnt/yuhan/FastWAM
bash scripts/train_gembench_4gpu.sh
```

Baseline training wrapper:

```bash
cd /mnt/yuhan/FastWAM
bash scripts/train_gembench_baseline_4gpu.sh
```

This wrapper keeps the training hyperparameters in
`configs/task/gembench_keysteps_bbox_3cam224_1e-4.yaml`, which mirrors the
existing FastWAM RLBench 3-camera setup: `lr=1e-4`, cosine scheduler, `bf16`,
`batch_size=2`, `gradient_accumulation_steps=2`, `max_steps=50000`, and
`eval/save_every=2000`. It is strictly a training entrypoint: it forces
`GEMBENCH_PREP_ONLY=0`, adds strict GEMBench data checks, verifies one sample per
taskvar by default, and uses a `baseline_YYYYmmdd_HHMMSS` run id.

Useful overrides:

```bash
STRICT_GEMBENCH_COMPLETE=1 bash scripts/train_gembench_4gpu.sh

VERIFY_GEMBENCH_CONTRACT=0 bash scripts/train_gembench_4gpu.sh

bash scripts/train_gembench_4gpu.sh \
  "max_steps=20" \
  "eval_every=0" \
  "save_every=0" \
  "wandb.enabled=false"
```


## VAE Cache 与等价加速实验

这一分支的加速路线按 gate 分阶段推进：先把冻结的 Wan VAE 输出离线缓存下来，减少训练 forward 中的 VAE encode 杂音；之后所有 batch/DeepSpeed/attention/compile 实验都优先在 VAE-cache 路径上做 50-step profile。任何阶段都必须先过语义检查，再做 speed A/B。

关键语义约束：cache 只保存 RGB video 经过 Wan VAE encode 后的 latent，shape 固定为 `[N, 48, 3, 14, 42]`。`action`、`proprio`、`context/context_mask`、normalization 和 text cache 仍由 `GEMBenchKeystepsDataset` 动态生成，所以不会把后续数据处理变化静默固化进 cache。

一个容易踩坑的点是 autocast。正式训练里 `training_loss()` 被 `accelerator.autocast()` 包住，因此 RGB 路径的在线 VAE encode 也处在 bf16 autocast 语义下。旧的 no-autocast cache 虽然能通过 VAE-only verifier，但和真实训练的 RGB path 不完全一致，会导致 loss/grad parity 失败。因此当前 cache 版本升级为 `gembench_vae_latents_v2`，manifest 必须记录：

```json
"vae": {
  "encode_autocast": true,
  "autocast_dtype": "bfloat16"
}
```

dataset 读取 cache 时会校验这个字段；旧 no-autocast cache 会被拒绝，避免误跑。

### 预计算 VAE Latent Cache

默认 cache 路径仍是：

```bash
/mnt/yuhan/datasets/GEMBench/fastwam_cache/vae_latents/keysteps_bbox_seed0_3cam224x672_t9_v1
```

用 4 张 GPU 重建 autocast cache：

```bash
cd /mnt/yuhan/FastWAM
GEMBENCH_VAE_CACHE_REBUILD=1 \
GEMBENCH_VAE_CACHE_NUM_SHARDS=4 \
bash scripts/precompute_gembench_vae_latents_4gpu.sh
```

这个 wrapper 会把旧 cache 目录重命名成 `.no_autocast_backup_YYYYmmdd_HHMMSS`，再用 4 个 shard 写同一个 memmap。完成后自动跑 verifier。单进程调试可以直接用：

```bash
/mnt/miniconda3/envs/fastwam/bin/python scripts/precompute_gembench_vae_latents.py \
  --root /mnt/yuhan/datasets/GEMBench \
  --cache-dir /tmp/fastwam_vae_autocast_smoke \
  --limit 2 \
  --no-resume
```

### 必须先过的 Parity Gate

1. cache contract / latent parity：

```bash
/mnt/miniconda3/envs/fastwam/bin/python scripts/verify_gembench_vae_cache.py \
  --root /mnt/yuhan/datasets/GEMBench \
  --vae-cache-dir /mnt/yuhan/datasets/GEMBench/fastwam_cache/vae_latents/keysteps_bbox_seed0_3cam224x672_t9_v1 \
  --samples 4 \
  --latent-atol 1e-3
```

2. loss / grad parity：

```bash
/mnt/miniconda3/envs/fastwam/bin/python scripts/check_gembench_loss_grad_parity.py \
  --batch-size 1 \
  --seed 1234 \
  --backward \
  --json-output runs/gembench_verification/loss_grad_parity_b1.json \
  --markdown-output runs/gembench_verification/loss_grad_parity_b1.md
```

loss-only 可以去掉 `--backward`，用于快速检查。脚本会固定 batch 和 seed，比较 RGB path 与 VAE-cache path 的 `loss_total/loss_video/loss_action`；加 `--backward` 时会额外比较 grad norm 和若干参数的 grad cosine。

### VAE-cache 训练与 profile

VAE-cache 训练入口：

```bash
cd /mnt/yuhan/FastWAM
bash scripts/train_gembench_vae_cache_4gpu.sh
```

当前唯一通过 100-step gate 且明显改善 backward 的训练入口：

```bash
cd /mnt/yuhan/FastWAM
bash scripts/train_gembench_vae_cache_b4a1_4gpu.sh
```

它只改变 microbatch 排布：从 `batch_size=2/GPU, grad_accum=2` 改成
`batch_size=4/GPU, grad_accum=1`，4 卡 effective batch 仍为 16。模型、loss、
optimizer、scheduler、VAE latent、action/proprio/context 数据链路都不变。

profile harness 默认关闭，不影响正式训练。50-step A/B benchmark 入口：

```bash
cd /mnt/yuhan/FastWAM
bash scripts/benchmark_gembench_acceleration_4gpu.sh vae_zero2
bash scripts/benchmark_gembench_acceleration_4gpu.sh vae_b4a1_zero2
bash scripts/benchmark_gembench_acceleration_4gpu.sh vae_zero2_tuned
bash scripts/benchmark_gembench_acceleration_4gpu.sh vae_zero1_tuned
```

候选项含义：

- `vae_zero2`: VAE-cache + 当前 ZeRO2，作为后续加速 baseline。
- `vae_b4a1_zero2`: `batch_size=4/GPU, grad_accum=1`，effective batch 仍为 16，用来测试减少 microbatch/backward 调度开销是否有效。
- `vae_zero2_tuned`: ZeRO2 开启 `overlap_comm/contiguous_gradients` 并使用较大 bucket。
- `vae_zero1_tuned`: 显存足够时测试 ZeRO1 是否比 ZeRO2 更快。

profile JSONL 默认写到 run 目录的 `profile/step_times.jsonl`。汇总方式：

```bash
/mnt/miniconda3/envs/fastwam/bin/python scripts/summarize_gembench_profile.py \
  runs/<run>/profile/step_times.jsonl
```

只有某一阶段先过 parity，并且 50-step 后 30 step 的稳定收益达到约 `1.15x`，才应该把它并入最终组合；低于阈值就只记录结论，不作为默认训练路径。

### Backward 加速结论（2026-06-03）

当前 profile 结论很集中：VAE-cache 去掉了在线 VAE encode，但训练主瓶颈仍然在
DiT 的 backward。`vae_zero2` 后 30 个 step 平均约 `29.65s/step`，其中
backward 平均约 `29.42s`，forward/loss 只有约 `0.23s`。因此后续优化优先看
backward 调度、通信和 GEMM，而不是继续围绕 VAE 做文章。

已测候选：

| candidate | 语义 gate | step 均值 | backward 均值 | peak memory | 结论 |
| --- | --- | ---: | ---: | ---: | --- |
| `vae_zero2` | RGB/cache loss-grad parity 通过 | `29.65s` | `29.42s` | `40.69GB` | VAE-cache 后的加速基线 |
| `vae_b4a1_zero2` | VAE cache、RGB/cache loss-grad、seeded accumulation parity 均通过；100-step smoke 通过 | `14.90s` | `14.74s` | `43.96GB` | 当前唯一接受的 backward 加速候选，约 `1.99x` |
| `vae_zero2_tuned` | RGB/cache parity 通过 | `29.91s` | `29.67s` | `43.87GB` | ZeRO2 tuned 无收益，不作为默认 |
| `vae_zero1_tuned` | RGB/cache parity 通过 | `29.75s` | `29.51s` | `51.12GB` | ZeRO1 无收益且显存更高，不作为默认 |
| `vae_b4a1_zero2_tuned` | 三道 gate 通过 | `15.25s` | `15.10s` | `47.40GB` | 比普通 b4a1 慢且显存更高，reject |
| `vae_b4a1_zero1_tuned` | 三道 gate 通过 | `15.39s` | `15.23s` | `47.40GB` | 比普通 b4a1 慢且显存更高，reject |

100-step gate 证据目录：

```bash
runs/gembench_acceleration_gates/vae_b4a1_zero2_20260603_222737
```

该 run 统计 step 21-100 共 80 个有效 step：`step_total_s_rank_avg.mean=14.8985`，
`backward_s_rank_avg.mean=14.7402`，`forward_loss_s_rank_avg.mean=0.1536`，
`optimizer_steps_per_sec=0.0671`。前置 gate 数字为：

- VAE cache：`train=3038`、`val=62`、`cache_rows=3100`，抽样 latent `max_abs=0`。
- RGB/cache loss-grad parity：`loss_action/loss_total/loss_video` 完全一致，
  `grad_norm_rel=1.38e-4`。
- accumulation parity：`loss_action rel=7.85e-4`、`loss_total rel=6.24e-4`、
  `loss_video rel=1.22e-6`、`grad_norm_rel=2.15e-4`。

注意：accumulation parity verifier 必须在模型实例化前固定 seed，否则未被 checkpoint
覆盖的随机初始化参数会让 full-batch vs microbatch 的数值敏感性在不同运行之间漂移。
这个修复只影响 verifier 的可复现性，不改变训练代码。

DeepSpeed tuned 组合的补测证据目录：

```bash
runs/gembench_acceleration_gates/vae_b4a1_zero2_tuned_20260603_230329
runs/gembench_acceleration_gates/vae_b4a1_zero1_tuned_20260603_232325
```

二者都通过 VAE cache、RGB/cache loss-grad、seeded accumulation parity，但后 30 step
均慢于普通 `vae_b4a1_zero2`，且 peak memory 从 `43.96GB` 上升到 `47.40GB`。因此
DeepSpeed ZeRO tuning 不进入默认路径。

单进程 profiler 的证据在：

```bash
runs/gembench_profiler/single_step_b1_key_averages.txt
runs/gembench_profiler/single_step_b4_key_averages.txt
```

`batch=4` 单步 CUDA 表里，`aten::mm`、`aten::addmm` 和 bf16 GEMM kernel 是主要
开销；`aten::_efficient_attention_backward` 约占 `5%`，说明 attention 已经走
PyTorch efficient attention path，短期不优先做 FlexAttention。更合理的下一步是：

1. 用 `vae_b4a1_zero2` 启动正式 accelerated baseline，并和原 `vae_zero2`/RGB baseline
   分开 W&B subproject。
2. 后续只在 `vae_b4a1_zero2` 基础上探索局部 `torch.compile` / FFN 融合等严格等价项。
3. DeepSpeed ZeRO1/ZeRO2 bucket tuning 已经先标为低优先级；FlexAttention 也不优先，
   因为 profiler 显示 attention backward 不是主瓶颈。

100-step gate 命令：

```bash
cd /mnt/yuhan/FastWAM
DISABLE_WANDB=1 WANDB_ENABLED=false \
GEMBENCH_PROFILE_STEPS=100 GEMBENCH_PROFILE_WARMUP_STEPS=20 \
bash scripts/run_gembench_acceleration_gate.sh vae_b4a1_zero2
```

100-step gate 通过后，再启动正式 accelerated run：

```bash
cd /mnt/yuhan/FastWAM
bash scripts/train_gembench_vae_cache_b4a1_4gpu.sh
```

## Official Success-Rate Eval

GEMBench leaderboard numbers are simulator success rates, not the trainer's
offline val loss. The v1 success-rate adapter follows the official
robot-3dlotus protocol:

- validation: `val_dataset/microsteps/seed100` on `taskvars_train`
- test L1: `test_dataset/microsteps/seed{200,300,400,500,600}` on
  `taskvars_train`
- test L2/L3/L4: official `taskvars_test_l2/l3/l4`
- action mode: end-effector pose planning + discrete gripper
- action format: `[x, y, z, qx, qy, qz, qw, gripper_open]`
- output format: official-compatible `seed*/results.jsonl` rows with
  `checkpoint/task/variation/num_demos/sr`

Prepare microsteps:

```bash
cd /mnt/yuhan/FastWAM
bash scripts/extract_gembench_microsteps.sh
```

Check simulator readiness:

```bash
cd /mnt/yuhan/FastWAM
bash scripts/check_gembench_success_eval_ready.sh
```

Run a small validation smoke after `rlbench`, `pyrep`, CoppeliaSim, and Xvfb are
available:

```bash
cd /mnt/yuhan/FastWAM
bash scripts/eval_gembench_success_rate.sh \
  --splits val \
  --taskvars push_button+0 \
  --num-demos 1 \
  --max-steps 3 \
  --checkpoint runs/<run>/checkpoints/weights/step_XXXXXX.pt \
  --output-root runs/gembench_success_eval_smoke
```

Run the official validation split:

```bash
bash scripts/eval_gembench_success_rate.sh \
  --splits val \
  --checkpoint runs/<run>/checkpoints/weights/step_XXXXXX.pt \
  --output-root runs/gembench_success_eval_val
```

Run all four official test levels:

```bash
bash scripts/eval_gembench_success_rate.sh \
  --splits test_l1 test_l2 test_l3 test_l4 \
  --checkpoint runs/<run>/checkpoints/weights/step_XXXXXX.pt \
  --output-root runs/gembench_success_eval_test
```

## Current Status

Offline FastWAM training and the built-in trainer eval are already runnable.
The official success-rate code path is implemented and the jinshan runtime has
been prepared under `/mnt/yuhan/gembench_sim`:

- CoppeliaSim V4.1.0:
  `/mnt/yuhan/gembench_sim/CoppeliaSim_Edu_V4_1_0_Ubuntu20_04`
- PyRep: `/mnt/yuhan/gembench_sim/PyRep`
- GEMBench-modified RLBench: `/mnt/yuhan/gembench_sim/RLBench`
- Xvfb: available as `xvfb-run`

The verified smoke command was:

```bash
RUN_WITH_XVFB=1 bash scripts/eval_gembench_success_rate.sh \
  --splits val \
  --taskvars push_button+0 \
  --num-demos 1 \
  --max-steps 1 \
  --num-inference-steps 1 \
  --replan-steps 1 \
  --checkpoint runs/gembench_full31_eval_smoke_4gpu/checkpoints/weights/step_000001.pt \
  --output-root /tmp/gembench_success_smoke_push_button0
```

It launched CoppeliaSim, reset to the official `seed100` microstep demo,
executed one FastWAM action, and wrote official-compatible `seed100/results.jsonl`.
