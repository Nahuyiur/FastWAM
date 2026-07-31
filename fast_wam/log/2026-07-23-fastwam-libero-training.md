# Megatron Fast-WAM 官方 LIBERO 训练复现工作日志

Last updated: 2026-07-24

## Objective

以当前官方 Fast-WAM 主线为 oracle，在不修改 `megatron/` 的前提下，把
Wan2.2 VideoDiT 初始化、ActionDiT 插值合成、官方 LIBERO-v2.1 数据处理、
UMT5/VAE frozen encoding、联合 video/action FlowMatch、Megatron TP/DP
训练、完整 DCP save/resume 和最终 2,000-episode rollout 写入现有
`fast_wam/` patch。

用户确认的最终实验长度是 20,000 optimizer steps，随后做 2,000 episode 评测。
正确性和数值一致性优先于 attention backend 性能。

## Status

训练实现、资产准备、CPU parity、真实 8-card step/save/resume 硬门禁已经完成。
正式 20k + 2k 链路入口已落地。首次 20k run 因 DSW 容器整体重启停在 step 890，
且早于首个 2,000-step checkpoint；当前 run 已修正 fresh-DCP load 的显存清理，
改为每 500 step 保存，并在 8 张 PPU 上重新稳定运行。

## Implementation

主要文件：

- `fast_wam/config.py`：官方 VideoDiT/ActionDiT、FlowMatch、loss contract。
- `fast_wam/scheduler.py`：shifted continuous FlowMatch training/inference。
- `fast_wam/model.py`：joint Video/Action MoT、官方 mask、loss、SDPA/Flex。
- `fast_wam/pretrain.py`：Megatron pretrain entry、leader-only VAE、TP broadcast、
  DP metric reduction、resume-stable stochastic policy。
- `fast_wam/train/data.py`：四 suite、33 observation/32 action、camera/video、
  padding、stats 和 fail-fast dataset。
- `fast_wam/train/text_encoder.py`、`prepare_text.py`：官方 UMT5 cache。
- `fast_wam/train/vae.py`：Wan2.2 VAE encoder。
- `fast_wam/train/initialization.py`：Wan VideoDiT load、ActionDiT exact
  interpolation/alpha scaling、官方随机 I/O。
- `fast_wam/train/sampler.py`：官方 seed-42 epoch sampler 和 validation index。
- `fast_wam/scripts/prepare_libero_training.sh`：资产准备。
- `fast_wam/scripts/train_libero_official.sh`：底层 8-card 训练与 DCP save/resume。
- `fast_wam/scripts/run_libero_official_20k_and_eval.sh`：自动恢复的 20k + 2k 链。
- `fast_wam/scripts/eval_libero_trained_2k.sh`：训练 DCP 的最终 LIBERO runner。

`megatron/` 未修改。

## Exact contracts checked

数据 release：

- 四 suite 固定顺序；
- 1,712 episodes；
- 277,713 frames；
- 40 tasks；
- LeRobot v2.1、20 Hz；
- sample 0 对官方 loader bitwise 一致。

Frozen components：

- 40 个 text cache 完整；
- 首个真实 prompt 的 UMT5 context/mask 与官方一致；
- 真实 VAE probe 输出 `[1,48,3,14,28]`，BF16 elementwise exact。

初始化：

- Video copied 825；
- Action copied 300；
- Action interpolated 520；
- Action random backbone skips 4；
- official random action I/O + proprio tensors 6；
- initial BF16 TP2 DCP：
  `outputs/fast_wam_libero_training_assets/initial_dcp_bf16`。

模型/训练：

- 30 layers；
- Video hidden/FFN `3072/14336`；
- Action hidden/FFN `1024/4096`；
- 24 heads、head dim 128；
- video/action shift 5.0；
- loss weights 1.0/1.0；
- AdamW `(0.9,0.95)`、epsilon `1e-8`、weight decay `1e-2`；
- BF16、global batch 128、microbatch 16、gradient accumulation 1；
- LR `1e-4`、5% linear warmup、cosine 到 `1e-6`；
- save 2,000、validation 200、seed 42。

## Deliberate deviations

1. 用户选择 20,000 steps；官方 config 的 10 epochs 在当前 release/global batch
   下是 21,700 steps，release checkpoint metadata 也是 step 21,700。
2. dataset decode error fail-fast，不做官方随机 sample replacement。
3. diffusion RNG keyed by phase/iteration/microbatch/DP rank，保持官方四次 draw
   的分布和顺序，同时解除 DataLoader bookkeeping 对断点恢复的影响。
4. 官方 online eval 每 200 steps 还会生成/解码视频并写 PSNR/SSIM/action
   diagnostics；当前 Megatron 中间门禁计算 deterministic validation loss，最终质量
   由 2,000-episode rollout 判定。
5. FlexAttention 保留并通过 CPU parity。无 mask 的 compiled Flex
   forward/backward PPU smoke 通过，但训练 MoT 自定义 BlockMask/特殊 mask 路径
   编译反向出现进程级 `SIGSEGV`；正式训练使用 BF16 Q/K/V 的 SDPA，不做显式
   FP32 upcast。mask 和 attention 数学 contract 不变。

## Validation

CPU：

```text
FAST_WAM_DISABLE_MCORE=1 python -m pytest -q fast_wam/tests
11 passed
```

真实 PPU：

- 无 mask、`[2,8,1024,64]` Q/K/V 的 Flex compiled forward/backward：通过；
- 训练 MoT 自定义 BlockMask/特殊 mask 的 Flex compiled backward：
  `SIGSEGV`，未用于正式训练；
- SDPA TP1+DP8、microbatch 16、global batch 128：
  forward/backward/optimizer step 通过；
- `--grad-reduce-in-bf16`、distributed optimizer、overlap grad reduce、
  200,000,000-element DDP bucket：通过；
- full optimizer/RNG DCP save/load：通过。

Resume hard gate artifacts：

```text
outputs/fast_wam_resume_gate_continuous/
outputs/fast_wam_resume_gate_interrupted/
outputs/fast_wam_resume_gate_resumed/
```

八个 ranks 的 indices、latent、action/context/proprio 和四类随机 tensor digest
逐 rank exact。中断 step-1 DCP hash
`5ed96fd88c06f4d618f33e8cd1a4e0ce10a8cc449c715a675aff89572abe0981`
与 resumed live pre-step-2 hash exact。

step-2：

| Metric | Continuous | Resumed | Difference |
| --- | ---: | ---: | ---: |
| Global loss | 4.342782914625 | 4.3425183595 | 2.64555e-4 |
| Grad norm | 7.854 | 7.855 | 0.001 |

per-rank loss max difference 为 `8.71658e-4`。

最终两个独立 launch 的 6,020,710,599 个 BF16 parameter elements：

- max abs `1.220703125e-4`；
- mean abs `1.4283151089858059e-6`；
- unequal `297,961,473 / 6,020,710,599 = 4.9489%`；
- 1,349/1,651 tensors 含至少一个不同 element。

这是 PPU/PCCL BF16 reduction 的跨 launch 非确定包络。DCP 自身 roundtrip 对全部
1,651 tensors bitwise exact。

## Artifacts and storage

- Initial model DCP：约 12 GiB。
- Full optimizer training DCP：当前 topology 每 step checkpoint 约 79 GiB。
- filesystem 在启动正式 run 前剩余约 11 TiB；原 2k interval 预计约 790 GiB，
  当前 500-step interval 若保留全部 checkpoint 预计约 3.1 TiB。
- checkpoint、dataset、raw logs 和 rollout case JSON 均在 ignored/shared path，
  不进入 Git。

## Formal run

入口：

```bash
bash fast_wam/scripts/run_libero_official_20k_and_eval.sh
```

Canonical outputs：

```text
outputs/fast_wam_libero_training_20k/
outputs/fast_wam_libero_training_20k_eval_2k/
```

注意：Python training log 使用 DSW 本地时间 UTC+8；PPU monitor CSV 显式使用 UTC。

首次启动时间：2026-07-23 21:17 UTC+8（13:17 UTC）。

tmux session：

```text
fastwam_libero_20k
```

启动 preflight：

- `torch.cuda.is_available() == True`；
- 8/8 devices 均为 `PPU-ZW810E`；
- formal output 是全新目录；
- 40 个 text cache、initial DCP、stats、VAE 均存在；
- Wan base 无 broken symlink；
- Fast-WAM checkpoint tree 无 `.incomplete`。

首个正式 optimizer step 于 21:19 UTC+8 完成：

| Metric | Value |
| --- | ---: |
| Loss | 4.026476 |
| Video loss | 2.692963 |
| Action loss | 1.333513 |
| Grad norm | 7.461 |
| Learning rate after step | 1.999e-7 |
| Skipped / NaN iterations | 0 / 0 |

该 step-1 loss、两个分项 loss 和 grad norm 与 resume hard gate 的 continuous
step 1 完全一致。首次 run 稳定运行到 step 890（loss `0.3547449`，skipped/NaN
`0/0`），log 在 22:31:33 UTC+8 无 traceback/OOM/NCCL error 地停止。新容器 PID 1
于 22:37:50 UTC+8 启动，原 tmux/worker 全部消失，因此判断为 DSW 容器整体重启或
回收，而不是训练代码异常。由于默认 save interval 为 2,000，输出目录没有 DCP 或
`latest_checkpointed_iteration.txt`，必须从初始 DCP 重跑。原日志保留为：

```text
outputs/fast_wam_libero_training_20k/train.interrupted_step0890_20260723.log
```

冷启动后的早期稳态：

| Step | Loss | Window time | Skipped / NaN |
| ---: | ---: | ---: | ---: |
| 10 | 4.239196 | 9.068 s/step | 0 / 0 |
| 20 | 3.905135 | 5.027 s/step | 0 / 0 |

按 step-20 window 的速度，20k 主训练约 28 小时量级；这只是早期 ETA，不是最终
throughput 报告。

## Formal restart and monitoring

首次重启于 23:36 UTC+8 发起。step 1 的 loss/grad norm 再次 exact，但 fresh-DCP
load 后有一套约 11.5 GiB 的临时 BF16 state tensors 未及时回收：step-1
allocated/max 从健康 run 的 `35.1/80.7 GiB` 增至 `46.6/92.3 GiB`，随后 rank
4/7 在 step 2 forward OOM。失败日志保留为：

```text
outputs/fast_wam_libero_training_20k/train.failed_oom_step0001_20260723.log
```

`fast_wam/checkpoint.py` 现在在 `model.load_state_dict()` 后显式删除 DCP state、
执行 Python GC 并释放 CUDA allocator cache。正式 launch 同时使用
`PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`，只改变 allocator 行为，不改变
训练数值。修改后 CPU gate 仍为 `11 passed`。

最终重启于 23:42 UTC+8 发起，tmux 仍为 `fastwam_libero_20k`，save interval 从
2,000 缩短为 500。step 1 再次 exact：

| Metric | Value |
| --- | ---: |
| Loss | 4.026476 |
| Video loss | 2.692963 |
| Action loss | 1.333513 |
| Grad norm | 7.461 |
| Skipped / NaN | 0 / 0 |
| Allocated / peak after step 1 | 35.1 / 80.3 GiB |
| Peak after step 10 | 86.5 GiB |

step 30 已通过，skipped/NaN 仍为 `0/0`。step-20/30 window 均约
`2.93 s/step`；这是重启后的早期采样，不作为最终 throughput。

同一 tmux 内的 monitor 每 30 秒记录每张 PPU 的 utilization、显存、温度和功率：

```text
outputs/fast_wam_libero_training_20k/ppu_metrics.csv
```

训练早期 32 个 per-device 活跃采样的 aggregate 为：平均功率 `295.42 W/card`
（`275.51–320.79 W`），平均利用率 `99.4%`，最高温度 `62°C`，最大 device memory
`90,252 MiB`。monitor 脚本和 CSV 都是 ignored run artifacts。

当前 canonical live log：

```text
outputs/fast_wam_libero_training_20k/train.log
```

2026-07-24 13:35 UTC+8 提交前复核时，正式 run 到 step 16,620/20,000
（83.1%），skipped/NaN 仍为 `0/0`，最近 50 个 log window 平均
`2.949 s/step`。step 16,500 DCP 已完整落盘并更新 tracker；当时 8 张 PPU
均在运行，单卡约占 `89.7–90.6 GiB`。这些是进行中的健康检查，不是最终训练或
2,000-episode 验收结论。

2026-07-24 13:54 UTC+8，step 17,000 full optimizer DCP 完整落盘后，为运行固定
8-case 闭环 sanity eval 暂停正式训练。暂停前多运行到 step 17,010；恢复时从
tracker 17,000 重算这 10 步。复核发现 Megatron tracker 是无末尾换行的 ASCII
`17000`，而两个 resume/eval wrapper 在 `set -e` 下使用 `read -r < tracker`；
`read` 虽赋值但在 EOF 返回非零，导致 wrapper 在 load 前退出。DCP 内容不受影响：
step 16,500 metadata 实查有 2,359 项，其中 93 项覆盖 31 个 distributed optimizer
bucket 的 `param/exp_avg/exp_avg_sq`。两个 wrapper 已改用 Bash file command
substitution，兼容有/无末尾换行的 tracker。

首次从 full training DCP 启动 eval 时，模型 load 已得到 `missing=[]`，但
`load_megatron_dcp()` 将 `args/checkpoint_version/iteration/optimizer/
opt_param_scheduler/num_floating_point_operations_so_far/content_metadata` 这些
training-only 顶层公共状态当成 unexpected model keys 而 fail-fast。eval loader
现仅忽略这组精确白名单；任何其他 unexpected key 和所有 missing model key 仍会
报错。该修改只影响 full-training-DCP 的 model-only eval load，不改变训练 resume
所用的 Megatron model/optimizer/RNG loader。

修复后 full DCP 的 model-only load 已通过，并进入 LIBERO 环境创建；当前容器随后
因系统 `libosmesa6` 在重启后缺失而在 PyOpenGL 初始化处退出，尚未产生 8-case
rollout 结果。此前成功的本机 2,000-episode 运行明确记录安装过该系统包；本次没有
用其他 GL backend 改变评测环境，也没有改动 checkpoint。

2026-07-24 14:08 UTC+8，正式训练从 step 17,000 真实恢复，默认
`LOAD_OPTIM=1`，参数中 `no_load_optim=None`、`no_load_rng=None`。Megatron 日志
直接确认：

- DCP metadata 为 `distrib_optim_sharding_type=dp_reshardable`；
- `Loading distributed optimizer sharded state of type dp_reshardable`；
- optimizer scheduler 的 LR/min-LR/warmup/total-samples/cosine 配置均取 checkpoint
  值；
- `successfully loaded ... at iteration 17000`，八 rank load 用时约 19.99 秒；
- 恢复后 step 17,001 的 consumed samples 为 2,176,128、LR
  `6.962155e-6`、loss `0.1013142`，step 17,010 为 2,177,280、
  `6.927154e-6`、loss `0.1001166`，两者 skipped/NaN 均为 0。

因此这次不是仅恢复模型权重：distributed Adam 主参数、`exp_avg/exp_avg_sq`、
optimizer scheduler、iteration 与 RNG 都走完整 Megatron resume 路径。训练已在
tmux `fastwam_libero_20k:train` 继续；暂停前超出 tracker 的 17,001–17,010 十步
按预期重算。

2026-07-24 14:35 UTC+8，step 17,500 DCP 完整落盘，tracker 原子更新且
`iter_0017500/.metadata` 存在，目录约 79 GiB。经用户明确授权，系统重新安装
`libosmesa6 25.1.7-1ubuntu2~24.04.2`；安装前无代理刷新过 apt 索引，没有改动
PyTorch 或 Transformer Engine。运行时验证为 `libOSMesa.so.8`、MuJoCo 3.1.6，
PyOpenGL `GL.glGetError` 可加载。

随后使用 step 17,500、TP1+DP8、BF16、10 diffusion steps 和固定
`fast_wam/eval/manifest.json` 运行 8-case 闭环 sanity eval，结果 **8/8**：

| Suite | Cases | Successes | Environment steps |
|---|---:|---:|---|
| LIBERO-Spatial | 2 | 2 | 113, 131 |
| LIBERO-Object | 2 | 2 | 151, 159 |
| LIBERO-Goal | 2 | 2 | 90, 88 |
| LIBERO-10 | 2 | 2 | 294, 360 |

结果位于
`outputs/fast_wam_libero_training_step17500_eval8_20260724/summary.json`，八个
per-case JSON 均存在。该结果验证了训练 DCP model-only load、VAE、文本条件、
动作去噪、归一化、夹爪和 LIBERO 闭环控制链，但八条样本不能替代最终
2,000-episode 精度验收。

评测后训练再次从 step 17,500 恢复。日志确认
`Loading distributed optimizer sharded state of type dp_reshardable` 和
`successfully loaded ... at iteration 17500`。恢复后的 step 17,501 loss
`0.08785976`、LR `5.165933e-6`；step 17,510 loss `0.09435428`、LR
`5.136405e-6`，均为 0 skipped/0 NaN。训练继续运行中。

Resume loss-curve continuity was audited directly from the appended canonical
log. Step 17,510 was computed once before the evaluation pause and once after
resuming from the step-17,500 DCP. Total loss was `0.09435233` versus
`0.09435428` (absolute difference `1.95e-6`, about `0.0021%`); video loss was
`0.08686904` versus `0.08687041`, action loss `0.007483280` versus
`0.007483865`, while consumed samples and LR were identical. The six logged
points from 17,450–17,500 averaged `0.098733808`; the six resumed points from
17,510–17,560 averaged `0.098185835`, a `-0.555%` change within normal batch
variation. Deterministic validation loss was also continuous: `0.07493986` at
17,400 and `0.07332309` at 17,600. As a cross-check, the earlier step-17,000
resume reproduced step-17,010 within `4.9e-6` total loss. There is no loss,
LR, sample-count, grad-norm, skipped-iteration, or NaN discontinuity at either
resume boundary.

## Final 20,000-step training status

正式训练于 2026-07-24 16:48 UTC+8 到达 step 20,000。最终 tracker 为
`20000`，完整 model/distributed optimizer/scheduler/RNG DCP 位于：

```text
outputs/fast_wam_libero_training_20k/iter_0020000/
```

最终 step 的训练指标为：

| Metric | Value |
| --- | ---: |
| Total loss | 0.09907796 |
| Video loss | 0.09414858 |
| Action loss | 0.004929387 |
| Learning rate | 1.000000e-6 |
| Grad norm | 0.070 |
| Consumed samples | 2,560,000 |
| Skipped / NaN iterations | 0 / 0 |

step 20,000 后的 deterministic validation loss 为 `0.08667219`，其中 video
loss `0.08114253`、action loss `0.005529673`。训练日志中没有 skipped optimizer
update 或 NaN iteration。训练期间两次正式恢复均加载完整 distributed optimizer
state，且 resume loss continuity 检查通过；最终 DCP 不是 weight-only checkpoint。

## Final 2,000-episode LIBERO evaluation

完整四-suite 评测于 2026-07-24 20:29 UTC+8 完成。8 个 DP worker 生成了全部
2,000 个原子 case JSON，随后成功生成：

```text
outputs/fast_wam_libero_training_20k_eval_2k/summary.json
```

最终结果为 **1,923/2,000（96.15%）**：

| Suite | Successes | Rate | Paper target | Meets suite target |
| --- | ---: | ---: | ---: | :---: |
| LIBERO-Spatial | 478/500 | 95.60% | 98.20% | No |
| LIBERO-Object | 497/500 | 99.40% | 100.00% | No |
| LIBERO-Goal | 491/500 | 98.20% | 97.00% | Yes |
| LIBERO-10 | 457/500 | 91.40% | 95.20% | No |
| **Overall** | **1,923/2,000** | **96.15%** | **97.60%** | **No** |

该结果比论文的 1,952/2,000（97.60%）少 29 个成功、低 1.45 个百分点，不能报告
为复现论文精度。相同本机评测栈下，转换后的官方 release Megatron DCP 为
1,938/2,000（96.90%）；本次训练 checkpoint 少 15 个成功、低 0.75 个百分点：

| Suite | Trained 20k | Release DCP | Difference |
| --- | ---: | ---: | ---: |
| LIBERO-Spatial | 478 | 485 | -7 |
| LIBERO-Object | 497 | 497 | 0 |
| LIBERO-Goal | 491 | 486 | +5 |
| LIBERO-10 | 457 | 470 | -13 |
| **Overall** | **1,923** | **1,938** | **-15** |

本次结果与同机 LeRobot local run 的 1,922/2,000（96.10%）基本持平，但 aggregate
相近不代表逐 episode 数值一致。

Spatial 的 500 个 case 可与 release DCP 逐一配对：464 个两边都成功，21 个仅
release 成功，14 个仅本次训练成功，1 个两边都失败。35/500 个闭环结果发生翻转，
净差为 -7；下降主要集中在 task 7（49 -> 43）、task 4（47 -> 44）和 task 6
（50 -> 48），其他若干 task 有小幅提升。这更符合不同 checkpoint 在长时域 MuJoCo
轨迹中的分叉，而不是统一方向的推理链路故障。

## Post-run training-hyperparameter audit

最终掉点后重新对照了当前官方 Fast-WAM 主线和本次 Megatron 实际日志。以下显式
训练参数一致：

- per-device batch 16、global batch 128、gradient accumulation 1；
- BF16；
- AdamW，betas `(0.9, 0.95)`，epsilon `1e-8`；
- nominal weight decay `1e-2`、gradient clip `1.0`；
- peak LR `1e-4`、5% linear warmup、cosine minimum LR `1e-6`；
- video/action loss weights `1.0/1.0`；
- video/action FlowMatch shift `5.0/5.0`、1,000 training timesteps；
- seed 42、epoch permutation `seed + epoch`；
- SDPA joint attention、相同 MoT mask、无 gradient checkpointing。

发现三项会改变训练轨迹的差异：

1. **训练长度和 scheduler 不同。** 官方 config 是 10 epochs；在 277,713 frames
   和 global batch 128 下为
   `ceil(277713 / 128) * 10 = 21,700` optimizer steps。本次按用户预先指定的
   20,000 steps 运行，仅相当于约 9.2166 epochs，少 1,700 steps /
   217,600 sample presentations。官方 warmup 为 1,085 steps、初始 LR
   `1e-4 / 1085 = 9.2166e-8`；本次 warmup 为 1,000 steps、初始 LR `1e-7`。
   官方在 step 20,000 的 LR 仍为约 `2.6519e-6`，本次已到 `1e-6`。按逐 step
   LR 求和，本次累计 LR 为官方 21,700-step schedule 的约 92.17%。
2. **weight-decay 参数分组不同。** 官方把全部 trainable DiT/proprio parameters
   放入同一个 `torch.optim.AdamW(weight_decay=0.01)`。Megatron 默认对所有 bias
   和一维参数设置 `wd_mult=0`。initial DCP metadata 中该集合为 3,059,911
   elements，占 6,020,710,599 trainable elements 的约 0.0508%。数量占比小，但
   包含 bias、normalization 和 modulation 参数，是明确的参数合同偏差。
3. **seed 数值相同但 stochastic stream 不同。** 官方在进程内用
   `seed + global_rank` 的 CUDA generator 顺序产生 video noise/timestep 和 action
   noise/timestep，并且每 200 steps 的 validation 也会推进 RNG。本 overlay 为保证
   Megatron DCP resume 不受 DataLoader iterator bookkeeping 影响，按
   `(phase, iteration, microbatch, DP rank)` 派生 private generator seed。分布和
   四次 draw 顺序相同，但并非官方同一串随机数。另外，当前官方代码在实例化
   ActionDiT/proprio 后才在 Trainer 内调用 `set_global_seed(42)`，因此其
   action encoder/head/proprio 随机初始化未被配置中的 seed 完整约束；本 overlay
   明确在 seed-42 CPU RNG scope 中初始化这些模块。没有作者当时的初始随机 state，
   无法从公开配置恢复发布 checkpoint 的逐元素初始化。

此外，官方使用 PyTorch AdamW + Accelerate/DeepSpeed ZeRO-1，本次使用
Transformer Engine FusedAdam + Megatron distributed optimizer，并在 PPU 上做
BF16 gradient reduction。公式超参一致，但 fused optimizer、collective reduction
顺序和硬件 backend 会产生额外数值轨迹差异。已通过的 DCP/model/input/loss 门禁
说明这些不是明显实现错误，但也不能把本次训练视作发布 checkpoint 的逐 step
bitwise reproduction。

## Final conclusion and follow-up

20,000-step 训练、full optimizer resume 和完整 2,000-episode evaluation 均已
端到端完成，工程链路通过；最终精度 **96.15%**，未达到论文 97.60%，也比相同评测
环境下的 release DCP 低 0.75 个百分点。

若继续追求官方训练参数复现，下一次正式 run 至少需要：

1. 从 initial DCP 重新运行完整 21,700-step cosine schedule；当前 20k DCP 已到
   minimum LR，简单续训 1,700 steps 不能还原官方 scheduler；
2. 在 overlay 侧显式覆盖 Megatron optimizer param groups，使所有 trainable
   parameters 都应用 weight decay 0.01；
3. 明确实验目标是“完全采用官方顺序 RNG”还是“保持当前 resume-stable private
   RNG”。后者可复现自身，但应作为不同 stochastic run 报告；
4. 在重跑整套 2,000 episodes 前，优先用已保存的 18k/19k/20k checkpoints 对
   Spatial task 4/7 和 LIBERO-10 的退化 task 做小规模 step-selection 诊断。

## Compiled FlexAttention host smoke after the formal run

正式 2,000-episode eval 释放全部设备后，在宿主机运行：

```bash
/opt/ac2/bin/python3 -m pytest -q \
  fast_wam/tests/test_compiled_flex_attention_ppu.py
```

测试使用无 attention mask 的 Q/K/V `[2,8,1024,64]`，执行
`torch.compile(flex_attention, fullgraph=False)`、forward、`out.sum().backward()`
和 device synchronize。结果为 **1 passed**，总用时 `62.85s`，其中 test call
`56.73s`。这证明当前 PPU 栈的基础 compiled Flex backward 可用；此前训练路径的
`SIGSEGV` 应缩小到 MoT 自定义 BlockMask/特殊 mask 或其与模型图组合的路径，不能再
笼统描述为所有 Flex backward 都会崩溃。
