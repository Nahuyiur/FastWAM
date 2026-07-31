# Fast-WAM 官方 LIBERO 训练在 Megatron Core 中的复现

Last updated: 2026-07-25

## 结论与范围

本实现把当前官方 Fast-WAM 主线的 LIBERO `fastwam` recipe 放进
`Megatron-Wan/fast_wam/` overlay，不修改 `megatron/`。范围是：

1. 从官方 Wan2.2 TI2V-5B VideoDiT 构造 Video expert；
2. 按官方预处理脚本逐维线性插值并做 alpha scaling，合成 1024-hidden ActionDiT
   backbone；
3. 保留官方随机初始化的 action input/output 和 proprio 模块；
4. 用四个官方 LIBERO `no_noops` suite 联合训练 video/action FlowMatch；
5. 保存可跨 topology reshard、包含 optimizer/RNG 的 Megatron distributed
   checkpoint；
6. 训练到用户指定的 20,000 step 后，执行四 suite、2,000 episode 的闭环评测。

官方仓库当前代码是逻辑与数值 contract 的 oracle。LeRobot checkpoint 只用于已有
action inference parity、评测配置和 stats，不作为训练初始化来源。

## 端到端数据流

```text
Wan2.2 TI2V-5B safetensors
  ├─ exact copy ───────────────> 3072-hidden VideoDiT
  └─ sequential 1-D interpolate
       + sqrt(d_video/d_action) > 1024-hidden ActionDiT backbone
                                   + official random action I/O
                                   + official random proprio encoder
                                                │
four LIBERO v2.1 no_noops suites                │
  ├─ 33 observations / 32 actions               │
  ├─ two 224x224 cameras -> 224x448 video        │
  ├─ UMT5 context cache                          │
  └─ Wan VAE -> [B,48,3,14,28] latent ──────────┤
                                                v
                           joint Video/Action MoT FlowMatch
                                                │
                          Megatron TP/DP + distributed AdamW
                                                │
                          full optimizer/RNG torch_dist DCP
                                                │
                                  20k checkpoint -> LIBERO 2k
```

## Step 1：固定 oracle 和输入资产

训练数据根目录：

```text
/mnt/world_foundational_model/ruibin/data/Fast-WAM/official/libero_mujoco3.3.2
```

suite 顺序固定为：

1. `libero_spatial_no_noops_lerobot`
2. `libero_object_no_noops_lerobot`
3. `libero_goal_no_noops_lerobot`
4. `libero_10_no_noops_lerobot`

loader 启动时强校验 LeRobot v2.1、20 Hz、camera/state/action schema、suite 顺序以及
总量 `1,712 episodes / 277,713 frames`。dataset stats 使用官方 release 中的
`libero_uncond_2cam224_dataset_stats.json`。

Wan 资产固定为本地官方 `Wan-AI/Wan2.2-TI2V-5B` VideoDiT、VAE 和 UMT5 encoder，
tokenizer 来自 `Wan-AI/Wan2.1-T2V-1.3B/google/umt5-xxl`。

## Step 2：精确复现数据和 frozen encoder

每个全局 frame index 查询当前 episode 内连续 33 个 observation 和 32 个 action；
越过 episode 尾部的索引 clamp 到最后一帧，并保留 padding mask。padded delta action
的 0..5 维先清零，再应用 release 的全局 MIN_MAX transform，并 clamp 到 `[-5,5]`。

两个 camera 分别按官方 LeRobot-v2.1 PyAV timestamp contract 解码和 resize 到
`224x224`，按固定 key 顺序沿 width 拼成 `224x448`。选择 observation
`0,4,...,32`，形成 9 帧输入，并只在 VAE 边界执行一次 `[0,1] -> [-1,1]`。

文本 prompt 固定为：

```text
A video recorded from a robot's point of view executing the following instruction: {task}
```

40 个 task 的 UMT5 context 预先缓存为 128 tokens。padding embedding 清零，但传给
DiT 的 context mask 全为可见，这一点与当前官方代码一致。训练时只有每个 TP group
的 rank 0 跑 frozen Wan VAE，BF16 latent 再在 TP group 内广播。

生产训练可进一步把固定 pixels-to-latent 变换离线化：

```bash
bash fast_wam/scripts/prepare_libero_latents.sh
```

输出默认位于：

```text
/mnt/world_foundational_model/ruibin/data/Fast-WAM/cache/libero_mujoco3.3.2_wan22_bf16
```

缓存为约 29 GiB 的 BF16 `[48,3,14,28]` memory-mapped shards。`build.json`
固定 dataset metadata/payload sizes、stats、预处理 contract 和 VAE checkpoint
SHA-256；每个 shard 先写临时文件、`fsync` 后原子 rename，完整扫描通过后才写
`manifest.json`。重复运行会跳过大小正确的已完成 shard。训练侧会再次校验样本数、
shape/dtype、dataset fingerprint 和预处理 fingerprint；contract 不一致直接报错，
不会静默使用陈旧 latent。

构建器按 episode 一次性解码和预处理两路 camera，再把所有重叠的 9 帧窗口分批送入
VAE；因此不会为 277,713 个全局 frame sample 重复打开/解码 MP4。episode 首部、
中部和尾部 clamp window 与原 sample-wise 路径的逐像素比较均为 bitwise exact，
maximum error 为 0。shard 写入中断不会产生可见的已完成 shard，恢复时只跳过大小
严格正确的原子文件。

cached training 启动时还会在 DataLoader workers fork 前把约 18 MB 的
state/action/timestamp parquet payload 全部物化到只读 tensor；worker 通过
copy-on-write 共享，稳态随机 batch 不再反复解析 parquet。latent shard 本身仍使用
memory map，不复制 29 GiB cache。

真实资产 parity：

- 官方 sample 0 的完整 dataset sample bitwise 一致；
- 第一个真实 prompt 的 UMT5 context/mask 一致；
- 真实 sample VAE 输出 shape 为 `[1,48,3,14,28]`，BF16 元素逐个一致。

正式输入的 9 个 RGB frames 经 Wan VAE temporal factor 4 压缩为 3 个 latent
temporal steps。

## Step 3：从 VideoDiT 合成 ActionDiT

入口：

```bash
bash fast_wam/scripts/prepare_libero_training.sh
```

该入口先生成 text cache，再构造 initial DCP。ActionDiT backbone 复刻官方
`scripts/preprocess_action_dit_backbone.py`：

- shape 已一致的 tensor 直接 copy；
- 其余 tensor 按每个不匹配 dimension 依次移到最后一维；
- 用 FP32 `linear` interpolation、`align_corners=True` 调整；
- 若最后一维发生变化，乘
  `sqrt(source_last_dim / target_last_dim)`；
- `action_encoder.*` 和 `head.*` 不从 VideoDiT 插值。

官方 ActionDiT 是普通 PyTorch `nn.Linear`，其随机模块使用 Kaiming-uniform，而
Megatron parallel linear 的默认初始化不同。因此实现会在 seed 42 的 CPU RNG scope
中按官方构造顺序创建一个 dense ActionExpert 和 proprio `nn.Linear`，再把
action encoder/head/proprio 参数切到对应 TP shard。

已生成 initial DCP：

```text
outputs/fast_wam_libero_training_assets/initial_dcp_bf16
```

manifest 统计：

| 类别 | tensor 数 |
| --- | ---: |
| VideoDiT exact copy | 825 |
| ActionDiT exact copy | 300 |
| ActionDiT interpolated | 520 |
| ActionDiT backbone skipped/random | 4 |
| Official random action I/O + proprio | 6 |

initial DCP 是 TP2 保存的 BF16 DCP，可由正式 TP1+DP8 训练直接 reshard load。

## Step 4：联合 Video/Action FlowMatch

正式 model 同时训练 VideoDiT、ActionDiT 和 proprio encoder，共
`6,020,710,599` 个 BF16 parameter elements。

每个样本的 RNG 顺序固定为：

1. video noise；
2. video timestep；
3. action noise；
4. action timestep。

video 和 action 分别用 shift 5.0、1,000 training timesteps 的连续
FlowMatch。首个 clean video latent step覆盖 noisy video 的第一步，video loss 只计算
未来 2 个 temporal latent steps；action loss 计算 32 个 action tokens。两个 loss
都按 padding mask 做 per-sample mean，再乘官方 timestep weight，最终
`loss = loss_video + loss_action`。

联合 attention mask 与官方 `_build_mot_attention_mask` 一致：

- 第一帧 video query 只能读第一帧 video；
- future-video query 可双向读取全部 video；
- action query 可读取 clean 第一帧 video 和全部 action；
- video query 不能读取 action。

训练时向 attention backend 传入 model dtype 的 BF16 Q/K/V，不做显式 FP32
upcast；内部 logits、softmax 和 value accumulation dtype 由 PyTorch/PPU backend
决定。代码同时实现 `flex`、`sdpa` 和 `structured_sdpa` backend。后者把精确 mask
编译成三个无 mask 的 dense SDPA：

```text
Q(first video)  -> KV(first video)
Q(future video) -> KV(all video)
Q(action)       -> KV(first video + action)
```

它不计算 mask 禁止的 pair，也不依赖 PPU 自定义 BlockMask backward。optimized
kernel mode 另外在 TP1 训练时使用 BF16-input LayerNorm/RMSNorm、complex64 RoPE
和 Megatron fused bias-GELU；TP>1 RMSNorm、reference 模式和 action inference
仍保留已验收实现。参数名、DCP key 和 tensor shape 均不改变。

CPU 前向/反向 parity 已覆盖 Flex 和 structured optimized path。
宿主机 PPU-ZW810E 上，无 mask、`[2,8,1024,64]` Q/K/V 的 compiled Flex
forward/backward smoke 已通过；触发进程级 `SIGSEGV` 的是训练 MoT 自定义
BlockMask/特殊 mask 路径，因此正式运行默认 `sdpa`。这不是 mask 语义变化，只是
当前设备 backend 选择。对实际 batch-16 的 video/action self-attention 和
cross-attention 五组 shape，强制 `SDPBackend.FLASH_ATTENTION` 均通过前向/反向；
auto 与 forced-Flash 的合计延迟同为约 `3.88 ms`，而显式 Transformer Engine
FlashAttention 约为 `5.33 ms`。所以性能配置保留 `structured_sdpa` 的 auto
dispatch，不强制 backend，也不切换到 TE。

## Step 5：optimizer、采样和并行 contract

用户指定的正式 run 是 20,000 optimizer steps：

| 项目 | 值 |
| --- | --- |
| precision | BF16 |
| topology | TP1 + DP8 |
| local microbatch | 16 |
| global batch | 128 |
| gradient accumulation | 1 |
| optimizer | AdamW |
| betas / epsilon | `(0.9,0.95)` / `1e-8` |
| weight decay | `1e-2` |
| gradient clip | `1.0` |
| LR | `1e-4` |
| warmup | 前 5%，从 `1e-7` 线性到 `1e-4` |
| decay | cosine 到 `1e-6` |
| seed | 42 |
| save / validation interval | 2,000 / 200 |

官方 config 写的是 10 epochs。按当前 release 和 global batch 128，
`ceil(277713/128) * 10 = 21,700` steps，release checkpoint metadata 也记录
step 21,700。本次按用户明确目标把 scheduler 和终点都设为 20,000；这是有意的实验
长度差异，不应把最终 checkpoint 称为官方 21,700-step release 的逐步复刻。

epoch sampler 精确构造 `seed 42 + epoch` 的全局 permutation，再按 Accelerate 的
consecutive local batches 分给 DP ranks，并用同一 permutation 开头补齐 epoch 尾部。

两项批准的确定性偏差：

- dataset 解码失败立即报错，不执行官方的随机 replacement；
- diffusion 随机张量由 `(phase, iteration, microbatch, DP rank)` 映射到 private
  generator seed。四次随机抽样的分布和调用顺序不变，但不依赖 Megatron 与
  DeepSpeed 不同的 iterator bookkeeping RNG 时点，因此断点恢复稳定。

官方每 200 steps 的 online eval 会额外生成视频、解码并保存 PSNR/SSIM/action
诊断视频；本 overlay 的训练中间门禁只计算同一确定性采样规则下的 validation loss，
不参与 parameter update。最终质量门禁使用完整 2,000-episode LIBERO rollout。

## Step 6：保存与断点恢复

正式 checkpoint 使用 Megatron `torch_dist` DCP，默认包含 model、distributed
optimizer、scheduler 和 RNG，每 2,000 steps 保存一次。单个完整 checkpoint 在当前
TP1+DP8 topology 约 79 GiB。

一键训练并在完成后评测：

```bash
bash fast_wam/scripts/run_libero_official_20k_and_eval.sh
```

若 `outputs/fast_wam_libero_training_20k/latest_checkpointed_iteration.txt` 存在，
脚本会校验 step 后自动从同一目录恢复，并把新日志追加到 `train.log`。也可以直接使用
底层训练入口并通过环境变量覆盖：

```bash
SAVE_DIR=outputs/my_fast_wam_run \
TRAIN_ITERS=20000 \
ATTENTION_BACKEND=sdpa \
bash fast_wam/scripts/train_libero_official.sh
```

## Step 7：硬正确性门禁

已在 8 张 PPU-ZW810E 上完成 TP1+DP8、microbatch 16、global batch 128 的真实
forward/backward/optimizer/save/resume gate。

连续两步、step 1 中断、从完整 DCP 恢复后执行 step 2 的诊断结果：

- 八个 DP ranks 的 sample indices 完全一致；
- VAE latent、action/context/proprio batch SHA-256 完全一致；
- video noise/timestep 和 action noise/timestep SHA-256 完全一致；
- step-1 DCP 的 1,651 个 live model parameter tensors 装载前后 bitwise 一致；
- continuous step-2 global loss：`4.342782914625`；
- resumed step-2 global loss：`4.3425183595`；
- global loss absolute difference：`2.64555e-4`；
- per-rank loss maximum difference：`8.71658e-4`；
- grad norm：`7.854` 对 `7.855`。

两个独立 PPU/PCCL launch 的最终 BF16 model 比较覆盖
`6,020,710,599` elements：

- maximum absolute difference：`1.220703125e-4`；
- mean absolute difference：`1.4283151089858059e-6`；
- unequal elements：`297,961,473`，占 `4.9489%`。

输入、随机张量和 checkpoint roundtrip 已严格相同；剩余差异来自独立进程启动中的
PPU/PCCL BF16 collective reduction 非确定顺序。因此验收标准是上述数值包络，而不是
不真实的跨 launch bitwise 参数一致性。

CPU gate：

```bash
PYTHONPATH=. FAST_WAM_DISABLE_MCORE=1 \
python -m pytest -q fast_wam/tests/test_fast_wam.py
```

当前结果：`14 passed`。真实 MCore BF16 tiny forward/backward gate 的 reference
和 optimized loss 完全相同，代表性 gradient maximum absolute difference 为
`1.220703e-3`。

## Step 7.5：吞吐、MFU 与功率

同机 TP1+DP8、microbatch 16、global batch 128、在线 VAE、丢弃前两步初始化开销
后的短跑结果：

| 路径 | 稳态 step 中位数 | samples/s | 相对 reference |
| --- | ---: | ---: | ---: |
| reference SDPA | 2.8877 s | 44.33 | baseline |
| structured + optimized kernels | 2.6187 s | 48.88 | +10.27% |
| 上述路径 + param-gather overlap | 2.5899 s | 49.42 | +11.50% |
| 上述路径 + 离线 latent cache | 1.5562 s | 82.25 | +85.57% |

这里“提升”按 throughput 计算；按 step-time reduction 分别为 9.32% 和 10.31%。
前三行均是在线 VAE；第四行使用 29.199 GiB 完整 cache，其相对 reference 的
step-time reduction 为 46.11%，相对在线 optimized profile 的吞吐提升为 66.43%。

MFU 使用本 workload 的 useful model FLOPs `8.52455 TF/sample`，以及同机 BF16
`8192^3` GEMM 实测 `135.479 TF/s/device` 作为经验 roofline，而不是厂商理论峰值。
由此 reference 约 `34.86%`，optimized 在线路径约 `38.87%`，cached optimized
路径约 `64.69%`。400 W 是功率
上限而不是验收目标；正式 20k reference 的 30 秒采样为平均约 291.7 W、平均利用率
96.8%、最大约 330.7 W。cached 30-step 短跑中的 96 个 device observations 为
平均 314.51 W、平均利用率 99.67%、最大 336.08 W。新路径仍以 step time/MFU
为主指标，功率只作为没有明显 host starvation 的诊断。

已通过的性能 profile 可用：

```bash
bash fast_wam/scripts/train_libero_optimized.sh
```

该入口要求完整 latent cache，并默认设置
`structured_sdpa + optimized + overlap-param-gather`。原
`train_libero_official.sh` 默认仍是 reference，便于回退和做严格 A/B；在完整长跑
质量门禁完成前，不把 optimized profile 宣称为新的 20k accuracy baseline。

正式 optimized 20k→2k 串行入口为：

```bash
bash fast_wam/scripts/run_libero_optimized_20k_and_eval.sh
```

默认使用独立目录
`outputs/fast_wam_libero_training_20k_optimized_20260725/`，每 2,000 steps 保存
完整 optimizer DCP；如果 tracker 已存在则从最新 DCP 恢复。训练完成后自动以最终
DCP 启动 `outputs/fast_wam_libero_training_20k_optimized_20260725_eval_2k/` 的
完整可恢复评测。该实验于 2026-07-25 02:19 UTC+8（2026-07-24 18:19 UTC）
启动；step 1 loss 为
`4.026691`，与已验收 cached smoke exact，step-20 window 为 `1.5579 s`，无
skip/NaN。20,000 steps 于 11:02 UTC+8 完成，最终 validation loss 为
`0.08776479`；2,000-episode gate 于 14:38 UTC+8 完成：

| Suite | Optimized 20k | Paper target |
| --- | ---: | ---: |
| Spatial | 479/500（95.8%） | 491/500（98.2%） |
| Object | 495/500（99.0%） | 500/500（100%） |
| Goal | 489/500（97.8%） | 485/500（97.0%） |
| LIBERO-10 | 463/500（92.6%） | 476/500（95.2%） |
| **Overall** | **1,926/2,000（96.30%）** | **1,952/2,000（97.60%）** |

只有 Goal 达到 suite target。总体比 reference 20k 的 1,923 多 3 个成功 episode，
但比本地 release DCP 的 1,938 少 12 个，也比论文目标少 26 个，不能宣称精度复现。
评测器在完整写出 2,000 cases 和 `summary.json` 后因 `meets_target=false` 按设计
返回 exit code 1；这不是 rollout 或聚合故障。

官方 10 epochs 对应的 21,700-step optimized fresh run 使用：

```bash
bash fast_wam/scripts/run_libero_optimized_21700.sh
```

该入口把 `TRAIN_ITERS` 和 LR schedule 都设为 21,700，并以 2,170 steps 作为
checkpoint interval，保留十个 epoch-boundary optimizer DCP。它与当前 20k
artifact 隔离、支持 tracker resume，训练成功后自动追加可恢复的完整 2k eval。

该 run 已于 2026-07-25 14:48 UTC+8 fresh 启动，artifact 为
`outputs/fast_wam_libero_training_21700_optimized_20260725/`，tmux
socket/session 为 `fastwam_opt_21700_20260725` / `fastwam_opt_21700`。
首步 loss `4.026691` 与验收基线 exact；step 20/30 分别为
`1.5572/1.5563 s`，8 卡均为 100% utilization，未出现 skip/NaN。本次训练早于
自动 eval 改动启动，因此通过独立 `eval_watcher_21700` tmux session 接续；
watcher 仅在训练 session 退出且 tracker 精确为 21,700 时启动 eval，异常或
未完成退出不会误跑。训练于 2026-07-26 00:18 UTC+8 完成，最终
train/validation loss 为 `0.0878931/0.1158151`。2k eval 于 09:01 UTC+8
启动、12:48 完成：

| Suite | 21.7k optimized | Optimized 20k | Paper target |
| --- | ---: | ---: | ---: |
| Spatial | 477/500（95.4%） | 479/500（95.8%） | 491/500（98.2%） |
| Object | 498/500（99.6%） | 495/500（99.0%） | 500/500（100%） |
| Goal | 485/500（97.0%） | 489/500（97.8%） | 485/500（97.0%） |
| LIBERO-10 | 453/500（90.6%） | 463/500（92.6%） | 476/500（95.2%） |
| **Overall** | **1,913/2,000（95.65%）** | **1,926/2,000（96.30%）** | **1,952/2,000（97.60%）** |

只有 Goal 精确达到论文 suite target。21.7k 总体比 optimized 20k 少 13 个成功
episode，比本地 release DCP 的 1,938 少 25 个，比论文目标少 39 个，不能宣称
精度复现。2,000 个 case 和最终 `summary.json` 均完整；evaluator 因
`meets_target=false` 按设计返回 exit code 1，并非 rollout 或聚合故障。

## Step 8：最终 2,000-episode gate

训练完成后：

```bash
TRAIN_DCP=outputs/fast_wam_libero_training_20k \
OUTPUT_DIR=outputs/fast_wam_libero_training_20k_eval_2k \
bash fast_wam/scripts/eval_libero_trained_2k.sh
```

runner 从训练 DCP tracker 解析最终 iteration，按 TP1+DP8 reshard load。四个 suite
各 10 tasks、每 task 50 init states，共 2,000 episodes。每个 case 原子写 JSON，
中断后 `--resume` 跳过已有 case；只有 2,000 case 齐全且根目录生成
`summary.json` 才算完成。

最终判读分两层：

1. 与同环境 release checkpoint 的完整本地 run 比较，要求没有统计显著回退；
2. 单独报告是否达到论文 1,952/2,000（97.6%），不能用第一层替代论文门槛。

正式 20k 与其后 2k rollout 是长时间作业；运行中的 canonical artifacts 为：

```text
outputs/fast_wam_libero_training_20k/
outputs/fast_wam_libero_training_20k_eval_2k/
```
