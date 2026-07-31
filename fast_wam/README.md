# Megatron Fast-WAM

Last updated: 2026-07-25

## RoboCasa 入口

本仓库的 RoboCasa Megatron 训练、latent cache、跨 topology DCP 和正式评估接入见
[`../MEGATRON_FASTWAM_ROBOCASA_中文说明.md`](../MEGATRON_FASTWAM_ROBOCASA_中文说明.md)。
RoboCasa 路线直接复用原 FastWAM baseline 的 Hydra dataset 与 success evaluator；
不会把 LIBERO 数值当作 RoboCasa 验证结论。

这是一个不修改 `megatron/` 的 Fast-WAM overlay，现已支持：

- 官方主线 LIBERO joint video/action training recipe；
- Wan2.2 VideoDiT 到 ActionDiT 的官方插值初始化；
- Megatron Core TP/DP、distributed optimizer 和跨 topology DCP；
- 与 LeRobot 一致的 Fast-WAM action inference；
- 固定 observation parity、LIBERO 50-episode 和完整 2,000-episode 评测。

详细训练 contract、逐阶段映射、已批准偏差和真实设备数值门禁见
[`docs/libero_official_training_zh.md`](docs/libero_official_training_zh.md)。

## 官方 LIBERO 训练

训练数据必须是四个独立 suite 的官方 MuJoCo-3.3.2、LeRobot-v2.1 `no_noops`
release，合计 1,712 episodes、277,713 frames。不要用 workspace 中那个 combined
LeRobot LIBERO symlink 替代。

### 1. 准备 UMT5 cache 和初始 DCP

```bash
bash fast_wam/scripts/prepare_libero_training.sh
```

这一步：

1. 对 40 个 task 生成官方 prompt 的 UMT5 128-token cache；
2. 从 Wan2.2 TI2V-5B safetensors exact load VideoDiT；
3. 按官方 sequential 1-D linear interpolation、`align_corners=True` 和 alpha
   scaling 合成 1024-hidden ActionDiT backbone；
4. 按官方 seed-42 dense `nn.Linear` 构造顺序初始化 action I/O 和 proprio；
5. 保存可 reshard 的 BF16 TP2 initial DCP。

默认 artifact：

```text
outputs/fast_wam_libero_training_assets/text_embeds/
outputs/fast_wam_libero_training_assets/initial_dcp_bf16/
```

### 2. 可选：构建离线 VAE latent cache

```bash
bash fast_wam/scripts/prepare_libero_latents.sh
```

默认在 Ruibin-owned data 目录生成约 29 GiB 的 BF16 memory-mapped cache。构建器
按 episode 只解码一次两路 MP4，再批量编码所有重叠 window；每个 shard 原子落盘并
可恢复。训练加载时会强校验数据、stats、预处理、VAE SHA-256、样本数、dtype 和
shape，contract 不匹配会 fail fast。

cache 完成后使用性能 profile：

```bash
bash fast_wam/scripts/train_libero_optimized.sh
```

该入口选择 exact structured SDPA mask、优化训练 kernel、parameter-gather overlap
和离线 latent。reference-compatible 的 `train_libero_official.sh` 默认仍保持
`sdpa + reference kernels + online VAE`。

同机 TP1+DP8 短跑中，reference/online-optimized/cached-optimized 的稳态中位 step
分别为 `2.8877 / 2.5899 / 1.5562 s`；cached profile 达到 `82.25 samples/s` 和
经验 MFU `64.69%`，相对 reference 吞吐提高 `85.57%`。这是性能和短程数值门，
不是新的 20k convergence/2k rollout 质量结论。

### 3. 训练 20k 并执行 2k rollout

```bash
bash fast_wam/scripts/run_libero_official_20k_and_eval.sh
```

默认是 8 张 PPU、TP1+DP8、local microbatch 16、global batch 128、BF16、AdamW、
20,000 steps、每 2,000 steps 保存完整 optimizer/RNG DCP。若 output 中存在合法
`latest_checkpointed_iteration.txt`，入口会自动恢复并追加训练日志。

默认输出：

```text
outputs/fast_wam_libero_training_20k/
outputs/fast_wam_libero_training_20k_eval_2k/
```

本地 20k run 已完成到 step 20,000，完整 trained-DCP gate 为
`1,923/2,000（96.15%）`：Spatial 478、Object 497、Goal 491、LIBERO-10 457。
它没有达到论文的 97.60%，也低于本地 release-DCP 的 96.90%；不能描述为精度复现。

底层训练入口可独立调用：

```bash
SAVE_DIR=outputs/my_fast_wam_run \
TRAIN_ITERS=20000 \
ATTENTION_BACKEND=sdpa \
bash fast_wam/scripts/train_libero_official.sh
```

官方 config 的 10 epochs 在当前数据量/global batch 下等于 21,700 steps；本次
20,000-step scheduler/终点是用户明确选择的实验长度。不要把 20k checkpoint 描述成
官方 21,700-step release 的逐 step 复刻。

### 4. 单独评测训练 DCP

```bash
TRAIN_DCP=outputs/fast_wam_libero_training_20k \
OUTPUT_DIR=outputs/fast_wam_libero_training_20k_eval_2k \
bash fast_wam/scripts/eval_libero_trained_2k.sh
```

runner 从 DCP tracker 选择最新 iteration，按 TP1+DP8 reshard load。每个 episode
原子写入 `cases/<case-id>.json`，支持 `--resume`。只有 2,000 个 case 齐全并生成
`summary.json` 才是完整结果。

## Training numerical contract

- 数据：33 observations、32 actions、两相机各 `224x224` 后横向拼成
  `224x448`，选择 RGB frames `0,4,...,32`。
- VAE：只在边界执行一次 `[0,1] -> [-1,1]`，9 个 RGB frames 编码为
  `[B,48,3,14,28]` BF16 latent。
- 文本：固定官方 prompt，UMT5 128 tokens，zero padding 对 DiT 保持可见。
- FlowMatch：video/action 独立 noise 和 timestep，shift 都是 5.0，首个 video
  latent step 保持 clean。
- MoT mask：普通 FastWAM 中 first video frame 只看自己，future video 看全部
  video，action 看 first video frame 和全部 action，video 不看 action。RoboCasa
  `fastwam_joint` 入口显式设置 `joint_action_video_attention=true`，此时 action 看
  全部联合去噪 video 和全部 action；两种合同不能混用。
- loss：未来 2 个 video latent steps 与 32 action tokens 分别 masked mean，
  timestep weighting 后以 1:1 相加。
- optimizer：AdamW `(0.9,0.95)`、epsilon `1e-8`、weight decay `1e-2`、
  clip `1.0`；LR `1e-4`，5% linear warmup from `1e-7`，cosine to `1e-6`。

代码同时支持 `flex`、`sdpa` 和 `structured_sdpa` training attention。structured
路径把官方 mask 精确分解为 first-video、future-video 和 action 三个无 mask SDPA，
不计算禁止的 attention pair。optimized kernel mode 进一步启用 BF16-input norm、
complex64 RoPE 和 Megatron fused bias-GELU，parameter key/shape 不变。Flex 的 CPU
forward/backward parity 已通过；当前 PPU-ZW810E 上 compiled Flex backward 会
`SIGSEGV`，因此 reference launcher 默认使用 BF16 Q/K/V 的 SDPA；代码不显式
upcast，内部 accumulation dtype 由 PyTorch/PPU backend 决定。Flex 前向可用不等于
当前 PPU 上可安全长训。Fast-WAM 的五组真实 batch-16 attention shape 均能强制进入
PyTorch Flash backend；auto 与 forced-Flash 前后向合计约 `3.88 ms`，显式
Transformer Engine FlashAttention 约 `5.33 ms`。因此 optimized profile 保持
`structured_sdpa` 的 auto dispatch，这是当前实测更快且兼容性更好的路径。

完整 optimized 复现实验入口：

```bash
bash fast_wam/scripts/run_libero_optimized_20k_and_eval.sh
```

它使用完整 BF16 latent cache、`structured_sdpa`、optimized kernels 和
parameter-gather overlap，训练 20,000 steps，每 2,000 steps 保存完整 optimizer
DCP，随后自动运行可恢复的 2,000-episode LIBERO gate。2026-07-25 完成的默认
artifact 是 `outputs/fast_wam_libero_training_20k_optimized_20260725/` 和对应的
`..._eval_2k/`。最终结果为 **1,926/2,000（96.30%）**：Spatial 479/500、
Object 495/500、Goal 489/500、LIBERO-10 463/500。它比 reference 20k 多 3 个
成功 episode，但仍低于本地 release DCP 的 1,938 和论文目标 1,952。

与官方 10 epochs 对齐的 21,700-step optimized 训练入口为：

```bash
bash fast_wam/scripts/run_libero_optimized_21700.sh
```

该入口从 initial DCP fresh start，使用相同的加速 profile，并按每个约 2,170-step
epoch 保存完整 optimizer DCP；tracker 存在时自动恢复。训练成功完成后会自动启动
可恢复的完整 2,000-episode eval。默认 artifact 为
`outputs/fast_wam_libero_training_21700_optimized_20260725/` 和
`outputs/fast_wam_libero_training_21700_optimized_20260725_eval_2k/`。

该正式 run 已于 2026-07-25 14:48 UTC+8 启动；tmux socket/session 为
`fastwam_opt_21700_20260725` / `fastwam_opt_21700`。首步 loss 为
`4.026691`（video `2.693185`、action `1.333506`），与已验收 fresh baseline
exact；step 20/30 为 `1.5572/1.5563 s`，8 张 PPU 均为 100% utilization，
无 skip/NaN。由于该 run 在自动 eval 接入前已经启动，当前任务另由
`eval_watcher_21700` tmux session 接续；只有训练 session 退出且 checkpoint
tracker 精确等于 21,700 时才会启动 eval。训练于 2026-07-26 00:18 UTC+8
完成，最终 train/validation loss 为 `0.0878931/0.1158151`，无 skip/NaN；
完整 2k eval 于 12:48 UTC+8 完成。最终结果为
**1,913/2,000（95.65%）**：Spatial 477/500、Object 498/500、Goal
485/500、LIBERO-10 453/500。它比 optimized 20k 少 13 个成功 episode，
比本地 release DCP 少 25 个，比论文目标少 39 个；只有 Goal 精确达到论文
suite target，不能宣称精度复现。

已完成真实 TP1+DP8 save/resume gate：indices、输入 latent/condition、四类 diffusion
随机张量和 DCP roundtrip exact；独立 PPU/PCCL launch 的 step-2 global loss 差
`2.65e-4`，最终 BF16 参数 max abs 差 `1.22e-4`。这是当前接受的分布式数值包络，
不宣称跨 launch bitwise 参数确定。

CPU gate：

```bash
FAST_WAM_DISABLE_MCORE=1 python -m pytest -q fast_wam/tests
```

当前结果：`14 passed`。真实 MCore BF16 reference/optimized smoke 的 loss exact，
代表性 gradient maximum absolute difference 为 `1.220703e-3`。

## Action inference 与输入 contract

- camera pixels 保持 `[0,1]`，沿 width 拼成 `224x448`；
- 当前 LeRobot revision 不应再次应用发布 checkpoint 中序列化的
  `VISUAL=MEAN_STD(.5,.5)`，否则会 double normalization；
- state/action 使用 checkpoint stats 的 MIN_MAX transform；
- gripper 输出使用 LeRobot 相同的 binary toggle；
- 每个 TP group 只有 TP rank 0 加载 frozen VAE、UMT5 和 LIBERO env，编码结果在
  TP group 内广播；
- action inference 支持 TP1/2/4、DP 和多 topology DCP reshard。

固定 8-episode acceptance：

```bash
torchrun --nproc_per_node=4 -m fast_wam.eval.acceptance \
  --checkpoint "$FASTWAM_CKPT" \
  --assets "$WAN22_DIFFUSERS_SNAPSHOT" \
  --tokenizer "$FASTWAM_CKPT/google/umt5-xxl" \
  --reference outputs/fast_wam_reference \
  --output outputs/fast_wam_tp2_dp2 \
  --tp 2 --dtype float32 --n-action-steps 10
```

门槛是完整 32-step action chunk `max_abs <= 1e-3`、gripper sign exact、闭环总成功数
不回退。已验证最大误差 `8.37e-4`，Megatron/LeRobot 都是 6/8。

## Release checkpoint 的本地 LIBERO 基线

50-episode Spatial：

```bash
bash fast_wam/scripts/run_libero_spatial_bf16.sh
```

本机 Megatron BF16 为 48/50（96%），standalone Fast-WAM 同协议为 47/50（94%）。

完整 release-checkpoint 2,000 episodes：

```bash
bash fast_wam/scripts/run_libero_full_2k_bf16.sh
```

本机结果 1,938/2,000（96.90%）；论文为 1,952/2,000（97.60%），因此没有通过论文
严格门槛。同机 LeRobot 为 1,922/2,000（96.10%）。训练后 checkpoint 要同时报告
相对同环境 release 是否有统计显著回退，以及是否达到论文门槛，二者不能混写。

## 当前不支持

- Fast-WAM training 的 PP、CP、SP、FP8/FP4 和 activation recompute；
- RoboTwin training；
- 官方每 200 steps online eval 中的视频生成、VAE decode 和 PSNR/SSIM 视频诊断。
  训练中间门禁是 deterministic validation loss，最终质量门禁是完整 LIBERO 2k。

## 深入阅读

- [`docs/libero_official_training_zh.md`](docs/libero_official_training_zh.md)：
  完整训练复现、数值 contract 和硬门禁。
- [`docs/fast_wam_paper_code_deep_dive_zh.md`](docs/fast_wam_paper_code_deep_dive_zh.md)：
  论文、官方代码、attention mask 和 FlowMatch 深入对照。
- [`docs/libero_spatial_bf16_eval_zh.md`](docs/libero_spatial_bf16_eval_zh.md)：
  release checkpoint 的 50-episode 基线。
- [`docs/libero_full_2k_bf16_eval_zh.md`](docs/libero_full_2k_bf16_eval_zh.md)：
  release checkpoint 的完整 2,000-episode 基线。
- [`log/2026-07-23-fastwam-libero-training.md`](log/2026-07-23-fastwam-libero-training.md)：
  实现、测试、artifact 和正式长跑状态。
- [`log/2026-07-24-fastwam-training-efficiency.md`](log/2026-07-24-fastwam-training-efficiency.md)：
  attention/kernel/data/communication profiling、A/B 吞吐与 MFU。
