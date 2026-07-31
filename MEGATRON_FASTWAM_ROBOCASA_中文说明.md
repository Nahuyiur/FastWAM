# Megatron FastWAM RoboCasa 中文说明

Last updated: 2026-08-01

## 1. 目标与代码来源

本分支的目标不是重写 FastWAM，而是在**保持 RoboCasa baseline 数据、模型数学和
评估指标不变**的前提下，把训练执行层替换为 Megatron Core，使同一份模型可以用
DP/TP 扩展到 4、8、16 卡，并使用可跨拓扑恢复的分布式 checkpoint。

代码来源如下：

- RoboCasa baseline：本仓库提交 `f86adf7`，原始目录保持不动；
- Megatron-Wan：`KuangzhiGe/Megatron-Wan` 的提交
  `2ed4ae185bfdb8368a7522895baa273554e9a535`；
- 本分支：`megatron-fastwam-robocasa`；
- 远端独立目录：`/mnt/yuhan/FastWAM_megatron_robocasa`。

`megatron/`、`wan/` 和 `fast_wam/` 是从上述 Megatron-Wan 提交引入的源码；
RoboCasa baseline 的 `src/fastwam`、`configs/` 和数据定义没有被复制后再暗改。
Megatron-Wan 的完整上游许可证保留在
`third_party/Megatron-Wan-LICENSE`，来源说明保留在
`third_party/Megatron-Wan-NOTICE.md`。

## 2. 保持不变的训练合同

Megatron 版本继续复用 baseline 的 Hydra task
`robocasa_acg_v1_fastwam_8gpu`，因此以下内容保持一致：

- 数据划分：`train_id=3440 episodes`、`val_id=430 episodes`；
- 两相机：`robot0_agentview_left` 与 `robot0_eye_in_hand`；
- RGB：9 帧，索引为 `0,4,...,32`，每路 `224x224`，沿宽度拼为 `224x448`；
- 动作：32 步、12 维；proprio：32 步、16 维；
- 文本：已有 UMT5 128-token embedding cache；
- 归一化：沿用 baseline 的 train-id stats 和 MIN/MAX 处理；
- FlowMatch、video/action loss、attention 可见性和 action inference 均以 RoboCasa
  task 实际选中的 `fastwam_joint` 为基准；
- 正式成功率仍由 `info["success"]` 给出，不把额外诊断指标混成成功率。

## 3. 在 FastWAM 基础上的具体修改

### 3.1 Megatron 模型与初始化

主要文件：

- `fast_wam/model.py`：用 Megatron Core linear/attention 构造 VideoDiT 与 ActionDiT；
- `fast_wam/components.py`：保持 FastWAM MoT block、mask、RoPE、loss 的原有数学；
- `fast_wam/pretrain.py`：接入 Megatron `pretrain()`、forward step 与 loss reduction；
- `fast_wam/train/initialization.py`：将 Wan2.2 VideoDiT 和 FastWAM ActionDiT 初始化
  转为可重分片 DCP；
- `fast_wam/checkpoint.py`：支持 safetensors 流式加载和跨 TP/DP 的 DCP 加载。

RoboCasa 需要的 action/proprio 尺寸不是 LIBERO 默认值，所以初始化入口新增
`action_dim=12`、`proprio_dim=16`。ActionDiT backbone 仍采用 FastWAM 的顺序一维
线性插值初始化；action/proprio I/O 层在固定 seed 42 下初始化。

当前初始 DCP manifest 记录：VideoDiT 复制 825 个 tensor；ActionDiT 直接复制 300、
插值 520、结构随机 4；action/proprio I/O 随机初始化 6 个 tensor。所有 tensor 使用
BF16 保存，DCP 可在 TP1/TP2/TP4 间重分片。

### 3.2 注意力等价优化

RoboCasa task 通过 `override /model: fastwam_joint` 选择的合同是：首帧 video query
只看首帧；未来 video query 看全部 video；action query 看全部联合去噪 video 和全部
action；video 不看 action。这里的 future video 是当前 diffusion step 的带噪/预测
latent，不是数据中的真值未来 RGB。

普通 LIBERO FastWAM 的 action 只看首帧 video。两种合同现在由
`joint_action_video_attention` 显式区分：普通入口默认 `false`，RoboCasa 训练脚本和
评估 backend 强制设为 `true`，避免修改 RoboCasa 时暗中改变 LIBERO。当前提供两个
语义相同的优化后端：

- `structured_sdpa`：按三类 query 拆成三个无 mask SDPA，仍是正式默认；
- `flex`：将同一谓词编译成 FlexAttention BlockMask。BlockMask 按
  `device/video长度/action长度/首帧长度` 缓存，不再依赖临时 dense mask 的内存地址，
  也不在每个 step 构造二次方 dense mask。

两条路径使用同一套 Q/K/V 投影、RoPE、残差、cross-attention、FFN 和 loss。所选
FastWAM 变体的访问图
逐元素完全相同；但 BF16 kernel 的归约顺序不同，因此只保证受门限约束的数值等价，
不承诺位级相同。可通过 `ATTENTION_BACKEND=flex` 显式启用 Flex，默认仍为
`structured_sdpa`。

这里的 SDPA 是 PyTorch 的 scaled dot-product attention 计算接口，不等同于
Megatron。`structured_sdpa` 只是利用当前 mask 的块结构，把一次带 dense mask 的
attention 等价拆成三个 SDPA 调用；Megatron 负责的是 TP/DP、分布式优化器和 DCP。

### 3.3 RoboCasa 数据桥接

主要文件：

- `fast_wam/train/robocasa_data.py`；
- `fast_wam/train/prepare_robocasa_latents.py`；
- `fast_wam/pretrain_robocasa.py`。

数据桥不复制 baseline 的数据处理逻辑，而是直接实例化原 Hydra dataset。启动时会
读取一个真实样本，强校验 action 最后一维为 12、proprio 最后一维为 16。多 rank
读取 metadata 时按 rank 分阶段打开，避免所有进程同时冲击共享文件系统。

可选 latent cache 只缓存冻结 Wan VAE 的 BF16 输出。cache loader 直接读取原 parquet
中的 action/proprio/text metadata，并加载对应 latent，**不会先解码 MP4 再丢弃
video**。该性质已有专门测试，且通过把 baseline `_load_video` 替换为强制抛错进行了
真实集成验证。

### 3.4 分布式训练与断点续训

主要文件：

- `fast_wam/scripts/train_robocasa_megatron.sh`；
- `fast_wam/pretrain_robocasa.py`。

训练启用：

- Megatron Core TP + DP；
- distributed optimizer；
- overlap grad reduce 与可选 overlap param gather；
- BF16 parameter/gradient reduction；
- `torch_dist` DCP，完整保存模型、优化器、调度器、RNG 和 consumed samples。

当前机器没有 Transformer Engine/Apex，因此使用 Megatron 官方 Torch optimizer
fallback，并显式关闭依赖 Apex 的 gradient accumulation fusion。原生 Torch Adam
在首次加载前没有 state，Megatron upstream 的 DCP template 会提前断言；
`pretrain_robocasa.py` 仅在 load 路径为该 fallback 初始化 dummy optimizer state，
然后继续使用 upstream sharded-state load，不更改优化器数值。

启动器支持 `EXIT_INTERVAL`。严格续训测试使用相同的 `TRAIN_ITERS=3`：第 2 step
保存并正常退出，再从完整 DCP 恢复到第 3 step，避免通过修改总步数造成学习率调度器
不一致。

### 3.5 正式 RoboCasa 评估

主要文件：

- `scripts/robocasa_acg_policy_backends.py`；
- `scripts/robocasa_acg_eval.py`。

新增 `fastwam_megatron` backend：

1. 从训练 DCP model-only 加载 FastWAM；
2. 用冻结 Wan VAE 编码当前两路 RGB；
3. 读取同一份 UMT5 cache 和 normalization stats；
4. 同时初始化未来 video latent 与 action latent，按 10 个 diffusion step 联合去噪；
5. 每步 action 读取当前预测的全部 video latent，运行
   `infer_action_encoded()` 得到 action；
6. 反归一化为 `(32,12)` action chunk；
7. 复用原 evaluator 的环境、replan、视频和 `info["success"]` 统计。

训练和评估退出时都会显式销毁 NCCL process group，避免连续作业留下通信上下文。

### 3.6 语义审计发现与纠错

2026-08-01 对 RoboCasa Hydra 实际入口进行反向追踪后，确认 task 选择的是
`fastwam_joint`，而不是普通 FastWAM。旧 Megatron 实现存在两处会改变策略输出的
偏差：

1. action query 只读取首帧 video，训练和评估均没有复现
   `FastWAMJoint` 的“action 读取全部联合去噪 video”合同；
2. VideoDiT 输出 patch 的反展开顺序写成了 `[channel, pt, ph, pw]`，而 baseline
   实际采用 `[pt, ph, pw, channel]`。单帧 action-only smoke 不会暴露该错误，但联合
   视频滚动会把预测 latent 排列打乱，并继续影响 action。

当前修复包括：显式区分 ordinary/Joint 两种模型合同；RoboCasa train/eval 强制 Joint；
恢复联合 video-action 去噪；修正 patch 反展开；新增直接调用 baseline 源模块的
跨实现回归。修复前的 Megatron/Flex/cache 性能数字只能说明错误实现的运行速度，不能
再作为“与 RoboCasa baseline 同语义”的加速证据。

## 4. 准备与启动

远端默认 Python：

```bash
export PYTHON_BIN=/mnt/yuhan/envs/motus-rebuilt-v2_10/bin/python
cd /mnt/yuhan/FastWAM_megatron_robocasa
```

当前环境只额外需要 `pybind11==2.13.6` 来编译 Megatron Core 的 dataset helper；
没有升级 PyTorch，也没有安装不匹配的 Transformer Engine。

### 4.1 生成初始 DCP

```bash
bash fast_wam/scripts/prepare_robocasa_megatron.sh
```

默认输出：

```text
outputs/robocasa_megatron_assets/initial_dcp_bf16/
```

### 4.2 可选 latent cache

```bash
bash fast_wam/scripts/prepare_robocasa_latents.sh
```

cache 构建支持 shard、原子写入和 resume。必须等 `manifest.json` 中
`complete=true` 才能用于正式训练。

### 4.3 4 卡训练

模型可以在单张 A800 80GB 上容纳，因此吞吐优先默认 TP1+DP4：

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 \
GPUS_PER_NODE=4 TP_SIZE=1 \
MICRO_BATCH_SIZE=1 GLOBAL_BATCH_SIZE=32 \
TRAIN_ITERS=50000 \
SAVE_DIR=outputs/robocasa_megatron_50k \
bash fast_wam/scripts/train_robocasa_megatron.sh
```

显存优先可用 TP2+DP2 或 TP4+DP1：

```bash
TP_SIZE=2 GLOBAL_BATCH_SIZE=32 bash fast_wam/scripts/train_robocasa_megatron.sh
TP_SIZE=4 GLOBAL_BATCH_SIZE=32 bash fast_wam/scripts/train_robocasa_megatron.sh
```

`GLOBAL_BATCH_SIZE` 必须能被 `MICRO_BATCH_SIZE * DP_SIZE` 整除。

### 4.4 8/16 卡建议

如果每卡仍为 80GB，吞吐优先建议：

| 卡数 | 推荐拓扑 | 说明 |
|---:|---|---|
| 4 | TP1+DP4 | 当前实测最快的主配置 |
| 8 | TP1+DP8 | 优先增加数据并行吞吐 |
| 16 | TP1+DP16 | 单机/多机均可，需设置 NNODES/NODE_RANK |
| 8 | TP2+DP4 | 显存紧张或增大 local batch 时使用 |
| 16 | TP2+DP8 | 显存与吞吐折中 |

8/16 卡启动器和 DCP 拓扑合同已经实现，但当前 Jinshan 节点只有 4 张 A800，故
8/16 卡没有物理实测，正式长训前仍需分别做 20--100 step smoke。

多机示例：

```bash
NNODES=2 NODE_RANK=0 MASTER_ADDR=<node0-ip> MASTER_PORT=29603 \
GPUS_PER_NODE=8 TP_SIZE=1 bash fast_wam/scripts/train_robocasa_megatron.sh
```

另一节点只把 `NODE_RANK` 改为 1，其余训练参数必须一致。

### 4.5 恢复训练

```bash
LOAD_DIR=outputs/robocasa_megatron_50k \
SAVE_DIR=outputs/robocasa_megatron_50k \
TRAIN_ITERS=50000 \
bash fast_wam/scripts/train_robocasa_megatron.sh
```

恢复时不要任意改变 LR、warmup、总步数或 optimizer 参数；Megatron 会检查调度器
合同。不一致应被视为实验配置变化，而不是普通 resume。

### 4.6 评估

```bash
$PYTHON_BIN scripts/robocasa_acg_eval.py \
  --policy-backend fastwam_megatron \
  --fastwam-repo /mnt/yuhan/FastWAM_megatron_robocasa \
  --fastwam-checkpoint outputs/robocasa_megatron_50k \
  --fastwam-vae-checkpoint /mnt/yuhan/FastWAM/checkpoints/Wan-AI/Wan2.2-TI2V-5B/Wan2.2_VAE.pth \
  --fastwam-norm-stats /mnt/yuhan/experiments/robocasa_acg_v1/fastwam/cache/norm_stats/robocasa_acg_v1_train_id_dataset_stats.json \
  --fastwam-text-cache /mnt/yuhan/experiments/robocasa_acg_v1/fastwam/cache/text_embeds/robocasa_acg_v1 \
  --output-dir outputs/robocasa_megatron_eval
```

正式评估仍应使用既定 plan、episode 数和 video policy；不能用单步 inference smoke
替代成功率评估。

可重复的真实数据/DCP 推理回归入口为
`fast_wam/tests/run_robocasa_eval_smoke.py`。它会从 train_id 读取一条真实两相机样本，
检查输出 shape、有限值和进程组释放，但仍只是链路 smoke。

## 5. 已完成的验证

### 5.1 功能闭环

完整 VideoDiT + ActionDiT 参数量由训练入口实测为 `6,020,753,612`。

- CPU：25 个 FastWAM 模型回归测试通过；其中固定同一组权重、真实 baseline 源模块、
  固定 batch/noise/timestep，对比了 forward、video/action loss、全部参数梯度、AdamW
  单步更新和 Joint action inference；
- 真实 TP2 小模型：forward、backward、AdamW、DCP roundtrip exact；
- 6B 模型 TP1+DP4：训练、验证、完整保存、严格恢复通过；
- 6B 模型 TP2+DP2：训练与验证通过，峰值约 40.5GB/卡；
- 6B 模型 TP4+DP1：训练与验证通过，峰值约 34.9GB/卡；
- Megatron eval backend：从训练 DCP 加载，真实 RoboCasa 图像/文本输入，输出
  `(32,12)` 且全为有限值。
- 编译版 FlexAttention 独立 forward/backward 通过；Flex mask 谓词对三组不同长度
  与 dense mask 逐元素完全一致；
- 正式 RoboCasa 尺寸的 `6,020,753,612` 参数 BF16 Joint 模型用 seed 17/29/43 做
  同权重 structured/Flex A/B，三组均通过。最差 action loss 相对差 `0.1149%`，
  全层参数梯度范数相对差 `0.4136%`，全层抽样梯度最大绝对差 `0.0234375`。显式
  `joint_action_video_attention=true` 同步到远端后又独立复跑 seed 17，通过相同门禁；
- 远端最终代码的 25 个模型回归测试及 Python 编译检查均通过。

严格断点测试关键结果：iteration 2 的 DCP 保存约 30.1 秒；恢复 load 约 20.6 秒；
恢复后 iteration 精确为 2，下一步 consumed samples 从 8 增到 12；step 3 和 validation
均无 skipped/nan iteration。

### 5.2 4 卡长时间性能测试

硬件统一为 4 张 A800-SXM4-80GB，BF16，单卡 microbatch=1。每组丢弃 warmup 后
统计 20、40 或 80 个稳态 step；无 W&B、无保存。baseline 使用原仓库 ZeRO-1 与原
activation checkpointing；Megatron 当前没有 activation recompute，因此速度提升
伴随更高显存，不能只报速度而隐去该差异。

修正 Joint attention、联合推理和 video patch 顺序后，重新运行 online-VAE、
`structured_sdpa`、global batch 4 的 100-step 训练。两边都丢弃前 20 step，统计
step 21--100：

| 协议 | mean | median | P90 | 稳态 step | skipped/nan |
|---|---:|---:|---:|---:|---:|
| baseline FastWAMJoint | 0.9364 s | 0.8941 s | 1.0827 s | 80 | 0/0 |
| corrected Megatron Joint | 0.7325 s | 0.7234 s | 0.7480 s | 80 | 0/0 |

按 mean 计算为 `1.278x`，即平均 step 时间降低 `21.77%`。baseline profiler 的峰值
约 `41.0 GiB/卡`；corrected Megatron rank-0 日志的 max allocated 为
`52,118 MB`（约 `50.9 GiB`）。速度提升伴随更高显存，主要原因是 baseline 启用了
activation checkpointing，而当前 Megatron 路径没有 recompute。

修复前的 `1.290x`、Flex、global batch 32 和 8-sample cache 表格已撤销。它们不是
Joint 同语义结果。当前 Flex 只有 6B BF16 数值等价证据，尚无修正语义后的正式吞吐
优势；8-sample cache 也被完整数据合同正确拒绝，不能冒充 286101-window 正式 cache。

复测入口：

```bash
BENCH_ROOT=outputs/benchmarks/baseline_repeat \
TRAIN_ITERS=60 WARMUP_ITERS=20 \
bash fast_wam/scripts/benchmark_robocasa_baseline.sh

BENCH_ROOT=outputs/benchmarks/megatron_repeat \
TRAIN_ITERS=60 WARMUP_ITERS=20 \
bash fast_wam/scripts/benchmark_robocasa_megatron.sh

BENCH_ROOT=outputs/benchmarks/megatron_joint_repeat \
ATTENTION_BACKEND=structured_sdpa TRAIN_ITERS=100 WARMUP_ITERS=20 \
MICRO_BATCH_SIZE=1 GLOBAL_BATCH_SIZE=4 \
bash fast_wam/scripts/benchmark_robocasa_megatron.sh
```

## 6. 当前局限与正式训练前门禁

1. 当前没有 Transformer Engine/Apex，未启用 fused Adam、FP8 或依赖 Apex 的 fusion；
2. 当前不支持 FastWAM training 的 PP、CP、SP 和 activation recompute；
3. TP2/TP4 主要降低显存，当前 4 卡模型规模下未体现吞吐优势；
4. 8/16 卡与多机只完成代码合同，尚未物理实测；
5. 性能测试不是收敛质量测试，不能据此宣称成功率不回退；
6. FlexAttention 已通过结构与 6B BF16 数值门禁，但修正 Joint 语义后尚未证明吞吐
   优势，也没有 50k 收敛与完整 RoboCasa 成功率证据，因此不应替换正式默认；
7. 正式 50k 前至少要完成：目标拓扑 100-step 数值 smoke、save/resume、固定 seed
   open-loop action parity，以及与 baseline 相同 plan 的完整在线 RoboCasa eval。
