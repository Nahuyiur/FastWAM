# Video Generation 分布式训练 Trick 调研

> **Status:** 调研记录，2026-05-19
> **Scope:** 面向视频生成基座模型，重点关注 DiT / MMDiT / flow-matching / diffusion transformer 的大规模训练并行栈。
> **结论一句话:** 头部视频基座训练不是靠单个 trick，而是 `FSDP/ZeRO 或 distributed optimizer + TP + SP/CP + activation recompute + sharded checkpoint + mixed precision/fused kernels + 多分辨率训练配方` 的组合；闭源报告通常只讲到高层，开源报告和 repo 才能看到比较 concrete 的实现。

## Reading Path

| 想了解 | 读 |
| --- | --- |
| 30 秒结论 | [Executive Summary](#executive-summary) |
| 具体有哪些 trick | [Trick Taxonomy](#trick-taxonomy) |
| 头部模型/报告是否提到 | [Model Evidence Matrix](#model-evidence-matrix) |
| 对 Wan Megatron port 的直接要求 | [Implications For Wan Megatron Port](#implications-for-wan-megatron-port) |
| 来源链接 | [Sources](#sources) |

## Executive Summary

当前大视频生成模型训练的主要瓶颈有三类：

- **参数和 optimizer state:** 1B+ DiT 加 Adam 很快超过单卡显存，5B/14B/30B 级别必须 sharding。
- **长视频 token 序列:** latent token 规模约等于 `T * H * W`。720p、16s、长上下文训练会把 attention activation 和通信压力推到主瓶颈。
- **I/O 和容错:** checkpoint 体积大、训练周期长，必须支持 sharded save/load、异步或高吞吐 checkpoint、不同并行度 resume/reshard。

因此头部方案通常是多层组合：

```text
Data/bucket curriculum
  + 3D VAE latent compression
  + BF16/FP8/fused attention
  + activation recompute
  + TP for matmuls
  + SP/CP for long visual sequence
  + FSDP/ZeRO/distributed optimizer for params/grads/optimizer
  + sharded distributed checkpoint
```

Tech report 披露粒度差异很大：

- **Sora / Veo:** 主要讲 latent patches、variable size、caption/data/safety，不披露具体训练并行实现。
- **Movie Gen / Goku / Open-Sora / Wan / Hunyuan / Seedance:** 会不同程度提到 parallelization、FSDP、sequence parallel、checkpoint、kernel/system optimization。
- **代码更可靠:** Wan2.1 repo 里直接能看到 FSDP 和 xDiT/USP context parallel；Open-Sora 报告明确写 sequence parallel 的实现取舍；Goku 公开材料提到 FSDP HYBRID_SHARD、activation checkpointing、ByteCheckpoint。

## Trick Taxonomy

### 1. FSDP / ZeRO / Distributed Optimizer

**用途:** shard parameters、gradients、optimizer states，解决 Adam state 和 full replica 显存问题。

**常见形式:**

- PyTorch FSDP `FULL_SHARD` 或 `HYBRID_SHARD`
- DeepSpeed ZeRO / ColossalAI ZeroDP
- Megatron distributed optimizer
- FSDP + activation checkpointing + mixed precision

**公开证据:**

- Wan2.1 `wan/distributed/fsdp.py` 使用 PyTorch `FullyShardedDataParallel`，默认 `ShardingStrategy.FULL_SHARD`，mixed precision 为 BF16 params / FP32 reduce / FP32 buffer。
- Goku 公开材料提到 FSDP `HYBRID_SHARD`：shard group 内 FULL_SHARD，group 间 replication，用于减少通信成本。
- Open-Sora 2.0 公开摘要/解读提到 ColossalAI，MMDiT training 使用 ZeroDP + Context Parallelism。

**工程判断:** 这是 5B+ 视频 DiT 的基础能力。没有 sharded optimizer，就只能做小模型或短 smoke。

### 2. Tensor Parallelism, TP

**用途:** shard transformer matmul，降低每 rank 权重和 optimizer state，并提升大矩阵吞吐。

**常见切法:**

- Q/K/V projection: Column parallel
- Attention output projection: Row parallel
- FFN: Column up-proj + Row down-proj
- Text/time conditioning MLP: Column + Row
- Output head: 视 shape 和并行策略选 Row/Column

**公开证据:**

- Movie Gen 最大 video model 是 30B transformer，最大上下文 73K video tokens，并明确说 paper 涵盖 parallelization techniques。
- Movie Gen 相关公开材料提到 model parallelism / 3D parallelism，用于 30B backbone 和 73K-token long-context training。

**工程判断:** TP 不能盲目套 LLM pattern。视频 DiT 里 QK norm、3D RoPE、cross-attention、patchify/unpatchify 都会影响切法。比如 Wan QK RMSNorm 如果在 full hidden dimension 上做，就必须保证 TP 下 numerics 正确。

### 3. Sequence Parallelism / Context Parallelism, SP/CP

**用途:** 沿 sequence/token 维切分长视频 latent tokens，降低 attention activation 和 per-rank sequence memory。

**常见形式:**

- Ulysses / all-to-all sequence parallel
- Ring Attention / long-context attention
- xDiT / xFuser USP
- Megatron Context Parallelism

**公开证据:**

- Wan2.1 `xdit_context_parallel.py` 使用 xFuser/xDiT sequence parallel group；RoPE 会按 SP rank 取对应 freqs，attention 用 `xFuserLongContextAttention`。
- Open-Sora 1.2 报告有 “Sequence parallelism” 一节，说明基于 Ulysses，用 all-to-all 支持 long-sequence；当时训练分辨率较小所以主要用于推理 OOM 场景。
- ColossalAI 文档列出 TP+SP、DeepSpeed-Ulysses、Ring Attention 三类 sequence parallel。

**工程判断:** SP/CP 是高分辨率/长视频训练的关键进阶项，但非常容易错。必须同时处理 3D RoPE、padding、attention mask、variable grid、loss reduction、CFG/context dropout RNG 和 checkpoint reshard。

### 4. Pipeline Parallelism, PP

**用途:** 沿层切分模型，降低单 rank 权重/activation 压力，常与 TP/FSDP/CP 组成 3D/4D parallelism。

**公开证据:**

- Movie Gen 公开材料提到对 30B model 使用多轴 model parallelism。
- Megatron-Core 文档把 DP/FSDP、TP、PP、CP 作为可组合并行维度。

**工程判断:** 1.3B/5B 阶段通常可以先用 TP + distributed optimizer/FSDP；14B/30B 或更长上下文才更可能需要 PP。PP 支持必须验证 DCP、resume、loss、inference，不应只靠配置开关。

### 5. Activation Checkpointing / Recompute

**用途:** backward 时重算 activation，换取显存下降。视频 DiT activation 很大，recompute 几乎是标配。

**公开证据:**

- Goku 公开材料明确提到 activation checkpointing，并与 FSDP、ByteCheckpoint 一起作为大规模训练基础设施。
- NVIDIA NeMo 文档总结 activation checkpointing 通常与 FSDP 参数 sharding 组合使用。

**工程判断:** 自定义 DiT port 不能只传 `--recompute-granularity full`。如果模型不是 Megatron 内置 Transformer block，必须在自定义 block forward 里显式接入 checkpoint/recompute，并在日志和显存上验证。

### 6. Sharded / Async / Reshardable Checkpointing

**用途:** 支持大模型训练容错、并行保存、不同并行拓扑加载。

**常见能力:**

- per-rank sharded checkpoint
- async save / pipeline I/O
- TP/DP/FSDP reshard
- optimizer/scheduler/RNG state resume
- checkpoint load 和 weight transfer pipeline

**公开证据:**

- Goku 公开材料提到 ByteCheckpoint 支持 partitioned checkpoint 的高效 save/load 和 distributed checkpoint resharding。
- Open-Sora 2.0 公开解读提到 checkpoint optimization，包括 pinned memory、async disk writing、shard reading 与 weight transfer pipeline。

**工程判断:** Megatron port 应坚持 `torch_dist` / DCP，而不是单文件 `.pt`。验收至少包括 official ckpt -> Megatron DCP、DCP resume、optimizer/scheduler resume、TP=N checkpoint load/infer。

### 7. Mixed Precision / FP8 / Kernel Fusion / Flash Attention

**用途:** 提升吞吐、降低显存。

**常见形式:**

- BF16 baseline
- FP8 training/inference
- FlashAttention / SDPA / xFormers
- fused QKV/MLP/norm kernels
- heterogeneous quantization / sparsity

**公开证据:**

- HunyuanVideo 公开页面有 FP8 weights/inference release，并提供 xDiT parallel inference。
- Seedance 1.0 blog 提到 kernel fusion、heterogeneous quantization and sparsity、adaptive hybrid parallelism、async offloading、distributed VAE hybrid parallelism。

**工程判断:** BF16 是安全 baseline；FP8 需要 layer policy、amax recipe、norm/loss FP32 策略和 overfit/consistency 验证，不能只开 flag。

### 8. Multi-Resolution / Frame Packing / Bucketed Batching

**用途:** 让 variable duration/resolution/aspect ratio 训练可吞吐、可收敛、可控显存。

**公开证据:**

- Sora report 说训练 joint images/videos，支持 variable durations/resolutions/aspect ratios，并把视觉数据变成 spacetime latent patches。
- CogVideoX 使用 progressive training 和 multi-resolution frame pack。
- Open-Sora 1.2 支持 0s-16s、144p-720p、多 aspect ratios，并引入 temporal compression 3D VAE。

**工程判断:** 这不是通信 primitive，但决定训练系统是否真实可用。真实基座训练需要按 latent token count / frames / resolution / aspect 分桶，避免 batch token 量大幅波动。

### 9. Progressive Curriculum / Joint Image-Video Training

**用途:** 降低训练成本，先学图像/低分辨率/短视频，再逐步提高帧数和分辨率。

**公开证据:**

- HunyuanVideo 提到 data curation、image-video joint training、高效大规模训练基础设施，并训练 13B+ video model。
- CogVideoX 采用 progressive training。
- Seedance 1.0 采用 unified efficient pre-training framework，支持 T2I/T2V/I2V joint learning、多镜头、多任务建模。

**工程判断:** 这影响 data loader、bucket sampler、condition schema 和 checkpoint schedule。分布式实现要能支持不同 shape 的 curriculum 逐步切换。

## Model Evidence Matrix

| 模型 / 项目 | 披露程度 | 提到/可见的分布式或系统 trick | 备注 |
| --- | --- | --- | --- |
| Sora | 低 | latent patches、variable duration/resolution/aspect、scaling | 报告明确说不包含 model 和 implementation details。 |
| Veo 3 | 低 | 数据、caption、安全、post-training | 技术报告不披露训练并行栈。 |
| Movie Gen | 中 | 30B transformer、73K video tokens、parallelization techniques、3D/model parallelism | 头部闭源里相对更明确提到 parallelization。 |
| HunyuanVideo | 中 | 13B+ model、efficient infrastructure、xDiT parallel inference、FP8 inference | 训练并行细节没有完全展开。 |
| Wan2.1 | 高，代码可查 | FSDP FULL_SHARD、USP/xDiT sequence/context parallel、CPU/model offload | repo 代码最 concrete。 |
| Open-Sora / Open-Sora 2.0 | 高 | ColossalAI、ZeroDP、Context Parallelism、Sequence Parallelism、checkpoint optimization | 开源报告对训练系统更透明。 |
| Goku | 高 | FSDP HYBRID_SHARD、activation checkpointing、ByteCheckpoint、reshardable checkpoint | 报告偏系统工程，信息很有参考价值。 |
| Seedance 1.0 | 中 | kernel fusion、heterogeneous quantization/sparsity、adaptive hybrid parallelism、async offloading、distributed VAE | 更偏 inference/system acceleration，但对训练/部署栈有参考。 |
| CogVideoX | 中低 | progressive training、multi-resolution frame pack、3D VAE | 训练配方披露多，并行细节较少。 |

## Implications For Wan Megatron Port

### 必须支持并实测

- **TP:** `ColumnParallelLinear` / `RowParallelLinear` 实切参数和 optimizer state；official full checkpoint 需要按 TP rank 做 slicing。
- **DDP / DP:** noise、timestep、context dropout RNG 在 TP rank 内一致；DP rank 正常 all-reduce。
- **Distributed optimizer:** 至少验证 Megatron distributed optimizer 或等价 ZeRO/FSDP optimizer state sharding。
- **DCP:** official ckpt load 后保存 Megatron DCP；DCP resume；DCP inference；optimizer/scheduler state resume。
- **Activation recompute:** 自定义 Wan block 必须真的 checkpoint/recompute，不能只把 Megatron flag 打进日志。
- **BF16 baseline:** 先保证 BF16 下 official ckpt load、train、resume、infer、overfit 成立。
- **真实样本 overfit:** 用官方 VAE/UMT5 latent，不把 pseudo sample 当主验证。

### 暂时不应声称支持，除非单独实现并验证

- **SP/CP:** 需要重做 3D RoPE slicing、QK norm、attention all-to-all/all-gather、padding 和 variable grid。
- **PP:** 需要 Wan blocks pipeline stage 化，并验证 DCP/resume/loss/inference。
- **FP8:** 需要 layer policy、amax recipe、FP32 norm/loss 策略、数值一致性和 overfit。
- **FSDP:** 如果改用 Megatron custom FSDP 或 Torch FSDP2，必须与 TP、DCP、official load 一起验。

### 推荐验收阶梯

1. CPU unit smoke：scheduler / forward shape / checkpoint key conversion。
2. TP=1 official ckpt train smoke：strict load `missing=0 unexpected=0`。
3. TP=2 official ckpt train smoke：每 rank 参数约为 TP=1 的一半，DCP save。
4. DP=2 train smoke：world size 2 / data parallel size 2，DCP save。
5. TP=2 + distributed optimizer smoke：确认 optimizer state sharded save。
6. TP=1 真实 1-sample overfit：loss 接近 0，DCP inference latent MSE 显著下降，VAE decode 可人工查看。
7. TP=2 + recompute 真实 overfit：确认 recompute 实际生效，DCP inference/decode 可背样本。
8. 后续再上 SP/CP/PP/FP8，每个 trick 单独加一致性校验，不混在一次大改里。

## Sources

- Sora report: [Video generation models as world simulators](https://openai.com/index/video-generation-models-as-world-simulators/)
- Veo 3 report: [Veo 3 Tech Report PDF](https://storage.googleapis.com/deepmind-media/veo/Veo-3-Tech-Report.pdf)
- Movie Gen: [arXiv 2410.13720](https://arxiv.org/abs/2410.13720), [Hugging Face paper page](https://huggingface.co/papers/2410.13720)
- HunyuanVideo: [arXiv 2412.03603](https://arxiv.org/abs/2412.03603), [Hugging Face model page](https://huggingface.co/tencent/HunyuanVideo)
- Wan2.1 distributed processing: [DeepWiki overview](https://deepwiki.com/Wan-Video/Wan2.1/8.1-distributed-processing), [FSDP source](https://raw.githubusercontent.com/Wan-Video/Wan2.1/main/wan/distributed/fsdp.py), [xDiT context parallel source](https://raw.githubusercontent.com/Wan-Video/Wan2.1/main/wan/distributed/xdit_context_parallel.py)
- Open-Sora: [Open-Sora 1.2 report](https://github.com/hpcaitech/Open-Sora/blob/main/docs/report_03.md), [Open-Sora 2.0 arXiv](https://arxiv.org/abs/2503.09642), [Open-Sora arXiv](https://arxiv.org/abs/2412.20404)
- Open-Sora Plan: [arXiv 2412.00131](https://arxiv.org/abs/2412.00131)
- Goku: [arXiv 2502.04896](https://arxiv.org/abs/2502.04896), [CVPR 2025 paper PDF](https://openaccess.thecvf.com/content/CVPR2025/papers/Chen_Goku_Flow_Based_Video_Generative_Foundation_Models_CVPR_2025_paper.pdf)
- CogVideoX: [arXiv 2408.06072](https://arxiv.org/abs/2408.06072)
- Seedance 1.0: [ByteDance paper page](https://seed.bytedance.com/en/public_papers/seedance-1-0-exploring-the-boundaries-of-video-generation-models), [Seedance blog](https://seed.bytedance.com/en/blog/tech-report-of-seedance-1-0-is-now-publicly-available), [arXiv 2506.09113](https://arxiv.org/abs/2506.09113)
- ColossalAI: [Sequence Parallelism](https://colossalai.org/docs/features/sequence_parallelism/), [ZeRO with chunk](https://colossalai.org/docs/features/zero_with_chunk/)
- Megatron-Core: [Parallelism Strategies Guide](https://docs.nvidia.com/megatron-core/developer-guide/0.17.0/user-guide/parallelism-guide.html)
- NVIDIA NeMo: [Activation checkpointing](https://docs.nvidia.com/nemo/automodel/0.4.0/guides/gradient-checkpointing.html)
