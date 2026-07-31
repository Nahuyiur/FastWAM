# 音视频联合生成基座方案调研

> **Status:** 调研记录，2026-05-19
> **Scope:** 成熟或接近成熟的 audio-video joint generation 方案，覆盖闭源商业模型、开源/开放权重模型、带 technical report 的研究方案，以及视频后配音/音效模型。
> **结论一句话:** 当前最成熟的方向正在从“先生成视频再配音”迁移到“原生 audio-video joint generation”。闭源头部是 Sora 2 / Veo 3 / Seedance 1.5-2.0 / Kling-Omni；开源可落地的代表是 LTX-2 / LTX-2.3；研究和社区可复现路线集中在 Apollo/Klear、ALIVE、UniVerse-1、JavisDiT++、UniAVGen、ProAV-DiT，以及 HunyuanVideo-Foley / Kling-Foley 这类 TV2A 强基线。

## Reading Path

| 想了解 | 读 |
| --- | --- |
| 30 秒结论 | [Executive Summary](#executive-summary) |
| 方案分类 | [Taxonomy](#taxonomy) |
| 闭源成熟模型 | [Closed / Commercial Systems](#closed--commercial-systems) |
| 开源或开放报告模型 | [Open / Research Systems](#open--research-systems) |
| 视频后配音强基线 | [Video-To-Audio / Foley Systems](#video-to-audio--foley-systems) |
| 对 Wan 继续演进的启发 | [Implications For Wan-Based Work](#implications-for-wan-based-work) |
| 来源链接 | [Sources](#sources) |

## Executive Summary

音视频联合生成现在分成三条清晰路线：

1. **Native joint audio-video generation**
   一个模型或紧耦合模型族在同一生成过程中输出 video + speech / sound effects / ambience / music。代表：Sora 2、Veo 3、Seedance 1.5 pro / 2.0、LTX-2、Apollo/Klear、ALIVE。

2. **Video backbone + audio branch adaptation**
   复用强视频基座（例如 Wan / T2V MMDiT），加 audio branch、cross-modal attention、temporal-aligned RoPE、audio codec/VAE，再通过继续预训练和偏好优化把视频模型改造成 T2VA/I2VA。代表：ALIVE、JavisDiT++、UniVerse-1、部分 Wan-based 社区路线。

3. **Video-to-audio / Foley 后处理模型**
   输入已有视频和文本，生成同步音效/环境音/音乐/语音。它们非常实用，也常被接到视频生成 pipeline 后面，但不等同于原生 audio-video joint base。代表：Movie Gen Audio、HunyuanVideo-Foley、Kling-Foley、DreamFoley、ControlFoley。

成熟度判断：

- **闭源商业最成熟:** Sora 2、Veo 3/3.1、Seedance 2.0、Kling-Omni。它们通常有产品级 audio sync、dialogue、ambient、foley、multi-shot，但训练细节披露较少。
- **开源可落地最成熟:** LTX-2 / LTX-2.3。它明确是 open-source/open-weights audio-video foundation model，paper 公开架构：14B video stream + 5B audio stream + bidirectional cross-attention。
- **研究路线最有参考价值:** Apollo/Klear、ALIVE、UniVerse-1、JavisDiT++。它们给出可借鉴的架构模块和训练策略，尤其适合从 Wan 这类 T2V 基座继续扩展。
- **Foley 线不可忽视:** HunyuanVideo-Foley、Kling-Foley 在 TV2A 上更成熟，适合作为分阶段系统中的 audio module，也能为 joint model 训练提供 teacher / preference / data engine。

## Taxonomy

### A. 原生联合生成 Native A/V

**目标:** 输入 text / image / video / audio reference，直接输出同步的视频与音轨。

**典型能力:**

- dialogue + lip sync
- ambient soundscape
- action-timed foley
- background music
- multi-speaker / multilingual speech
- reference voice / identity / style

**常见架构:**

- dual-stream DiT：video stream + audio stream + cross-modal joint module
- single-tower unified DiT：audio/video token 放到统一 attention 中
- MMDiT + audio-video branch：从已有 T2V 基座继续预训练
- modality-aware CFG / cross-modal AdaLN / temporally aligned RoPE

### B. 视频基座扩音频 Video Backbone Adaptation

**目标:** 不从零训练，而是在 Wan / LTX / Sora-like T2V 模型上加 audio latent branch。

**常见步骤:**

1. 选视频基座和 audio codec / audio VAE。
2. 对齐 audio token 时间轴与 video latent frames。
3. 加 cross-modal attention 或 joint self-attention。
4. 用 video-audio-caption 数据继续预训练。
5. 加 SFT / DPO / RLHF / reward model 优化同步、自然度、叙事一致性。

**优点:** 成本低，继承视频模型画质和 motion prior。
**风险:** audio branch 容易拖垮原视频质量；lip-sync、foley timing、speaker identity 需要强数据和偏好优化。

### C. Video-To-Audio / Foley 模块

**目标:** 已有视频 -> 生成同步音频。

**优点:** 工程简单，能给任意 silent video 补音效；适合和现有 Wan/Hunyuan/CogVideoX pipeline 拼接。
**缺点:** 不能反向约束视频生成；复杂 scene 中音画因果一致性不如原生 joint model。

## Closed / Commercial Systems

### Sora 2

**类型:** 闭源，OpenAI video-audio generation model。
**资料:** OpenAI research blog + system card，不是完整 architecture technical report。

公开信息：

- Sora 2 被描述为 video and audio generation model，支持 synchronized audio、speech、sound effects、background soundscapes。
- 支持多镜头指令、世界状态保持、real-world “characters” 注入，角色功能需要一次性 video/audio recording。
- 系统卡主要讲能力、安全、风险、部署限制，不披露训练架构和并行细节。

判断：

- 产品成熟度高，audio-video 一体化能力明确。
- 对实现参考有限，因为没有公开模型结构、训练数据、tokenizer、loss、并行栈。

### Veo 3 / Veo 3.1

**类型:** 闭源，Google DeepMind video model with native audio。
**资料:** Veo 3 tech report 偏安全和数据；产品材料和第三方测试强调 audio sync。

公开信息：

- Veo 3 report 主要描述大规模 image/video 数据、Gemini caption、多层安全过滤和 red-teaming。
- 产品侧普遍将 Veo 3 视为支持 dialogue、sound effects、ambient audio 的原生音视频生成模型。
- 训练架构、audio branch、tokenizer、loss 未公开。

判断：

- 闭源产品成熟度高，但 technical report 对复现价值有限。

### Seedance 1.5 pro

**类型:** 闭源商业模型，ByteDance Seed。
**资料:** arXiv technical report + 官方 paper page。

公开信息：

- 明确是 “Native Audio-Visual Joint Generation Foundation Model”。
- 架构是 dual-branch Diffusion Transformer，带 cross-modal joint module。
- 有 specialized multi-stage data pipeline。
- 后训练包含 SFT 和 RLHF，多维 reward models。
- 报告称推理加速框架带来 10x+ speedup。
- 强调 multilingual/dialect lip-sync、cinematic camera control、narrative coherence。

判断：

- 披露质量高，是当前最值得参考的闭源 joint A/V tech report 之一。
- 关键模块：dual branch + cross-modal joint + SFT/RLHF + acceleration。

### Seedance 2.0

**类型:** 闭源商业模型，Seedance 1.5 pro 后继。
**资料:** arXiv technical report。

公开信息：

- 被描述为 native multi-modal audio-video generation model。
- 采用 unified、large-scale architecture，支持 text / image / audio / video 四类输入。
- 支持 4-15s audio-video content generation，native 480p/720p。
- 平台侧支持多参考输入：最多 3 个 video clips、9 张 images、3 段 audio clips。
- 有 Fast variant 面向低延迟。

判断：

- 更偏多模态编辑/引用/生产工作流。
- 如果目标是“多输入 reference + A/V 生成”，Seedance 2.0 是重要参考。

### Kling-Omni / Kling-O1

**类型:** 闭源/商业，Kuaishou Kling 系列。
**资料:** arXiv technical report。

公开信息：

- Kling-Omni 是 generalist generative framework，面向 multimodal visual language inputs。
- 统一生成、编辑、智能推理任务，强调 in-context generation、reasoning-based editing、multimodal instruction following。
- 技术报告摘要未明确展开 audio branch 细节，但外部产品资料将 Kling Omni/Kling 3.0 Omni 描述为支持 native audio output。

判断：

- 更像“统一 multimodal video creation framework”，音频能力公开细节不如 Seedance 1.5/2.0、LTX-2 具体。
- 对多模态输入、编辑、推理式生成很有参考价值。

### Movie Gen Audio

**类型:** Meta 闭源研究系统，视频生成 + 音频生成 suite。
**资料:** Movie Gen technical report。

公开信息：

- Movie Gen 是一组 media foundation models：video generation、personalization、editing、video-to-audio、text-to-audio。
- 最大 video model 是 30B transformer，73K video tokens；audio 部分用于给视频生成匹配音频。

判断：

- 更接近“视频基座 + audio generation module”的强系统，而不是一个单体 native A/V joint base。
- 对 production pipeline 很有价值：先生成/编辑视频，再做 video-to-audio。

## Open / Research Systems

### LTX-2 / LTX-2.3

**类型:** 开源 / open weights，Lightricks。
**资料:** arXiv technical report + LTX open-source docs。

公开信息：

- LTX-2 是 open-source foundational model，统一生成 synchronized audiovisual content。
- 架构：asymmetric dual-stream transformer，14B video stream + 5B audio stream。
- 通过 bidirectional audio-video cross-attention、temporal positional embeddings、cross-modality AdaLN 共享 timestep conditioning。
- 引入 modality-aware CFG 改善 A/V alignment 和 controllability。
- LTX-2.3 docs 描述其为 DiT-based audio-video foundation model，open weights，可本地执行，支持约 20s synchronized audio/video。

判断：

- 目前最值得优先研究的开源 audio-video foundation model。
- 对 Wan 扩音频的启发：非对称双流、cross-modal AdaLN、modality-CFG、audio stream 参数小于 video stream。

### Apollo / Klear

**类型:** 开放 technical report，Kling Team。
**资料:** arXiv 2601.04151；部分页面称 Klear，arXiv 标题为 Apollo。

公开信息：

- 目标是 unified multi-task audio-video joint generation。
- 采用 single-tower design、unified DiT blocks、Omni-Full Attention。
- 训练策略：progressive multitask，从 random modality masking 到 joint optimization，多阶段 curriculum。
- 数据：大规模 audio-video dense-caption dataset，自动 annotation/filtering，强调严格对齐的 audio-video-caption triplets。
- 报告称在 joint 和 unimodal settings 都能生成，并接近 Veo 3。

判断：

- 与 LTX-2 的 dual-stream 路线不同，Apollo/Klear 是 single-tower 统一建模路线。
- 对想做“一个模型统一 T2AV、T2V、T2A、V2A、A2V”的方案很有参考价值。

### ALIVE

**类型:** 开放 technical report / project page，面向从 T2V 基座适配到 A/V。
**资料:** arXiv 2602.08682 + project page。

公开信息：

- 目标：把 pretrained T2V model 适配成 Sora-style audio-video generation and animation。
- 支持 T2VA、I2VA、S2VA、A2VA、V2VA、VACE 等多种任务。
- 核心是 audio-video diffusion transformer，配合 synchronized A/V 数据构建。
- 项目页强调可生成语音、音乐、环境音、音效和对应视频。

判断：

- 最贴近“从 Wan 这种 T2V 基座继续扩成 A/V”的研究路线。
- 值得重点看它如何对齐 audio latent 时间轴、如何做多任务 masking 和 condition dropout。

### UniVerse-1

**类型:** 开放 research report。
**资料:** arXiv 2603.13775。

公开信息：

- 目标是 unified audio-video generation via conditional diffusion transformers。
- 框架由 shared foundation + task-specific lightweight experts 组成。
- 覆盖 T2V、T2A、T2AV、V2A、A2V、I2AV。
- 称其参数量和 compute 比单独训练多模型少约 30%，并在 T2AV 质量上超过若干开源模型。

判断：

- 适合多任务产品线：不同任务共享 backbone，再用轻量 expert 解决任务差异。
- 对工程落地比 pure single-tower 更稳，便于逐步上线。

### JavisDiT++

**类型:** 开放 technical report / arXiv，Wan-based 社区路线。
**资料:** arXiv 2508.09739。

公开信息：

- JavisDiT++ 是 Audio-Video-Text joint generation model。
- 基于 Wan 2.1 继续训练，用约 240k A/V paired videos。
- 使用 audio-visual fusion module，把 heterogeneous features unified 后接入 existing T2V model。
- 目标是在保持 pretrained visual generation capability 的同时获得 audio-video synchronization。

判断：

- 对当前 Wan Megatron port 最直接：它说明了“不要从零做 joint A/V，可以在 Wan 上加融合模块继续训”。
- 风险是数据规模、audio codec 和 fusion module 细节会决定上限。

### UniAVGen

**类型:** 开放 research report。
**资料:** arXiv 2605.06119。

公开信息：

- 目标是 joint audio-video generation，支持五类任务：video-to-audio、text-to-audio、text-to-video、text-to-audio-video、audio-to-video。
- 采用 unified Diffusion Transformer，带 audio-video interaction layer。
- 提出 cross-modal rotary position embedding，对齐连续音频和视频 token。
- 训练数据包含约 3M high-quality audio-video clips。

判断：

- 技术上适合参考 cross-modal RoPE 和 audio-video interaction layer。
- 更像研究原型，成熟度低于 LTX-2，但设计点很有价值。

### ProAV-DiT

**类型:** 开放 research report。
**资料:** arXiv 2512.06195。

公开信息：

- 目标是 efficient audio-visual generation。
- 把 video 和 audio 放入 independent projected latent spaces。
- 用 projected DiT 建模音视频 latent，强调比 existing joint models 更低算力、更快 inference。

判断：

- 值得关注的是“projected latent spaces”降低联合建模成本的思路。
- 如果目标是把 Wan 扩音频但不想显著增加 token/attention 成本，这条路线值得做 ablation。

## Video-To-Audio / Foley Systems

### HunyuanVideo-Foley

**类型:** 开源/开放权重 TV2A 模型，Tencent。
**资料:** arXiv + GitHub + Hugging Face。

公开信息：

- 输入 silent video + text prompt，生成 high-fidelity synchronized audio。
- 采用 multimodal diffusion transformer。
- 提供 8B 参数模型，支持 up to 10s / 48kHz audio。
- 数据 pipeline 包含 audio-visual alignment、captioning、quality filtering。

判断：

- TV2A 线的强基线，适合接到 Wan/HunyuanVideo/CogVideoX 后面。
- 也适合作为 joint model 的 teacher 或 preference data generator。

### Kling-Foley

**类型:** 开放 research / project，Kling Team。
**资料:** project page + arXiv。

公开信息：

- 文本指导视频音效生成，强调 high-quality、semantically aligned、temporally synchronized audio。
- 采用 multimodal diffusion transformer。
- 有专门的数据 pipeline 和 benchmark。

判断：

- 和 HunyuanVideo-Foley 一样是 V2A/Foley 强模型，不是完整 native A/V joint base。
- 对数据 pipeline、同步评估和 sound event alignment 很有参考价值。

### MMAudio / DreamFoley / ControlFoley

**类型:** 研究型 video-to-audio / text-video-to-audio 模型。
**资料:** arXiv + project pages。

公开信息：

- MMAudio 聚焦从 video + text 生成 synchronized audio。
- DreamFoley / ControlFoley 强调 controllable foley、sound event timing、semantic alignment。

判断：

- 它们不是“基座级”联合生成，但在数据、评价指标、同步控制和 foley 任务定义上有价值。

## Comparison Matrix

| 方案 | 开放性 | 是否原生 A/V | 主要架构 | 技术报告价值 | 适合借鉴点 |
| --- | --- | --- | --- | --- | --- |
| Sora 2 | 闭源 | 是 | 未公开 | 中低 | 产品级 synchronized speech / SFX / ambience 目标形态 |
| Veo 3 / 3.1 | 闭源 | 是 | 未公开 | 中低 | 产品级 dialogue + foley + safety pipeline |
| Seedance 1.5 pro | 闭源 | 是 | dual-branch DiT + cross-modal joint module | 高 | dual branch、SFT/RLHF、reward models、推理加速 |
| Seedance 2.0 | 闭源 | 是 | unified large-scale multimodal architecture | 中高 | 多输入 reference、多任务、多分辨率产品能力 |
| Kling-Omni | 闭源/报告开放 | 部分公开 | generalist multimodal generation framework | 中 | in-context generation、reasoning-based editing |
| Movie Gen Audio | 闭源研究 | 更偏后处理 | video model + audio model suite | 中高 | 生产 pipeline、video-to-audio、large video transformer |
| LTX-2 / 2.3 | 开源/开放权重 | 是 | 14B video + 5B audio dual-stream transformer | 高 | 最值得实操研究的 open A/V foundation |
| Apollo / Klear | 报告开放 | 是 | single-tower DiT + Omni-Full Attention | 高 | unified multitask、random modality masking |
| ALIVE | 报告开放 | 是 | T2V backbone adaptation + A/V diffusion transformer | 高 | 从视频基座扩音频 |
| UniVerse-1 | 报告开放 | 是 | shared foundation + task-specific lightweight experts | 中高 | 多任务共享 backbone + expert |
| JavisDiT++ | 报告开放 | 是 | Wan 2.1 + A/V fusion module | 高 | Wan-based joint A/V 继续训练路线 |
| UniAVGen | 报告开放 | 是 | unified DiT + A/V interaction layer + cross-modal RoPE | 中 | token 时间对齐、跨模态 RoPE |
| ProAV-DiT | 报告开放 | 是 | projected latent spaces + DiT | 中 | 降低 joint A/V token 成本 |
| HunyuanVideo-Foley | 开源/开放权重 | 否，TV2A | multimodal diffusion transformer | 高 | TV2A 强基线、teacher/data engine |
| Kling-Foley | 报告开放 | 否，TV2A | multimodal diffusion transformer | 高 | Foley 数据 pipeline 和同步评估 |

## Implications For Wan-Based Work

### 如果目标是短期可用

最稳路线是 **Wan T2V + HunyuanVideo-Foley / Kling-Foley-style TV2A**：

1. Wan 生成 silent video。
2. 抽视频语义、动作、事件、caption。
3. TV2A 生成音效/环境音/音乐。
4. 对口型/语音类任务另接 speech/lip-sync 模块。

优点：实现快、风险低、能快速拿到可看 demo。
缺点：视频生成时不知道未来音频，复杂 causality 和 dialogue consistency 受限。

### 如果目标是中期研究可发表

推荐 **Wan + audio branch + cross-modal joint module**，参考 Seedance 1.5 / LTX-2 / JavisDiT++：

- 保留 Wan video DiT 主干。
- 增加 audio latent codec/VAE。
- 加 audio stream，规模可以小于 video stream。
- 在若干 block 加 bidirectional cross-attention 或 audio-video interaction layer。
- 对齐 video latent frames 和 audio latent windows。
- 训练任务覆盖 T2V、T2A、T2AV、V2A、A2V，使用 modality dropout。
- 用真实 synchronized A/V 数据做 SFT，再用 reward/preference 优化 audio-video alignment。

这是最接近“从当前 Wan Megatron port 继续演进”的路线。

### 如果目标是长期基座

推荐比较三条路线：

| 路线 | 代表 | 优点 | 风险 |
| --- | --- | --- | --- |
| Dual-stream | Seedance 1.5、LTX-2 | 模态参数可独立扩展，音视频能力都强 | cross-modal module 设计复杂 |
| Single-tower | Apollo/Klear、UniAVGen | 任务统一，天然支持多模态 masking | token 成本高，训练稳定性难 |
| Shared backbone + experts | UniVerse-1 | 多任务工程友好，逐步扩展 | expert 设计和路由会影响泛化 |

### 对分布式训练的额外要求

音视频联合训练比纯视频更需要：

- **跨模态 token bucket:** 按 video tokens + audio tokens 的总 budget 分桶。
- **audio/video 时间轴对齐:** dataloader 需要保存精确 fps、sample rate、codec hop、latent frame mapping。
- **SP/CP 更重要:** joint attention token 更长，单纯 TP/FSDP 不够。
- **modality dropout RNG 一致性:** TP rank 内必须一致，否则同一样本不同 rank 条件不同。
- **多 loss / 多 head checkpoint:** video loss、audio loss、sync loss、preference/reward state 都要进入 DCP。
- **评估不只看 FID/FVD:** 还要看 audio quality、AV sync、lip sync、semantic event alignment、music/ambience naturalness。

## Recommended Next Reading Order

1. **LTX-2 / LTX-2.3**：先看最完整的 open A/V foundation model 架构。
2. **Seedance 1.5 pro / 2.0**：看闭源头部如何组织 dual-branch、数据、SFT/RLHF 和产品能力。
3. **JavisDiT++**：直接看 Wan-based audio-video fusion 路线。
4. **ALIVE / UniVerse-1 / Apollo-Klear**：比较 single-tower、backbone-adaptation、shared-expert 三类路线。
5. **HunyuanVideo-Foley / Kling-Foley**：作为短期 TV2A baseline 和 joint model data/teacher。

## Sources

- OpenAI Sora 2: [Sora 2 research blog](https://openai.com/research/sora-2/), [Sora 2 System Card](https://cdn.openai.com/sora/2/System_Card.pdf)
- Google Veo 3: [Veo 3 Tech Report PDF](https://storage.googleapis.com/deepmind-media/veo/Veo-3-Tech-Report.pdf)
- Seedance 1.5 pro: [arXiv 2601.01615](https://arxiv.org/abs/2601.01615), [ByteDance paper page](https://seed.bytedance.com/en/public_papers/seedance-1-5-pro-native-audio-visual-joint-generation-foundation-model)
- Seedance 2.0: [arXiv 2605.07009](https://arxiv.org/abs/2605.07009), [ByteDance paper page](https://seed.bytedance.com/en/public_papers/seedance-2-0-exploring-the-boundaries-of-unified-audio-video-generation), [Seedance 2.0 product page](https://seed.bytedance.com/en/seedance)
- Kling-Omni / Kling-O1: [arXiv 2512.14814](https://arxiv.org/abs/2512.14814), [Kling official site](https://klingai.com/)
- Movie Gen: [arXiv 2410.13720](https://arxiv.org/abs/2410.13720), [Movie Gen paper page](https://ai.meta.com/research/publications/movie-gen-a-cast-of-media-foundation-models/)
- LTX-2: [arXiv 2601.12262](https://arxiv.org/abs/2601.12262), [LTX-2 docs](https://docs.ltx.video/docs/video-generation), [GitHub docs](https://github.com/Lightricks/LTX-Video)
- Apollo / Klear: [arXiv 2601.04151](https://arxiv.org/abs/2601.04151), [Hugging Face paper page](https://huggingface.co/papers/2601.04151)
- ALIVE: [arXiv 2602.08682](https://arxiv.org/abs/2602.08682), [project page](https://humanaigc.github.io/alive)
- UniVerse-1: [arXiv 2603.13775](https://arxiv.org/abs/2603.13775)
- JavisDiT++: [arXiv 2508.09739](https://arxiv.org/abs/2508.09739)
- UniAVGen: [arXiv 2605.06119](https://arxiv.org/abs/2605.06119)
- ProAV-DiT: [arXiv 2512.06195](https://arxiv.org/abs/2512.06195)
- HunyuanVideo-Foley: [arXiv 2508.16930](https://arxiv.org/abs/2508.16930), [GitHub](https://github.com/Tencent-Hunyuan/HunyuanVideo-Foley), [Hugging Face](https://huggingface.co/tencent/HunyuanVideo-Foley)
- Kling-Foley: [project page](https://klingfoley.github.io/), [arXiv 2506.08299](https://arxiv.org/abs/2506.08299)
- MMAudio: [arXiv 2412.15322](https://arxiv.org/abs/2412.15322), [project page](https://hkchengrex.github.io/MMAudio/)
- DreamFoley: [arXiv 2502.15482](https://arxiv.org/abs/2502.15482)
- ControlFoley: [arXiv 2501.08222](https://arxiv.org/abs/2501.08222)
