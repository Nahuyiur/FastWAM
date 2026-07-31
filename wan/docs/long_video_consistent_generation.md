# 长视频与一致性生成方案调研

> **Status:** 调研记录，2026-05-19
> **Scope:** 长视频、multi-shot、角色/物体/场景一致、世界状态保持、长程 motion coherence。覆盖闭源商业模型、开源模型、训练范式、推理扩展方法和创作工作流。
> **结论一句话:** 当前“长 video 且一致”的成熟方案不止一种：闭源产品靠 scaling + reference/ingredient workflow 提供跨镜头一致性；开源可跑路线以 SkyReels-V2、FramePack、LongVie/LongVie2 最有影响；研究路线集中在 long-context tuning、memory bank、context compression、diffusion forcing、chunk autoregression；实际生产仍大量依赖 keyframe/storyboard/reference image 工作流。

## Reading Path

| 想了解 | 读 |
| --- | --- |
| 30 秒结论 | [Executive Summary](#executive-summary) |
| 长视频一致性到底指什么 | [Consistency Dimensions](#consistency-dimensions) |
| 技术路线分类 | [Taxonomy](#taxonomy) |
| 头部闭源/商业方案 | [Closed / Commercial Systems](#closed--commercial-systems) |
| 开源/开放报告方案 | [Open / Research Systems](#open--research-systems) |
| 对 Wan/Megatron 的启发 | [Implications For Wan-Based Work](#implications-for-wan-based-work) |
| 来源链接 | [Sources](#sources) |

## Executive Summary

现在视频生成的“长”至少有三种含义：

1. **单镜头长视频:** 同一个 shot 连续 20s、60s、甚至数分钟，要求 motion 和场景不崩。
2. **多镜头叙事:** 多个 shot 之间角色、服装、物体、场景、光照、风格一致。
3. **世界状态一致:** 物体离开画面后再次出现仍一致，动作造成的状态变化能持久存在，角色关系和空间布局不漂移。

当前比较成熟的方案可以分成四层：

- **闭源产品级:** Sora / Sora 2、Veo 3.1、Runway Gen-4、Seedance 1.0/2.0、Kling-Omni。这些模型的产品能力强，但训练细节大多不公开。
- **开源长视频模型:** SkyReels-V2、FramePack、LongVie/LongVie2、部分 LTX/Wan/SkyReels 社区路线。它们更适合工程复现和拆解。
- **研究范式:** Long Context Tuning、StoryMem、FilmWeaver、LoViC、Mixture of Contexts、StreamingT2V、FreeLong/FreeNoise、VideoTetris。
- **生产工作流:** reference image / character sheet / keyframe storyboard / I2V per-shot animation / edit stitching。直到 2026，很多长片段仍是多短片段“编排”出来，不是一个 prompt 一次生成完整长片。

一个现实判断：

- **最像产品可用:** Runway Gen-4 / Veo 3.1 / Sora 2 / Seedance 2.0。
- **最适合本地研究复现:** SkyReels-V2、FramePack、LongVie2、StoryMem。
- **最适合改造 Wan:** SkyReels-V2 的 diffusion forcing、FramePack 的固定长度 context packing、StoryMem 的 visual memory、Long Context Tuning 的 scene-level full attention、LongVie2 的 history-context guidance。

## Consistency Dimensions

长视频一致性不是一个指标，而是一组约束：

| 维度 | 要求 | 常见失败 |
| --- | --- | --- |
| 角色身份 | 同一角色跨角度、镜头、光照仍保持脸、体型、服装 | 变脸、发型变化、服装漂移 |
| 物体一致 | 道具、产品、车辆、logo、纹理保持 | 物体形状/颜色变化，细节丢失 |
| 场景布局 | 房间、街道、地形、镜头空间关系稳定 | 空间翻转、门窗消失、背景重排 |
| 风格光照 | 色调、镜头、质感、grade 一致 | 每个 shot 像不同模型生成 |
| 动作连贯 | 动作速度、方向、接触关系连续 | 运动跳变、肢体变形、物理不成立 |
| 世界状态 | 前一镜头发生的变化在后续保持 | 破坏/移动/吃掉的物体复原 |
| 叙事连贯 | 镜头顺序符合故事和 cinematic grammar | cut 不连贯，角色位置关系混乱 |
| 长程记忆 | 多镜头后仍能找回同一实体 | recurrence distance 越长 identity 越差 |

## Taxonomy

### 1. Scale Native Long-Context Video Transformer

**代表:** Sora, Sora 2, Veo, Seedance 2.0。

核心思想是直接把长时长、多分辨率、多宽高比视频训练进大模型，让模型在 scale 下涌现 object permanence、3D consistency、long-range coherence。

优点：

- 上限最高。
- 能原生处理 variable duration / resolution / aspect。
- 世界状态和物理一致性可能从数据和规模中涌现。

风险：

- 成本极高。
- 训练实现和数据策略通常闭源。
- 仍会出现长时长 incoherence 和 spontaneous object appearance。

### 2. Reference / Ingredient Consistency Workflow

**代表:** Runway Gen-4, Veo 3.1 Ingredients to Video。

核心思想是用 reference images / ingredient images 锚定角色、物体、背景、风格，再让视频模型在不同镜头里复用这些视觉实体。

优点：

- 产品上最直接可用。
- 对广告、产品、短片、多镜头 storyboard 非常实用。
- 不需要用户训练 LoRA。

风险：

- 真正的世界状态推理有限。
- 跨很多镜头后仍可能 drift。
- 往往是 workflow 能力，不等于模型一次生成完整长视频。

### 3. Autoregressive Chunking With Memory

**代表:** FramePack, SkyReels-V2, StreamingT2V, StoryMem, FilmWeaver, ShotStream。

核心思想是把长视频拆成多个 chunk 或 shot，生成下一段时显式带上前文记忆：

- previous frames / keyframes
- compressed context
- global memory bank
- local temporal cache
- shot-level memory
- initial appearance reference

优点：

- 可扩展到分钟级。
- 能复用已有短视频模型。
- 工程上最适合落地和迭代。

风险：

- 误差会随时间累积。
- 对多角色、多物体、遮挡后再出现仍难。
- chunk boundary 需要专门处理，否则会抖动或跳变。

### 4. Long-Context Tuning / Scene-Level Attention

**代表:** Long Context Tuning, LoViC, Mixture of Contexts。

核心思想是把单 shot 模型扩成 scene-level model，让 attention 或 compressed context 能跨多个 shot 建模。

优点：

- 直接学习 scene-level consistency。
- 比纯 workflow 更接近模型能力。
- 可以支持 multi-shot extension 和 compositional generation。

风险：

- 长 context 训练贵。
- SP/CP、sparse attention、context compression 都要同时做。
- 数据需要 scene-level captions 和 shot boundary 标注。

### 5. Training-Free Long Video Extension

**代表:** FreeNoise, FreeLong, FreeLong++, Ouroboros-Diffusion。

核心思想是不重新训练模型，通过 noise rescheduling、sliding-window temporal attention、frequency blending、look-back regularization 等推理技巧扩展短视频模型。

优点：

- 成本低。
- 可快速套到已有模型上。
- 对 16f -> 128f 这类扩展有效。

风险：

- 上限受限，通常不解决真正的叙事一致性。
- 对复杂角色和场景长期记忆较弱。
- 更像增强 sampler，不是新基座。

### 6. Storyboard / Keyframe / I2V Production Workflow

**代表:** 现实影视/广告生成工作流；Runway/Veo/Seedance/Kling 的 reference workflows。

核心思想是先生成角色表、场景表、关键帧和镜头脚本，再逐 shot 做 I2V/extension，最后剪辑。

优点：

- 当前最可靠。
- 可人工修正每个 shot。
- 适合商业交付。

风险：

- 工作流重，不是端到端。
- 自动化程度低。
- 需要很多人工筛选和后期。

## Closed / Commercial Systems

### Sora / Sora 2

**成熟度:** 产品级闭源；Sora 1 technical report 有较多高层信息，Sora 2 release 主要讲能力。
**长视频/一致性要点:**

- Sora 1 report 称最大模型可生成一分钟高保真视频。
- 采用 visual spacetime patches，训练在可变时长、分辨率、宽高比的视频和图像上。
- report 明确提到 object permanence、3D consistency、long-range coherence、多镜头同角色外观保持。
- Sora 2 强调更好的 physical accuracy、controllability、world state persistence 和 multi-shot instruction following。

**限制:**

- Sora 1 报告明确不包含模型和实现细节。
- Sora 2 也没有公开训练架构、tokenizer、long-context 并行栈。
- 仍承认模型会犯错，只是相对 prior systems 更稳定。

**判断:** 最重要的闭源 scaling baseline。对工程复现的直接信息有限，但它定义了“长视频世界模拟”的目标形态。

### Veo / Veo 3.1

**成熟度:** 产品级闭源。
**长视频/一致性要点:**

- Google 早期 Veo 宣称可生成超过一分钟的 1080p 视频。
- Veo 3.1 Ingredients to Video 强调 identity consistency、background/object consistency、reference image ingredients、1080p/4K upscaling。
- 产品侧强调多场景、vertical video、professional workflow。

**限制:**

- 训练架构和 long-video 机制不公开。
- 官方公开资料更多是 product feature，而不是完整 technical report。

**判断:** 在 reference-image-driven consistency workflow 上非常有代表性，适合学习“ingredient/reference as product interface”。

### Runway Gen-4

**成熟度:** 产品级闭源。
**长视频/一致性要点:**

- 官方称 Gen-4 是 “world consistency” 模型。
- 支持用单张 reference image 保持角色一致，跨 lighting、location、treatment 生成。
- 支持 consistent objects、locations、coverage shots、production-ready video。

**限制:**

- 主要是产品/研究公告，没有训练细节。
- 很多“长视频”仍是多个 shot/clip 组合，不一定是一次长 rollout。

**判断:** 当前商业创作中最重要的“跨镜头一致性 workflow”之一。

### Seedance 1.0 / 2.0

**成熟度:** 产品级闭源，但 technical report 信息相对多。
**长视频/一致性要点:**

- Seedance 1.0 report 提到 natively supporting multi-shot generation，并 joint learning T2V/I2V。
- 使用多源数据、精细 caption、efficient architecture、post-training、video-specific RLHF、多维 reward。
- Seedance 2.0 是 unified multi-modal audio-video generation architecture，强调世界复杂度、参考输入和高效推理。

**限制:**

- 具体模型结构、长程记忆、训练并行细节仍不完整。

**判断:** 对“multi-shot + reward/post-training + production quality”的启发价值很高。

### Kling / Kling-Omni

**成熟度:** 产品级闭源，部分报告开放。
**长视频/一致性要点:**

- Kling-Omni/Kling-O1 更偏 generalist video generation/editing/in-context framework。
- 产品侧强调多镜头、角色一致、编辑控制。

**限制:**

- audio/video/long-context 技术细节公开程度不如 Seedance、Runway、LCT/StoryMem 等。

**判断:** 对多模态指令、视频编辑、in-context generation 方向有参考意义，但作为长视频技术拆解资料不够 concrete。

## Open / Research Systems

### SkyReels-V2

**成熟度:** 开源/开放权重，影响力高。
**核心机制:** MLLM + multi-stage pretraining + RL + diffusion forcing。

要点：

- 目标是 infinite-length film generative model。
- 使用结构化视频表示：MLLM general description + sub-expert shot language。
- 用 SkyCaptioner-V1 标注视频数据。
- progressive-resolution pretraining + 四阶段 post-training。
- diffusion forcing framework 使用 non-decreasing noise schedules，支持长视频合成。
- GitHub / Hugging Face 有 DF 14B 540P/720P 等模型和 long video scripts。

判断：

- 当前最值得研究的开源长视频方案之一。
- 对 Wan 直接相关：社区里已有 SkyReels-V2 与 Wan/SkyReels diffusion forcing 结合路线。

### FramePack

**成熟度:** 开源，实践影响力高。
**核心机制:** next-frame / next-section prediction + frame context packing + anti-drifting sampling。

要点：

- 把输入帧 context 压缩到固定长度，使 transformer context 长度不随视频长度增长。
- 支持把已有 video diffusion model finetune 成 next-frame prediction。
- 通过 inverted temporal order 和 early-established endpoints 缓解 exposure bias / drift。
- 社区实践中常用于分钟级本地长视频生成。

判断：

- 对“长视频不爆 context”非常关键。
- 比单纯 sliding window 更系统，适合改造 Wan 这种 fixed-clip DiT。

### LongVie / LongVie2

**成熟度:** 开放报告/权重，研究影响力高。
**核心机制:** controllable ultra-long video world model，autoregressive generation + multimodal control + history-context guidance。

要点：

- LongVie 识别长视频失败的关键：separate noise initialization、independent control normalization、single-modality guidance limitation。
- 使用 unified noise initialization、global control signal normalization、dense+sparse multimodal control、degradation-aware training。
- LongVie2 进一步加入三阶段训练：multi-modal guidance、degradation-aware input-frame training、history-context guidance。
- 报告称支持连续生成最多 5 分钟，并提出 LongVGenBench。

判断：

- 最接近“视频世界模型”的开放路线。
- 对需要几分钟级连续 controllable video 的任务，比纯 multi-shot storytelling 更相关。

### Long Context Tuning, LCT

**成熟度:** ICCV 2025，研究影响力高。
**核心机制:** 把 single-shot diffusion model 的 attention context 扩到 scene-level。

要点：

- full attention 从单个 shot 扩到 scene 中所有 shots。
- 使用 interleaved 3D position embedding 和 asynchronous noise strategy。
- 可做 joint multi-shot generation，也可 fine-tune context-causal attention 后做 autoregressive generation with KV-cache。
- 目标是直接从数据学习 scene-level consistency。

判断：

- 对 Megatron/Wan 最重要的训练范式之一。
- 如果要让 Wan 原生支持 multi-shot，而不是外部 stitching，LCT 是必须重点复现/对照的路线。

### StoryMem

**成熟度:** 2025-12 开放报告，long-form storytelling 方向影响力高。
**核心机制:** explicit visual memory bank + Memory-to-Video。

要点：

- 将 long-form storytelling 改写为 iterative shot synthesis。
- 维护 compact memory bank，保存历史 shots 的 keyframes。
- 通过 latent concatenation 和 negative RoPE shifts 把 memory 注入单 shot diffusion model。
- 只需要 LoRA fine-tuning。
- 引入 semantic keyframe selection、aesthetic preference filtering、ST-Bench。

判断：

- 对“多镜头角色一致”特别直接。
- 比单纯长 context 更省，因为保留的是精选 memory，不是所有 tokens。

### FilmWeaver / OneStory / ShotStream

**成熟度:** 研究型，多镜头记忆方向。
**核心机制:** global/shot/local cache, autoregressive shot generation, interactive streaming。

要点：

- FilmWeaver 使用 dual-level cache：shot memory 保持 character/scene identity，temporal memory 保持当前 shot motion。
- OneStory 关注 coherent multi-shot generation，通过 global memory 维护长程上下文。
- ShotStream 关注 streaming interactive multi-shot generation，用 dual-cache memory 降低延迟并保持一致。

判断：

- 这组方法对产品化 storytelling 很重要。
- 技术共同点是“显式 memory + 逐 shot 生成”，比一次生成长视频更可控。

### LoViC / Mixture of Contexts

**成熟度:** 研究型，context compression / retrieval 方向。
**核心机制:** 将长视频 context 压缩或检索为可用记忆。

要点：

- LoViC 使用 FlexFormer，将 video/text 压缩成 unified latent representations，支持 prediction、retrodiction、interpolation、multi-shot generation。
- Mixture of Contexts 把 long-context video generation 看作内部信息检索，用 sparse attention routing 做 long-term memory retrieval。

判断：

- 适合解决“全量 context 太贵”的问题。
- 对 SP/CP 之外的 memory-efficient long context 建模很有价值。

### StreamingT2V

**成熟度:** 早期但影响力高。
**核心机制:** short-term memory + long-term appearance preservation + randomized blending。

要点：

- conditional attention module 保持 chunk transition。
- appearance preservation module 从第一个 chunk 提取高层 scene/object features，防止忘记初始场景。
- randomized blending 让 video enhancer autoregressive 应用时不产生明显不一致。

判断：

- 早期长视频 chunking 方案，思想仍然有效。
- 对 Wan 的简单 extension pipeline 有参考价值。

### FreeNoise / FreeLong / FreeLong++

**成熟度:** training-free sampler 系列，实用但上限有限。
**核心机制:** noise rescheduling、sliding temporal attention、SpectralBlend / SpectralFusion。

要点：

- FreeNoise 用 noise rescheduling 和 temporal attention over sliding windows 延长短视频模型。
- FreeLong 用 global low-frequency + local high-frequency blending 保持全局一致和局部细节。
- FreeLong++ 将该思路扩展到多频段，并在 Wan2.1 / LTX-Video 等模型上报告更好的 temporal consistency。

判断：

- 适合作为低成本 baseline。
- 不适合作为真正 long-video foundation 的唯一方案。

### VideoTetris

**成熟度:** 研究型，compositional long video 方向。
**核心机制:** progressive compositional prompts + reference frame attention。

要点：

- 面向 compositional text-to-video 和 progressive long video prompts。
- 引入 reference frame attention，提高 autoregressive video generation 一致性。
- 强调 position information、scene transition、sub-object composition。

判断：

- 对复杂 prompt、逐步引入新角色/物体的长视频有参考价值。

## Comparison Matrix

| 方案 | 开放性 | 长度/场景 | 一致性核心机制 | 成熟度判断 |
| --- | --- | --- | --- | --- |
| Sora / Sora 2 | 闭源 | Sora 1 报告称 up to 1 min；Sora 2 强调 multi-shot | scaling, spacetime patches, world simulation | 产品级目标形态，高层信息多 |
| Veo 3.1 | 闭源 | 产品侧支持 longer clips / multi-scene workflow | ingredient/reference consistency | 产品可用，训练细节少 |
| Runway Gen-4 | 闭源 | multi-shot narrative workflow | reference image, world consistency, character/object/location consistency | 商业创作强 |
| Seedance 1.0/2.0 | 闭源/报告开放 | native multi-shot / multi-modal A/V | data curation, post-training, RLHF, unified architecture | 头部闭源里披露较多 |
| SkyReels-V2 | 开源/开放权重 | long/infinite-length film generation | diffusion forcing, MLLM caption, progressive training, RL | 开源长视频重点方案 |
| FramePack | 开源 | minute-level practical generation | fixed-length context packing, anti-drift sampling | 本地实践影响力高 |
| LongVie2 | 开放报告/权重 | up to 5 min | multimodal control, degradation-aware training, history-context guidance | ultra-long controllable route |
| Long Context Tuning | 论文/项目 | scene-level multi-shot | scene-level full attention, interleaved 3D position, async noise | 训练范式重要 |
| StoryMem | 论文/项目 | minute-long storytelling | visual memory bank, latent concat, negative RoPE, LoRA | multi-shot consistency 强 |
| FilmWeaver / OneStory / ShotStream | 论文/项目 | arbitrary shots / streaming | global/shot/local caches | storytelling memory route |
| LoViC / MoC | 论文 | segment-wise long video | context compression / sparse retrieval | 长 context 效率路线 |
| StreamingT2V | 论文 | extendable long video | short-term memory + appearance preservation | 早期经典 chunking |
| FreeNoise / FreeLong | 论文/插件 | 16f -> 128f 等 | training-free noise/attention/frequency tricks | sampler baseline |
| VideoTetris | 论文/项目 | progressive compositional prompts | reference frame attention | compositional long video |

## What Is Actually Mature

### 最成熟的产品能力

- **Runway Gen-4:** 角色/物体/场景 reference 一致性产品化最好之一。
- **Veo 3.1:** Ingredients to Video 是强 reference workflow，强调 identity/background/object consistency。
- **Sora 2:** 世界状态、物理、multi-shot instruction following 的闭源标杆。
- **Seedance 2.0:** 多模态音视频、世界复杂度、reference 输入和商业生成能力很强。

### 最值得工程复现

- **SkyReels-V2:** 有开放模型和 diffusion forcing long video 路线。
- **FramePack:** context 长度固定化，极其实用。
- **LongVie2:** 几分钟级 controllable world model 的清晰训练 recipe。
- **StoryMem:** 最直接解决 multi-shot cross-shot identity consistency。

### 最适合做 Wan extension

1. **FramePack-style context packing:** 解决 context 长度随视频长度线性增长。
2. **SkyReels-style diffusion forcing:** 解决长视频 chunk 生成和非递减 noise schedule。
3. **StoryMem-style visual memory:** 解决跨 shot 角色/场景一致。
4. **LCT-style scene-level attention:** 从数据学习 shot 间一致性。
5. **LongVie-style history-context guidance:** 让相邻 clips 共享历史上下文，减少 drift。

## Implications For Wan-Based Work

### 短期路线：生产工作流

用 Wan 做短 clip / shot-level generation，再加：

- reference image / character sheet
- fixed scene prompt anchors
- I2V per shot
- keyframe storyboard
- shot transition smoothing
- VAE latent-space interpolation 或 video editing

这是最快拿到“看起来长且一致”的方法。

### 中期路线：Wan Long-Video Extension

在 Wan DiT 上加：

- chunk autoregressive generation
- context frame encoder
- first-frame / keyframe appearance memory
- latent memory bank
- cross-chunk noise schedule
- recompute + CP/SP long-context training
- DCP 支持长视频 dataset resume

目标：从 49f/81f clip 扩到 10s/20s/60s，并保持同一角色/场景。

### 长期路线：Multi-Shot Wan Foundation

训练数据和模型都要升级：

- 数据：scene-level multi-shot videos，带 shot boundary、角色 ID、scene ID、camera language、dense captions。
- 模型：scene-level attention 或 memory retrieval；支持 per-shot prompt + global story prompt。
- 训练：single-shot pretrain -> multi-shot long context tuning -> autoregressive/memory fine-tune -> preference/RL。
- 评估：不只 FVD/VBench，还要 cross-shot identity、object permanence、state persistence、story coherence。

### 推荐验收指标

| 指标 | 怎么测 |
| --- | --- |
| Identity consistency | face/person re-id, DINO/CLIP image similarity, human preference |
| Object consistency | object crop embedding similarity, detection/classification stability |
| Scene layout | depth/segmentation/keypoint stability, camera trajectory plausibility |
| Motion coherence | optical flow smoothness, motion direction continuity |
| Boundary smoothness | chunk boundary LPIPS/flow jump |
| State persistence | scripted events before/after comparison |
| Long-range recurrence | entity reappears after N shots, similarity vs first appearance |
| Story coherence | LMM/VLM judge + human scoring |

## Recommended Reading Order

1. **Sora 1 report + Sora 2 release:** 明确 scaling 目标和 world-state 需求。
2. **Runway Gen-4 + Veo 3.1 Ingredients:** 看产品级 reference consistency workflow。
3. **SkyReels-V2:** 看开源 long/infinite video generation 的完整工程路线。
4. **FramePack:** 看 fixed context compression 和 anti-drift sampling。
5. **Long Context Tuning:** 看 scene-level attention 如何训练。
6. **StoryMem / FilmWeaver / OneStory:** 看 memory bank 如何解决 multi-shot identity。
7. **LongVie2:** 看 controllable ultra-long world model 的训练 recipe。
8. **FreeNoise / FreeLong / StreamingT2V:** 作为 low-cost extension baseline。

## Sources

- OpenAI Sora: [Video generation models as world simulators](https://openai.com/index/video-generation-models-as-world-simulators/)
- OpenAI Sora 2: [Sora 2 is here](https://openai.com/index/sora-2/)
- Google Veo 3.1: [Ingredients to Video update](https://blog.google/innovation-and-ai/technology/ai/veo-3-1-ingredients-to-video/)
- Runway Gen-4: [AI Video Generation with World Consistency](https://runwayml.com/research/introducing-runway-gen-4)
- Seedance 1.0: [arXiv 2506.09113](https://arxiv.org/abs/2506.09113)
- Seedance 2.0: [arXiv 2604.14148](https://arxiv.org/abs/2604.14148)
- SkyReels-V2: [arXiv 2504.13074](https://arxiv.org/abs/2504.13074), [GitHub](https://github.com/SkyworkAI/SkyReels-V2), [Hugging Face DF 14B 720P](https://huggingface.co/Skywork/SkyReels-V2-DF-14B-720P)
- FramePack: [arXiv 2504.12626](https://arxiv.org/abs/2504.12626), [GitHub](https://github.com/lllyasviel/FramePack), [project page](https://lllyasviel.github.io/frame_pack_gitpage)
- LongVie: [arXiv 2508.03694](https://arxiv.org/abs/2508.03694)
- LongVie2: [arXiv 2512.13604](https://arxiv.org/abs/2512.13604), [Hugging Face](https://huggingface.co/Vchitect/LongVie2)
- Long Context Tuning: [ICCV 2025 page](https://openaccess.thecvf.com/content/ICCV2025/html/Guo_Long_Context_Tuning_for_Video_Generation_ICCV_2025_paper.html), [project page](https://guoyww.github.io/projects/long-context-video/), [arXiv 2503.10589](https://arxiv.org/abs/2503.10589)
- StoryMem: [arXiv 2512.19539](https://arxiv.org/abs/2512.19539)
- FilmWeaver: [arXiv 2512.11274](https://arxiv.org/abs/2512.11274)
- ShotStream: [arXiv 2603.25746](https://arxiv.org/abs/2603.25746)
- LoViC: [arXiv 2507.12952](https://arxiv.org/abs/2507.12952)
- Mixture of Contexts: [arXiv 2508.21058](https://arxiv.org/abs/2508.21058)
- StreamingT2V: [arXiv 2403.14773](https://arxiv.org/abs/2403.14773)
- FreeNoise: [arXiv 2310.15169](https://arxiv.org/abs/2310.15169)
- FreeLong: [NeurIPS 2024 page](https://proceedings.neurips.cc/paper_files/paper/2024/hash/ed67dff7cb96e7e86c4d91c0d5db49bb-Abstract-Conference.html), [arXiv 2407.19918](https://arxiv.org/abs/2407.19918)
- FreeLong++: [arXiv 2507.00162](https://arxiv.org/abs/2507.00162)
- VideoTetris: [project page](https://videotetris.github.io/)
- EntityBench: [arXiv 2605.15199](https://arxiv.org/abs/2605.15199)
- Survey: [A Survey on Long-Video Storytelling Generation](https://research.adobe.com/publication/a-survey-on-long-video-storytelling-generation-architectures-consistency-and-cinematic-quality/)
