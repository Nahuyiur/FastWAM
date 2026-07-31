# Video Generation 数据管线与清洗策略调研

> **Status:** 调研记录，2026-05-19
> **Scope:** text-to-video / image-to-video / video foundation model 的训练数据配方、清洗、标注、过滤、采样与课程训练数据组织。覆盖工业 tech report、开源报告、数据集论文和 NVIDIA/Google/Meta/OpenAI/ByteDance/Tencent/Alibaba/Stability AI 等可信来源。
> **结论一句话:** 现在高质量 video generation 的关键差异已经从“有没有大模型”扩展到“有没有工业级 video data factory”。头部报告普遍提到多阶段 data curation：shot splitting、motion/quality/aesthetic/OCR/watermark/filter、dense recaption、semantic dedup、concept balancing、duration/resolution bucketing、progressive high-quality SFT；学界也有 SVD、Panda-70M、VidGen-1M、OpenVid-1M、MiraData/LVD-2M 这类专门研究数据配方的 high-impact 工作。

## Reading Path

| 想了解 | 读 |
| --- | --- |
| 30 秒结论 | [Executive Summary](#executive-summary) |
| 工业级 pipeline 长什么样 | [Canonical Industrial Pipeline](#canonical-industrial-pipeline) |
| 哪些 paper 专门研究数据配方 | [Data-Centric Papers](#data-centric-papers) |
| 头部基座报告怎么写数据 | [Foundation Reports](#foundation-reports) |
| 清洗策略清单 | [Filtering Strategy](#filtering-strategy) |
| caption/recaption 怎么做 | [Captioning Strategy](#captioning-strategy) |
| 对 Wan 训练的落地建议 | [Implications For Wan](#implications-for-wan) |
| 来源链接 | [Sources](#sources) |

## Executive Summary

有。video generation 的数据配方和清洗策略已经有明确的论文研究，也在多个基座模型 tech report 里被当作核心能力披露。

当前可信来源可以分成三类：

1. **工业/基座报告明确写 data curation**
   - **Sora:** 强调 variable duration/resolution/aspect 原生训练，以及 DALL-E 3 式 video recaptioning。
   - **Stable Video Diffusion:** 系统研究 data curation 对 video LDM 的影响，包含 cut detection、captioning、CLIP/OCR/optical-flow/aesthetic filtering，并证明 curated pretraining dataset 很关键。
   - **Movie Gen:** 明确公开 pretraining data curation pipeline：visual filtering、motion filtering、content filtering、captioning；O(100)M video-text + O(1)B image-text。
   - **Open-Sora Plan:** 专门一节写 Data Curation Pipeline，包含 jump cut、motion、OCR cropping、aesthetic、technical quality、dense caption。
   - **HunyuanVideo / HunyuanVideo 1.5:** 把 data curation、captioning、progressive training 写成核心贡献。
   - **Wan:** 报告强调 large-scale data curation、scalable pretraining、automated evaluation，14B 训练在 billions images/videos。
   - **NVIDIA Cosmos / NeMo Curator:** 公开了可工程复用的 video curation pipeline，覆盖 100PB+ 级别处理、split、transcode、filter、embedding、caption、dedup。

2. **专门研究 video 数据集/清洗/标注的学界论文**
   - **Panda-70M:** 从 HD-VILA-100M 中抽取 3.8M 高分辨率长视频，split 成 70.8M 语义一致 clip，用多个 cross-modality teacher 生成 caption，再用人工小集训练 retrieval model 选择最佳 caption。
   - **VidGen-1M:** 明确指出低 caption 质量、低视频质量、temporal inconsistency、分布不均衡会限制 T2V；提出 coarse curation -> captioning -> fine curation。
   - **OpenVid-1M:** ICLR 2025，强调高质量 million-scale 数据和 expressive caption，比单纯大规模 WebVid/Panda 更有效。
   - **MiraData / LVD-2M:** 聚焦 long-duration、structured captions、temporal consistency，面向长视频生成。

3. **数据工具链/基础设施**
   - **NVIDIA NeMo Curator / Cosmos Curate:** 工业级 video data factory 的开源/文档化版本：GPU decode/encode、scene split、motion/aesthetic filter、embedding、Qwen-VL caption、semantic dedup、Ray/Xenna executor。

一句工程判断：

- **pretrain 阶段要大而杂，但必须经过基本质量/运动/去重/分桶。**
- **finetune/SFT 阶段要小而精，宁愿少也不要脏，caption 必须 dense 且时间语义正确。**
- **长视频/一致性不是靠模型结构单独解决，必须有 long-take / multi-shot / entity-consistent / camera-motion-rich 数据和 structured caption。**
- **音视频联合生成还要额外做 audio quality、A/V sync、speech/lip、diegetic/non-diegetic 分类与 caption。**

## Canonical Industrial Pipeline

下面是把 SVD、Movie Gen、Open-Sora Plan、HunyuanVideo、Cosmos/NeMo Curator、Panda-70M、VidGen-1M 的公开实践归纳后的工业级管线。

| 阶段 | 目标 | 典型做法 | 产物 |
| --- | --- | --- | --- |
| Source registry | 明确数据来源、授权、domain、风险 | 记录 source、license、region、language、privacy/safety risk | `source_manifest.jsonl` |
| Ingest & probe | 批量读取视频元信息 | `ffprobe` / GPU decoder / object storage manifest | duration, fps, size, codec, audio tracks |
| Transcode | 统一可训练格式 | H.264/H.265、固定 pixel format、可 seek、保留原始 aspect | normalized clips |
| Shot splitting | 去掉多场景拼接 | PySceneDetect、FFmpeg scene boundary、LPIPS/ImageBind/CLIP 相似度二次校正 | single-shot segments |
| Clip windowing | 控制训练长度 | 4s/8s/16s/32s bucket；长视频按窗口滑动或 scene-level chunk | clip-level records |
| Video quality filtering | 去掉低画质 | resolution、bitrate、blur、noise、DOVER/technical quality、aesthetic score | quality scores |
| Motion filtering | 保留有效动态 | optical flow、VMAF motion、motion vectors、static detector、jitter detector | motion scores |
| Artifact filtering | 去广告/水印/边框/字幕 | OCR、watermark/logo detector、border detector、grid/stitching detector、padding detector | artifact tags |
| Safety filtering | 去风险内容 | NSFW/violence/face/privacy/copyright/logos/policy classifiers | safety labels |
| Captioning | 建 text-video alignment | VLM/video captioner + LLM recaption + ASR/OCR/camera/action tags | dense captions |
| Text-video alignment | 剔除错配 caption | CLIP/UMT/VideoCLIP/InternVideo/Qwen-VL scoring，人工小集校准阈值 | alignment score |
| Dedup | 降低重复和 dominant concepts | perceptual hash、copy-detection embedding、CLIP/ImageBind/Cosmos-Embed clustering | dedup cluster id |
| Resampling | 控制数据分布 | concept balancing、human/action/camera/style buckets、domain mixture | sampling weights |
| Bucketing | 让训练 batch 高效 | resolution/aspect/duration/FPS latent-shape buckets | bucket id |
| Tokenization cache | 降低训练 IO/compute | video VAE latents、text embeddings、metadata cache、可选 audio latents | training shards |
| SFT/HQ set | 提升最终质感 | 人工或强模型筛选 high-aesthetic/high-motion/high-caption set | curated finetune set |
| Preference data | 做 DPO/RLHF/奖励模型 | human A/B、model-rank、multi-axis reward labels | preference pairs |

## Data-Centric Papers

### Stable Video Diffusion

**定位:** 第一批系统研究 video generation data curation 的高影响工作之一。

公开要点：

- 认为 video generation 领域长期低估了数据选择问题。
- 将训练拆成 text-to-image pretraining、video pretraining、high-quality video finetuning 三阶段。
- 对 WebVid 等视频数据做 cut detection、synthetic captioning、CLIP/OCR/optical-flow/aesthetic filtering。
- 通过 human preference / Elo-like 方式比较不同筛选子集对生成质量的影响。
- 结论是 curated pretraining dataset 的收益会延续到后续 high-quality finetune。

工程启发：

- 不要把“原始视频多”当作等价于“有效训练数据多”。
- filter threshold 不是纯启发式，应该用小模型训练 + 人评/自动评测回归选择。
- HQ finetune set 是必要阶段，不是可选项。

### Panda-70M

**定位:** CVPR 2024，video caption/data pipeline 代表作。

公开要点：

- 从 HD-VILA-100M 中筛出 3.8M 高分辨率长视频。
- 两阶段 semantics-aware splitting：先 shot boundary detection，再用 ImageBind frame embedding 合并误切/过短片段。
- 提出 Max Running LPIPS 衡量 clip 内语义一致性。
- 使用多个 cross-modality teacher：video VQA、image captioning、image VQA、subtitle/description 等。
- 人工标注 100K 小集，训练 fine-grained video-to-text retrieval model，从多个 caption 候选中选最佳 caption。
- 最终得到 70.8M video-caption pairs。

工程启发：

- 单一 captioner 对开放域视频不够，multi-teacher + retrieval selection 更稳。
- ASR/subtitle 不能直接当 caption，容易描述旁白而不是画面。
- “切得短”不一定最好，要在语义一致和足够运动/上下文之间平衡。

### VidGen-1M

**定位:** 直接研究“什么样的视频数据适合 T2V 训练”的数据集论文。

公开要点：

- 明确指出四类问题：低质量 captions、低质量 videos、temporal inconsistency、data imbalance。
- 使用 coarse curation -> captioning -> fine curation。
- coarse 阶段做 scene splitting、tagging、filtering、sampling。
- tagging 维度包括 aesthetic/OCR、temporal consistency、category、motion。
- 使用 VILA captioner 生成 descriptive synthetic captions，再用 LLM 做 fine curation，修正 scene transition 和 caption 错误。
- caption 平均长度约 89 words，强调 motion/action/camera 信息。

工程启发：

- 先 coarse filtering 缩小数据，再跑昂贵 captioner/LLM。
- caption 的动作、相机运动、时间顺序信息对视频生成非常关键。
- 需要显式处理类别分布，否则模型会偏向 dominant scenes。

### OpenVid-1M

**定位:** ICLR 2025，强调 million-scale high-quality 比盲目大规模更适合研究和训练。

公开要点：

- 目标是高质量、expressive caption、open scenario 的 text-video pairs。
- 约 1M 高质量 clips，并构建 433K 1080p HD subset。
- 关注 aesthetic、clarity、motion、temporal consistency、caption expressiveness。
- 通过模型和 ablation 展示数据质量对 T2V 生成效果的重要性。

工程启发：

- 大规模 pretrain 可以用大池子，但面向可复现实验和 SFT，million-scale 高质量数据更可控。
- HD subset 应独立建，不要简单从低分辨率训练集上采样。

### MiraData / LVD-2M / Long-Take Datasets

**定位:** 面向长视频生成和 long-range consistency 的数据工作。

公开要点：

- 强调长时长、高 motion intensity、structured captions、temporal consistency。
- LVD-2M 从多个大规模公开视频源中构建 long-take video-caption pairs。
- 这些工作通常不仅给 global caption，还给 hierarchical / progressive / segment captions。

工程启发：

- 如果目标是长视频和角色/场景一致性，普通 4s/8s single-shot 数据不够。
- 需要 long-take、multi-shot、entity recurrence、camera path、事件前后因果的标注。

## Foundation Reports

### Sora

公开信息：

- Sora 训练 text-conditional diffusion model，联合训练 variable duration、resolution、aspect ratio 的 images/videos。
- 使用 spacetime patches，把视觉数据统一成 patch sequence。
- 应用 DALL-E 3 的 recaptioning 思路：先训练 descriptive captioner，再对训练视频生成更详细 caption。
- 报告明确说不包含完整模型和实现细节。

对数据的启发：

- 不要强制所有视频 resize/crop 到单一正方形；native aspect 和 variable duration 能保留真实构图。
- caption 长度和细节要对齐训练目标，否则 prompt following 上限低。

### Movie Gen

公开信息：

- pretraining data 约 O(100)M video-text pairs + O(1)B image-text pairs。
- 原始视频 4 秒到 2 分钟，最终 clip 是 4-16 秒 single-shot 且有 non-trivial motion。
- 数据管线包含 visual filtering、motion filtering、content filtering、captioning。
- visual filtering 包含最低分辨率、aspect ratio mix、OCR 去过多文字、scene boundary detection、aesthetic/quality/border/visual effects。
- motion filtering 去掉 static、低 motion、jittery camera、slideshow/special motion。
- content filtering 做 perceptual dedup 和 concept resampling。
- 训练按 aspect ratio、duration、FPS 做 bucket，使同 bucket latent shape 一致。
- finetune 数据更严格，包含自动和人工过滤，并对 human/action 概念做平衡。

对数据的启发：

- final pretrain set 不是“所有可用视频”，而是从大池子经过三层过滤得到的小得多的 clip-prompt pairs。
- portrait/landscape 比例、human motion、camera motion 要显式控制。
- duration bucket 要和 latent frame 数绑定，否则 batch 效率差。

### Open-Sora Plan

公开信息：

- 数据管线包含 jump cut 检测、clip、fast/slow motion filtering、edge subtitle cropping、aesthetic score、technical quality、captioning。
- 使用 LPIPS 计算相邻帧变化，既做 jump cut，也作为 motion proxy。
- OCR cropping 不是简单丢弃所有含文字视频，而是裁边缘字幕；中心文字可能保留。
- aesthetic filtering 使用 LAION aesthetic predictor，公开阈值 4.75，约过滤 40%。
- technical quality 使用 DOVER technical score，过滤低 bitrate、压缩伪影、temporal jitter。
- captioning 用 InternVL2、Qwen2-VL、ShareGPT4Video 等，并清理 “This video/image” 这类模板前缀。

对数据的启发：

- OCR 不一定是 delete；字幕在边缘时可 crop，中心文字有时是场景语义。
- motion filter 要在 OCR/crop 后复查，避免字幕变化被误当运动。
- technical quality 和 aesthetic quality 是两个维度，不能只用一个分数。

### HunyuanVideo / HunyuanVideo 1.5

公开信息：

- 报告把 data curation、architecture、progressive scaling/training、infrastructure 作为完整框架。
- HunyuanVideo 1.5 公开摘要强调 meticulous data curation、progressive pretraining/post-training。
- 二级资料和报告摘要描述其使用多阶段视频过滤、caption model post-training、camera motion/caption tokens 等。

对数据的启发：

- 开源模型如果参数规模小于闭源模型，数据质量和 post-training 会更关键。
- bilingual / glyph-aware / camera-motion caption 这类标签会直接影响 prompt following 和 motion controllability。

### Wan

公开信息：

- Wan 报告强调 novel VAE、scalable pre-training、large-scale data curation、automated evaluation metrics。
- 14B 模型训练在包含 billions images and videos 的大规模数据上，并展示 data/model scaling law。
- 公开摘要没有给出完整数据清洗细节，但明确把 data curation 作为关键贡献。

对数据的启发：

- 对 Wan continuation/pretrain，最好不要只用公开小数据集做“功能验证”；要同时构建 HQ overfit/smoke、small curated SFT、large noisy pretrain 三层数据。
- 如果要复刻官方质量，需要补齐 dense captions、camera/motion tags、quality/motion filters、automated eval，而不是只迁移模型结构。

### NVIDIA Cosmos / NeMo Curator

公开信息：

- Cosmos 平台包含 video curation pipeline、world foundation models、tokenizers。
- NeMo Curator 公开工业级 video curation stages：load、clip、encode、filter、frame extraction、embedding、caption/preview、duplicate removal。
- filtering 包含 motion pass 和 aesthetic pass。
- embedding 可用 Cosmos-Embed1 / InternVideo2，caption 可用 Qwen-VL，dedup 用 semantic clustering / pairwise similarity / k-means。
- NVIDIA blog 描述 petabyte-scale pipeline：download、cutscene detection、clip extraction、transcoding、quality filters、captioning、embeddings。

对数据的启发：

- 对集群训练，数据管线要按 streaming/distributed executor 设计，不能靠单机脚本扫大目录。
- metadata schema 必须可增量更新：每个 stage 写自己的 score/tag，不要覆盖原始信息。

## Filtering Strategy

### Video Quality

建议保留分数，不要只保留 pass/fail：

| 维度 | 指标/模型 | 典型用途 |
| --- | --- | --- |
| resolution | width/height/min side | 去低清，做 bucket |
| bitrate/compression | ffprobe bitrate, VMAF, no-reference quality | 去糊、马赛克、压缩伪影 |
| blur/sharpness | Laplacian variance, DOVER technical | 去失焦 |
| exposure/noise | technical quality model | 去过曝、过暗、噪声 |
| aesthetic | LAION aesthetic, DOVER aesthetic | SFT/HQ set |
| frame stability | optical flow variance, jitter detector | 去抖动、伪运动 |

### Motion

不能只用 “motion 越大越好”：

- 太静：训练会学成 still image / Ken Burns。
- 太快：object identity、temporal consistency、caption alignment 会变差。
- 镜头抖动：高 motion 但低质量。
- 字幕/水印变化：会污染 optical flow/LPIPS motion score。

建议分桶：

| bucket | 用途 |
| --- | --- |
| static/low motion | 少量保留给 camera slow pan、I2V 微动 |
| normal motion | T2V 主训练 |
| high action | action/camera motion 专项 SFT |
| jitter/special transition | 默认过滤 |

### Shot / Temporal Consistency

常见策略：

- PySceneDetect / FFmpeg scene boundary 作为第一层。
- LPIPS、CLIP/ImageBind frame embedding 做二次校验。
- 对 long-take 数据保留 global caption + segment captions，而不是硬切成短片丢掉长程信息。
- 对 transition/fade/slideshow 单独标签，默认从主训练移除，可作为编辑/transition controller 数据。

### OCR / Watermark / Borders

策略要区分：

- **边缘字幕:** 可 crop 或过滤。
- **中心文字:** 可能是街牌、书本、广告牌、UI，不能全删；Wan/AnyText 类任务反而需要保留。
- **水印/logo:** 如果目标是通用创作，默认过滤；如果目标是品牌生成，需走授权数据。
- **黑边/白边/grid:** 默认过滤或 crop。

### Dedup And Diversity

需要两层去重：

1. **Exact / near duplicate:** hash、perceptual hash、copy-detection embedding。
2. **Semantic over-representation:** CLIP/ImageBind/Cosmos embeddings 聚类后 resampling，防止 dominant concepts 过多。

Movie Gen 类报告说明，content filtering 不只是去重，还包括 concept balancing。对 video model 来说，人像、动物、室内 vlog、游戏录屏、舞台表演、风景航拍等比例会强烈影响输出风格。

## Captioning Strategy

### Caption Schema

建议不要只有一个 `caption` 字段。工业可用 schema 至少包含：

```json
{
  "id": "source__video__clip",
  "source": "licensed_pool_A",
  "path": "clips/...",
  "duration_sec": 8.0,
  "fps": 24,
  "width": 1280,
  "height": 720,
  "global_caption": "...",
  "dense_caption": "...",
  "segment_captions": [
    {"start": 0.0, "end": 2.0, "caption": "..."},
    {"start": 2.0, "end": 4.0, "caption": "..."}
  ],
  "camera": {"motion": "slow pan left", "shot": "medium shot", "angle": "low angle"},
  "style": {"lighting": "soft natural light", "color": "warm", "medium": "live action"},
  "entities": [{"type": "person", "description": "..."}],
  "actions": ["walks", "turns", "opens a door"],
  "ocr": {"text": [], "score": 0.02},
  "quality": {"aesthetic": 6.1, "technical": 0.83, "motion": 0.42},
  "safety": {"nsfw": 0.01, "violence": 0.0},
  "bucket": {"aspect": "16:9", "duration": "8s", "resolution": "720p"}
}
```

### Multi-Teacher Captioning

Panda-70M 的核心经验是：单个 captioner 在开放域视频上不够稳。推荐做法：

- video caption model 描述时间变化和动作。
- image VLM 描述物体、风格、细节。
- ASR/OCR 提供画面文字和语音内容，但不能直接替代视觉 caption。
- LLM 整合多候选 caption，清理模板话术和 hallucination。
- 用人工小集训练 retrieval/ranker 选择最佳 caption。

### Dense Recaption

Sora/DALL-E 3、Open-Sora Plan、VidGen-1M 都支持一个结论：dense synthetic captions 能显著提升 prompt following。

但 dense caption 有风险：

- VLM hallucination 会教模型生成不存在的物体。
- 过长 caption 可能降低训练吞吐，并造成 text encoder 截断。
- caption 风格不一致会造成 prompt domain gap。

建议：

- 保留 raw captions、captioner outputs、final caption，不要覆盖。
- 对关键字段做结构化抽取：subject/action/camera/style/audio/OCR。
- 对 final training caption 做长度 bucket 和 prompt style randomization。

## Data Recipe By Training Stage

| 阶段 | 数据目标 | 数据质量 | 建议数据 |
| --- | --- | --- | --- |
| VAE/tokenizer | 重建质量和运动保真 | 高分辨率、低压缩、domain wide | raw image/video，caption 可无 |
| Low-res pretrain | 学概念、构图、基础 motion | 大规模，基本过滤即可 | image-text + video-text |
| Mid-res video pretrain | 学真实运动和 camera | 过滤 shot/motion/quality | 4-16s single-shot clips |
| High-res finetune | 提升质感和 prompt adherence | 严格 HQ + dense caption | aesthetic/motion top subset |
| I2V / FLF / continuation | 锚定条件和长程一致性 | motion-rich, start/end clear | frame-masked video clips |
| Long video tuning | 学 long-range consistency | long-take/multi-shot structured | global + segment captions |
| Preference alignment | 人类审美和可控性 | 小而精，多样 prompt | A/B pairs, rank data |

## Implications For Wan

如果要把 Wan 训练/续训做成接近官方基座训练路径，建议把数据工作拆成四层，而不是只做一个 overfit mp4：

### 1. Smoke / Overfit Layer

用途：验证 loader、VAE encode/decode、official ckpt load、Megatron DCP save/load、loss backward、TP/CP/recompute。

建议：

- 1 条真实短视频，保留原始 fps/resolution，并派生多个训练 bucket。
- 存储原始 mp4、标准化 mp4、latent cache、metadata json。
- caption 要人工写，包含动作、画面、相机、风格。

### 2. Small Curated SFT Layer

用途：验证模型能在几十到几千条 HQ 样本上 loss 下降，并改善指定 domain。

建议：

- 从 `/aifs4su` 中选 100-1000 条短视频。
- 做 scene split、OCR/watermark/motion/quality 过滤。
- 用 VLM 生成 caption，再人工 spot check 50 条。
- 训练固定 resolution/duration bucket，例如 480p 5s、720p 5s。

### 3. Medium Pretrain Layer

用途：验证 distributed training、data sharding、bucket sampling、checkpoint reshard。

建议：

- 10K-1M clips，按 source/domain 分布写 manifest。
- 自动生成 quality/motion/aesthetic/OCR/alignment scores。
- latent cache + text embedding cache 分离，支持 resume。
- 每个 shard 有 stats：duration/aspect/motion/caption length/source。

### 4. HQ Evaluation / Preference Layer

用途：验证续训后是否真正提升，而不是只记住数据。

建议：

- 固定 prompt suite：人物、动物、风景、室内、快速动作、相机运动、文字、长视频。
- 保存 base Wan vs finetuned Wan 的 paired outputs。
- 人评或 VLM-as-judge 只做辅助，核心要有 artifact audit：motion、identity、prompt adherence、temporal flicker。

## Minimal Industrial Checklist

对任何新视频训练数据进入 Wan/Megatron 训练前，至少要有这些字段：

- `source_id`, `license`, `original_path`, `clip_path`
- `duration_sec`, `fps`, `width`, `height`, `aspect_bucket`, `duration_bucket`
- `shot_boundary_score`, `temporal_consistency_score`
- `motion_score`, `jitter_score`
- `aesthetic_score`, `technical_quality_score`, `blur_score`
- `ocr_score`, `watermark_score`, `border_score`
- `dedup_cluster_id`, `semantic_cluster_id`
- `caption_raw`, `caption_dense`, `caption_final`
- `camera_motion`, `shot_type`, `style_tags`, `action_tags`
- `safety_tags`
- `latent_path`, `text_embedding_path`
- `sampling_weight`, `split`

## Sources

| Source | Why It Matters |
| --- | --- |
| [Sora: Video generation models as world simulators](https://openai.com/research/video-generation-models-as-world-simulators) | Variable duration/resolution/aspect, spacetime patches, video recaptioning. |
| [Stable Video Diffusion](https://arxiv.org/abs/2311.15127) | Systematic study of video data curation and staged video LDM training. |
| [Movie Gen](https://arxiv.org/abs/2410.13720) | Industrial data curation pipeline: visual/motion/content filtering, captioning, bucketing. |
| [Open-Sora Plan](https://arxiv.org/abs/2412.00131) | Open report with detailed filtering: LPIPS jump cut, OCR crop, aesthetic, DOVER, captioning. |
| [HunyuanVideo](https://arxiv.org/abs/2412.03603) | Open-source video foundation model report emphasizing curation/training/infrastructure. |
| [HunyuanVideo 1.5](https://arxiv.org/abs/2511.18870) | Lightweight high-performance model with meticulous data curation and post-training. |
| [Wan](https://arxiv.org/abs/2503.20314) | Open large-scale video model report; data/model scaling and curation are core claims. |
| [Cosmos](https://arxiv.org/abs/2501.03575) | NVIDIA world foundation model platform with video curation pipeline. |
| [NeMo Curator video docs](https://docs.nvidia.com/nemo/curator/curate-video) | Industrial video data factory stages: split, filter, embed, caption, dedup. |
| [Panda-70M](https://arxiv.org/abs/2402.19479) | Multi-teacher captioning and semantic clip splitting for 70M video-caption pairs. |
| [VidGen-1M](https://arxiv.org/abs/2408.02629) | Coarse-to-fine curation, captioning, temporal consistency, balanced distribution. |
| [OpenVid-1M](https://arxiv.org/abs/2407.02371) | ICLR 2025 high-quality million-scale T2V dataset with expressive captions. |
| [MiraData](https://papers.nips.cc/paper_files/paper/2024/hash/57f6683e550eb067936c9e9f0bcb8e31-Abstract-Datasets_and_Benchmarks_Track.html) | Long-duration video data and structured captions. |
| [LVD-2M](https://papers.nips.cc/paper_files/paper/2024/file/1df493ec1c2530c038d94d7300b5b368-Paper-Datasets_and_Benchmarks_Track.pdf) | Long-take video dataset construction and hierarchical captions. |
