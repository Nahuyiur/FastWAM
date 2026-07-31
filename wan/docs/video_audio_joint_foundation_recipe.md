# Video-Audio 联合生成基座模型综合方案

> **Status:** 方案文档，2026-05-19
> **Scope:** 结合 Sora 2、Veo 3、Seedance 1.5 pro、LTX-2、Movie Gen Audio、UniVerse-1、JavisDiT++、HunyuanVideo-Foley、MMAudio、SVD/Movie Gen/Open-Sora 的公开报告，给出一个可工程落地的 video-audio joint foundation model recipe。
> **结论一句话:** 最稳妥的路线不是从零训练一个全统一大模型，而是以 Wan/LTX/Movie-Gen 类视频 DiT 为视觉主干，加入较小 audio stream 和双向 cross-modal fusion，先把 video-to-audio / text-to-audio 能力训稳，再逐步解冻到 native text-to-audio-video，最后用 AV-DPO/RLHF 对齐同步、音质、画质、prompt following 和人类偏好。

## Reading Path

| 想了解 | 读 |
| --- | --- |
| 30 秒方案 | [Executive Summary](#executive-summary) |
| 支持哪些任务 | [Target Capabilities](#target-capabilities) |
| 数据配方 | [Data Recipe](#data-recipe) |
| 模型架构 | [Model Recipe](#model-recipe) |
| 训练阶段 | [Training Recipe](#training-recipe) |
| 分布式训练策略 | [Distributed Training](#distributed-training) |
| 评测和验收 | [Evaluation And Acceptance](#evaluation-and-acceptance) |
| 风险和取舍 | [Risks And Tradeoffs](#risks-and-tradeoffs) |
| 参考来源 | [Sources](#sources) |

## Executive Summary

当前 audio-video generation 有三条路线：

1. **闭源 native joint A/V:** Sora 2、Veo 3、Seedance 1.5 pro/2.0。产品能力强，但训练细节披露有限。Seedance 1.5 pro 明确写 dual-branch Diffusion Transformer、cross-modal joint module、multi-stage data pipeline、SFT 和 RLHF。
2. **开源/开放报告 joint A/V:** LTX-2、UniVerse-1、JavisDiT++、Apollo/Klear、ALIVE、ProAV-DiT。它们给出了可复现的架构模块：dual-stream、single-tower、stitching-of-experts、temporal-aligned RoPE、modality CFG、AV-DPO。
3. **Video-to-audio / Foley 强模块:** Movie Gen Audio、Google DeepMind V2A、HunyuanVideo-Foley、MMAudio。它们不一定是单体 joint base，但非常适合作为 data engine、teacher、pretraining stage 和 production fallback。

推荐工程路线：

- **主干:** 复用 Wan 类 video DiT + Wan VAE，保留现有 video 质量和 checkpoint 兼容性。
- **音频表示:** 使用 continuous audio latent autoencoder，而不是直接生成 waveform；可参考 Stable Audio / LTX-2 的 latent diffusion 思路。DAC/EnCodec 可作为替代或辅助 tokenizer，但 DiT/flow matching 下 continuous latent 更自然。
- **架构:** 非对称 dual-stream DiT：video stream 大、audio stream 小；每隔若干层插入 bidirectional cross-modal blocks；cross-modal gates 零初始化，避免一开始破坏视频主干。
- **训练:** 先 train audio branch / V2A，再 joint T2AV；逐步从低分辨率短 clip 到高分辨率长 clip；最后 SFT + AV-DPO/RLHF。
- **数据:** 不是简单视频+音轨。必须构建 video-audio-caption triplets，包含 video dense caption、audio caption、ASR transcript、audio event timeline、A/V sync score、diegetic/non-diegetic 标签、speech/lip 标签。
- **评测:** 不能只看 FVD 或 CLIPScore。必须同时评估 video quality、audio quality、audio-video synchrony、lip-sync、prompt adherence、event timing、human preference。

## Target Capabilities

建议把基座定义为一个 multi-task A/V model，而不是只做 text-to-video-with-sound。

| 任务 | 输入 | 输出 | 训练数据 |
| --- | --- | --- | --- |
| T2AV | text | video + audio | text-video-audio triplets |
| I2AV | image + text | video + audio | first-frame video-audio |
| V2A / TV2A | video + optional text | audio | video-audio + audio captions |
| T2V | text | video only | image/video-text |
| T2A | text | audio only | text-audio |
| A2V | audio + optional text/image | video | speech/music/action aligned A/V |
| AV continuation | partial video/audio + text | continuation | long clips / masked frames/audio |
| AV editing | video/audio + instruction | edited video/audio | synthetic edit pairs + real edits |

第一阶段不必全部上线，但训练时应该用 condition dropout 支持这些任务，否则模型会过拟合某一种条件路径，后续扩展困难。

## Data Recipe

### Data Sources

建议按用途维护多个数据池，而不是混在一个 JSONL：

| 数据池 | 用途 | 典型来源 | 核心过滤 |
| --- | --- | --- | --- |
| General A/V | 学常见音画关联 | 授权公开视频、影视片段、纪录片、UGC 授权池 | video quality、audio quality、sync、dedup |
| Speech/Lip | 学 dialogue/lip-sync | 访谈、vlog、演讲、影视对话、配音数据 | face track、speaker diarization、ASR、SyncNet |
| Foley/Event | 学动作音效 | 运动、交通、自然、工具、厨房、爆炸、脚步 | audio event detection、onset alignment |
| Music/Performance | 学音乐和舞台 | 演奏、演唱、舞蹈、MV 授权池 | beat/action alignment、music quality |
| Ambient/Scene | 学环境声 | 城市、自然、室内、雨、风、海、工厂 | long ambience、low speech contamination |
| Video-only | 保视频能力 | Wan/SVD/Movie Gen 式 video-text | video caption/quality |
| Audio-only | 保音频质量 | Freesound/FMA/AudioSet/VGGSound/内部授权音频 | audio caption、loudness、noise |
| Image-text | 保静态视觉概念 | HQ image-text、SAM、captioned images | aesthetic、OCR、安全 |

### Required Metadata Schema

每条训练样本建议至少包含：

```json
{
  "id": "source__video__clip",
  "modalities": ["video", "audio", "text"],
  "video_path": "clips/...",
  "audio_path": "audio/...",
  "duration_sec": 8.0,
  "fps": 24,
  "sample_rate": 48000,
  "width": 1280,
  "height": 720,
  "global_caption": "...",
  "video_caption": "...",
  "audio_caption": "...",
  "transcript": [{"start": 1.2, "end": 3.4, "speaker": "S1", "text": "..."}],
  "audio_events": [
    {"start": 0.8, "end": 1.1, "label": "door slam", "confidence": 0.92}
  ],
  "camera": {"motion": "dolly in", "shot": "close-up", "angle": "eye-level"},
  "av": {
    "sync_score": 0.88,
    "lip_sync_score": 0.81,
    "diegetic_ratio": 0.74,
    "speech_ratio": 0.25,
    "music_ratio": 0.10,
    "silence_ratio": 0.02
  },
  "quality": {
    "video_aesthetic": 6.2,
    "video_technical": 0.86,
    "motion": 0.41,
    "audio_snr": 24.1,
    "audio_loudness_lufs": -16.2,
    "audio_clipping_ratio": 0.0
  },
  "bucket": {
    "aspect": "16:9",
    "duration": "8s",
    "resolution": "720p",
    "audio_latent_rate": "25hz"
  },
  "sampling_weight": 1.0
}
```

### Video Cleaning

沿用 video generation 数据管线：

- shot splitting：PySceneDetect/FFmpeg + LPIPS/ImageBind/CLIP 二次校正。
- quality：resolution、bitrate、blur、DOVER technical、aesthetic。
- motion：optical flow、VMAF motion、static detector、jitter detector。
- artifact：OCR、水印、logo、边框、拼接/grid、slideshow。
- dedup：perceptual hash + semantic embedding cluster。
- sampling：concept/domain/action/camera/style balancing。

### Audio Cleaning

audio-video joint 模型比 T2V 多一套硬要求：

| 维度 | 过滤/标注方式 | 目的 |
| --- | --- | --- |
| loudness | LUFS normalize / peak check | 避免音量分布污染 |
| clipping | waveform peak / clipped sample ratio | 去爆音 |
| SNR/noise | speech/music/noise estimator | 去低质录音 |
| silence ratio | VAD / RMS | 去长静音 |
| speech/music/sfx | audio classifier / AED | 控制数据配比 |
| diegetic vs non-diegetic | classifier + LLM caption | 区分画面内声音和配乐 |
| sync offset | AV sync model / cross-correlation / lip-sync model | 去音画错位 |
| event onset | audio event detection | 训练 foley timing |
| ASR transcript | ASR + diarization | dialogue/lip-sync |
| language/accent | ASR/lang-id | 多语言配比 |

### Captioning

joint A/V 的 caption 应该拆成四层：

1. **Video caption:** 画面主体、动作、场景、风格、相机运动。
2. **Audio caption:** 环境声、音效、音乐、语音、情绪、混音。
3. **Transcript:** speaker/time-aligned speech text，不要混进 general caption。
4. **Event timeline:** 时间轴上的可见动作和可听事件对齐，例如 “0.8s door closes -> slam sound”。

建议的生成链：

- VLM/video captioner 生成画面 dense caption。
- audio captioner/AED/ASR 生成 audio labels、transcript、event tags。
- AV alignment model 给 sync/lip scores。
- LLM 整合成 final structured caption，严格标注哪些信息来自视觉、哪些来自音频。
- 人工小集训练 caption ranker / hallucination detector。

### Data Mixture

一个实用初始配比：

| 数据类型 | pretrain 比例 | SFT 比例 | 说明 |
| --- | --- | --- | --- |
| General A/V | 35% | 20% | 建基础音画共现 |
| Speech/Lip | 20% | 30% | 对话、口型、多语言 |
| Foley/Event | 15% | 20% | 动作音效和 timing |
| Ambient/Scene | 10% | 10% | 场景环境声 |
| Music/Performance | 5% | 5% | 控制音乐污染，保留音乐能力 |
| Video-only | 10% | 10% | 防 video quality 退化 |
| Audio-only | 5% | 5% | 防 audio quality 退化 |

注意：music-heavy 数据要谨慎。大量非 diegetic music 会让模型在任何视频里自动加背景音乐，降低可控性。

## Model Recipe

### Recommended Architecture

推荐从 Wan 类 T2V 模型扩成 **asymmetric dual-stream video-audio DiT**。

```text
text / image / audio / video conditions
          |
     condition encoders
          |
  +-------------------+       bidirectional cross-modal blocks       +-------------------+
  |  video DiT stream | <------------------------------------------> | audio DiT stream |
  |  large, Wan init  |                                             | smaller, new/init |
  +-------------------+                                             +-------------------+
          |                                                                   |
   video latent head                                                    audio latent head
          |                                                                   |
      Wan VAE decode                                                  audio AE decode
          \________________ synchronized video + audio _______________________/
```

为什么推荐 dual-stream，而不是 single-tower：

- 能复用 Wan 官方 video checkpoint，降低从零训练成本。
- video stream 和 audio stream 参数规模可以非对称，符合 LTX-2 的公开经验。
- 可以把 cross-modal gate 零初始化，先不破坏 video 主干。
- 推理时可以关闭 audio stream 做 T2V，或固定 video stream 做 V2A，任务组合更灵活。

single-tower/SoE 仍有价值：

- Apollo/Klear 类 single-tower 更统一，适合从零训练或极大规模训练。
- UniVerse-1 stitching-of-experts 适合把已有 video/music/audio experts 拼起来做低成本启动。
- JavisDiT++ 的 MS-MoE 和 TA-RoPE 很适合借到 dual-stream 里。

### Video Representation

- 使用 Wan VAE / causal 3D VAE，保持 official checkpoint path 和 latent shape 兼容。
- 不建议在 joint A/V 第一阶段重训 video VAE；先冻结，避免同时引入太多不确定性。
- 支持多 bucket：480p/720p、16:9/9:16/1:1、4s/8s/16s。

### Audio Representation

推荐 continuous audio latent autoencoder：

- waveform -> audio latent sequence -> DiT/flow -> audio latent -> waveform。
- sample rate：48k 或 44.1k；内部可用 mono/stereo 两种 head。
- latent frame rate 尽量和 video latent temporal grid 有整数或简单比例关系，例如 25Hz / 50Hz。
- 训练 audio AE 时覆盖 speech、foley、ambient、music，避免只会语音或只会音乐。

替代方案：

- DAC/EnCodec discrete tokens：适合 AR audio LM 或 codec-token transformer；对 diffusion/flow DiT 不是最自然，但可作为辅助 teacher/decoder。
- Mel spectrogram latent：实现简单，但高保真和相位重建可能弱于 neural audio autoencoder。

### Cross-Modal Fusion

推荐组合：

- **Bidirectional cross-attention:** video attends audio, audio attends video。
- **Temporal-aligned RoPE:** audio/video token 使用同一物理时间坐标，参考 JavisDiT++ 的 TA-RoPE 思路。
- **Cross-modality AdaLN:** timestep/condition 共享，但保留 modality-specific scale/shift。
- **Gated fusion:** cross-modal residual gate 零初始化，逐步放开。
- **Modality-specific MoE:** 可在中后期加 MS-MoE，让 speech/foley/music/video 的 FFN 专家分工。

### Classifier-Free Guidance

需要 modality-aware CFG，而不是单一 CFG：

| CFG 分量 | 控制 |
| --- | --- |
| text CFG | prompt following |
| video CFG | 画面质量和条件图像/视频 adherence |
| audio CFG | 音频语义、音质、音量 |
| cross-modal CFG | 音画同步和互相约束强度 |

训练时要做 condition dropout：

- drop text -> unconditional。
- drop audio -> T2V。
- drop video -> T2A。
- drop image/video condition -> T2AV。
- drop transcript -> non-dialogue sound。
- drop cross-modal links -> unimodal robustness。

## Training Recipe

### Stage 0: Tokenizer / Autoencoder Validation

目标：

- 冻结或确认 Wan VAE 的 video reconstruction。
- 训练或选择 audio autoencoder，保证 speech/foley/music 都能高保真重建。

验收：

- video VAE：重建无明显 temporal flicker，PSNR/LPIPS/VMAF 过线。
- audio AE：FAD、mel loss、multi-resolution STFT loss、主观听感过线。
- A/V 对齐：encode/decode 后不会引入音画时延。

### Stage 1: Audio Stream Warmup

目标：

- 先把音频生成能力训稳，避免 joint 训练时 audio branch 成为噪声源。

训练：

- T2A：text/audio caption -> audio latent。
- V2A/TV2A：video latent + text -> audio latent。
- 冻结 video stream，只用 video encoder features/cross attention 给 audio stream。

数据：

- audio-only + text-audio。
- V2A 数据：speech/lip、foley、ambient。

验收：

- audio fidelity 不差于 MMAudio/HunyuanVideo-Foley 小模型基线。
- V2A event timing 能跟上明显动作。

### Stage 2: Joint Low-Resolution Pretraining

目标：

- 建立 video/audio/text 三者共同分布。

训练：

- 低分辨率短 clip：例如 256p/360p，4s/8s。
- video stream 先部分冻结，只训练 cross-modal modules、audio stream、部分后层。
- 多任务混合：T2AV、T2V、T2A、V2A、I2AV。

损失：

```text
L = w_v * L_video_flow
  + w_a * L_audio_flow
  + w_sync * L_av_sync
  + w_lip * L_lip_sync
  + w_align * L_text_av_alignment
  + w_reg * L_cross_modal_regularization
```

注意：

- 早期 `w_a` 可大于 `w_v`，保护 audio branch 学起来。
- `w_sync` 不宜过早过强，否则可能牺牲音质或画质。
- video-only batch 要持续存在，防止 video degradation。

### Stage 3: Multi-Resolution / Multi-Duration Pretraining

目标：

- 扩到 480p/720p、4s/8s/16s，学习真实产品时长。

策略：

- 按 latent shape bucket 组 batch。
- progressive duration：4s -> 8s -> 16s。
- progressive resolution：360p -> 480p -> 720p。
- 混合 general A/V、speech/lip、foley、ambient、video-only、audio-only。

验收：

- 不同 bucket 的 loss 没有明显漂移。
- 同一 prompt 在 audio on/off 两种模式下 video quality 不应明显退化。
- V2A 与 T2AV 都能跑通，不出现只有一条路径可用。

### Stage 4: Capability SFT

目标：

- 把预训练能力收敛成可控产品能力。

建议拆四个 SFT sets：

| SFT set | 目标 |
| --- | --- |
| Dialogue/Lip | 多语言口型、speaker consistency、turn-taking |
| Foley/Event | 可见动作和音效严格对齐 |
| Ambient/Cinematic | 环境声、空间感、混音、情绪 |
| Visual Quality | 保护 Wan 原有画质、人物、相机运动 |

训练技巧：

- 对 dialogue 样本加 transcript condition，不要只靠 dense caption。
- 对 foley 样本加 event timeline condition。
- 对 cinematic 样本显式区分 diegetic sound 和 non-diegetic music。
- 保留 T2V-only SFT batch，防止所有视频都自动带音频偏好。

### Stage 5: Preference Alignment

参考 Seedance 1.5 pro 的 SFT/RLHF、多维 reward，JavisDiT++ 的 AV-DPO。

推荐 reward axes：

| 轴 | 判断 |
| --- | --- |
| video quality | 清晰、稳定、无畸变、无闪烁 |
| audio quality | 无爆音、无噪声、频响自然、混音合理 |
| A/V sync | 动作、接触、脚步、口型、爆炸等 timing |
| prompt adherence | 文本、风格、动作、声音是否符合 prompt |
| narrative coherence | 多镜头、世界状态、声音空间是否一致 |
| non-hallucination | 没有画面外不合理声音或不存在物体 |
| safety | 人声、身份、版权音乐、敏感内容 |

训练形式：

- pairwise AV-DPO：同 prompt 下两个 A/V 输出排序。
- reward model：多轴打分，不要合成一个黑盒分数。
- rejection sampling：先离线筛高质量样本做 SFT，再做 DPO。

### Stage 6: Distillation And Serving

目标：

- 降低推理成本，支持交互式生成。

策略：

- flow/diffusion step distillation。
- CFG distillation，减少多 CFG forward。
- KV/cache / temporal cache。
- audio stream 可选低步数先生成草稿，再 refine。
- 对 V2A 单独蒸馏一个轻量 Foley model 作为低成本路径。

## Distributed Training

Video-audio joint model 比纯 T2V 更吃显存和 IO。推荐：

| 维度 | 策略 |
| --- | --- |
| Data parallel | FSDP / distributed optimizer / ZeRO-style sharding |
| Tensor parallel | video stream attention/MLP TP，audio stream 可较小 TP |
| Sequence/context parallel | 对 video token 的 temporal/spatial sequence 做 CP/SP；cross-modal attention 也要分片 |
| Activation recompute | 对 video blocks、cross-modal blocks、long-duration buckets 开 recompute |
| Mixed precision | bf16 起步；成熟后评估 FP8/TE，注意 audio loss 数值稳定 |
| Checkpoint | distributed checkpoint，支持 TP/CP/DP reshard；video/audio/cross-modal 分 module 保存 |
| Data loading | WebDataset/manifest shards；video/audio latent cache；caption/text embedding cache |
| Bucket sampler | 按 video latent shape + audio latent length 组合分桶，避免 padding 爆炸 |

关键风险：

- cross-modal attention 的 sequence length 可能成为 bottleneck，必须限制交互频率或使用 window/segment cross-attention。
- audio latent rate 太高会把 token 数拉爆；太低会损失瞬态音效。
- A/V batch 的 decode/IO 很重，最好预先缓存 latents。

## Evaluation And Acceptance

### Metrics

| 类别 | 指标 |
| --- | --- |
| Video quality | FVD、VBench、LPIPS/tLPIPS、temporal flicker、human preference |
| Audio quality | FAD、CLAPScore、PESQ/STOI for speech、loudness/clipping/noise |
| A/V sync | SyncNet/LSE-C/LSE-D、AVSync score、event onset error、lip-sync score |
| Alignment | text-video CLIP、text-audio CLAP、VLM judge、caption consistency |
| Controllability | CFG sensitivity、negative prompt compliance、condition ablation |
| Robustness | audio-only/video-only/joint paths、multi-language、long duration |

### Acceptance Ladder

建议按下面顺序验收：

1. **Tokenizer reconstruction:** video/audio encode-decode 无明显损坏。
2. **V2A smoke:** 给真实视频生成同步音频，明显动作有对应音效。
3. **T2A smoke:** 文本生成合理音频。
4. **T2V parity:** 加入 audio branch 后，纯 T2V 不比原 Wan 明显变差。
5. **T2AV smoke:** 文本一次生成视频+音频。
6. **1-sample overfit:** 同一 A/V 样本能被模型背出画面和声音。
7. **Small SFT overfit:** 100 条样本 loss 降、生成风格明显贴近数据。
8. **Bucket stress:** 480p/720p、4s/8s/16s、横竖屏都能训练和推理。
9. **Preference win:** 在固定 prompt suite 上优于 base Wan + external Foley baseline。
10. **Distributed resume:** TP/CP/recompute/FSDP 配置下 ckpt load、续训、推理 reshard 全通过。

## Risks And Tradeoffs

| 风险 | 表现 | 缓解 |
| --- | --- | --- |
| video degradation | 加 audio 后画质下降 | 冻结/低 LR video stream，保留 video-only batch |
| audio hallucination | 无关背景音乐/音效乱入 | diegetic/non-diegetic 标签，negative prompts，AV-DPO |
| lip-sync 不稳 | 口型和语音错位 | speech/lip 数据池，SyncNet reward，transcript condition |
| audio quality 差 | 爆音、噪声、混音混乱 | audio AE 质量、LUFS/SNR/clipping filters，audio-only SFT |
| cross-modal collapse | 模型只听 text，不看 video 或反之 | modality dropout、cross-modal loss、condition ablation |
| caption hallucination | caption 写了不存在物体/声音 | multi-teacher + ranker + human audit |
| copyright/privacy | 训练出受保护声音/人物 | source registry、安全过滤、licensed data、voice consent |
| compute explosion | token 太多训练慢 | bucket、latent cache、cross-attention frequency、CP/SP |

## Recommended Wan-Based Milestones

### Milestone A: Wan + Foley Module Baseline

- 保持 Wan 不动。
- 接 HunyuanVideo-Foley/MMAudio 类 V2A 模块。
- 建立 A/V evaluation suite。
- 目标：得到 “Wan video + external audio” 的强 baseline。

### Milestone B: Wan Audio Branch Warmup

- 冻结 Wan video stream。
- 新增 audio stream + cross-attention。
- 训练 V2A/T2A。
- 目标：模型能根据 Wan latent/video frames 生成同步音频。

### Milestone C: Native T2AV Low-Res

- 解冻 Wan 后几层 + cross-modal modules。
- 训练 360p/480p 4s/8s T2AV。
- 目标：文本一次生成视频+音频，T2V parity 不明显退化。

### Milestone D: HQ SFT + AV-DPO

- 用 HQ dialogue/foley/ambient 数据做 SFT。
- 构建 paired outputs 做 AV-DPO。
- 目标：在固定 prompt suite 上超过 external Foley baseline。

### Milestone E: Long / Multi-Reference AV

- 加 long-take、multi-shot、reference image/audio。
- 支持 audio reference voice、image reference identity、video continuation。
- 目标：接近 Sora 2 / Veo 3 / Seedance 2.0 的产品工作流。

## Sources

| Source | Why It Matters |
| --- | --- |
| [Sora 2](https://openai.com/research/sora-2/) | OpenAI video-audio generation model，公开 synchronized dialogue、sound effects、soundscapes 能力。 |
| [Sora 2 System Card](https://openai.com/index/sora-2-system-card/) | 安全、能力边界和部署风险。 |
| [Veo 3 Technical Report](https://storage.googleapis.com/deepmind-media/veo/Veo-3-Tech-Report.pdf) | Google DeepMind native audio-video product report，偏安全与评测。 |
| [Google DeepMind V2A](https://deepmind.google/blog/generating-audio-for-video/) | Video-to-audio diffusion pipeline，视频像素 + 文本 prompt 生成同步音频。 |
| [Seedance 1.5 pro](https://arxiv.org/abs/2512.13507) | Native audio-visual joint foundation model，dual-branch DiT、cross-modal joint、SFT/RLHF。 |
| [LTX-2](https://arxiv.org/abs/2601.03233) | 开源 joint audio-video foundation model，14B video stream + 5B audio stream、bidirectional cross-attention、modality CFG。 |
| [Movie Gen](https://arxiv.org/abs/2410.13720) | Meta media foundation model suite，video model + video-to-audio/text-to-audio，工业数据和训练配方。 |
| [UniVerse-1](https://arxiv.org/abs/2509.06155) | Stitching-of-experts，把预训练 video/music experts 融合成 unified A/V generation。 |
| [JavisDiT++](https://arxiv.org/abs/2602.19163) | 基于 Wan2.1-1.3B，MS-MoE、TA-RoPE、AV-DPO；对 Wan 扩 A/V 最直接。 |
| [HunyuanVideo-Foley](https://arxiv.org/abs/2508.16930) | 100k-hour multimodal data pipeline、representation alignment、TV2A DiT。 |
| [MMAudio](https://arxiv.org/abs/2412.15322) | CVPR 2025 V2A 强基线，joint train video/text/audio，157M 参数、低延迟。 |
| [Stable Audio Open](https://huggingface.co/docs/diffusers/api/pipelines/stable_audio) | Latent audio diffusion recipe：audio autoencoder + text encoder + DiT。 |
| [Descript Audio Codec](https://huggingface.co/descript/descript-audio-codec) | 高保真 neural audio codec，可作为替代 audio tokenizer 或辅助工具。 |
| [SoundStream](https://research.google/pubs/soundstream-an-end-to-end-neural-audio-codec/) | neural audio codec 基础工作。 |
| [Video data pipeline doc](video_generation_data_pipeline.md) | 配套的 video data curation 详细调研。 |
