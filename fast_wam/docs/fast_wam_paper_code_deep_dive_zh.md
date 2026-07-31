# Fast-WAM 深度解读：训练时学会预测世界，推理时不再预测世界

Last updated: 2026-07-23

> 论文：*Fast-WAM: Do World Action Models Need Test-time Future Imagination?*
> arXiv：2603.16666
> 原始分析：本地论文 PDF/LaTeX source，以及官方代码 `main@45d8e14`
> 移植来源：`Fast-WAM/docs/fast_wam_paper_code_deep_dive_zh.md`
> 本文同时补充 `Megatron-Wan/fast_wam/` inference overlay 的实现对应和边界。

## 0. 先说结论：它到底在干什么？

Fast-WAM 想回答一个很具体的问题：

> World Action Model（WAM）效果好，到底是因为它在**训练时被迫学习预测未来**，
> 还是因为它在**推理时真的生成了未来视频，再据此做动作**？

作者的答案是：

> 主要收益来自前者。视频预测是一个很强的训练期辅助任务，会把当前观测编码成更懂
> 物理、运动和交互的表示；但部署时未必需要把未来视频真正生成出来。

所以 Fast-WAM 的做法可以浓缩成一句话：

> 训练时，同时学习未来视频的 flow matching 和动作 chunk 的 flow matching；
> 推理时，删掉未来视频分支，只用当前帧经过一次 Video DiT 得到的逐层 K/V 表示，
> 迭代生成动作。

它并不是：

- 一个在测试时高速生成未来视频的模型；
- 一个经典意义上“对候选动作做世界模型 rollout、比较后再规划”的
  model-based planner；
- 一个一步前向就直接回归动作的普通行为克隆策略。

它更准确的定位是：

> 一个使用视频预测进行辅助表示学习、但部署形态仍然是 direct
> diffusion/flow policy 的机器人策略。

这也是理解整篇论文最重要的视角。

### 0.1 在 Megatron-Wan 中实现了什么？

`Megatron-Wan/fast_wam/` 只移植了论文模型的部署快路径：

- 当前帧 Video expert prefill；
- 逐层 video K/V cache；
- Action expert 的 10-step FlowMatch denoising；
- LeRobot-compatible camera/state/action preprocessing；
- Megatron Core TP、DP、safetensors 和 DCP；
- LIBERO replay/closed-loop evaluation。

当前没有移植：

- Fast-WAM training；
- future-video flow loss；
- Fast-WAM-Joint；
- Fast-WAM-IDM；
- PP/CP/SP；
- RoboTwin；
- joint video/action generation。

因此本文关于训练、消融和 Joint/IDM 的分析仍指向官方 Fast-WAM code；关于部署快路径、
TP/DP/DCP 和本地 LIBERO 验收的补充才对应 Megatron overlay。

Megatron 主要代码落点：

| 论文/官方概念 | Megatron-Wan 实现 |
|---|---|
| Video expert / Action expert / MoT | [`model.py`](../model.py) |
| current-frame K/V prefill | `MoT.prefill_video_cache` |
| action-only cached inference | `MoT.forward_action_with_video_cache` |
| FlowMatch action scheduler | [`scheduler.py`](../scheduler.py) |
| LeRobot checkpoint / TP slicing / DCP | [`checkpoint.py`](../checkpoint.py) |
| VAE、UMT5、camera composition | [`components.py`](../components.py) |
| MIN_MAX、gripper、TP broadcast | [`policy.py`](../policy.py) |
| LIBERO observation 和 rollout | [`libero.py`](../libero.py) |

---

## 1. 论文真正拆开的两个变量

### 1.1 普通 VLA/direct policy

标准视觉语言动作策略直接学习：

\[
p(a_{1:H}\mid o,l),
\]

其中：

- \(o\)：当前视觉观测；
- \(l\)：语言指令；
- \(a_{1:H}\)：长度为 \(H\) 的动作 chunk。

模型看到现在，直接输出接下来一段动作。

### 1.2 常见 WAM：imagine then execute

典型 WAM 显式引入未来视频 \(v_{1:T}\)：

\[
p(a_{1:H}\mid o,l)
=
\int p(v_{1:T}\mid o,l)\,
p(a_{1:H}\mid o,l,v_{1:T})\,dv_{1:T}.
\]

直觉是：

1. 先想象“如果任务顺利执行，未来画面应该长什么样”；
2. 再根据这个未来视觉轨迹反推动作。

但视频 diffusion/flow model 通常要对大量时空 token 做多轮去噪，部署开销很大。

### 1.3 以前被绑在一起的两个因素

已有 WAM 通常同时具有：

1. **训练期视频建模**：优化未来视频预测损失；
2. **推理期未来想象**：真的生成未来视频，并让动作依赖它。

因此，性能提升究竟来自哪一个因素并不清楚。

Fast-WAM 的实验设计就是把二者拆开：

| 设计 | 训练期视频损失 | 推理期生成未来视频 |
|---|---:|---:|
| Fast-WAM | 是 | 否 |
| Fast-WAM-Joint | 是 | 是，视频和动作同步去噪 |
| Fast-WAM-IDM | 是 | 是，先视频、后动作 |
| Fast-WAM w.o. video co-train | 否 | 否 |

论文最有价值的并不是又提出了一个模型名字，而是构造了这组相对受控的比较。

![论文 Figure 1：三类 WAM 范式及 Fast-WAM 的核心区别](../../../Fast-WAM/docs/assets/teaser_main.png)

*论文 Figure 1。A：video/action 联合去噪；B：先想象视频、再由 IDM 出动作；
C：Fast-WAM 只在训练时保留视频共训练，推理时直接从当前观测表示生成动作。图中
“single forward pass”指 Video DiT 只做一次 current-frame prefill，Action DiT
仍迭代 10 步。[矢量原图](../../../Fast-WAM/paper/latex-source/figs/teaser_main_new.pdf)*

---

## 2. 总体架构

Fast-WAM 的主体由四部分组成：

| 组件 | 作用 | 官方代码中的实现 |
|---|---|---|
| Wan2.2-5B Video DiT | 处理当前帧和训练期未来视频 latent；提供 world-grounded visual representation | `WanVideoDiT` |
| Wan2.2 VAE | 把多相机拼接图像/视频压到 48-channel latent | `WanVideoVAE38` |
| T5 text encoder | 编码机器人任务指令 | `WanTextEncoder`，训练时通常预计算 |
| 约 1B Action DiT | 对 32-step action chunk 做 flow matching | `ActionDiT` |

![论文 Figure 2（左）：Fast-WAM 模型架构](../../../Fast-WAM/docs/assets/model_arch.png)

*架构图的关键不是“两个网络并排”，而是每一层 Video/Action expert 先各自产生
Q/K/V，再在 mask 约束下做 mixed attention，之后回到各自的 FFN 和输出头。训练时有
video/action 两个 flow loss；推理时移除 noisy future latent，只保留当前帧
Video DiT 表示。[矢量原图](../../../Fast-WAM/paper/latex-source/figs/model_arch.pdf)*

Video DiT 和 Action DiT 被组织为一个 Mixture-of-Transformer（MoT）。

### 2.1 这里的 MoT 不是常规 MoE

名字容易造成误解。它不是“多个专家由 router 按 token 选择”的稀疏 MoE。

实际做法是：

1. Video expert 和 Action expert 各自有独立的输入投影、Transformer block、FFN、
   cross-attention 和输出头；
2. 每一层分别算出各自的 \(Q/K/V\)；
3. 在 attention 内把两组 \(Q/K/V\) 沿 token 维拼接；
4. 使用结构化 mask 做一次 mixed attention；
5. 再把结果按 token 段切开，分别进入各自 expert 的输出投影、残差、
   cross-attention 和 FFN。

官方代码落点是
[`mot.py`](../../../Fast-WAM/code/src/fastwam/models/wan22/mot.py)：

- `MoT._build_expert_attention_io`
- `MoT._mixed_attention`
- `MoT.forward`

两个 expert 的 hidden dimension 不相同：

- Video DiT hidden dimension：3072；
- Action DiT hidden dimension：1024。

但它们都有：

- 30 层；
- 24 个 attention head；
- 每个 head 128 维。

因此两边投影后的 attention 总维度都是 \(24\times128=3072\)，可以在 shared
attention 中拼接。Action expert 的 residual stream 是 1024 维，但其 Q/K/V
空间扩展到了 3072 维。

Megatron overlay 保留相同的 expert/mixed-attention 数学结构，但把适用的线性层换成
Megatron Core `ColumnParallelLinear` / `RowParallelLinear`。Q/K RMSNorm 需要完整
hidden dimension，所以对 TP-local 平方和做 all-reduce。

### 2.2 Action DiT 怎么初始化？

Action DiT 并不是全部随机初始化。

官方
[`preprocess_action_dit_backbone.py`](../../../Fast-WAM/code/scripts/preprocess_action_dit_backbone.py)
会：

1. 读取 Wan2.2 Video DiT 权重；
2. 把适配 Action DiT backbone 的 tensor 逐维线性插值到目标 shape；
3. 当最后一维发生缩放时，默认乘上
   \[
   \alpha=\sqrt{d_v/d_a};
   \]
4. 保留 action input encoder 和 action output head 为随机初始化。

所以“1B action expert”并不是从零学 Transformer 基础能力，而是从视频 DiT
压缩/插值得到一个初始化。Megatron overlay 直接加载已经训练好的 LeRobot checkpoint，
不执行这一步训练前预处理。

---

## 3. 输入到底长什么样？

### 3.1 数据时间轴

LIBERO 和 RoboTwin 的当前官方配置都使用：

- 33 个连续 observation 时刻；
- 对应 32 个 action；
- 视频每隔 4 个环境步取一帧；
- 取样下标为 \(0,4,8,\dots,32\)，共 9 帧；
- action 不降采样，仍然保留 32 步。

官方代码见
[`robot_video_dataset.py`](../../../Fast-WAM/code/src/fastwam/datasets/lerobot/robot_video_dataset.py)：

```python
self.video_sample_indices = list(range(0, num_frames, action_video_freq_ratio))
```

因此训练样本是“9 帧视频 + 32 步动作”，而不是 9 帧对应 8 步动作。

Megatron overlay 是 inference-only，只输入当前 observation，不构造 9 帧训练样本。

### 3.2 多相机如何进入视频模型？

模型没有为每台相机单独建立 token stream，而是先把多相机画面拼成一张图：

| Benchmark | 相机布局 | 最终图像 |
|---|---|---:|
| LIBERO | 主相机与腕部相机水平拼接 | \(224\times448\) |
| RoboTwin | 头部相机在上，左右腕部相机缩小后并排放在下方 | \(384\times320\) |

这意味着相机身份主要由固定空间位置隐式表达，而不是显式 camera embedding。

Megatron LIBERO 路径按 camera key 排序、分别 resize 后沿 width 拼接，保持
`224x448`。输入像素必须保持 `[0,1]`，VAE 边界只做一次 `[0,1]\to[-1,1]`；
不要再次应用发布 LeRobot checkpoint 中序列化的 `VISUAL=MEAN_STD(.5,.5)`。

### 3.3 从像素到 token 的准确形状

Wan2.2 VAE 的关键参数是：

- latent channel：48；
- spatial downsample factor：16；
- temporal downsample factor：4。

Video DiT 再使用 `patch_size=[1,2,2]`。

于是：

| 项目 | LIBERO | RoboTwin |
|---|---:|---:|
| 拼接图像 | \(224\times448\) | \(384\times320\) |
| VAE latent 空间 | \(14\times28\) | \(24\times20\) |
| DiT 每个 latent time step 的空间 token | \(7\times14=98\) | \(12\times10=120\) |
| 9 个视频帧经 VAE 后的 latent time steps | 3 | 3 |
| 训练期总 video token | \(3\times98=294\) | \(3\times120=360\) |
| 推理期当前帧 video token | 98 | 120 |
| action token | 32 | 32 |

论文说“每个 chunk 有 9 个视频帧”是像素/VAE 输入层面的说法；真正进入 Video DiT
attention 的时间维已经进一步压成 3 个 latent time steps。

### 3.4 语言和 proprioception

语言使用固定模板：

```text
A video recorded from a robot's point of view executing the following instruction: {task}
```

训练时通常先运行 `scripts/precompute_text_embeds.py`，把 T5 embedding 缓存下来，
从而不用在每个训练 step 重复运行 T5。

当前 proprioception 会先通过一个线性层投影到 4096 维，然后作为一个额外 context
token 追加到文本序列。Video expert 和 Action expert 都通过各自的 cross-attention
读取同一组语言/proprio context。

尽管训练数据包含一段 proprio 序列，`FastWAM.build_inputs` 最终只取：

```python
proprio = proprio[:, 0, :]
```

即当前时刻状态，而不是未来状态。Megatron overlay 同样只接收当前 8-D robot state，
并使用 checkpoint stats 做 MIN_MAX normalization。

---

## 4. 最关键的设计：attention mask

下面的矩阵中，行为 query，列为 key/value。“是”表示该行 token 能读取该列 token。

<p align="center">
  <img src="../../../Fast-WAM/docs/assets/mask.png" alt="论文 Figure 2（右）：Fast-WAM 训练与推理 attention mask" width="440">
</p>

*论文 Figure 2（右）。训练时的非对称 mask 保证 action 看不到 ground-truth future；
推理时 future token 整段消失，action 仍读取与训练条件一致的 current-frame token。
有色填充格表示该行 query 可以读取对应列的 key/value，灰框白底格表示被屏蔽。
[矢量原图](../../../Fast-WAM/paper/latex-source/figs/mask.pdf)*

### 4.1 Fast-WAM 训练 mask

| Query \ Key | 当前帧 \(f_0\) | 未来视频 \(f_{1:h}\) | 动作 \(a_{1:H}\) |
|---|---:|---:|---:|
| 当前帧 \(f_0\) | 是 | 否 | 否 |
| 未来视频 \(f_{1:h}\) | 是 | 是 | 否 |
| 动作 \(a_{1:H}\) | 是 | 否 | 是 |

这张表包含四个关键事实。

#### 事实一：当前帧表示不能偷看未来

当前帧 token 只能在当前帧空间 token 内做 attention，不读取未来视频或动作。

因此，推理期只提供当前帧时，其计算图与训练期当前帧表示的因果条件一致。

#### 事实二：未来视频读取当前帧

未来视频 token 可以读取当前帧和其他未来视频 token，用于学习从当前视觉上下文恢复
未来 latent。

#### 事实三：动作不能读取训练集中的真实未来

动作 token 只能读取：

- 当前帧；
- 其他动作 token；
- 通过 cross-attention 读取语言/proprio。

它不能读取未来视频 token。否则训练时动作会看到由 ground-truth future 加噪得到的
token，而推理时却没有，形成严重的信息泄漏和 train-test mismatch。

#### 事实四：视频分支也不读取动作

配置中 `action_conditioned=false`，视频生成分支建模的是：

\[
p(v_{1:T}\mid o,l),
\]

而不是：

\[
p(v_{1:T}\mid o,l,a_{1:H}).
\]

所以 Fast-WAM 并没有学习一个可以输入候选动作、再预测其后果的
action-conditioned dynamics model。这进一步说明它不是经典的 model-based planning。

### 4.2 视频损失如何帮助动作？

乍看之下，动作不看未来视频，两项任务似乎互不相关。真正的耦合发生在参数和逐层
attention 上：

1. 未来 token 为了预测未来，会在每层读取当前帧 token 的 K/V；
2. 视频损失的梯度因此会反向更新当前帧的 Video DiT 表示；
3. 动作 token 在同一层又读取当前帧 K/V；
4. Action DiT 因而消费一个被未来预测任务塑造过的视觉表示。

可以把当前帧表示理解为一个“要足够支持未来预测的状态摘要”。代码中没有单独定义论文
公式里的 \(z(o,l)\) tensor；它对应 Video DiT 各层当前帧 hidden state 及其 K/V。

这也是 Fast-WAM 最核心的机制：

> 未来视频是训练当前状态表示的监督信号，而不是部署时必须显式构造的中间结果。

### 4.3 训练数据流

![根据开源代码还原的 Fast-WAM 训练数据流](../../../Fast-WAM/docs/assets/code_training_path.svg)

图中最重要的约束是：`FV` 不向 action query 开放。Megatron overlay 没有训练期
future-video token，因此只实现推理图中的 current-frame/action dense attention。

---

## 5. Flow matching：视频和动作怎么生成？

### 5.1 论文中的基本目标

对目标 \(y\)（未来视频 latent 或动作）采样高斯噪声 \(\epsilon\)，构造：

\[
y_t=(1-t)y+t\epsilon.
\]

模型预测从数据到噪声的速度：

\[
u=\epsilon-y.
\]

损失为：

\[
\mathcal L_{\mathrm{FM}}(y)
=
\mathbb E\left[
\left\|f_\theta(y_t,t,o,l)-(\epsilon-y)\right\|_2^2
\right].
\]

总损失：

\[
\mathcal L
=
\lambda_{\mathrm{act}}\mathcal L_{\mathrm{act}}
+
\lambda_{\mathrm{vid}}\mathcal L_{\mathrm{vid}}.
\]

当前官方配置默认：

\[
\lambda_{\mathrm{act}}=\lambda_{\mathrm{vid}}=1.
\]

第一帧保持干净，且不计算第一帧的视频重建损失；视频 loss 只监督未来 latent。

### 5.2 action 与 video 的噪声时间独立采样

官方代码分别调用：

- `train_video_scheduler.sample_training_t`
- `train_action_scheduler.sample_training_t`

因此同一个样本的 video timestep 和 action timestep 可以不同。MoT 中两个 expert
也各自使用自己的 timestep modulation。

### 5.3 当前代码的 schedule 与论文描述不完全一致

论文 `exp.tex` 写的是 logit-normal distribution over \(t\)。

但当前
[`scheduler_continuous.py`](../../../Fast-WAM/code/src/fastwam/models/wan22/schedulers/scheduler_continuous.py)
实际实现为：

1. 采样 \(u\sim U(0,1)\)；
2. 使用 shift 映射
   \[
   \phi(u;s)=\frac{su}{1+(s-1)u},
   \]
   默认 \(s=5\)；
3. 令 \(\sigma=\phi(u;s)\)；
4. 用
   \[
   x_\sigma=(1-\sigma)x+\sigma\epsilon.
   \]

代码还对每个 timestep 的 MSE 乘了一个归一化权重，其主体形状为：

\[
w(t)\propto
\exp\left[
-2\left(\frac{t-T/2}{T}\right)^2
\right]-w_{\min}.
\]

这项 timestep weighting 在论文给出的简化目标中也没有写出。

因此，复现 release checkpoint 时应以代码为准，不能仅按论文里的
“logit-normal + standard FM”自行重写 scheduler。

### 5.4 推理更新

推理从高斯噪声 action 开始，沿从噪声到数据的 schedule 做 10 个 Euler-style 更新：

\[
x_{\mathrm{next}}=x+\Delta\sigma\,f_\theta(x,t,\cdot).
\]

因为 \(\Delta\sigma<0\)，样本逐步从噪声端走向数据端。

论文和配置都使用：

- 10 个 inference steps；
- CFG scale = 1.0。

scale 为 1.0 等价于不施加额外 classifier-free guidance。官方
`FastWAM.infer_action` 虽然保留 `negative_prompt` 和 `text_cfg_scale` 参数，但没有
实际运行负条件分支。

Megatron [`scheduler.py`](../scheduler.py) 复现 action inference 所需的 shifted
FlowMatch schedule；它不包含训练 timestep sampling 或 video loss。

---

## 6. Fast-WAM 为什么快？

### 6.1 真正的推理流程

![根据开源代码还原的 Fast-WAM 推理与逐层 K/V cache](../../../Fast-WAM/docs/assets/code_inference_kv_cache.svg)

官方代码关键路径：

1. `FastWAM.infer_action`
2. `video_expert.pre_dit`
3. `MoT.prefill_video_cache`
4. 循环 10 次 `FastWAM._predict_action_noise_with_cache`
5. `MoT.forward_action_with_video_cache`

Megatron 对应路径：

1. `FastWAMPolicy.predict_action_chunk`
2. `FastWAMModel.infer_action_encoded`
3. `VideoExpert.pre_dit`
4. `MoT.prefill_video_cache`
5. 循环 scheduler timesteps
6. `MoT.forward_action_with_video_cache`

### 6.2 “single forward pass”需要准确理解

论文多次说 Fast-WAM 在推理时使用 single forward pass。更准确地说：

> 5B Video DiT 只对当前帧运行一次，未来视频不做迭代去噪；但约 1B Action DiT
> 仍然运行 10 个 flow-matching steps。

所以它不是整个模型一步输出动作。

加速来自两个部分：

1. 删除未来视频时空 token 的 10 轮去噪；
2. 当前帧的 Video DiT K/V 只算一次，此后每个 action step 复用。

在每层 action attention 中：

- query 来自当前 step 的 action token；
- key/value 是“缓存的当前帧 K/V + 当前 step 的 action K/V”；
- action token 之间仍是双向 attention。

Megatron overlay 的首帧推理只有一个 video temporal group，当前 mask 实际为 dense，
因此使用 FP32 PyTorch SDPA 保持 LeRobot numerical parity。只有加入多帧/训练 mask
后才需要 BAGEL MoT 风格的 FlexAttention block mask。

### 6.3 action chunk 与 replanning

模型每次预测 32 步动作，但评测不会把 32 步全部无条件执行：

| Benchmark | 预测 horizon | 默认执行前缀 | 然后 |
|---|---:|---:|---|
| LIBERO | 32 | 10 | 重新取观测、重新预测 |
| RoboTwin | 32 | 24 | 重新取观测、重新预测 |

因此论文的 190 ms 是一次 chunk replan 的模型延迟，不是每个低层动作都付一次
190 ms。

若只做简单摊销、不计环境和通信开销，则模型推理开销约为：

- LIBERO：\(190/10\approx19\) ms/执行步；
- RoboTwin：\(190/24\approx7.9\) ms/执行步。

代价是 chunk 内部更偏 open-loop。RoboTwin 默认评测还设置
`skip_get_obs_within_replan=true`，在 action queue 没有执行完时不渲染/获取新 RGB。

Megatron LIBERO runner 同样预测 32 步并默认执行前 10 步，然后重新观测和规划。

---

## 7. 四种受控变体到底差在哪？

### 7.1 汇总

| 变体 | 训练时 action 能看什么 | 推理流程 | 论文延迟 |
|---|---|---|---:|
| Fast-WAM | 当前帧 + action | 当前帧 Video DiT 一次；Action DiT 10 步 | 190 ms |
| Fast-WAM-Joint | 当前帧、未来 noisy video、action | video/action 同步去噪 10 步 | 580 ms |
| Fast-WAM-IDM | teacher-forced future video + action | video 10 步，再 action 10 步 | 810 ms |
| w.o. video co-train | 当前帧 + action | 与 Fast-WAM 相同 | 190 ms |

延迟均在单张 NVIDIA RTX 5090D V2 32GB GPU 上测量。Megatron-Wan 的 PPU
集成评测没有复测这组 latency，不能把论文 GPU latency 当作 PPU 实测值。

### 7.2 Fast-WAM-Joint

官方
[`fastwam_joint.py`](../../../Fast-WAM/code/src/fastwam/models/wan22/fastwam_joint.py)
把 mask 改为：

- action 可以读取全部 video token；
- video 仍不读取 action；
- video 和 action 使用各自 timestep，但在相同的 10 次循环内同步更新。

所以它是“动作依赖同步生成的视频”，而不是完全对称、双向耦合的 joint model。

训练时 action 会读取由真实 future latent 加噪得到的视频 token。低噪声样本中包含较强
ground-truth future 信息；推理时这些 token 则来自模型自己的中间生成结果。这是
joint/teacher-forcing 类方案固有的 train-test gap 之一。

### 7.3 Fast-WAM-IDM

官方
[`fastwam_idm.py`](../../../Fast-WAM/code/src/fastwam/models/wan22/fastwam_idm.py)
的训练比论文一句话描述得更具体。它同时建立三条分支：

1. noisy video：负责视频 flow loss；
2. noisy action：负责 action flow loss；
3. conditioning video：给 action 提供 future condition。

conditioning video 对每个样本有 50% 概率再次加噪，这是为了减轻“训练看真实未来、
推理看生成未来”的 exposure bias。

推理时严格分两阶段：

1. 独立生成未来视频 latent，10 步；
2. 把生成结果当作干净条件，运行 Video DiT prefill，并对 action 再去噪 10 步。

这解释了它为什么最慢：两段生成是串行的。

### 7.4 无视频共训练

官方代码没有单独的 `fastwam_no_video.yaml`。对应做法是保持 Fast-WAM 架构不变，
并设置：

```text
model.loss.lambda_video=0
```

要准确理解这个消融：

- Video DiT 仍然存在；
- 当前帧仍通过 Video DiT；
- action loss 仍可通过当前帧 K/V 更新 Video DiT；
- 只是未来视频预测损失不再提供梯度。

当前代码仍会构造未来视频分支并计算未加权的 video loss，只是乘零后不贡献总 loss。
因此这个消融回答的是“显式视频预测监督是否有用”，而不是“有没有 Video DiT
是否有用”。

---

## 8. 训练实现

### 8.1 哪些模块更新？

`Wan22Trainer._apply_dit_only_train_mode` 会：

- 冻结 VAE；
- 冻结 T5 text encoder；
- 训练整个 `model.dit`，而 `model.dit` 实际指向 `MoT`；
- 因此 Video DiT 和 Action DiT 都会更新；
- 如果启用 proprio encoder，也一同训练。

这不是只训练一个小 action head，而是对约 6B 的 Video+Action DiT 做联合微调。

再次强调：Megatron overlay 当前不实现这条训练路径。

### 8.2 关键训练超参数

| 项目 | 设置 |
|---|---:|
| optimizer | AdamW |
| learning rate | \(10^{-4}\) |
| betas（代码） | \((0.9,0.95)\) |
| weight decay | 0.01 |
| LR schedule | 5% warmup + cosine |
| precision | BF16 |
| gradient clipping | 1.0 |
| action horizon | 32 |
| inference steps for eval | 10 |

论文报告：

- LIBERO 训练 20k steps；
- RoboTwin 训练 30k steps。

但当前 task config 以 `num_epochs` 为主、`max_steps=null`，实际 step 数由数据集长度、
world size、batch size 和 gradient accumulation 推导。要严格复现论文 step 数，
需要检查最终 Hydra 配置和数据集实际长度，不能只看静态 YAML。

### 8.3 训练成本并没有消失

Fast-WAM 优化的是**推理成本**。

训练期间它仍然：

- 编码未来视频；
- 处理全部未来 video token；
- 计算 video flow loss；
- 更新 5B Video DiT。

所以不能把“推理比 WAM 快 4 倍”外推成“训练也更便宜”。

---

## 9. 关键实验

### 9.1 实验设置

| Benchmark | 训练数据/设置 | 评测 |
|---|---|---|
| LIBERO | 4 suites；每 suite 10 tasks、每 task 50 demos，共 2,000 demos | 40 tasks，共 2,000 trials |
| RoboTwin 2.0 | 2,500 clean demos + 25,000 heavily randomized demos，超过 50 tasks | 每 task、每种场景 100 trials |
| 真机毛巾折叠 | Galaxea R1 Lite；60 小时遥操作数据 | success rate、average completion time、latency |

真机任务示意：

![论文 Figure 3：Galaxea R1 Lite 毛巾折叠任务](../../../Fast-WAM/docs/assets/bench_grid_2560x720.png)

*论文 Figure 3。模型需要在长时程闭环控制中完成抓取、对齐、折叠等阶段；毛巾的
可变形性使误差会跨阶段累积。*

### 9.2 RoboTwin

主要 baseline：

| 方法 | Embodied pretraining | Clean | Randomized | Average |
|---|---:|---:|---:|---:|
| \(\pi_0\) | 是 | 65.92 | 58.40 | 62.2 |
| \(\pi_{0.5}\) | 是 | 82.74 | 76.76 | 79.8 |
| Motus | 是 | 88.66 | 87.02 | 87.8 |
| Motus from Wan2.2 | 否 | 77.56 | 77.00 | 77.3 |
| LingBot-VA | 是 | 92.90 | 91.50 | **92.2** |
| LingBot-VA from Wan2.2 | 否 | 80.60 | — | 80.6 |
| **Fast-WAM** | 否 | 91.88 | 91.78 | **91.8** |

受控变体：

| 变体 | Clean | Randomized | Average |
|---|---:|---:|---:|
| Fast-WAM | 91.88 | 91.78 | **91.8** |
| Fast-WAM-Joint | 90.84 | 90.32 | 90.6 |
| Fast-WAM-IDM | 91.16 | 91.34 | 91.3 |
| w.o. video co-train | 82.76 | 84.80 | 83.8 |

关键差值：

- Fast-WAM 比无视频共训练高 **8.0** 个平均成功率点；
- Fast-WAM 反而比 Joint 高 1.2 点、比 IDM 高 0.5 点；
- clean 和 randomized 结果几乎相同，说明整体平均上对场景随机化较稳。

RoboTwin 的 per-task appendix 也说明结果不是每个任务都一致。例如
`Open Microwave` 上差距很大，`Hanging Mug` 对所有方法都较难，少数任务中无视频
共训练并不更差。因此视频共训练的结论是平均趋势，不是逐任务严格支配。

### 9.3 LIBERO

主要 baseline：

| 方法 | Embodied pretraining | Spatial | Object | Goal | Long | Average |
|---|---:|---:|---:|---:|---:|---:|
| OpenVLA | 是 | 84.7 | 88.4 | 79.2 | 53.7 | 76.5 |
| \(\pi_0\) | 是 | 96.8 | 98.8 | 95.8 | 85.2 | 94.1 |
| \(\pi_{0.5}\) | 是 | 98.8 | 98.2 | 98.0 | 92.4 | 96.9 |
| LingBot-VA | 是 | 98.5 | 99.6 | 97.2 | 98.5 | **98.5** |
| Motus | 是 | 96.8 | 99.8 | 96.6 | 97.6 | 97.7 |
| **Fast-WAM** | 否 | 98.2 | 100.0 | 97.0 | 95.2 | 97.6 |

受控变体：

| 变体 | Spatial | Object | Goal | Long | Average |
|---|---:|---:|---:|---:|---:|
| Fast-WAM | 98.2 | **100.0** | 97.0 | 95.2 | 97.6 |
| Fast-WAM-Joint | **99.6** | 99.4 | **98.2** | 96.8 | **98.5** |
| Fast-WAM-IDM | 98.8 | 97.8 | 97.8 | **97.6** | 98.0 |
| w.o. video co-train | 89.2 | 99.2 | 95.4 | 90.0 | 93.5 |

关键差值：

- 视频共训练带来 **4.1** 个平均成功率点；
- 最大退化发生在 Spatial（-9.0）和 Long（-5.2）；
- Object 已接近饱和，无视频共训练仍有 99.2；
- Joint 比 Fast-WAM 高 0.9，IDM 高 0.4，说明 test-time future 在 LIBERO 上可能
  有小幅收益，但远小于视频训练目标带来的 4.1 点。

### 9.4 真机毛巾折叠

论文只用图展示结果。下表 completion time 是按坐标轴读取的约数，不是作者提供的
精确表格值：

![论文 Figure 4：真机成功率、完成时间和单次推理延迟](../../../Fast-WAM/docs/assets/method_scatter_latency.png)

*论文 Figure 4。左图越靠左上越好，右图越短越好。Fast-WAM 的主要优势是保持
190 ms 延迟；IDM 的成功率更高但需串行生成视频和动作，延迟升至 810 ms。
[矢量原图](../../../Fast-WAM/paper/latex-source/figs/method_scatter_latency.pdf)*

| 方法 | Success rate | Average completion time | Latency |
|---|---:|---:|---:|
| pretrained \(\pi_{0.5}\) | 100% | 约 119 s | 180 ms |
| Fast-WAM-IDM | 90% | 约 177 s | 810 ms |
| Fast-WAM | 75% | 约 152 s | 190 ms |
| Fast-WAM-Joint | 70% | 约 227 s | 580 ms |
| \(\pi_{0.5}\) w.o. pretrain | 40% | 约 206 s | 图中未单列 |
| Fast-WAM w.o. video co-train | 10% | 约 241 s | 190 ms |

这里有三层结论：

1. 视频共训练极其重要：75% 对 10%，差 65 点；
2. Fast-WAM 在速度上显著优于 Joint/IDM；
3. 但 IDM 成功率比 Fast-WAM 高 15 点。

第三点不能忽略。它说明在长时程、可变形物体任务上，显式未来想象可能仍然提供有价值的
foresight，只是代价从 190 ms 增加到了 810 ms，而且论文没有给置信区间，无法判断
这个差异的统计稳定性。

pretrained \(\pi_{0.5}\) 在真机任务上同时有最高成功率、最短完成时间和最低延迟。
Fast-WAM 的强项不是“所有条件下绝对最优”，而是：

> 在没有 embodied pretraining 的 Fast-WAM 组内，以很小的推理成本获得较强的
> 数据效率和成功率。

---

## 10. 论文的关键结论应该怎么读？

### 10.1 最有说服力的证据不是 SOTA 表

与 LingBot-VA、Motus、\(\pi_{0.5}\) 的横向比较存在：

- backbone 不同；
- embodied pretraining 不同；
- 训练数据和实现 recipe 不同；
- 某些评测协议不完全一致。

所以“Fast-WAM 接近 SOTA”只能说明方法有竞争力。

真正支撑中心命题的是同一框架内的受控比较：

| 证据 | RoboTwin | LIBERO | 真机 |
|---|---:|---:|---:|
| 去掉视频共训练的损失 | -8.0 | -4.1 | -65 points |
| 换成显式未来想象 | -1.2 到 -0.5 | +0.4 到 +0.9 | -5 到 +15 points |

总体模式确实是：

> 去掉训练期视频目标的伤害，通常大于在推理期生成未来带来的变化。

### 10.2 不能推出“未来想象没用”

论文更稳妥的结论应该是：

> 在这套 Wan2.2 backbone、这些数据规模、这些 benchmark 和 10-step 生成设置下，
> 显式未来生成的边际收益通常不够覆盖其延迟成本。

不能直接推出：

- 所有机器人任务都不需要未来想象；
- 更长 horizon、更强 world model 或更好的 action-conditioned rollout 也没用；
- 规划、搜索和反事实预测都应该被删掉；
- future latent 没有任何额外信息。

真机 IDM 的结果本身就是一个反例提示：复杂可变形物体任务可能更能利用显式未来。

### 10.3 “world representation”可能包含多种效应

作者把提升解释为视频预测塑造了更好的 world representation，这是合理的，但实验尚未
区分：

- 真的学到了物理动力学；
- 学到了更强的时序和运动特征；
- 大规模视频预训练被更好地保留；
- 多任务学习带来的普通正则化；
- 多了一个 dense auxiliary loss，使 5B backbone 更容易优化。

因此“物理世界理解”是合理解释，但不是唯一被实验严格识别出的因果机制。

### 10.4 它更接近 predictive representation learning

从代码看，Fast-WAM 最接近下面这个范式：

1. 用 future prediction 定义什么样的当前状态表示是有用的；
2. 强迫当前帧表示携带足够的可预测动态信息；
3. 部署时只提取这个表示；
4. 用轻一些的 action generator 消费它。

这与很多“训练时使用 privileged target，测试时丢掉 target branch”的表示学习方法
在思想上相通。

---

## 11. 论文与开源实现的差异和复现 caveat

### 11.1 论文—代码对照

| 主题 | 论文表述 | 当前官方代码事实 | 影响 |
|---|---|---|---|
| noise schedule | logit-normal | shifted uniform continuous schedule | 复现应以代码为准 |
| FM loss | 简化的标准 MSE | 额外 timestep weighting | 实际优化目标不同于正文公式 |
| “single forward” | 容易理解成一步动作生成 | 只有 Video DiT 一次；Action DiT 仍 10 步 | 延迟解释必须拆开 |
| no-video variant | 一个消融模型 | 无独立 YAML，应设 `lambda_video=0` | 复现实验需手动 override |
| CFG=1.0 | 报告 CFG scale | 当前 Fast-WAM 不运行负分支 | 数学上与 scale=1 一致 |
| 训练 steps | LIBERO 20k、RoboTwin 30k | YAML 默认按 epoch 推导 | 要核对最终 resolved config |

代码提交时间晚于论文首次 arXiv 发布，因此这些差异也可能来自开源后的实现更新，
不能简单认定为错误。

### 11.2 “没有 embodied pretraining”不等于没有预训练

Fast-WAM 使用：

- 预训练 Wan2.2-5B Video DiT；
- 预训练 Wan VAE；
- 预训练 T5；
- 由 Video DiT 插值得到的 Action DiT backbone。

论文里的 `Embodied PT. = ✗` 仅表示没有先在大规模机器人轨迹上做 embodied
pretraining，不表示整个模型从随机权重训练。

这点对“data efficiency”尤其重要：少量机器人数据能够工作，部分原因很可能正是继承了
大规模视频/文本生成预训练。

### 11.3 RoboTwin 协议并非完全等价

当前官方 README 明确说明：

- Fast-WAM 默认评测 `instruction_type=unseen`，与 Motus 对齐；
- README 指出某个 LingBot-VA 实现使用 `seen` instruction；
- 改为 `seen` 理论上可再提高约 1–2 点。

因此 RoboTwin 表中与 LingBot-VA 的比较需要谨慎解读。协议不是完全统一的
apples-to-apples comparison。

### 11.4 真机实验的证据边界

论文没有报告：

- 每种方法的真机测试次数；
- 多随机种子；
- 标准差或置信区间；
- 失败类型；
- 不同毛巾、背景、初态的分层结果。

成功率以 10% 为粒度，很可能测试次数不大，但论文没有给出明确数字，不能自行断言。

当前官方仓库也只包含 LIBERO/RoboTwin 的训练和评测代码，没有真机毛巾折叠的数据、
配置和部署脚本。因此真机结果无法仅依靠当前仓库完整复现。

### 11.5 缺失的关键消融

如果要更强地证明中心命题，仍希望看到：

- \(\lambda_{\mathrm{vid}}\) 从 0 到 1 的连续 sweep；
- 只冻结/只训练 Video DiT 的比较；
- 不同 Video DiT 规模；
- 不同 embodied data 规模；
- 不同 action denoising step 数；
- 一步 latent prediction，而不是完整视频生成；
- action-conditioned world model；
- 使用生成 future feature 但不解码像素；
- 训练 compute、显存和 wall-clock 对比；
- 多随机种子和置信区间。

这些实验可以区分“视频任务的表示学习价值”“模型容量/正则化效应”和“显式
foresight”的真正贡献。

### 11.6 Megatron overlay 的额外复现 caveat

Megatron-Wan 使用的是转换后的 LeRobot checkpoint 和本地 LIBERO adapter，还要注意：

- 当前 LeRobot revision 要求 visual input 保持 `[0,1]`；发布 checkpoint 序列化的
  `VISUAL=MEAN_STD(.5,.5)` 会造成 double normalization，Megatron 和 reference
  exporter 都绕过它。
- 模型 BF16 时，当前 dense mixed attention 仍强制 FP32 SDPA 以保持 action parity。
- TP reduction 的亚 `1e-4` 差异可能在几百步 MuJoCo 闭环中放大，所以 exact success
  vector 只能用于诊断，不能替代固定 observation action gate。
- 本地 simulator 是 MuJoCo 3.1.6 / robosuite 1.4.0 / OSMesa，不等同于训练数据配置名
  中的 MuJoCo 3.3.2。
- 当前 overlay 是 inference-only，不能用它复现论文训练消融。

---

## 12. 一些进一步 insights

### Insight 1：昂贵的输出模态不一定是必要的推理中间件

视频生成要求模型解释所有像素变化，包括许多与控制无关的细节。动作决策可能只需要
少量任务相关状态。

训练视频预测有助于逼迫表示包含动态信息；但推理时把这些信息再次展开成完整视频，
可能是一个昂贵的信息“解码—再编码”过程。

Fast-WAM 的本质就是保留信息、删除可视化式解码。

### Insight 2：推理优化的关键是静态条件缓存

当前帧和语言在一个 action denoising chunk 内保持不变。

因此最昂贵的 5B visual backbone 可以 prefill 一次；每个 action step 只计算较小的
action expert，并读取静态 K/V。这与语言模型中 prefix KV cache 的思想非常接近。

Megatron overlay 又进一步让每个 TP group 只在 leader 加载 VAE、UMT5 和 env，
将 encoded inputs 广播给 TP ranks；DP replicas 则并行跑不同 episode。

### Insight 3：训练/推理一致性可能比“多看 future”更重要

Fast-WAM 的 action 在训练和推理中都只看当前帧表示。

Joint/IDM 在训练时可能看较干净的 ground-truth future，在推理时却看模型生成的
future。未来信息更丰富，但 distribution shift 也更大。RoboTwin 上 Fast-WAM 反而
略胜两个 imagination variant，可能部分来自这种一致性，而不只是 future 本身无用。

### Insight 4：任务饱和会掩盖机制差异

LIBERO Object 上所有强方法都接近 100%。在这种 ceiling effect 下，0.5–1 点差异很难
说明架构优劣。

更有信息量的是：

- LIBERO Spatial/Long；
- RoboTwin 的困难单任务；
- 真机可变形物体；
- 完成时间和失败模式。

### Insight 5：应把“是否想象”改成“何时值得想象”

结果更像是在支持分层策略：

- 普通、反应式、短 horizon 控制：使用 Fast-WAM 的快速路径；
- 高不确定性、长 horizon、可变形物体或即将失败时：按需启用 future imagination；
- 不必在每个 replan 都无条件支付完整视频生成成本。

这可能比“永远生成”或“永远不生成”的二元选择更有研究价值。

---

## 13. Megatron-Wan 本地验证状态

以下结果验证的是 inference overlay 集成，不是论文训练复现。

### 13.1 Checkpoint 与 CPU parity

- 发布 6B checkpoint 的 1,651 个 source/target tensors 一一匹配；
- zero missing、unexpected 或 shape mismatch；
- CPU tiny end-to-end inference 对 sibling LeRobot implementation 达到
  `atol=rtol=1e-5`；
- streaming checkpoint round-trip 和 TP2/TP4 slicing unit tests 通过。

CPU gate：

```bash
FAST_WAM_DISABLE_MCORE=1 python -m pytest -q fast_wam/tests
```

### 13.2 真实 PPU TP/DP gate

- TP2 和 TP4 full-checkpoint action output shape 都是 `(32, 7)`；
- TP ranks 间 `max_rank_diff=0.0`；
- 固定 8-episode TP2+DP2 gate：
  - action replay 8/8；
  - gripper sign 8/8；
  - maximum action error `8.370876312255859e-4`；
  - Megatron closed-loop 6/8，LeRobot 6/8；
  - `passed=true`。

硬门槛是固定 observation 上完整 action chunk `max_abs<=1e-3`、gripper exact，
以及闭环总成功数不回退。

### 13.3 BF16 LIBERO

50-episode LIBERO-Spatial：

- TP2+DP4；
- Megatron DCP 48/50（96%）；
- 同机 standalone Fast-WAM 47/50（94%）；
- 这是 integration benchmark，不是统计显著改进，也不是完整论文协议。

完整 2,000-episode runner：

```bash
bash fast_wam/scripts/run_libero_full_2k_bf16.sh
```

默认 TP1+DP8，从 TP2 BF16 DCP reshard，加 `--resume`。只有 2,000 个 case 齐全并生成
最终 `summary.json` 后才算完成；中间成功率不能汇报为最终 benchmark。

详细记录：

- [`2026-07-23-megatron-fastwam.md`](../log/2026-07-23-megatron-fastwam.md)
- [`2026-07-23-megatron-fastwam-bf16-libero.md`](../log/2026-07-23-megatron-fastwam-bf16-libero.md)
- [`libero_spatial_bf16_eval_zh.md`](libero_spatial_bf16_eval_zh.md)

---

## 14. 代码阅读地图

### 14.1 论文

| 内容 | 文件 |
|---|---|
| 完整 PDF | [Fast-WAM_arXiv-2603.16666.pdf](../../../Fast-WAM/paper/Fast-WAM_arXiv-2603.16666.pdf) |
| 问题设定和贡献 | [intro.tex](../../../Fast-WAM/paper/latex-source/intro.tex) |
| 方法、mask、FM 目标、variants | [method.tex](../../../Fast-WAM/paper/latex-source/method.tex) |
| 实验设置和结论 | [exp.tex](../../../Fast-WAM/paper/latex-source/exp.tex) |
| RoboTwin 主表 | [tab_robotwin.tex](../../../Fast-WAM/paper/latex-source/tables/tab_robotwin.tex) |
| LIBERO 主表 | [tab_libero.tex](../../../Fast-WAM/paper/latex-source/tables/tab_libero.tex) |
| RoboTwin per-task appendix | [tab_robotwin_detail.tex](../../../Fast-WAM/paper/latex-source/tables/tab_robotwin_detail.tex) |

### 14.2 官方模型

| 想看什么 | 符号/文件 |
|---|---|
| Fast-WAM 总体、loss、mask、推理 | `FastWAM` in [fastwam.py](../../../Fast-WAM/code/src/fastwam/models/wan22/fastwam.py) |
| 两 expert 如何共享 attention | `MoT` in [mot.py](../../../Fast-WAM/code/src/fastwam/models/wan22/mot.py) |
| Action expert | `ActionDiT` in [action_dit.py](../../../Fast-WAM/code/src/fastwam/models/wan22/action_dit.py) |
| Joint variant | `FastWAMJoint` in [fastwam_joint.py](../../../Fast-WAM/code/src/fastwam/models/wan22/fastwam_joint.py) |
| IDM variant | `FastWAMIDM` in [fastwam_idm.py](../../../Fast-WAM/code/src/fastwam/models/wan22/fastwam_idm.py) |
| 实际 scheduler | `WanContinuousFlowMatchScheduler` in [scheduler_continuous.py](../../../Fast-WAM/code/src/fastwam/models/wan22/schedulers/scheduler_continuous.py) |

### 14.3 官方数据与运行

| 想看什么 | 文件 |
|---|---|
| 33→9 帧、相机拼接、T5 cache | [robot_video_dataset.py](../../../Fast-WAM/code/src/fastwam/datasets/lerobot/robot_video_dataset.py) |
| action/proprio 处理和归一化 | [fastwam_processor.py](../../../Fast-WAM/code/src/fastwam/datasets/lerobot/processors/fastwam_processor.py) |
| 模型实例化与 loss 默认值 | [runtime.py](../../../Fast-WAM/code/src/fastwam/runtime.py) |
| 冻结逻辑、optimizer、训练循环 | [trainer.py](../../../Fast-WAM/code/src/fastwam/trainer.py) |
| LIBERO 数据配置 | [libero_2cam.yaml](../../../Fast-WAM/code/configs/data/libero_2cam.yaml) |
| RoboTwin 数据配置 | [robotwin.yaml](../../../Fast-WAM/code/configs/data/robotwin.yaml) |
| Fast-WAM 模型尺寸 | [fastwam.yaml](../../../Fast-WAM/code/configs/model/fastwam.yaml) |
| LIBERO chunk/replan | [sim_libero.yaml](../../../Fast-WAM/code/configs/sim_libero.yaml) |
| RoboTwin chunk/replan | [sim_robotwin.yaml](../../../Fast-WAM/code/configs/sim_robotwin.yaml) |

### 14.4 Megatron inference overlay

| 想看什么 | 文件 |
|---|---|
| Expert、MoT、cache、action inference | [`model.py`](../model.py) |
| Parallel config/fallback | [`mcore.py`](../mcore.py)、[`distributed.py`](../distributed.py) |
| Streaming checkpoint 和 DCP | [`checkpoint.py`](../checkpoint.py) |
| VAE/UMT5 与 camera image | [`components.py`](../components.py) |
| MIN_MAX、TP broadcast、action semantics | [`policy.py`](../policy.py) |
| LIBERO env 与 rollout | [`libero.py`](../libero.py) |
| 8-episode acceptance | [`eval/acceptance.py`](../eval/acceptance.py) |
| Resumable LIBERO evaluation | [`eval/evaluate_libero.py`](../eval/evaluate_libero.py) |
| DCP conversion | [`eval/convert_to_dcp.py`](../eval/convert_to_dcp.py) |

---

## 15. 最终 takeaway

如果只记住五件事：

1. **Fast-WAM 不是测试时生成未来视频更快，而是测试时根本不生成未来视频。**
2. **训练期视频预测通过梯度塑造当前帧的 Video DiT 表示，动作分支只读取这个当前表示。**
3. **推理时 5B Video DiT 只运行一次并缓存 K/V，但 1B Action DiT 仍做 10 步 flow matching。**
4. **受控实验支持“视频共训练比显式未来想象更重要”，但不支持“未来想象永远没用”。**
5. **从实现看，它是 predictive representation learning + direct action diffusion policy，而不是经典的 world-model planning。**

对 Megatron-Wan 还应额外记住：

> 当前 overlay 只实现 inference fast path。它用 Megatron Core TP/DP/DCP 扩展部署，
> 但没有把官方训练、future-video loss、Joint/IDM 或 RoboTwin 一并移植过来。

这篇工作的最大 insight 不是一个具体结构，而是一个很实用的系统设计原则：

> 世界模型的训练目标和世界模型的部署接口不必相同。可以在训练时用昂贵的预测任务
> 学习状态表示，再在推理时只保留真正服务决策的那部分计算。
