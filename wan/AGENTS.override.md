# Wan Video FlowMatch on Megatron-LM

Last updated: 2026-07-25

## Scope and precedence

- This is the canonical handoff for work under `wan/`. Read the repository root
  `AGENTS.md` first for the Megatron baseline, environment immutability,
  artifact policy, work-log requirements, and the shared `Done When` contract.
- The commands, paths, hardware measurements, and validation results below
  describe the May 2026 DGX/H800 Wan training environment. In particular,
  `/aifs4su/...`, NVCR 25.09, enroot, Slurm, `dgx-043`, and `dgx-067` are
  historical environment details; they are not the active Ruibin PPU Fast-WAM
  environment.
- Keep Wan video training details here instead of duplicating them in the root
  handoff. New Wan work should replace stale conclusions and link to a dated
  `wan/log/YYYY-MM-DD-<topic>.md`.
- `wan/` and `fast_wam/` are separate overlays. The support for training,
  PP/CP/SP, pre-encoded video datasets, and VAE decode documented here does not
  imply that `fast_wam/` supports those features.

> **TL;DR**: `wan/` ports the DiffSynth/official Wan DiT core into this Megatron
> repo. Validated snapshot (2026-05-20): Wan2.1 T2V and Wan2.2 TI2V-5B core
> model, FlowMatch SFT loss, official/Diffusers checkpoint loader,
> Megatron-native TP/DP/distributed optimizer/DCP/recompute/PP/CP/SP smoke and
> 8-GPU combinations, TransformerEngine flash attention with CP p2p/ring, real
> Wan VAE/UMT5 pre-encode/decode, official checkpoint inference, fixed TP2 DCP
> reshard/load, and official-checkpoint 1-sample overfit are validated.
> Short-sequence best was `PP2+CP2`; full-duration Wan2.1-1.3B used
> `CP4+DP2` full recompute. Wan2.1-14B large-GPU load/training and a unified
> full text-to-video CLI remain unvalidated/incomplete.

## Reading Path

| 想 ... | 读 |
| --- | --- |
| 30 秒理解现在有什么 | §"What & Why" |
| 跑 smoke / overfit | §"Quick Start" |
| 准备数据 schema | §"Inputs / Outputs / Paths" |
| 看官方 checkpoint 怎么 load | §"Checkpoint / Inference" |
| 看与 DiffSynth 对齐点 | §"Architecture" |
| 看已验证结果 | §"Verified Results" |
| 看坑和剩余风险 | §"Known Pitfalls" |
| 当前 TODO | §"Production Status" |
| 过程记录 | §"Deeper References" |

## What & Why

Wan 是视频 latent-space FlowMatch DiT。DiffSynth 训练逻辑是：

```text
video -> Wan VAE latents x
prompt -> UMT5 context
timestep sigma -> x_t = (1 - sigma) * x + sigma * noise
target = noise - x
Wan DiT(latents=x_t, timestep=t*1000, context) -> velocity
loss = MSE(velocity, target) * scheduler.training_weight(t)
```

本子项目只移植 trainable DiT + FlowMatch loss 到 Megatron training loop，输入默认是预编码好的 `latents/context`。这样可以把大而稳定的 VAE/T5 预处理留在外部 DiffSynth/官方 stack，Megatron 只负责可训练主干、DCP、DDP 和 resume。

为什么先这样切：
- `megatron/` 不改，符合仓库红线；所有入口在 `wan/`。
- `WanModel` 参数名保持 DiffSynth/官方命名，loader 在进入 Megatron TP rank 时做 key adaptation + rank-local shard slicing，便于直接 load `Wan2.1-T2V-1.3B`、`Wan2.2-TI2V-5B` 和 converted safetensors。
- overfit/smoke 可用预编码 tensor 机械验证，不被 VAE/T5 下载和视频 decode 阻塞。
- production validation 不依赖 pseudo fallback：当前主验证样本是用户提供的 `wan/scripts/overfit.mp4`。短序列 sanity 用官方 Wan VAE + UMT5 预编码为 `512x384 / 49 frames / 12fps`；full-duration overfit 用 `512x384 / 513 frames / 12fps`。

## Quick Start

### Prepare Real DiffSynth Pre-Encoded Data

**Input:** local overfit video `wan/scripts/overfit.mp4` + official Wan VAE / UMT5 assets under `/aifs4su/mmcode/codeclm/checkpoints/wan/Wan-AI/Wan2.1-T2V-1.3B/`
**Output:** `/workspace/checkpoints/wan_tmp/overfit_mp4_real_preencoded_512x384_49f_12fps.pt` with real Wan VAE latents `[16,13,48,64]` + UMT5 context `[512,4096]`
**Success check:** stdout contains `latents=(16, 13, 48, 64) context=(512, 4096)`

```bash
# Run in the standard enroot container for CUDA. The script does not download;
# it auto-discovers already downloaded assets from /workspace/checkpoints.
PYTHONPATH=. python \
  wan/scripts/prepare_diffsynth_sample.py \
  --video wan/scripts/overfit.mp4 \
  --output /workspace/checkpoints/wan_tmp/overfit_mp4_real_preencoded_512x384_49f_12fps.pt \
  --prompt 'a person or object moving in a short demo video' \
  --height 384 --width 512 --num-frames 49 --fps 12 \
  --device cuda --dtype bf16 --auto-search
```

If assets are not in the searched roots, pass them explicitly:

```bash
WAN_VAE_CKPT=/aifs4su/mmcode/codeclm/checkpoints/wan/Wan-AI/Wan2.1-T2V-1.3B/Wan2.1_VAE.pth \
WAN_TEXT_ENCODER_CKPT=/aifs4su/mmcode/codeclm/checkpoints/wan/Wan-AI/Wan2.1-T2V-1.3B/models_t5_umt5-xxl-enc-bf16.pth \
WAN_TOKENIZER_PATH=/aifs4su/mmcode/codeclm/checkpoints/wan/Wan-AI/Wan2.1-T2V-1.3B/google/umt5-xxl \
PYTHONPATH=. python wan/scripts/prepare_diffsynth_sample.py \
  --video wan/scripts/overfit.mp4 \
  --output /workspace/checkpoints/wan_tmp/overfit_mp4_real_preencoded_512x384_49f_12fps.pt \
  --prompt 'a person or object moving in a short demo video' \
  --height 384 --width 512 --num-frames 49 --fps 12
```

### Prepare Wan2.2 TI2V-5B Pre-Encoded Data

**Input:** `wan/scripts/overfit.mp4` + official `Wan-AI/Wan2.2-TI2V-5B` assets under `/workspace/checkpoints/wan/Wan-AI/Wan2.2-TI2V-5B/`
**Output:** `/workspace/checkpoints/wan_tmp/overfit_mp4_real_preencoded_wan22_ti2v_512x384_49f_12fps.pt` with Wan2.2 VAE38 latents `[48,13,24,32]`, UMT5 context `[512,4096]`, and first-frame latent `[48,1,24,32]`
**Success check:** stdout/sample metadata contains `wan_version=2.2`, `fuse_vae_embedding_in_latents=True`, and `first_frame_latents`

```bash
PYTHONPATH=. python wan/scripts/prepare_diffsynth_sample.py \
  --video wan/scripts/overfit.mp4 \
  --output /workspace/checkpoints/wan_tmp/overfit_mp4_real_preencoded_wan22_ti2v_512x384_49f_12fps.pt \
  --prompt 'wan-overfit-sample' \
  --height 384 --width 512 --num-frames 49 --fps 12 \
  --device cuda --dtype bf16 \
  --wan-version 2.2 --fuse-first-frame --tiled \
  --vae-ckpt /workspace/checkpoints/wan/Wan-AI/Wan2.2-TI2V-5B/Wan2.2_VAE.pth \
  --text-encoder-ckpt /workspace/checkpoints/wan/Wan-AI/Wan2.2-TI2V-5B/models_t5_umt5-xxl-enc-bf16.pth \
  --tokenizer-path /workspace/checkpoints/wan/Wan-AI/Wan2.2-TI2V-5B/google/umt5-xxl
```

### Prepare Smoke-Only Pseudo Data

**Input:** local overfit video `wan/scripts/overfit.mp4` (42.92s, 960x720, ignored by git)
**Output:** `/workspace/checkpoints/wan_tmp/overfit_mp4_tiny_sample.pt` with pseudo latents `[4,3,8,8]` + context `[32,64]` for `tiny` preset
**Success check:** stdout contains `saved /tmp/wan_overfit_sample.pt`

```bash
PYTHONPATH=. /aifs4su/mmcode/codeclm/miniconda3/envs/vllm/bin/python \
  wan/scripts/prepare_overfit_sample.py \
  --video wan/scripts/overfit.mp4 \
  --output /workspace/checkpoints/wan_tmp/overfit_mp4_tiny_sample.pt \
  --preset tiny \
  --prompt 'wan overfit sample video' \
  --latent-frames 3 --latent-height 8 --latent-width 8
```

This path is intentionally smoke-only. It is not used as evidence for official checkpoint quality, VAE distribution matching, or final overfit.

### CPU Smoke

**Input:** code only
**Output:** forward/scheduler/backward smoke pass
**Success check:** three `ok` lines

```bash
PYTHONPATH=. /aifs4su/mmcode/codeclm/miniconda3/envs/vllm/bin/python - <<'PY'
from wan.tests.test_wan_smoke import (
    test_scheduler_matches_wan_endpoints,
    test_tiny_flow_loss_backward,
    test_tiny_wan_forward_shape,
)
for fn in [test_tiny_wan_forward_shape, test_scheduler_matches_wan_endpoints, test_tiny_flow_loss_backward]:
    fn()
    print(fn.__name__, "ok")
PY
```

### Train 1-Sample Overfit

**Input:** `/tmp/wan_overfit_sample.pt` or a real Wan VAE/UMT5 pre-encoded sample
**Output:** Megatron DCP under `$SAVE_DIR`
**Success check:** `overfit.log` reaches the requested final iteration and saves `iter_XXXXXXX`; after inference, stdout prints `latent_mse`

```bash
# Run inside the standard pyxis/enroot container, or a GPU shell with torchrun.
SAMPLE_PATH=/tmp/wan_overfit_sample.pt \
SAVE_DIR=/workspace/checkpoints/wan_overfit_tiny \
WAN_PRESET=tiny TRAIN_ITERS=1000 SAVE_INTERVAL=1000 \
  bash wan/scripts/overfit.sh
```

The 2026-05-18 validated path used `dgx-067` plus the standard enroot image:

```bash
ssh dgx-067 'NAME=wan_train_$$; enroot create -n $NAME /aifs4su/mmcode/codeclm/containers/pytorch-25.09-py3.sqsh >/tmp/${NAME}_create.log 2>&1 && enroot start -r -w --mount /aifs4su/mmcode/codeclm/Megatron-Wan:/workspace/megatron --mount /aifs4su/mmcode/codeclm/dataset:/workspace/dataset --mount /aifs4su/mmcode/codeclm/checkpoints:/workspace/checkpoints --mount /aifs4su/mmdata:/aifs4su/mmdata --mount /aifs4su/mmcode/codeclm:/aifs4su/mmcode/codeclm $NAME bash -lc "cd /workspace/megatron && export CUDA_VISIBLE_DEVICES=0 WAN_PRESET=tiny SAMPLE_PATH=/workspace/checkpoints/wan_tmp/wan_overfit_sample.pt SAVE_DIR=/workspace/checkpoints/wan_overfit_tiny_dgx067_enroot_20260518_2209 TRAIN_ITERS=100 SAVE_INTERVAL=50 LOG_INTERVAL=5 LR=0.002 MIN_LR=0.0002 MASTER_PORT=25310 && bash wan/scripts/overfit.sh"'
```

The 2026-05-18 real official-ckpt path overfit the user-provided sample:

```bash
ssh dgx-067 'NAME=wan_overfit_real_512_resume1000_$$; enroot create -n $NAME /aifs4su/mmcode/codeclm/containers/pytorch-25.09-py3.sqsh >/tmp/${NAME}_create.log 2>&1 && enroot start -r -w --mount /aifs4su/mmcode/codeclm/Megatron-Wan:/workspace/megatron --mount /aifs4su/mmcode/codeclm/dataset:/workspace/dataset --mount /aifs4su/mmcode/codeclm/checkpoints:/workspace/checkpoints --mount /aifs4su/mmdata:/aifs4su/mmdata --mount /aifs4su/mmcode/codeclm:/aifs4su/mmcode/codeclm $NAME bash -lc "cd /workspace/megatron && export CUDA_VISIBLE_DEVICES=0 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True WAN_PRESET=t2v-1.3b SAMPLE_PATH=/workspace/checkpoints/wan_tmp/overfit_mp4_real_preencoded_512x384_49f_12fps.pt LOAD_DIR=/workspace/checkpoints/wan_overfit_mp4_real_512x384_49f_official_20260518 SAVE_DIR=/workspace/checkpoints/wan_overfit_mp4_real_512x384_49f_official_resume1000_20260518 TRAIN_ITERS=1000 SAVE_INTERVAL=1000 LOG_INTERVAL=20 LR=1e-4 MIN_LR=1e-5 WAN_GRADIENT_CHECKPOINTING=1 OVERRIDE_OPT_PARAM_SCHEDULER=1 MASTER_PORT=25342 && bash wan/scripts/overfit.sh"'
```

The 2026-05-19 Wan2.2 TI2V-5B path uses TP2, distributed optimizer, full block recompute, and official checkpoint initialization:

```bash
WAN_PRESET=ti2v-5b \
SAMPLE_PATH=/workspace/checkpoints/wan_tmp/overfit_mp4_real_preencoded_wan22_ti2v_512x384_49f_12fps.pt \
WAN_LOAD_OFFICIAL_CKPT=/workspace/checkpoints/wan/Wan-AI/Wan2.2-TI2V-5B \
SAVE_DIR=/workspace/checkpoints/wan2p2_ti2v5b_tp2_overfit_512x384_49f_1000_20260519 \
TRAIN_ITERS=1000 SAVE_INTERVAL=1000 LOG_INTERVAL=50 \
GPUS_PER_NODE=2 TP_SIZE=2 LR=1e-4 MIN_LR=1e-5 \
WAN_GRADIENT_CHECKPOINTING=1 USE_DISTRIBUTED_OPTIMIZER=1 \
  bash wan/scripts/overfit.sh
```

### CPU Tiny Overfit Verifier

**Input:** smaller pseudo-latent sample from the same video
**Output:** `/tmp/wan_overfit_cpu_result_240.pt` with loss curve and latent MSE
**Success check:** final loss below initial loss; 2026-05-18 short sandbox run reached `final_loss=0.45512381`, `latent_mse=0.14147638`

```bash
PYTHONPATH=. /aifs4su/mmcode/codeclm/miniconda3/envs/vllm/bin/python \
  wan/scripts/prepare_overfit_sample.py \
  --video /aifs4su/mmcode/codeclm/LLaMA-Factory/data/mllm_demo_data/3.mp4 \
  --output /tmp/wan_overfit_cpu_sample.pt \
  --preset tiny \
  --prompt 'a person or object moving in a short demo video' \
  --latent-frames 3 --latent-height 8 --latent-width 8

PYTHONPATH=. /aifs4su/mmcode/codeclm/miniconda3/envs/vllm/bin/python \
  wan/scripts/overfit_tiny_cpu.py \
  --sample /tmp/wan_overfit_cpu_sample.pt \
  --output /tmp/wan_overfit_cpu_result_40.pt \
  --iters 40 --lr 0.002 --train-timesteps 16 --eval-steps 8 --seed 1234
```

### Latent-Space Inference / Overfit Eval

**Input:** sample + official ckpt or Megatron DCP
**Output:** `.pt` with `pred_latents`, `gt_latents`, optional `latent_mse`
**Success check:** stdout prints `saved <output>`; overfit DCP should reduce `latent_mse`

```bash
# Random-init smoke (CPU)
PYTHONPATH=. /aifs4su/mmcode/codeclm/miniconda3/envs/vllm/bin/python \
  wan/scripts/infer.py \
  --sample /tmp/wan_overfit_sample.pt \
  --output /tmp/wan_infer_smoke.pt \
  --preset tiny --steps 4 --device cpu --dtype fp32 --seed 7

# DCP overfit eval (GPU recommended)
PYTHONPATH=. python wan/scripts/infer.py \
  --sample /tmp/wan_overfit_sample.pt \
  --output /tmp/wan_overfit_eval.pt \
  --preset tiny --dcp-ckpt /workspace/checkpoints/wan_overfit_tiny \
  --steps 50 --device cuda --dtype bf16 --seed 7
```

### Decode Latents to MP4

**Input:** latent `.pt` with `input_latents` or `pred_latents` + official `Wan2.1_VAE.pth`
**Output:** human-viewable MP4 outside the codebase
**Success check:** `ffprobe` reports the intended `width/height/fps/nb_frames`; 2026-05-18 validated `512x384`, `49` frames, `12fps`

```bash
PYTHONPATH=. python wan/scripts/decode_latents.py \
  --latents /workspace/checkpoints/wan_tmp/overfit_mp4_real_512x384_49f_dcp_iter1000_infer_20step.pt \
  --latent-key pred_latents \
  --output /workspace/checkpoints/wan_outputs/overfit_mp4_real_512x384_49f_dcp_iter1000_infer_20step_decode.mp4 \
  --fps 12 --device cuda --dtype bf16 --tiled \
  --vae-ckpt /workspace/checkpoints/wan/Wan-AI/Wan2.1-T2V-1.3B/Wan2.1_VAE.pth
```

Wan2.2 TI2V-5B uses VAE38:

```bash
PYTHONPATH=. python wan/scripts/decode_latents.py \
  --latents /workspace/checkpoints/wan_tmp/overfit_mp4_wan22_ti2v_tp2_overfit1000_20step.pt \
  --latent-key pred_latents \
  --output /workspace/checkpoints/wan_outputs/overfit_mp4_wan22_ti2v_tp2_overfit1000_20step_decode.mp4 \
  --fps 12 --device cuda --dtype bf16 \
  --wan-version 2.2 --tiled \
  --vae-ckpt /workspace/checkpoints/wan/Wan-AI/Wan2.2-TI2V-5B/Wan2.2_VAE.pth
```

## Inputs / Outputs / Paths

Pre-encoded sample `.pt` schema:

```python
{
    "input_latents": torch.Tensor,  # [C,F,H,W] or [1,C,F,H,W]
    "context": torch.Tensor,        # [L,text_dim] or [1,L,text_dim]
    "first_frame_latents": torch.Tensor,  # optional Wan2.2 TI2V [C,1,H,W]
    "fuse_vae_embedding_in_latents": bool, # optional Wan2.2 TI2V flag
    "prompt": str,                  # optional metadata
    "video_path": str,              # optional metadata
}
```

JSONL schema for real training:

```json
{"latents": "/path/to/latents.pt", "context": "/path/to/context.pt", "first_frame_latents": "/path/to/first_frame.pt", "prompt": "...", "video_path": "..."}
```

Production packed-shard schema:

```json
{"shard_path": "/path/to/shard_000001.safetensors", "index": 42, "latents_key": "input_latents", "context_key": "context", "first_frame_latents_key": "first_frame_latents", "prompt": "...", "video_path": "..."}
```

Supported tensor files: `.pt/.pth`, `.npy`, `.safetensors` (requires safetensors in the active env). The real production path should use official Wan VAE latents plus UMT5 context (`text_dim=4096`): Wan2.1 T2V uses `C=16`; Wan2.2 TI2V-5B uses `C=48` and should carry `first_frame_latents` with `fuse_vae_embedding_in_latents=True`. For scale, use pre-extracted packed safetensors shards via `wan/scripts/pack_preencoded_dataset.py`; per-sample `.pt` rows are acceptable for overfit/debug only.

## Architecture

Implemented files:
- `wan/model/wan_dit.py`: DiffSynth-compatible `WanModel`, T2V path, SDPA attention, 3D RoPE, DiT blocks, head, and `WanFlowTrainingModel`.
- Its shared `_Linear` wrapper has opt-in `skip_bias_add=False`; only the
  Fast-WAM optimized MLP enables bias return for Megatron bias-GELU fusion.
  All Wan modules retain the default path and unchanged checkpoint keys.
- `wan/model/scheduler.py`: DiffSynth Wan FlowMatch scheduler (`shift=5`, timesteps `sigma*1000`, timestep weighting).
- `wan/pretrain.py`: Megatron `pretrain()` entry with collate patch, NullTokenizer, DCP save/resume.
- `wan/model/checkpoint.py`: official/DiffSynth `.pth/.pt/.safetensors` loader.

DiffSynth alignment:
- timestep embedding uses `sinusoidal_embedding_1d(freq_dim, timestep)` with timestep in `[0,1000]`.
- target is `noise - clean`.
- scheduler `step()` uses `sample + pred * (sigma_next - sigma)`.
- core parameter names intentionally match DiffSynth `WanModel`.

Current scope:
- T2V DiT core is implemented.
- Wan2.2 TI2V-5B core is implemented for the DiffSynth path that fuses the first-frame VAE latent into the latent tensor and uses separated timestep conditioning.
- I2V/VACE/S2V/Animate/LongCat/TeaCache branches are not ported.
- VAE/T5/image encoder are not ported; use external preprocessing or DiffSynth for full video I/O.

## Checkpoint / Inference

Official/DiffSynth ckpt load:

```bash
PYTHONPATH=. python wan/scripts/infer.py \
  --preset t2v-1.3b \
  --official-ckpt /path/to/Wan2.1-T2V-1.3B_or_converted_safetensors \
  --sample /path/to/preencoded_sample.pt \
  --output /tmp/wan_official_latents.pt \
  --steps 50 --device cuda --dtype bf16
```

Wan2.2 TI2V-5B uses the `ti2v-5b` preset and a sample containing `first_frame_latents`:

```bash
PYTHONPATH=. python wan/scripts/infer.py \
  --preset ti2v-5b \
  --official-ckpt /workspace/checkpoints/wan/Wan-AI/Wan2.2-TI2V-5B \
  --sample /workspace/checkpoints/wan_tmp/overfit_mp4_real_preencoded_wan22_ti2v_512x384_49f_12fps.pt \
  --output /workspace/checkpoints/wan_tmp/overfit_mp4_wan22_ti2v_official_1step.pt \
  --steps 1 --device cuda --dtype bf16
```

Checkpoint loader behavior:
- file/dir input is supported;
- directory input prefers DiT shards such as `diffusion_pytorch_model*.safetensors` and skips obvious VAE/T5/CLIP files;
- directory input is loaded incrementally shard-by-shard, so 14B official checkpoints do not need to materialize all DiT shards in one Python dict;
- original/DiffSynth keyspace is loaded directly;
- Diffusers keyspace (`blocks.*.attn1.*`, `condition_embedder.*`, `proj_out.*`) is converted locally using the same mapping as DiffSynth `WanVideoDiTFromDiffusers`.

Continue training from official ckpt:

```bash
torchrun --nproc_per_node 8 wan/pretrain.py \
  --wan-preset t2v-1.3b \
  --wan-load-official-ckpt /path/to/official_or_diffsynth_ckpt \
  --wan-data-path /path/to/train.jsonl \
  ... Megatron args ...
```

`--wan-strict-load` can enforce exact key match. Without it, the loader reports missing/unexpected key counts and continues, which is useful for inspecting checkpoint flavor.

Continue from a Megatron DCP:

```bash
LOAD_DIR=/workspace/checkpoints/wan_overfit_tiny \
SAVE_DIR=/workspace/checkpoints/wan_overfit_tiny_resume \
SAMPLE_PATH=/tmp/wan_overfit_sample.pt \
WAN_PRESET=tiny TRAIN_ITERS=120 OVERRIDE_OPT_PARAM_SCHEDULER=1 \
  bash wan/scripts/overfit.sh
```

`OVERRIDE_OPT_PARAM_SCHEDULER=1` is required when extending a checkpoint beyond the original `--train-iters`; otherwise Megatron rejects the scheduler mismatch during optimizer load.

## Megatron Distributed Support

Current support is Megatron-native where it is claimed: linear projections are real `ColumnParallelLinear` / `RowParallelLinear` modules when `megatron_config` is present, official full checkpoints are sliced to the local TP rank before `load_state_dict`, and training uses Megatron DDP/optimizer/DCP instead of an outer fallback wrapper. Wan2.2 TI2V additionally carries per-token timestep tensors through TP/CP/SP and PP payloads.

| Feature | Status | Validation |
| --- | --- | --- |
| Tensor parallelism | ✅ validated TP=2 | `dgx-043`, official Wan2.1-T2V-1.3B, 2 ranks, per-rank params `709897024`, torch_dist DCP save |
| Wan2.2 TI2V TP | ✅ validated TP=2 | `dgx-127`, official Wan2.2-TI2V-5B, 2 ranks, per-rank params `2500887744`, torch_dist DCP save |
| TP DCP reshard/load | ✅ validated TP2→TP1 and TP2→TP2 | `wan_tp2_fixeddcp3_official_smoke_20260519` resharded to TP1 inference; fixed TP2 overfit DCP loaded with TP2 inference |
| Wan2.2 TI2V DCP load/resume | ✅ validated TP=2 | `wan2p2_ti2v5b_tp2_official_smoke_20260519` resumed with `OVERRIDE_OPT_PARAM_SCHEDULER=1`; TP2 DCP inference loaded final overfit checkpoint |
| DCP resume | ✅ validated TP=2 | loaded `iter_0001000`, resumed to `iter_0001001`, restored optimizer/scheduler/RNG, saved new torch_dist DCP |
| Data parallelism | ✅ validated DP=2 | world size 2 / data-parallel size 2, official ckpt load, one train step, DCP save |
| Distributed optimizer | ✅ validated TP=2 smoke | `--use-distributed-optimizer`, log contains `Storing distributed optimizer sharded state of type dp_reshardable`, DCP saved at `wan_tp2_distopt_fixeddcp_smoke_20260519` |
| Full block recompute | ✅ validated TP=2 | `--recompute-granularity full --recompute-method uniform --recompute-num-layers 1`, 1000-step official-ckpt overfit completed |
| Official ckpt load under TP | ✅ validated | full official safetensors load with `missing=0 unexpected=0`, rank-local shape slicing from full DiT weights |
| DCP latent inference + VAE decode | ✅ validated TP=2 | fixed TP2 DCP 20-step inference `latent_mse=0.00155023`; decoded MP4 `512x384`, `49` frames, `12fps` |
| Pipeline parallelism | ✅ validated PP=2 | layer partition + P2P payload + DCP save smoke passed at `wan_pp2_official_smoke_20260519c` |
| Context parallelism | ✅ validated CP=2 | video-token sequence split/gather + full self-attn K/V gather smoke passed at `wan_cp2_official_smoke_20260519b` |
| Sequence parallelism | ✅ validated TP=2+SP | MCore sequence scatter/gather around Row/Column TP linears passed at `wan_tp2_sp_official_smoke_20260519b` |
| Wan2.2 TI2V PP/CP/SP | ✅ validated individually and in 4GPU combinations | `dgx-043` Slurm job `183133`, random-init 5B real-shape smoke with distributed optimizer + full recompute: TP2, CP2, TP2+SP, TP2+CP2, TP2+CP2+SP, PP2, PP2+CP2, PP2+TP2+SP all completed one train step with nonzero grad norm and no skip/NaN |
| Full PP+TP+CP+SP | ✅ validated TP2+PP2+CP2+SP | `dgx-067`, 8 H800 direct enroot, random-init Wan2.2 5B real-shape matrix with distributed optimizer + full recompute; `TP2+PP2+CP2+SP` completed 8 train steps with no skip/NaN |
| FP8/FP4/FSDP | ❌ fail-fast | `--fp8`, `--fp4`, Megatron FSDP, and Torch FSDP2 are blocked until explicitly ported and verified |

The q/k/v projections deliberately use ColumnParallel with `gather_output=True`: Wan applies RMSNorm over the full hidden dimension before RoPE/attention. This preserves official numerics while still sharding projection weights and optimizer state. FFN, text/time MLPs, attention output projections, and the head use the standard Column→Row pattern where the math permits local intermediate activations.

## Performance / Scaling

Measured on `dgx-067` H800 with Wan2.2-TI2V-5B, real `wan/scripts/overfit.mp4` pre-encoded sample (`512x384`, `49` frames, `12fps`, `2496` video tokens + `512` text tokens), distributed optimizer, and full block recompute. Warmup iterations 1-2 are excluded.

| Setting | GPUs | Avg step | Max allocated | Useful MFU | Recompute HFU | Status |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| `PP2+CP2` | 4 | 294.6 ms | 29.4 GB | 6.53% | 8.68% | best current short-seq default |
| `TP4` | 4 | 333.7 ms | 22.3 GB | 5.77% | 7.66% | fastest TP-only |
| `TP4+PP2` | 8 | 333.8 ms | 11.9 GB | 2.88% | 3.83% | best full-node single MP group when CP is not needed |
| `TP4+CP2` | 8 | 334.9 ms | 14.8 GB | 2.87% | 3.82% | best full-node single MP group with CP enabled |
| `TP2+PP2+CP2+SP` | 8 | 390.7 ms | 14.9 GB | 2.46% | 3.27% | full TP/PP/CP/SP correctness path |
| DiffSynth `WanModel` DDP | 8 | 472.8 ms | 56.0 GB | 16.28% | 21.63% | original model path, DDP not official ZeRO3 |

Recommendation:
- Current 1-sample/short-sequence experiments: use `PP2+CP2`. It is the fastest measured setting and keeps CP in the exercised path.
- Full 8GPU single model-parallel group: use `TP4+PP2` for short sequences, or `TP4+CP2` when sequence length is the scaling target. Leave SP off at this length; every SP case was slower in the matrix.
- 128-card balanced estimate: `TP2+PP2+CP16+DP2` supports about `19,968` video tokens while still leaving two data-parallel replicas. At `512x384` this is about `413` raw frames, or `34.4s` at `12fps`; at `832x480`, about `201` frames; at `1280x704`, about `85` frames.
- 128-card max-length estimate: `TP2+PP2+CP32+DP1` supports about `39,936` video tokens. At `512x384` this is about `829` raw frames, or `69.1s` at `12fps`. This is a topology/memory extrapolation from the 8GPU matrix, not a completed 128-card long-sequence run.

DiffSynth comparison caveat: the official DiffSynth Zero3 config failed in the standard 25.09 container because `deepspeed` is not installed. The measured DiffSynth row uses the original DiffSynth `WanModel` and training function path under plain DDP, with the same pre-encoded sample and no VAE/UMT5 work in the timed loop.

Full-duration Wan2.1-1.3B (`513` frames, `99,072` video tokens) was measured separately because it is sequence-length dominated and no-recompute OOMs on H800 80GB:

| Setting | GPUs | Avg step/update | Max allocated | Useful MFU | Recompute HFU | Power / result |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| `CP4+DP2`, TE flash + CP p2p | 8 | 7.76 s | 15.3 GB | 20.19% | 26.92% | selected overfit path after attention fix; official ckpt load, `missing=0 unexpected=0`, no skip/NaN |
| `CP1+DP8`, TE flash | 8 | 29.28 s | 29.8 GB | 21.40% | 28.54% | best full-duration throughput/MFU probe, slower optimizer-update cadence for 1-sample overfit |
| `TP2+CP4`, local-QKV TE, no recompute | 8 | 33.79 s | 66.6 GB | 2.32% | 2.32% | fits full 513-frame no-recompute after TP-aware Q/K RMSNorm rewrite, but too slow to select |
| `CP4+DP2`, old SDPA all-gather | 8 | 8.12 s | 16.1 GB | 19.30% | 25.74% | original overfit run; avg `4.79 kW`, peak `5.51 kW`, raw training energy `32.016795 kWh` |
| `CP1+DP8`, old SDPA | 8 | 30.66 s | 30.6 GB | 20.40% | 27.30% | previous higher-throughput baseline |
| `CP4+DP2`, no recompute | 8 | OOM | ~79 GB in use | n/a | n/a | SDPA and TE probes both OOM before first full step; full-duration training requires activation checkpointing/recompute |

## Verified Results

| Date | Check | Result | Notes |
| --- | --- | --- | --- |
| 2026-05-18 | DiffSynth source inspection | Done | `/aifs4su/mmcode/codeclm/DiffSynth-Studio` cloned at `699e9e1b5cc9b4ce5cdc251c3c65c47bf45507ac` |
| 2026-05-18 | Video sample search | Done | Chose `/aifs4su/mmcode/codeclm/LLaMA-Factory/data/mllm_demo_data/3.mp4`, 6.09s / 640x360 / 270 KB |
| 2026-05-18 | Overfit video sample | Done | `wan/scripts/overfit.mp4`, 42.92s / 960x720 / 515 frames / 5.9 MB; ignored by `wan/.gitignore` |
| 2026-05-18 | Wan2.1-T2V-1.3B ckpt download | Passed | `/aifs4su/mmcode/codeclm/checkpoints/wan/Wan-AI/Wan2.1-T2V-1.3B`, 17 GB: DiT safetensors, UMT5 encoder, VAE, tokenizer |
| 2026-05-18 | Real DiffSynth pre-encode asset check | Passed | auto-search finds VAE, UMT5 encoder, tokenizer under `/aifs4su/mmcode/codeclm/checkpoints/wan/Wan-AI/Wan2.1-T2V-1.3B` |
| 2026-05-18 | Real Wan VAE/UMT5 pre-encode | Passed | `dgx-067` enroot wrote `/workspace/checkpoints/wan_tmp/wan_real_preencoded_128_17.pt`, latents `(16,5,16,16)`, context `(512,4096)` |
| 2026-05-18 | Real Wan VAE/UMT5 pre-encode on `overfit.mp4` | Passed | `dgx-067` enroot wrote `/workspace/checkpoints/wan_tmp/overfit_mp4_real_preencoded_128_17.pt`, latents `(16,5,16,16)`, context `(512,4096)` |
| 2026-05-18 | 512x384 real pre-encode on `overfit.mp4` | Passed | `/workspace/checkpoints/wan_tmp/overfit_mp4_real_preencoded_512x384_49f_12fps.pt`, latents `(16,13,48,64)`, context `(512,4096)` |
| 2026-05-18 | VAE reconstruction video for human review | Passed | `/aifs4su/mmcode/codeclm/checkpoints/wan_outputs/overfit_mp4_vae_recon_512x384_49f_12fps.mp4`, ffprobe `512x384`, `49` frames, `12fps`; source-vs-recon PSNR average `25.61` |
| 2026-05-18 | Sample prep smoke | Passed | `/tmp/wan_overfit_sample.pt`, latents `(4,5,16,16)`, context `(32,64)` |
| 2026-05-18 | CPU unit smoke | Passed | forward shape, scheduler, backward, Diffusers key converter |
| 2026-05-18 | DiffSynth FlowMatch consistency | Passed | direct import of DiffSynth `FlowMatchScheduler("Wan")`: sigmas/timesteps/add_noise/target/weight/step all `allclose=True`, max diff `0.0` |
| 2026-05-18 | CPU latent inference smoke | Passed | random-init tiny, 4 steps, `latent_mse=1.54815555` |
| 2026-05-18 | CPU tiny overfit | Passed smoke | pseudo-latent tiny sample, loss `1.63734245 -> 0.45512381`, 8-step latent MSE `0.14147638` |
| 2026-05-18 | GPU Megatron overfit save | Passed | `dgx-067` enroot, `TRAIN_ITERS=100`, DCP saved at `.../wan_overfit_tiny_dgx067_enroot_20260518_2209`, MSE `0.6373244 -> 0.2041151` |
| 2026-05-18 | Megatron DCP resume | Passed | loaded `iter_0000100`, resumed to `iter_0000120` with `OVERRIDE_OPT_PARAM_SCHEDULER=1`, final MSE `0.1391399` |
| 2026-05-18 | DCP latent inference | Passed | loaded resume DCP `iter_0000120`, 8-step eval `latent_mse=0.14424200` |
| 2026-05-18 | Official ckpt inference branch | Passed | real Wan2.1-T2V-1.3B DiT loaded `missing=0 unexpected=0`; 2-step latent inference on real preencoded sample saved `wan_official_1p3b_infer_2step.pt`, `latent_mse=1.66672158` |
| 2026-05-18 | Official ckpt inference on `overfit.mp4` | Passed | real Wan2.1-T2V-1.3B DiT loaded `missing=0 unexpected=0`; 2-step latent inference saved `overfit_mp4_official_1p3b_infer_2step.pt`, `latent_mse=2.03031325` |
| 2026-05-18 | Official ckpt 512x384 decoded inference | Passed | 20-step official DiT inference `latent_mse=2.38715410`; decoded MP4 `/aifs4su/mmcode/codeclm/checkpoints/wan_outputs/overfit_mp4_official_1p3b_infer_512x384_49f_20step_decode.mp4`, ffprobe `512x384`, `49` frames, `12fps` |
| 2026-05-18 | Official ckpt Megatron train smoke | Passed | real Wan2.1-T2V-1.3B ckpt + real VAE/UMT5 sample, `TRAIN_ITERS=1`, MSE `3.406981E-01`, DCP saved at `wan_official_1p3b_train_smoke_20260518` |
| 2026-05-18 | Tiny Megatron overfit on `overfit.mp4` | Passed | `TRAIN_ITERS=100`, DCP saved at `wan_overfit_mp4_tiny_dgx067_enroot_20260518`, MSE `0.6315210 -> 0.1695125`; DCP eval `latent_mse=0.06636329` |
| 2026-05-18 | Official ckpt 512x384 Megatron overfit on `overfit.mp4` | Passed | `TRAIN_ITERS=50` saved DCP at `wan_overfit_mp4_real_512x384_49f_official_20260518`, final MSE `1.381323E-01`, DCP eval `latent_mse=1.56933236` |
| 2026-05-18 | Official ckpt 512x384 Megatron overfit resume | Passed | resumed to `iter_0001000` at `wan_overfit_mp4_real_512x384_49f_official_resume1000_20260518`; loss reached `2.939545E-03` and final `8.699312E-03`; DCP eval `latent_mse=0.01415021`; decoded MP4 `overfit_mp4_real_512x384_49f_dcp_iter1000_infer_20step_decode.mp4`, `512x384`, `49` frames, `12fps` |
| 2026-05-19 | TP2 DCP sharding fix | Passed | replicated tensors now pass TP replica metadata; `wan_tp2_fixeddcp3_official_smoke_20260519` saved, then resharded TP2→TP1 for inference |
| 2026-05-19 | TP2 + full recompute overfit | Passed | `wan_native_tp2_recompute_overfit_512x384_49f_official_1000_fixed_dcp_20260519`; final train loss `7.561287E-03`; DCP TP2 inference `latent_mse=0.00155023` |
| 2026-05-19 | TP2 overfit VAE decode for human review | Passed | `/aifs4su/mmcode/codeclm/checkpoints/wan_outputs/overfit_mp4_native_tp2_recompute_fixed_dcp_iter1000_infer_20step_decode.mp4`; ffprobe `512x384`, `49` frames, `12fps`; PSNR vs VAE recon average `34.345020` |
| 2026-05-19 | TP2 DCP resume | Passed | loaded fixed `iter_0001000`, resumed to `iter_0001001`, loss `1.010879E-03`, saved `wan_tp2_recompute_resume_iter1001_20260519` |
| 2026-05-19 | Distributed optimizer after sharding fix | Passed | `wan_tp2_distopt_fixeddcp_smoke_20260519`, log contains `Storing distributed optimizer sharded state of type dp_reshardable`, DCP saved |
| 2026-05-19 | PP/CP/SP smoke | Passed | PP2: `wan_pp2_official_smoke_20260519c`; CP2: `wan_cp2_official_smoke_20260519b`; TP2+SP: `wan_tp2_sp_official_smoke_20260519b` |
| 2026-05-19 | Packed-shard dataloader | Passed | `WanJsonlDataset` supports `sample_path`, per-sample paths, and `shard_path/index`; CPU smoke reads torch shard and safetensors pack script output |
| 2026-05-19 | Wan2.2-TI2V-5B ckpt download | Passed | `/aifs4su/mmcode/codeclm/checkpoints/wan/Wan-AI/Wan2.2-TI2V-5B`, 32 GB: 3 DiT safetensors shards, UMT5 encoder, VAE38, tokenizer; no `.incomplete` files |
| 2026-05-19 | Wan2.2 TI2V real pre-encode | Passed | `wan/scripts/overfit.mp4` encoded to `/workspace/checkpoints/wan_tmp/overfit_mp4_real_preencoded_wan22_ti2v_512x384_49f_12fps.pt`, latents `(48,13,24,32)`, first-frame latents `(48,1,24,32)`, context `(512,4096)` |
| 2026-05-19 | Wan2.2 official ckpt inference/decode | Passed | official DiT loaded `missing=0 unexpected=0`; 1-step inference `latent_mse_without_first_frame=1.34833670`; decoded MP4 `512x384`, `49` frames, `12fps` |
| 2026-05-19 | Wan2.2 TP2 official train/save/resume | Passed | TP2 + distributed optimizer + recompute, per-rank params `2500887744`; smoke loss `2.454885E-01`; resumed with optimizer state and saved iter2 |
| 2026-05-19 | Wan2.2 TP2 DCP inference | Passed | TP2 DCP load `missing=0 unexpected=7` metadata keys only; iter2 1-step `latent_mse_without_first_frame=1.33042240` |
| 2026-05-19 | Wan2.2 TP2+SP / CP2 / PP2 smoke | Passed | After CP autograd gather fix: TP2+SP loss `6.390747E+00`, grad norm `43.432`; CP2 loss `4.737865E+00`, grad norm `12.212`; PP2 loss `8.303958E+00`, grad norm `17.222` |
| 2026-05-19 | Wan2.2 5B 4GPU parallel matrix on dgx-043 Slurm | Passed | job `183133` completed `0:0`: TP2, CP2, TP2+SP, TP2+CP2, TP2+CP2+SP, PP2, PP2+CP2, PP2+TP2+SP all passed with distributed optimizer + full recompute; no skip/NaN |
| 2026-05-19 | Wan2.2 5B 8GPU parallel matrix on dgx-067 | Passed | `wan22_5b_parallel_matrix_dgx067_20260519_speed`: all cases except expected `TP1` OOM passed, including `TP2+PP2+CP2+SP`; full table at `speed_mfu_summary.tsv` |
| 2026-05-19 | Wan2.2 5B performance baseline | Passed | best short-seq setting `PP2+CP2`: `294.6ms`, `29.4GB`, HFU `8.68%`; best 8GPU single MP group `TP4+PP2`: `333.8ms`, `11.9GB`, HFU `3.83%` |
| 2026-05-19 | DiffSynth original model DDP comparison | Passed with caveat | official Zero3 config blocked by missing `deepspeed`; original DiffSynth `WanModel` DDP benchmark passed at `472.8ms`, `56.0GB` per GPU, HFU `21.63%` with local batch 1 per rank |
| 2026-05-19 | Wan2.2 1-sample overfit | Passed | `wan2p2_ti2v5b_tp2_overfit_512x384_49f_1000_20260519`; final loss `8.661483E-03`; TP2 DCP 20-step `latent_mse_without_first_frame=0.00102816`; decoded MP4 `overfit_mp4_wan22_ti2v_tp2_overfit1000_20step_decode.mp4`, `512x384`, `49` frames, `12fps`; PSNR vs VAE recon average `37.449779` |
| 2026-05-19 | Wan2.2 1-sample overfit rerun on cleaned dgx-067 | Passed | cleaned current-user GPU process group `2119123`; reran TP2 official-ckpt overfit at `wan2p2_ti2v5b_tp2_overfit_cpfix_dgx067_512x384_49f_1000_20260519`; final loss `8.436272E-03`; TP2 DCP 20-step `latent_mse_without_first_frame=0.00110742`; decoded MP4 `overfit_mp4_wan22_ti2v_tp2_overfit_cpfix_dgx067_1000_20step_decode.mp4`, `512x384`, `49` frames, `12fps`; PSNR vs VAE recon average `37.361619` |
| 2026-05-19 | Wan2.1-T2V-14B ckpt download | Passed | `/aifs4su/mmcode/codeclm/checkpoints/wan/Wan-AI/Wan2.1-T2V-14B`, 65 GB: 6 DiT safetensors shards, UMT5 encoder, VAE, tokenizer/config; no `.incomplete` files |
| 2026-05-19 | Incremental official shard loader | Passed | official directory load now streams one shard at a time and slices to local TP rank; CPU smoke covers two-shard directory load with strict key match |
| 2026-05-19 | Wan2.1 1.3B full-duration overfit | Passed | `wan/scripts/overfit.mp4` pre-encoded to `512x384`, `513` frames, `12fps`; official ckpt load, Megatron DCP resume `1000->3000`, `CP4+DP2`, distributed optimizer, full recompute; final loss `6.336699E-03`, DCP `CP4` 50-step `latent_mse=0.00428428`; decoded MP4 `/aifs4su/mmcode/codeclm/checkpoints/wan_outputs/overfit_mp4_full_wan21_1p3b_cp4dp2_iter3000_50step_decode.mp4`, PSNR vs VAE recon `35.431011`, raw power integral `32.016795 kWh` |
| 2026-05-19 | TransformerEngine flash attention + CP p2p/ring | Passed | `wan_attention_backend=te` uses MCore `TEDotProductAttention`; tiny 1GPU, CP2 train/save, TP2, TP2+SP, and PP2 smoke all passed with TE flash-attn path and no skip/NaN; full Wan2.1 1.3B `CP4+DP2` probe shows TE flash warning and CP `send/recv` P2P, stable step `7.7606s`, useful MFU `20.19%`, HFU `26.92%`; TE DCP CP4 50-step inference `latent_mse=0.00428173`, decoded MP4 `/aifs4su/mmcode/codeclm/checkpoints/wan_outputs/overfit_mp4_full_wan21_1p3b_cp4dp2_iter3000_te_50step_decode.mp4`, PSNR vs VAE recon `35.463917` |
| 2026-05-20 | TP-local QKV + TP-aware Q/K RMSNorm | Experimental pass | `--wan-local-qkv` keeps Q/K/V activations TP-local and all-reduces only Q/K RMS denominator; TP2 bf16 parity vs full-gather TE: tiny `max_abs=6.25e-02 mse=1.75e-05`, Wan2.1-1.3B shape `max_abs=7.03e-02 mse=4.15e-04`; TP2 and TP2+SP 2-step smokes passed with no skip/NaN; official Wan2.1-1.3B TP2 train smoke loaded `missing=0 unexpected=0`; full 513-frame `TP2+CP4` no-recompute now fits at `66.6GB/rank` but is slow (`33.79s`, useful MFU `2.32%`) |

## Known Pitfalls

- **Pre-extracted packed shards are the training contract** — Why: on-the-fly VAE/UMT5 would burn GPU memory/compute in the Megatron step, makes dataloader latency nondeterministic, and repeats frozen encoder work every epoch. How to apply: run official Wan VAE + UMT5 offline, pack same-shape samples into safetensors shards, then train from `shard_path/index` manifest rows. See: `wan/scripts/prepare_diffsynth_sample.py` and `wan/scripts/pack_preencoded_dataset.py`.
- **Wan2.2 TI2V is not just a bigger T2V preset** — Why: DiffSynth uses VAE38 (`C=48`), fuses the first-frame latent into `latents[:, :, 0:1]`, applies separated timestep conditioning, and excludes the first latent frame from loss. How to apply: use `WAN_PRESET=ti2v-5b`, `--wan-version 2.2 --fuse-first-frame` during pre-encode, and keep `first_frame_latents` in dataset/packed-shard rows. See: `wan/log/2026-05-19_wan22_ti2v5b.md`.
- **Use TE attention for production CP** — Why: the old SDPA CP path gathered full K/V on every layer and was correctness-first; it worked, but left MFU on the table. How to apply: keep `WAN_ATTENTION_BACKEND=te` so Wan self/cross attention uses MCore `TEDotProductAttention`; with `--context-parallel-size > 1`, Megatron's default `--cp-comm-type p2p` drives CP send/recv ring instead of all-gather. See: `wan/model/wan_dit.py` and `wan/log/2026-05-19_wan21_full_overfit.md`.
- **dgx-043 debug access is Slurm-only** — Why: direct ssh was not a usable GPU workflow, and `audio-debug` enforced a per-user GPU cap during this validation. How to apply: submit `wan/scripts/slurm_validate_wan22_5b_parallel.sh` with `sbatch`; use `dgx-067` direct enroot for 8GPU local matrix when the node is free.
- **DiffSynth official Zero3 benchmark needs deepspeed in the container** — Why: the standard `pytorch-25.09-py3.sqsh` has `accelerate` but not `deepspeed`, so DiffSynth's official `accelerate_config_zero3.yaml` fails before model init. How to apply: either bake deepspeed into a dedicated non-redline image, or treat the current DDP benchmark as a model-path comparison rather than official Zero3 efficiency.
- **DCP replicated tensors must carry TP replica metadata** — Why: if ordinary `nn.Module` tensors are wrapped with only DP/CP group, TP ranks write identical `replica_id=(0,0,0)` and MCore checkpoint validation rejects or produces non-reshardable checkpoints. How to apply: Wan `sharded_state_dict` passes both TP group and DP/CP group for replicated tensors; do not bypass it with a plain `state_dict()` save.
- **Pseudo sample is only a smoke sample** — Why: `prepare_overfit_sample.py` can derive latents from RGB frames without Wan VAE when DiffSynth assets are missing. How to apply: use this only for codepath smoke; for scientific overfit use `prepare_diffsynth_sample.py` with real Wan VAE + UMT5 assets. See: `wan/log/2026-05-18_wan_port.md`.
- **Match decode FPS to the source sample** — Why: early smoke videos used low resolution and mismatched fps to validate codepaths quickly, which is not suitable for human review. How to apply: pass `--fps 12` for `wan/scripts/overfit.mp4` and confirm with `ffprobe`; use the 512x384 artifacts under `/workspace/checkpoints/wan_outputs/` for visual checks.
- **Official checkpoint flavor matters** — Why: original/DiffSynth Wan keys load directly, while Diffusers keys are converted through a local mapper and can still fail if the preset shape is wrong. How to apply: run without `--wan-strict-load` first and inspect missing/unexpected key counts; use `--wan-strict-load` once the preset/checkpoint flavor is confirmed. See: `wan/model/checkpoint.py`.
- **Resume train-iters is strict** — Why: Megatron stores optimizer scheduler arguments in DCP and rejects a different `--train-iters` by default. How to apply: set `OVERRIDE_OPT_PARAM_SCHEDULER=1` when continuing an overfit run to a larger iteration count; omit it when exact scheduler replay is required.
- **Host Python is not the training env** — Why: bare host may lack Megatron build deps or CUDA visibility (`pybind11` was missing on dgx-067 bare conda). How to apply: use `/aifs4su/mmcode/codeclm/miniconda3/envs/vllm/bin/python` only for CPU smoke, and use the standard pyxis/enroot container for GPU training.

## Production Status (historical 2026-05 snapshot)

This table records the final state of the May DGX/H800 validation campaign.
`IN-FLIGHT` describes the Wan product scope at that snapshot; it does not mean
that these jobs are currently running in the Ruibin PPU workspace.

| Workstream | Status |
| --- | --- |
| Wan T2V/TI2V DiT model port | 🚧 IN-FLIGHT (2026-05-19): Wan2.1 T2V and Wan2.2 TI2V-5B core paths implemented; CPU smoke and Megatron TP/PP/CP/SP plus 4GPU combo smokes passed |
| Official checkpoint load | 🚧 IN-FLIGHT (2026-05-19): Wan2.1-T2V-1.3B and Wan2.2-TI2V-5B official DiT/VAE/UMT5 validated; Wan2.1-T2V-14B downloaded but not GPU-loaded due size |
| Megatron training/DCP | 🚧 IN-FLIGHT (2026-05-19): TP2 DCP save, reshard load, resume, distributed optimizer, recompute overfit, latent inference, and decode passed for 2.1; full-duration Wan2.1 1.3B overfit passed with CP4+DP2; TP2 train/save/resume/infer plus TP/CP/SP/PP 8GPU matrix passed for 2.2 5B |
| Data pipeline | 🚧 IN-FLIGHT (2026-05-19): per-sample and packed-shard dataloaders implemented, including Wan2.2 `first_frame_latents`; production recommendation is offline VAE/UMT5 preextract + safetensors shard manifest |
| Full prompt/video inference | 🚧 IN-FLIGHT (2026-05-19): real VAE/UMT5 pre-encode and decode wrappers work for Wan2.1 and Wan2.2; full text-to-video CLI is still split across pre-encode, latent inference, and decode scripts |
| CI/functional tests | 🚧 IN-FLIGHT (2026-05-19): local smoke tests pass; pytest/isort packages absent in current vLLM env |

## Deeper References

| File | 内容 |
| --- | --- |
| `wan/log/2026-05-18_wan_port.md` | 本次移植过程、命令、验证结果、剩余风险 |
| `wan/log/2026-05-19_wan21_full_overfit.md` | Wan2.1 1.3B full-duration overfit、CP/DP speed probes、power integral、decode/PSNR 结果 |
| `wan/log/2026-05-19_wan22_ti2v5b.md` | Wan2.2-TI2V-5B 配置对照、checkpoint 下载、TP2/PP/CP/SP、resume、推理、overfit 和解码结果 |
| `wan/log/2026-07-24-linear-bias-return.md` | `_Linear` opt-in bias return、Fast-WAM fusion 使用边界和验证 |
| `wan/tests/test_wan_smoke.py` | CPU-level shape/scheduler/backward smoke |
| `wan/scripts/infer.py` | latent-space official ckpt / DCP inference |
| DiffSynth `diffsynth/models/wan_video_dit.py` | Wan DiT architecture source |
| DiffSynth `diffsynth/diffusion/loss.py` | FlowMatch SFT loss source |
| DiffSynth `diffsynth/diffusion/flow_match.py` | Wan FlowMatch scheduler source |
