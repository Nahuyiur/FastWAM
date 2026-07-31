# Wan2.1 1.3B Full-Video Overfit on `wan/scripts/overfit.mp4`

Date: 2026-05-19 HKT

## TL;DR

The user-provided `wan/scripts/overfit.mp4` was pre-encoded as a near-full-duration Wan2.1 sample (`512x384`, `513` frames, `12fps`, `42.75s`) and trained with the official `Wan2.1-T2V-1.3B` checkpoint under Megatron.

Current status:
- `CP4+DP2`, full Wan block checkpointing/recompute, distributed optimizer, DCP save/load, and CP DCP inference all run.
- `iter_0003000` saved cleanly with final train loss `6.336699e-03`; the best logged single timestep loss was `2.431511e-03`.
- 50-step full-video sampling from the `iter_0003000` DCP reached `latent_mse=0.00428428` against the VAE latents, and decoded-video PSNR against the VAE reconstruction reached `35.431011 dB`.
- This is the first validated full-duration `wan/scripts/overfit.mp4` memorization run for Wan2.1 T2V-1.3B in the Megatron port.

## Inputs

- Source video: `wan/scripts/overfit.mp4`
  - `960x720`
  - `12fps`
  - `515` frames
  - `42.916667s`
- Pre-encoded sample:
  - `/aifs4su/mmcode/codeclm/checkpoints/wan_tmp/overfit_mp4_full_real_preencoded_wan21_t2v_512x384_513f_12fps.pt`
  - latents: `(16, 129, 48, 64)`
  - text context: `(512, 4096)`
  - video tokens: `129 * 24 * 32 = 99072`
- Official checkpoint:
  - `/aifs4su/mmcode/codeclm/checkpoints/wan/Wan-AI/Wan2.1-T2V-1.3B`
- DiffSynth reference root:
  - `/aifs4su/mmcode/codeclm/DiffSynth-Studio`

## Pre-Encode Command

Ran in the standard `pytorch-25.09-py3.sqsh` enroot container on `dgx-067`:

```bash
python wan/scripts/prepare_diffsynth_sample.py \
  --video wan/scripts/overfit.mp4 \
  --output /workspace/checkpoints/wan_tmp/overfit_mp4_full_real_preencoded_wan21_t2v_512x384_513f_12fps.pt \
  --prompt wan-full-duration-overfit-sample \
  --height 384 --width 512 --num-frames 513 --fps 12 \
  --device cuda --dtype bf16 --wan-version 2.1 --tiled \
  --vae-ckpt /workspace/checkpoints/wan/Wan-AI/Wan2.1-T2V-1.3B/Wan2.1_VAE.pth \
  --text-encoder-ckpt /workspace/checkpoints/wan/Wan-AI/Wan2.1-T2V-1.3B/models_t5_umt5-xxl-enc-bf16.pth \
  --tokenizer-path /workspace/checkpoints/wan/Wan-AI/Wan2.1-T2V-1.3B/google/umt5-xxl
```

Success output:

```text
saved /workspace/checkpoints/wan_tmp/overfit_mp4_full_real_preencoded_wan21_t2v_512x384_513f_12fps.pt
latents=(16, 129, 48, 64) context=(512, 4096)
```

## Training: 0 -> 1000

Checkpoint root:

```text
/aifs4su/mmcode/codeclm/checkpoints/wan21_1p3b_full513_cp4dp2_overfit_1000_20260519
```

Config:

```text
WAN_PRESET=t2v-1.3b
WAN_LOAD_OFFICIAL_CKPT=/workspace/checkpoints/wan/Wan-AI/Wan2.1-T2V-1.3B
GPUS_PER_NODE=8
TP_SIZE=1
CP_SIZE=4
PP_SIZE=1
DP_SIZE=2
BATCH_SIZE=1
GLOBAL_BATCH_SIZE=2
TRAIN_ITERS=1000
SAVE_INTERVAL=500
WAN_GRADIENT_CHECKPOINTING=1
USE_DISTRIBUTED_OPTIMIZER=1
RECOMPUTE_GRANULARITY=full
RECOMPUTE_METHOD=uniform
RECOMPUTE_NUM_LAYERS=1
```

Result:

```text
iter 500:  saved iter_0000500
iter 1000: mse loss 9.827744e-03, skipped=0, nan=0
iter 1000: saved iter_0001000
max allocated: ~16080 MB/rank
```

Selected loss points:

```text
iter 1:    3.049538e-01
iter 500:  6.387627e-02
iter 720:  1.696671e-02
iter 840:  1.385431e-02
iter 960:  9.221754e-03
iter 1000: 9.827744e-03
```

## Power Integral: 0 -> 1000

Power was sampled with `nvidia-smi` at ~1s cadence into:

```text
/aifs4su/mmcode/codeclm/checkpoints/wan21_1p3b_full513_cp4dp2_overfit_1000_20260519/power.csv
```

Using the `iter20 -> iter1000` training window:

```text
duration:              7997.526875 s
raw energy:            38,228,192 J = 10.618942 kWh
idle baseline:         544.64 W total
incremental energy:    33,872,419 J = 9.409005 kWh
average raw power:     4780.00 W
peak sampled power:    5452.00 W
average GPU util:      98.75 %
raw energy / update:   39,008 J
```

## MFU / Throughput Probes

The Wan FLOP estimator used here matches the earlier probe convention: for the full sample, useful train FLOPs are about `6199.37 TFLOP/sample`, and recompute HFU FLOPs are about `8265.78 TFLOP/sample`. H800 BF16 peak is treated as `989 TFLOP/s/GPU`.

### CP4 + DP2 + Recompute

8-iter probe and the 1000-iter run agree on stable step time:

```text
global batch:          2
stable step:           ~8.14 s
useful MFU:            ~19.3 %
HFU with recompute:    ~25.7 %
peak sampled power:    ~5.45 kW
max allocated:         ~16 GB/rank
```

This setting is best for optimizer-update speed.

### CP1 + DP8 + Recompute

Probe root:

```text
/aifs4su/mmcode/codeclm/checkpoints/wan21_1p3b_full513_cp1dp8_probe_20260519
```

Result:

```text
global batch:          8
stable step:           ~30.66 s
useful MFU:            ~20.4 %
HFU with recompute:    ~27.3 %
average power:         ~4.86 kW
peak sampled power:    ~5.50 kW
max allocated:         ~30.6 GB/rank
```

This setting is better for sample throughput/MFU and energy per sample, but worse for single-sample overfit update cadence.

### CP4 + DP2 without Wan block checkpointing

Probe root:

```text
/aifs4su/mmcode/codeclm/checkpoints/wan21_1p3b_full513_cp4dp2_norecompute_probe_20260519
```

Result: OOM before completing the first iteration.

```text
torch.OutOfMemoryError: GPU has 79.11 GiB total, 79.07 GiB in use
```

2026-05-20 TE no-recompute recheck:

```text
/aifs4su/mmcode/codeclm/checkpoints/wan21_1p3b_full513_te_cp4dp2_norecompute_probe_20260520
  WAN_ATTENTION_BACKEND=te
  setting: TP1 CP4 DP2, distributed optimizer, no recompute, no save
  result: OOM before completing first iteration
  failure point: self-attn Q RMSNorm after Q projection
  rank4/rank7: tried to allocate 146 MiB with ~132 MiB free
  process memory in use: 78.97 GiB, PyTorch allocated: 76.52 GiB
```

Conclusion: full-duration 1.3B training at this resolution still requires activation checkpointing/recompute on H800 80GB after the TE attention fix.

## Inference / Decode After 1000

DCP inference command used `TP1+CP4` and distributed DCP load:

```bash
torchrun --nproc_per_node 4 \
  wan/scripts/infer.py \
  --sample /workspace/checkpoints/wan_tmp/overfit_mp4_full_real_preencoded_wan21_t2v_512x384_513f_12fps.pt \
  --output /workspace/checkpoints/wan_tmp/overfit_mp4_full_wan21_1p3b_cp4dp2_iter1000_50step.pt \
  --preset t2v-1.3b \
  --dcp-ckpt /workspace/checkpoints/wan21_1p3b_full513_cp4dp2_overfit_1000_20260519 \
  --steps 50 --sigma-shift 5.0 --seed 7 \
  --device cuda --dtype bf16 \
  --context-parallel-size 4 --distributed-dcp-load
```

Result:

```text
initialized distributed inference: world_size=4 tensor_model_parallel_size=1 context_parallel_size=4
loaded DCP .../iter_0001000
DCP load_state_dict: missing=0 unexpected=7
latent_mse=0.54899985
```

Decoded prediction:

```text
/aifs4su/mmcode/codeclm/checkpoints/wan_outputs/overfit_mp4_full_wan21_1p3b_cp4dp2_iter1000_50step_decode.mp4
512x384, 12fps, 513 frames, 42.75s
```

Decoded VAE reconstruction reference:

```text
/aifs4su/mmcode/codeclm/checkpoints/wan_outputs/overfit_mp4_full_wan21_vae_recon_512x384_513f_12fps.mp4
512x384, 12fps, 513 frames, 42.75s
```

Prediction-vs-VAE-reconstruction PSNR:

```text
average PSNR: 18.971667 dB
```

## Resume: 1000 -> 3000

Resume command uses the same checkpoint root and preserves the original 0-1000 log as `overfit_0000_1000.log`.

Important env:

```text
LOAD_DIR=/workspace/checkpoints/wan21_1p3b_full513_cp4dp2_overfit_1000_20260519
SAVE_DIR=/workspace/checkpoints/wan21_1p3b_full513_cp4dp2_overfit_1000_20260519
OVERRIDE_OPT_PARAM_SCHEDULER=1
TRAIN_ITERS=3000
SAVE_INTERVAL=1000
```

Result:

```text
iter 1001: mse loss 5.424746e-03
iter 2000: mse loss 7.943106e-03, saved iter_0002000
iter 2240: mse loss 3.071148e-03
iter 2480: mse loss 2.431511e-03
iter 3000: mse loss 6.336699e-03, saved iter_0003000
skip/nan: 0/0
max allocated: ~16092 MB/rank
```

### Power Integral: 1000 -> 3000

Power was sampled into:

```text
/aifs4su/mmcode/codeclm/checkpoints/wan21_1p3b_full513_cp4dp2_overfit_1000_20260519/power_resume_1000_3000.csv
```

Using the `iter1020 -> iter3000` training window:

```text
duration:              16091.709125 s
raw energy:            77,032,270 J = 21.397853 kWh
idle baseline:         545.485 W total
incremental energy:    68,254,484 J = 18.959579 kWh
average raw power:     4787.08 W
peak sampled power:    5510.99 W
average GPU util:      99.04 %
raw energy / update:   38,905 J
```

Combined `0 -> 3000` training windows (`iter20 -> 1000` plus `iter1020 -> 3000`):

```text
duration:              24089.236 s
raw energy:            115,260,462 J = 32.016795 kWh
incremental energy:    102,126,903 J = 28.368584 kWh
average raw power:     ~4784 W
```

MFU/HFU with the same Wan FLOP estimator:

```text
avg step from iter1020 onward: 8118.578 ms
useful MFU:                   19.30 %
HFU with recompute:           25.74 %
```

## Inference / Decode After 3000

DCP inference used `TP1+CP4`, distributed DCP load, 50 sampling steps, and the exact `iter_0003000` checkpoint.

```text
initialized distributed inference: world_size=4 tensor_model_parallel_size=1 context_parallel_size=4
loaded DCP .../iter_0003000
DCP load_state_dict: missing=0 unexpected=7
latent_mse=0.00428428
```

Output latent file:

```text
/aifs4su/mmcode/codeclm/checkpoints/wan_tmp/overfit_mp4_full_wan21_1p3b_cp4dp2_iter3000_50step.pt
```

Decoded prediction:

```text
/aifs4su/mmcode/codeclm/checkpoints/wan_outputs/overfit_mp4_full_wan21_1p3b_cp4dp2_iter3000_50step_decode.mp4
512x384, 12fps, 513 frames, 42.75s
```

Prediction-vs-VAE-reconstruction PSNR:

```text
average PSNR: 35.431011 dB
```

Conclusion: this run successfully overfits the full-duration sample at the VAE latent level and produces a full-length decoded video for manual inspection.

## Attention Backend Fix: TransformerEngine Flash + CP Ring

The original full-duration overfit used Wan's correctness-first SDPA attention path. Under CP this path split local video tokens but gathered full K/V before attention. That validated CP gradients and DCP, but it was not Megatron's optimized context attention path.

Code change:

```text
wan/model/wan_dit.py:
  WAN_ATTENTION_BACKEND=te -> MCore TransformerEngine TEDotProductAttention
  q/k/v remain Wan-native: projection -> full-hidden QK RMSNorm -> 3D RoPE
  TE input layout: [seq, batch, local_heads, head_dim]
  CP path: TEDotProductAttention with cp_comm_type=p2p

wan/scripts/overfit.sh:
  default WAN_ATTENTION_BACKEND=te
  NVTE_FLASH_ATTN=1
  NVTE_FUSED_ATTN=0
  NVTE_UNFUSED_ATTN=0
```

Verification:

```text
python3 -m py_compile wan/model/wan_dit.py wan/pretrain.py wan/patches.py wan/scripts/infer.py
bash -n wan/scripts/overfit.sh
CPU smoke with vllm python: test_tiny_wan_forward_shape / scheduler / backward all ok
```

GPU smoke:

```text
/workspace/checkpoints/wan_te_tiny_smoke_20260519
  WAN_ATTENTION_BACKEND=te, 1 GPU, tiny, 2 iters
  TE DotProductAttention entered forward/backward, DCP saved iter_0000002
  skip/nan: 0/0

/workspace/checkpoints/wan_te_cp2_tiny_smoke_20260519
  WAN_ATTENTION_BACKEND=te, CP2, tiny, 2 iters
  context_parallel_size=2, cp_comm_type=['p2p']
  TE DotProductAttention entered forward/backward, DCP saved iter_0000002
  skip/nan: 0/0

/workspace/checkpoints/wan_te_tp2_tiny_smoke_20260519
  WAN_ATTENTION_BACKEND=te, TP2, tiny, 2 iters
  TE DotProductAttention flash-attn path present
  iter2 loss=2.151123e+00, grad_norm=9.397
  skip/nan: 0/0

/workspace/checkpoints/wan_te_tp2sp_tiny_smoke_20260519
  WAN_ATTENTION_BACKEND=te, TP2+SP, tiny, 2 iters
  sequence_parallel=True
  TE DotProductAttention flash-attn path present
  iter2 loss=2.325224e+00, grad_norm=7.260
  skip/nan: 0/0

/workspace/checkpoints/wan_te_pp2_tiny_smoke_20260519
  WAN_ATTENTION_BACKEND=te, PP2, tiny, 2 iters
  pipeline-model-parallel size=2
  TE DotProductAttention flash-attn path present
  iter2 loss=3.391573e+00, grad_norm=1.307
  skip/nan: 0/0
```

Full Wan2.1 1.3B full-duration probes:

```text
/workspace/checkpoints/wan21_1p3b_full513_te_cp4dp2_probe_20260519
  official ckpt load: missing=0 unexpected=0
  setting: TP1 CP4 DP2, distributed optimizer, full recompute, no save
  evidence: TE DotProductAttention flash-attn warning + CP send/recv P2P warning
  stable iterations: 3..6
  avg step: 7760.6 ms
  max allocated: 15.2786 GB/rank
  useful MFU: 20.19 %
  HFU with recompute: 26.92 %
  skip/nan: 0/0

/workspace/checkpoints/wan21_1p3b_full513_te_cp1dp8_probe_20260519
  official ckpt load: missing=0 unexpected=0
  setting: TP1 CP1 DP8, distributed optimizer, full recompute, no save
  stable iterations: 3..4
  avg step: 29284.95 ms
  max allocated: 29.8498 GB/rank
  useful MFU: 21.40 %
  HFU with recompute: 28.54 %
  skip/nan: 0/0
```

Comparison to old SDPA path:

```text
CP4+DP2 old SDPA all-gather: 8118.6 ms, useful MFU 19.30 %, HFU 25.74 %
CP4+DP2 TE p2p ring:         7760.6 ms, useful MFU 20.19 %, HFU 26.92 %

CP1+DP8 old SDPA:            30660 ms, useful MFU 20.40 %, HFU 27.30 %
CP1+DP8 TE flash:            29284.95 ms, useful MFU 21.40 %, HFU 28.54 %
```

Functional check with the existing overfit checkpoint:

```text
torchrun --nproc_per_node 4 wan/scripts/infer.py \
  --sample /workspace/checkpoints/wan_tmp/overfit_mp4_full_real_preencoded_wan21_t2v_512x384_513f_12fps.pt \
  --output /workspace/checkpoints/wan_tmp/overfit_mp4_full_wan21_1p3b_cp4dp2_iter3000_te_50step.pt \
  --preset t2v-1.3b \
  --dcp-ckpt /workspace/checkpoints/wan21_1p3b_full513_cp4dp2_overfit_1000_20260519/iter_0003000 \
  --steps 50 --sigma-shift 5.0 --seed 7 \
  --device cuda --dtype bf16 \
  --context-parallel-size 4 --distributed-dcp-load \
  --wan-attention-backend te
```

Result:

```text
initialized distributed inference: world_size=4 tensor_model_parallel_size=1 context_parallel_size=4
loaded DCP .../iter_0003000
DCP load_state_dict: missing=0 unexpected=7
TE DotProductAttention flash-attn warning present
CP send/recv P2P warning present
latent_mse=0.00428173
saved /workspace/checkpoints/wan_tmp/overfit_mp4_full_wan21_1p3b_cp4dp2_iter3000_te_50step.pt
```

Decoded TE prediction:

```text
/aifs4su/mmcode/codeclm/checkpoints/wan_outputs/overfit_mp4_full_wan21_1p3b_cp4dp2_iter3000_te_50step_decode.mp4
512x384, 12fps, 513 frames, 42.75s, 7.2 MB
PSNR vs VAE reconstruction: 35.463917 dB
```

Conclusion: Wan now has a production TE attention path. CP full-video training and inference use Megatron Core's TransformerEngine DotProductAttention with `cp_comm_type=p2p`, so context parallel attention runs through Megatron's p2p ring send/recv path instead of the old K/V all-gather path. `NVTE_FLASH_ATTN=1` is enabled for the TE backend, and the train/infer logs enter `DotProductAttention` with the flash-attn warning, so the active path is TE flash/context attention rather than PyTorch SDPA. MFU improved modestly because Q/K/V projection, RMSNorm, RoPE, MLP, recompute, and optimizer still dominate, but attention is now on the correct Megatron Core backend.

## Aggressive Attention Rewrite: TP-local QKV

Problem: Wan's official Q/K RMSNorm is over the full hidden dimension. The conservative TE path therefore kept `gather_output=True` for Q/K/V projections, normalized full hidden, then split local heads for TE. That is numerically safe but leaves Q/K/V activation all-gathers in the TP path.

Implementation:

```text
--wan-local-qkv / WAN_LOCAL_QKV=1

ColumnParallel Q/K/V(gather_output=False)
  -> TensorParallelRMSNorm:
       local_sumsq = sum(local_q.float() ** 2)
       global_sumsq = autograd-aware all_reduce_sum(local_sumsq, TP group)
       denom = rsqrt(global_sumsq / full_hidden_dim + eps)
       local_weight = replicated_norm_weight[tp_rank_slice]
  -> local-head RoPE
  -> TE DotProductAttention(local heads, CP p2p/ring)
  -> RowParallel output
```

The first implementation used raw `dist.all_reduce` for the RMS denominator and triggered PyTorch's missing-autograd-kernel warning during backward. That was fixed by `_AllReduceWithGrad`, whose forward and backward both all-reduce on the TP group.

Validation:

```text
python3 -m py_compile wan/model/wan_dit.py wan/pretrain.py wan/patches.py wan/scripts/infer.py wan/scripts/check_local_qkv_parity.py
bash -n wan/scripts/overfit.sh
CPU smoke with vllm python: test_tiny_wan_forward_shape / scheduler / backward all ok
```

TP2 parity against the conservative full-gather TE path:

```text
torchrun --nproc_per_node 2 wan/scripts/check_local_qkv_parity.py --preset tiny --tp-size 2 --dtype bf16
  max_abs=6.25000000e-02
  mse=1.75379337e-05

torchrun --nproc_per_node 2 wan/scripts/check_local_qkv_parity.py \
  --preset t2v-1.3b --tp-size 2 --dtype bf16 \
  --frames 1 --height 16 --width 16 --context-len 8
  max_abs=7.03125000e-02
  mse=4.15076618e-04
```

`dtype=fp32` parity was not runnable through TE DotProductAttention in this container: TransformerEngine reported no available DPA backend for fp32 inputs. The practical Wan path is bf16, so parity is judged in bf16 tolerance.

Backward smoke:

```text
/workspace/checkpoints/wan_localqkv_tp2_tiny_smoke_20260520b
  setting: tiny, TP2, WAN_LOCAL_QKV=1
  iter2 loss=2.149834e+00, grad_norm=9.386
  skip/nan: 0/0
  no raw dist.all_reduce autograd warning after _AllReduceWithGrad fix

/workspace/checkpoints/wan_localqkv_tp2sp_tiny_smoke_20260520
  setting: tiny, TP2+SP, WAN_LOCAL_QKV=1
  iter2 loss=2.325376e+00, grad_norm=7.258
  skip/nan: 0/0
```

Official checkpoint smoke:

```text
/workspace/checkpoints/wan21_1p3b_localqkv_tp2_official_smoke_20260520
  setting: Wan2.1 1.3B, TP2, 49 frames, WAN_LOCAL_QKV=1
  official ckpt load: missing=0 unexpected=0
  iter1 loss=1.565293e-01, grad_norm=0.724
  max allocated: 32257.44 MB/rank
  skip/nan: 0/0
```

Full-duration no-recompute probe:

```text
/workspace/checkpoints/wan21_1p3b_full513_localqkv_tp2cp4_norecompute_probe_20260520
  setting: Wan2.1 1.3B, full 513 frames, TP2 CP4 DP1, distributed optimizer,
           WAN_LOCAL_QKV=1, no recompute, no save
  official ckpt load: missing=0 unexpected=0
  iter1 step: 33793.4 ms
  loss=3.630039e-01, grad_norm=2.956
  max allocated: 66558.10 MB/rank
  skip/nan: 0/0
```

MFU estimate for that full-duration no-recompute probe:

```text
useful train FLOPs/sample: 6199.37 TFLOP
8 * H800 BF16 peak:        8 * 989 TFLOP/s
step:                      33.7934 s
useful MFU:                2.32 %
HFU without recompute:     2.32 %
```

Conclusion: TP-local QKV with TP-aware full-hidden Q/K RMSNorm is mathematically consistent with official Wan's RMSNorm definition and makes full 513-frame no-recompute training fit under `TP2+CP4` on 8 H800s. It is not yet a better production setting: the tested no-recompute topology is much slower than the selected `CP4+DP2` full-recompute path (`33.79s` for batch 1 vs `7.76s` for batch 2). Keep it experimental until a faster topology/partial recompute setting is measured.
