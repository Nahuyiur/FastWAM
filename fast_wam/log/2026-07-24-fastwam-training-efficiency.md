# Fast-WAM Megatron training efficiency implementation

Last updated: 2026-07-25

## Objective

Raise TP1+DP8 Fast-WAM training throughput/MFU without changing checkpoint
layout or the accepted inference path. Audit and address attention masking,
normalization/RoPE precision overhead, unfused MLP activation, frozen-encoder
work in the hot path, DataLoader churn, and distributed-optimizer overlap.

Final status: implementation, unit/MCore numerical gates, full cache build,
and 8-card online/cached performance gates are complete. A new optimized
20k-training plus 2k-rollout convergence gate remains intentionally pending,
so reference stays the base-launcher default.

## Baseline and bottleneck evidence

The accepted 20k run was about 2.95 s/step. A fresh same-machine 12-step
reference run, excluding iterations 1-2 initialization, measured:

- median 2.8877 s/step;
- mean 2.8877 s/step;
- 44.33 samples/s at global batch 128.

Shape-accurate PPU forward/backward microbenchmarks found:

| Operation | Reference | Candidate |
| --- | ---: | ---: |
| joint attention, bool mask vs three exact rectangles | 4.455 ms | 2.458 ms |
| all-true cross attention mask vs `None` | 2.444 ms | 1.071 ms |
| FP64-complex vs complex64 RoPE | 0.625 ms | 0.367 ms |
| explicit FP32 vs BF16-input LayerNorm | 0.428 ms | 0.122 ms |
| manual FP32 vs native TP1 RMSNorm | 1.295 ms | 0.152 ms |
| three QKV linears vs one packed QKV | 7.696 ms | 7.307 ms |

QKV packing was therefore not implemented: its isolated gain was only 5% of
the projection operation and it would substantially enlarge checkpoint-loader
and TP compatibility risk.

Shape-level FLOP accounting also explains why replacing attention alone could
not produce a large end-to-end gain: video-expert MLPs account for about
54.43% of useful model FLOPs, while the QK/AV attention matmuls are below 2%.
The optimization therefore targets mask overhead, norms/RoPE, MLP pointwise
fusion, input encoding, and communication together.

Useful model work is approximately 8.52455 TF/sample. The same-device BF16
`8192^3` GEMM roofline measured 135.479 TF/s/device, so the fresh reference
empirical MFU is approximately 34.86%.

### SDPA Flash dispatch versus Transformer Engine

The five real batch-16 training shapes were checked with BF16
forward/backward:

```text
V0 self:       q=98,  kv=98,  heads=24, head_dim=128
Vf attention:  q=196, kv=294, heads=24, head_dim=128
Action:        q=32,  kv=130, heads=8,  head_dim=128
Video cross:   q=294, kv=129, heads=24, head_dim=128
Action cross:  q=32,  kv=129, heads=8,  head_dim=128
```

All five passed when PyTorch was restricted to
`SDPBackend.FLASH_ATTENTION`. Repeated aggregate timings were:

| Dispatch | Median forward + backward |
| --- | ---: |
| PyTorch SDPA auto | 3.881-3.883 ms |
| PyTorch forced Flash | 3.882-3.883 ms |
| Explicit Transformer Engine FlashAttention | 5.33 ms |

The auto and forced-Flash results are indistinguishable at measurement
precision, while explicit TE is about 37% slower in aggregate. TE is nearly
even on the larger video rectangles but loses on the short action rectangles.
The selected performance path therefore remains unmasked structured SDPA with
auto dispatch. No forced-backend wrapper or TE dependency was added.

## Implementation

### Model structure decisions

The full 6.021B BF16 model plus distributed optimizer fits one PPU at local
microbatch 16. TP1+DP8 is therefore intentional: TP2/4 would split already
large GEMMs while introducing Q/K RMSNorm and row-parallel collectives, and
would reduce the number of independent data replicas. PP/CP and activation
recompute similarly add scheduling or recomputation overhead without solving
a memory blocker in the accepted topology.

Video and action experts have different hidden/FFN dimensions and distinct
weights, so their projections cannot be collapsed into one checkpoint-neutral
GEMM. The layer-wise MoT execution remains interleaved to preserve the
official attention graph. Module registration/checkpoint hierarchy also
remains unchanged even though that makes Megatron's parameter-gather overlap
warning conservative. These are deliberate correctness choices rather than
unexplored optimizations.

### Training kernel profile

New public controls:

```text
--fast-wam-attention-backend {sdpa,flex,structured_sdpa}
--fast-wam-kernel-mode {reference,optimized}
```

`structured_sdpa` implements the exact joint mask as three unmasked dense
attention calls:

```text
Q(V0) -> KV(V0)
Q(Vf) -> KV(V0 + Vf)
Q(A)  -> KV(V0 + A)
```

The optimized mode adds:

- no context mask when the official all-visible context contract is proven by
  the CPU batch;
- BF16-input LayerNorm;
- native `torch.nn.functional.rms_norm` on TP1, with the existing
  tensor-parallel implementation retained for TP>1;
- FP32/complex64 RoPE intermediates;
- Megatron fused bias-GELU using `skip_bias_add`, without changing any
  parameter key or shape.

Reference remains the base launcher default. Inference continues to use the
accepted FP32 norms and original FP64-complex RoPE.

### Data path

The dataset now preloads the 40 UMT5 contexts once and uses persistent workers.
The parquet LRU default in the launcher increased from 4 to 32. When a latent
cache is selected, all state/action/timestamp parquet payloads (about 18 MB)
are materialized before workers fork, so random training batches no longer
parse parquet files in the steady-state path.

The offline latent path consists of:

- `fast_wam/train/prepare_latents.py`;
- `fast_wam/scripts/prepare_libero_latents.sh`;
- memory-mapped BF16 shard reads in `fast_wam/train/data.py`;
- `--fast-wam-latent-cache` in the Megatron entry.

The builder decodes and preprocesses both camera streams once per episode,
then batches all overlapping nine-frame windows through the VAE. This removes
the original prototype's repeated MP4 open/decode for every global frame.
Episode start, middle, and clamped-tail windows are bitwise identical to the
sample-wise preprocessing path (maximum pixel error 0).

The cache contract fingerprints official suite metadata, every parquet/video
relative name and byte size, stats, preprocessing semantics, and VAE SHA-256.
Shards are written with temporary files, `fsync`, and atomic rename. A
`manifest.json` is emitted only after all expected shard sizes pass.

### Distributed optimizer

`OVERLAP_PARAM_GATHER=1` exposes Megatron's parameter all-gather overlap. The
model's video/action subtree registration order does not exactly match its
interleaved MoT forward order, so Megatron emits a pre-dispatch warning for
some buckets. It still provided a small measured benefit; changing module
registration or checkpoint hierarchy solely to eliminate the warning was not
accepted.

`fast_wam/scripts/train_libero_optimized.sh` selects the accepted performance
profile and requires a complete latent cache. The original launcher remains a
reference-compatible fallback.

## Validation

CPU:

```bash
PYTHONPATH=. FAST_WAM_DISABLE_MCORE=1 \
pytest -q fast_wam/tests/test_fast_wam.py
```

Result: **14 passed**. Coverage includes upstream tiny inference/training
parity, Flex parity, structured optimized forward/backward parity, identical
state-dict keys, and latent manifest/mmap reading.

Real PPU BF16 MCore:

```bash
CUDA_VISIBLE_DEVICES=0 PYTHONPATH=. \
torchrun --standalone --nproc_per_node 1 \
fast_wam/tests/gpu_training_kernel_smoke.py
```

Result:

```text
loss_reference=7.658243179
loss_optimized=7.658243179
loss_error=0
grad_max_error=0.001220703
```

Full 6.021B-model TP1+DP8 online-data A/B, excluding iterations 1-2:

| Profile | Median | Mean | Throughput | Empirical MFU |
| --- | ---: | ---: | ---: | ---: |
| reference SDPA | 2.8877 s | 2.8877 s | 44.33 samples/s | 34.86% |
| structured + optimized | 2.6187 s | 2.6305 s | 48.88 samples/s | 38.44% |
| + parameter-gather overlap | 2.5899 s | 2.6047 s | 49.42 samples/s | 38.87% |
| + offline latent cache | 1.5562 s | 1.5557 s | 82.25 samples/s | 64.69% |

The final online profile reduces median step time by 10.31%, or raises
throughput by 11.50%, without relying on the offline cache.
The completed cached profile reduces median step time by 46.11%, or raises
throughput by 85.57%, relative to reference. Relative to the optimized online
profile, cached throughput is 66.43% higher.

Raw A/B logs are intentionally outside Git:

```text
/tmp/fastwam_ref_ab.log
/tmp/fastwam_opt_ab.log
/tmp/fastwam_opt_pg_ab.log
```

## Cache artifact

The full cache was completed with eight PPUs at:

```text
/mnt/world_foundational_model/ruibin/data/Fast-WAM/cache/libero_mujoco3.3.2_wan22_bf16
```

The completed manifest contains 277,713 samples in 272 shards:

```text
31,352,686,848 bytes (29.199465 GiB)
sample dtype/shape: bfloat16 [48,3,14,28]
bad-size shards: 0
temporary shards: 0
```

Three real 16-sample batches were recomputed through the online Wan VAE,
covering the original prototype shard, episode-wise batch-16 output, and the
later batch-64 builder output. All 48 samples were bitwise exact to cache:
maximum/mean error 0 and zero unequal BF16 elements. The cached full-model
step-1 loss was also exactly the online optimized value `4.026691`.

The longer 30-step cached TP1+DP8 run had 28 steady observations after
iterations 1-2: median `1.55615 s`, mean `1.555654 s`, range
`1.5497–1.5629 s`, no skipped or NaN iterations. During its steady section,
12 one-second samples across eight devices (96 observations) measured average
utilization `99.67%`, average power `314.51 W/device`, and maximum power
`336.08 W`. The accepted formal reference telemetry was approximately 96.8%,
291.7 W/device average, and 330.7 W maximum. Raw outputs are
`/tmp/fastwam_cached_ab.log` and `/tmp/fastwam_cached_power.log`.

## Formal optimized 20k training and 2k evaluation

Status: **complete; quality target not met**. The resume-safe chained
experiment was launched at 2026-07-25 02:19 UTC+8
(2026-07-24 18:19 UTC):

```bash
bash fast_wam/scripts/run_libero_optimized_20k_and_eval.sh
```

Configuration:

```text
topology: TP1 + DP8
precision: BF16
micro/global batch: 16 / 128
train iterations: 20,000
attention: structured_sdpa with auto dispatch
kernels: optimized
parameter gather overlap: enabled
data: validated 29.199 GiB offline latent cache
checkpoint interval: 2,000 steps
post-training gate: four suites, 2,000 LIBERO episodes, resume enabled
```

Artifacts:

```text
outputs/fast_wam_libero_training_20k_optimized_20260725/
outputs/fast_wam_libero_training_20k_optimized_20260725_eval_2k/
```

The first optimizer step reproduced the accepted cached smoke exactly:

```text
loss=4.026691
video_loss=2.693185
action_loss=1.333506
grad_norm=7.460
skipped=0
nan=0
```

The step-20 steady logging window was `1.5579 s/step`, consistent with the
`1.5562 s` performance gate. All eight devices reported 100% utilization;
the sampled per-device power range was approximately 296-326 W and memory was
about 72-73 GiB. Raw 30-second telemetry appends to
`ppu_metrics.log` under the training artifact.

Ten 79 GiB checkpoints were saved at the 2,000-step interval, versus 40 in the
old 500-step run. Training completed at 2026-07-25 11:02 UTC+8 with no skipped
or NaN iteration. Final train loss was `0.1000099`; deterministic
validation-set loss was `0.08776479`, split into video `0.08159905` and action
`0.006165743`. The final rollout case was written at 14:38 UTC+8, for about
8 hours 43 minutes of training and 3 hours 34 minutes of evaluation.

Final 2,000-episode result:

| Suite | Optimized 20k | Reference 20k | Local release DCP | Paper target |
| --- | ---: | ---: | ---: | ---: |
| Spatial | 479/500 | 478/500 | 485/500 | 491/500 |
| Object | 495/500 | 497/500 | 497/500 | 500/500 |
| Goal | 489/500 | 491/500 | 486/500 | 485/500 |
| LIBERO-10 | 463/500 | 457/500 | 470/500 | 476/500 |
| **Overall** | **1,926/2,000** | **1,923/2,000** | **1,938/2,000** | **1,952/2,000** |

The optimized model is 3 episodes / 0.15 percentage points above the
reference 20k run, but 12 episodes below the local release DCP and 26 episodes
below the paper target. Only Goal meets its paper suite target. This is a
performance-path quality gate, not a successful paper-accuracy reproduction.

All 2,000 atomic cases and the final `summary.json` exist. The evaluator
returned exit code 1 only after aggregation because `meets_target=false`;
this is the designed quality-gate status, not a rollout, NCCL, or aggregation
failure.

The requested next sequence was to commit and push these changes, then launch
a fresh official-length 21,700-step optimized run with:

```bash
bash fast_wam/scripts/run_libero_optimized_21700.sh
```

That run uses a separate
`outputs/fast_wam_libero_training_21700_optimized_20260725/` root and saves at
2,170-step epoch boundaries.

## Formal optimized 21.7k launch

The optimized implementation and the completed 20k/2k results were committed
as `0389d95`, and the launch record as `fae150f`. The first push attempt was
blocked because no usable GitHub SSH key was available, so the user explicitly
requested proceeding with the run. After the workspace key became available,
both commits were successfully pushed to `origin/dev` on 2026-07-25.

The fresh official-length run started at 2026-07-25 14:48 UTC+8:

```bash
bash fast_wam/scripts/run_libero_optimized_21700.sh
```

Operational handles:

```text
tmux socket: fastwam_opt_21700_20260725
tmux session: fastwam_opt_21700
artifact: outputs/fast_wam_libero_training_21700_optimized_20260725/
```

The actual torchrun command was checked for `--train-iters 21700`,
`--lr-decay-iters 21700`, `--save-interval 2170`, latent-cache input,
`structured_sdpa`, optimized kernels, parameter-gather overlap, and eight
TP1+DP8 ranks.

The first step matched the accepted fresh cached baseline exactly:

```text
loss=4.026691
video_loss=2.693185
action_loss=1.333506
grad_norm=7.460
skipped=0
nan=0
```

The step-20 and step-30 logging windows were 1.5572 and 1.5563 s/step. A
14:51 UTC+8 telemetry sample reported 100% utilization on all eight PPUs,
297--334 W per device, and about 72--73 GiB memory. At this throughput,
including observed startup/checkpoint overhead, training is expected to take
about 9.4 hours and finish near 2026-07-26 00:15 UTC+8 if uninterrupted.

## Automatic eval attachment

At the user's request, `run_libero_optimized_21700.sh` now launches the
resume-safe full 2,000-episode evaluation after successful training. Its
default eval artifact is:

```text
outputs/fast_wam_libero_training_21700_optimized_20260725_eval_2k/
```

The current training shell was already running the earlier launcher inode, so
editing the launcher alone could not safely alter that live shell. A separate
tmux session was attached on the same `fastwam_opt_21700_20260725` socket. It
waits for the training session to exit, then requires the checkpoint tracker
to equal exactly 21,700 before starting the resumable evaluator. A crash or
partial checkpoint therefore fails closed instead of evaluating an incomplete
model. Watcher output is written to `eval_watcher.log` in the training artifact.

The training completed at 2026-07-26 00:18 UTC+8. Step 21,700 had loss
`0.08789310`, split into video `0.08216834` and action `0.005724775`, with no
skipped or NaN iteration. The final deterministic validation loss was
`0.1158151`, split into video `0.1149315` and action `0.0008836285`. The final
79 GiB DCP save completed successfully and the tracker is exactly 21,700.

The initial live watcher name, `fastwam_opt_21700_eval_watcher`, shared the
training session's `fastwam_opt_21700` prefix. tmux target lookup accepted that
prefix, so after training exited the watcher incorrectly matched its own
session and continued waiting. At 2026-07-26 09:01 UTC+8 the live session was
renamed to `eval_watcher_21700`, which immediately released the gate and
started the full 2,000-episode eval. The reusable watcher now enumerates tmux
session names and compares them exactly instead of using ambiguous
`has-session -t` lookup.

## Formal optimized 21.7k evaluation

The full evaluation completed at 2026-07-26 12:48 UTC+8. All 2,000 atomic case
files and the final `summary.json` exist:

```text
outputs/fast_wam_libero_training_21700_optimized_20260725_eval_2k/summary.json
```

| Suite | Optimized 21.7k | Optimized 20k | Reference 20k | Local release DCP | Paper target |
| --- | ---: | ---: | ---: | ---: | ---: |
| Spatial | 477/500 | 479/500 | 478/500 | 485/500 | 491/500 |
| Object | 498/500 | 495/500 | 497/500 | 497/500 | 500/500 |
| Goal | 485/500 | 489/500 | 491/500 | 486/500 | 485/500 |
| LIBERO-10 | 453/500 | 463/500 | 457/500 | 470/500 | 476/500 |
| **Overall** | **1,913/2,000** | **1,926/2,000** | **1,923/2,000** | **1,938/2,000** | **1,952/2,000** |

The 21.7k run scored 95.65%. It is 13 episodes / 0.65 percentage points below
the optimized 20k run, 10 episodes below the reference 20k run, 25 below the
local release DCP, and 39 below the paper target. Only Goal exactly meets its
paper suite target. The largest task-level deficits are LIBERO-10 tasks 6 and
7 at 40/50 each, LIBERO-10 task 4 at 42/50, and Spatial task 8 at 43/50.

The evaluator returned exit code 1 after successfully writing the complete
summary because `meets_target=false`. This is the intended quality-gate exit
status, not an evaluation, PCCL, or aggregation failure.

## Limitations and remaining gates

- The optimized path completed its 20k plus 2,000-episode gate at 96.30%.
  This is slightly above the reference 20k result but below the local release
  DCP and paper target.
- The official-length optimized 21.7k path completed its 2,000-episode gate at
  95.65%, below both 20k runs, the local release DCP, and the paper target.
- The full-model A/B is a short performance/numerical smoke, not a convergence
  equivalence proof.
- MFU is empirical against the measured same-device GEMM roofline, not vendor
  theoretical peak.
- 400 W is a limit, not a target. The accepted formal reference telemetry was
  already 96.8% average utilization but only about 291.7 W average power;
  useful step time/MFU is the primary criterion.
