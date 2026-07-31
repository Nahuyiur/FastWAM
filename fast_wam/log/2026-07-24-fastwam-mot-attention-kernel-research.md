# Fast-WAM MoT attention kernel research

Last updated: 2026-07-24

## Objective

Survey current exact implementations for structured MoT attention masks, with
particular attention to training backward support and applicability to the
PPU-ZW810E Fast-WAM run.

## Local mask geometry

The formal recipe has 326 tokens per sample:

- first-frame video `V0`: 98
- future video `Vf`: 196
- action `A`: 32

The allowed attention rectangles are:

- `Q(V0) -> KV(V0)`
- `Q(Vf) -> KV(V0 + Vf)`
- `Q(A) -> KV(V0 + A)`

This is 71,388 allowed query/key pairs out of 106,276 dense pairs, or 67.17%
density. The ideal attention-matmul saving is therefore 32.83%; it is not an
end-to-end training speedup estimate.

For block masks that retain a tile whenever it contains at least one allowed
element, the current token order has the following geometric execution
fractions:

| Square block size | Score area in retained tiles |
| ---: | ---: |
| 128 | 100.00% |
| 64 | 86.27% |
| 32 | 80.13% |
| 16 | 73.05% |
| ideal token-level mask | 67.17% |

The default 128 block therefore has no sparse QK/PV tile saving for this short
sequence: all nine tiles contain some allowed elements.

## Survey result

- PyTorch FlexAttention is the most complete general PyTorch interface:
  arbitrary `mask_mod`, block-sparse iteration, and forward/backward. Its
  default block size is 128. Recent NVIDIA Hopper/Blackwell releases can use a
  FlashAttention-4 backend, but that backend is not a PPU path.
- BAGEL training uses compiled FlexAttention with a 128-block `BlockMask`.
  This is a good implementation reference, but its block geometry does not
  transfer efficiently to the 326-token Fast-WAM layout.
- MIT Han Lab Block-Sparse-Attention provides exact forward/backward kernels
  derived from FlashAttention-2, but uses 128-token blocks and CUDA/CUTLASS.
- Paddle FlashMask uses a compact column-boundary representation and supports
  training, plus a 128-block mask API. It is implemented in Paddle and is not a
  drop-in PyTorch/PPU operator.
- Transformer Engine arbitrary masks fall back to unfused attention in the
  documented path. Converting the mask to `post_scale_bias` can retain a fused
  cuDNN path on supported NVIDIA configurations, but remains dense compute and
  is not a PPU sparse solution.
- FlashInfer has fixed and variable block-sparse APIs and log-sum-exp state
  merging for shared-prefix decomposition, but its published APIs are
  inference-oriented rather than a training-backward contract.
- `flash-sparse-attn` 2.0.5 now publishes PPU-ZW810E forward/backward
  benchmarks and a dense varlen training API. Its current sparse API is
  threshold/gating sparsity, not a fixed arbitrary MoT mask. The old arbitrary
  mask branch states NVIDIA-only requirements. Version 2.0.5 also requires
  Triton 3.6, while the preserved local stack has Triton 3.5.
- The local immutable environment already has PPU FlashAttention
  `2.7.4.post1`; `flash_attn_varlen_func` is present and its autograd wrapper
  implements backward.

## Recommended experiment

Compile the mask graph into three exact, unmasked dense cross-attention calls
over the already-projected Q/K/V tensors:

```text
O0 = Attention(Q0, K0, V0)
Of = Attention(Qf, Kvideo, Vvideo)
Oa = Attention(Qa, concat(K0, Ka), concat(V0, Va))
```

This computes exactly the 71,388 allowed pairs and exposes only supported dense
attention primitives to the PPU fused backend. The first implementation should
use three regular calls because most K/V inputs remain views; a second variant
can pack the three rectangles as varlen cross-attention if launch overhead
dominates. Gradient accumulation through repeated/concatenated K/V must be
checked against the SDPA reference.

Do not modify the active formal training run. Benchmark after it releases the
devices, using forward/backward parity, peak memory, attention-only latency,
full-layer latency, and end-to-end step time. The practical winner cannot be
selected from sparsity ratios alone because the sequence is short and three
kernel launches or K/V packing may dominate.

## Primary references

- PyTorch FlexAttention documentation and FlashAttention-4 integration
- Dao-AILab FlashAttention API
- MIT Han Lab Block-Sparse-Attention
- PaddlePaddle FlashMask API and ICLR 2025 paper
- NVIDIA Transformer Engine attention guide
- FlashInfer sparse and cascade APIs
- HKUSTDial Flash-Sparse-Attention API
- Alibaba Cloud ZHENWU PPU training-image component matrix

## Changes and validation

- Added this research log only; no model, environment, or training changes.
- Verified local versions and API presence without importing an accelerator
  context: PyTorch `2.9.0+ali.10.ppu2.0.0.cu129`, Triton `3.5.0`,
  FlashAttention `2.7.4.post1`, varlen API present with autograd backward.
- No PPU benchmark was run because all eight devices are occupied by the
  formal 20,000-step training job.
