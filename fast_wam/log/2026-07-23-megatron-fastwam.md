# Megatron Fast-WAM inference patch

Date: 2026-07-23

## Objective and status

Implement a minimal inference-only Fast-WAM overlay under `Megatron-Wan/fast_wam` with
LeRobot-compatible LIBERO preprocessing/action semantics, Megatron Core TP, DP episode
sharding, direct LeRobot safetensors loading, optional Megatron DCP, and a reproducible
8-episode cross-suite consistency gate.

Status: complete. The implementation, released 6B checkpoint loading, real PPU TP1/2/4,
TP2+DP2 action parity, and the fixed 8-episode four-suite closed-loop gate were run.

The default command sandbox did not expose accelerator device nodes, but unrestricted
worker processes did. The existing environment was preserved: PyTorch
`2.9.0+ali.10.ppu2.0.0.cu129` and Transformer Engine `2.8+ppu2.0.0.oe`; neither package
was installed, removed, upgraded, or downgraded.

## Design

- No changes under `megatron/`; the implementation is a standalone patch/overlay.
- `infer_action` only: first-frame video K/V is prefetched once and reused for 10 action
  FlowMatch denoising steps.
- Megatron Core `ColumnParallelLinear`/`RowParallelLinear` cover embeddings, Q/K/V/O,
  FFN, action/proprio encoders, and heads. TP Q/K RMSNorm reduces full-dimension squared
  norms across the TP group.
- Frozen VAE/UMT5 and the LIBERO environment are instantiated only on TP rank 0. Encoded
  latent/context/state tensors are broadcast within the TP group; DP groups receive
  different manifest cases.
- First-frame inference has one video temporal group. The effective mask is dense, so
  FP32 SDPA is retained for exact LeRobot parity; FlexAttention is deferred until a
  multi-frame/training mask requires it.
- Direct safetensors loading streams one tensor at a time and slices the local TP shard.
  DCP conversion/loading supports resharding to another TP topology.

## Acceptance contract

- Manifest: `fast_wam/eval/manifest.json`, seed 20260723, two random cases from each of
  `libero_spatial`, `libero_object`, `libero_goal`, and `libero_10`.
- Replay complete action chunks on observations captured from LeRobot rollouts.
- FP32 action maximum absolute error must be at most `1e-3`; gripper sign is exact.
- Closed loop requires the Megatron success count not to regress from LeRobot. Exact
  vectors and per-case mismatches remain in `summary.json` for diagnosis. They are not a
  hard gate because sub-`1e-4` floating-point changes can bifurcate a hundreds-step
  MuJoCo trajectory.
- Required topology gate: TP2+DP2.

## Files

- Core: `config.py`, `mcore.py`, `model.py`, `scheduler.py`.
- Checkpoint/distributed runtime: `checkpoint.py`, `distributed.py`.
- Frozen preprocessing and policy semantics: `components.py`, `policy.py`, `libero.py`.
- Evaluation: `eval/export_lerobot_reference.py`, `eval/acceptance.py`,
  `eval/convert_to_dcp.py`, `eval/manifest.json`.
- Tests/docs: `tests/test_fast_wam.py`, `README.md`.
- Packaging: root `pyproject.toml` and `.gitignore`.

## Validation performed

```text
FAST_WAM_DISABLE_MCORE=1 python -m pytest -q fast_wam/tests
5 passed
```

- Tiny end-to-end inference uses the actual sibling LeRobot `WanVideoDiT`, `ActionDiT`,
  `MoT`, and scheduler as the reference. With copied weights and identical latent,
  context, proprio, and CPU-seeded action noise, the complete denoised chunk matches at
  `atol=rtol=1e-5`.
- Streaming checkpoint round-trip passed; TP2 and TP4 slicing unit cases passed.
- Camera ordering/resizing, MIN_MAX transforms, gripper toggle, and schedule passed.
- The released 6B checkpoint was inspected through a meta-device model: all 1,651 source
  tensors match one-for-one, with zero missing/unexpected keys and zero shape mismatches.
- `compileall`, `pyflakes`, all three CLI `--help` paths, and `git diff --check` passed.
- `ruff` was unavailable in the environment.

Real accelerator validation:

- Full released checkpoint TP2 inference: `(32, 7)` action output and
  `max_rank_diff=0.0`.
- Full released checkpoint TP4 inference: `(32, 7)` action output and
  `max_rank_diff=0.0`.
- LeRobot FP32 reference, fixed manifest: success vector
  `[T,T,T,T,F,T,T,F]`, 6/8.
- TP1 diagnostic: all eight closed-loop cases succeeded; fixed-observation action maximum
  error `9.236335754394531e-4`, gripper sign exact.
- Required TP2+DP2 final gate: `passed=true`, replay 8/8, gripper 8/8, action maximum
  error `8.370876312255859e-4`, closed-loop 6/8 versus LeRobot 6/8.
- TP2 exact success vector was `[T,F,T,T,F,T,T,T]`; the two explicitly recorded
  long-horizon bifurcations are `spatial-t4-i44` and `libero10-t0-i27`.

Artifacts:

- LeRobot reference: `outputs/fast_wam_reference_20260723/`.
- Final TP2+DP2 result:
  `outputs/fast_wam_tp2_dp2_20260723_final/summary.json`.

Known limitation: the overlay is inference-only. Training, PP/CP/SP, joint video/action
generation, and RoboTwin are intentionally out of scope.
