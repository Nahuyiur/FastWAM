# Megatron Fast-WAM BF16 LIBERO-Spatial evaluation

Date: 2026-07-23

## Objective and final status

Run the Megatron Fast-WAM checkpoint in BF16 on the same local 50-episode
LIBERO-Spatial integration protocol previously used by the official Fast-WAM repository,
then compare against its 47/50 (94%) result.

Status: **passed**. Megatron scored **48/50 (96%)**, exceeding the 94% target.

## Protocol

- 10 `libero_spatial` tasks; official init states 0–4 for each task.
- Seed 42, 30 reset no-op steps, at most 400 policy steps.
- Two cameras resized and concatenated to 224×448.
- Action horizon 32, 10 FlowMatch inference steps, replan every 10 simulator steps.
- BF16 model/VAE/UMT5, FP32 SDPA attention path, binary gripper action.
- MuJoCo 3.1.6, robosuite 1.4.0, OSMesa.
- Hardware/topology: 8 × PPU-ZW810E, TP2+DP4.

The fixed protocol is serialized in
`fast_wam/eval/manifest_libero_spatial_5trials.json`.

## Checkpoint path

Source checkpoint:

```text
/mnt/world_foundational_model/ruibin/checkpoints/Fast-WAM/lerobot/fastwam_libero_uncond_2cam224
```

It was converted with real MCore distributed checkpointing to:

```text
outputs/fast_wam_dcp_bf16_tp2_20260723/
```

The DCP is about 12 GiB: two `.distcp` shards of about 6 GiB each plus metadata.
The 8-PPU evaluation loaded this DCP through `load_megatron_dcp`; it did not use the
safetensors direct-load path for model weights.

Conversion command:

```bash
CUDA_VISIBLE_DEVICES=0,1 torchrun --standalone --nproc_per_node=2 \
  -m fast_wam.eval.convert_to_dcp \
  --checkpoint /mnt/world_foundational_model/ruibin/checkpoints/Fast-WAM/lerobot/fastwam_libero_uncond_2cam224 \
  --output outputs/fast_wam_dcp_bf16_tp2_20260723 \
  --tp 2 --dtype bfloat16
```

## Evaluation command

The full environment and command are preserved by
`fast_wam/scripts/run_libero_spatial_bf16.sh`. The core invocation was:

```bash
torchrun --standalone --nproc_per_node=8 \
  -m fast_wam.eval.evaluate_libero \
  --checkpoint /mnt/world_foundational_model/ruibin/checkpoints/Fast-WAM/lerobot/fastwam_libero_uncond_2cam224 \
  --dcp outputs/fast_wam_dcp_bf16_tp2_20260723 \
  --assets /mnt/world_foundational_model/ruibin/checkpoints/Fast-WAM/lerobot/hf_home/hub/models--Wan-AI--Wan2.2-TI2V-5B-Diffusers/snapshots/b8fff7315c768468a5333511427288870b2e9635 \
  --tokenizer /mnt/world_foundational_model/ruibin/checkpoints/Fast-WAM/lerobot/fastwam_libero_uncond_2cam224/google/umt5-xxl \
  --manifest fast_wam/eval/manifest_libero_spatial_5trials.json \
  --output outputs/fast_wam_megatron_dcp_bf16_spatial_5trials_20260723 \
  --tp 2 --dtype bfloat16 --n-action-steps 10 --target-success-rate 0.94
```

Wall-clock interval was approximately 12:54–13:09 local time, about 15 minutes.

## Results

| Task | Successes | Rate |
|---:|---:|---:|
| 0 | 5/5 | 100% |
| 1 | 5/5 | 100% |
| 2 | 5/5 | 100% |
| 3 | 5/5 | 100% |
| 4 | 5/5 | 100% |
| 5 | 5/5 | 100% |
| 6 | 5/5 | 100% |
| 7 | 5/5 | 100% |
| 8 | 5/5 | 100% |
| 9 | 3/5 | 60% |
| **Total** | **48/50** | **96.00%** |

Failures: `spatial-t9-i0` and `spatial-t9-i1`, both at the 400-step limit.

Official local comparison:

- Official standalone Fast-WAM BF16: 47/50 (94%).
- Megatron Fast-WAM BF16 DCP: 48/50 (96%).
- Megatron meets and exceeds the specified standard by one episode / two percentage points.

The per-episode outcome vector differs from the standalone run. This is expected for
long simulator trajectories under different BF16/TP reduction orders and does not imply
that 96% is a statistically significant improvement over 94%.

## Code changes

- Added rollout protocol overrides and a shared distributed rollout in `fast_wam/libero.py`.
- Added `fast_wam/eval/evaluate_libero.py`.
- Added the fixed 50-case manifest.
- Added clean process-group shutdown to DCP conversion.
- Added the reproduction script and Chinese report.

## Validation and artifacts

- CPU tests: 5/5 passed.
- `pyflakes`, `compileall`, CLI `--help`, and `git diff --check` passed before the run.
- Final summary:
  `outputs/fast_wam_megatron_dcp_bf16_spatial_5trials_20260723/summary.json`.
- Incremental per-DP results: `dp_0.json` through `dp_3.json` in the same directory.
- DCP and evaluation outputs are ignored by Git.
- PyTorch remained `2.9.0+ali.10.ppu2.0.0.cu129`; Transformer Engine remained
  `2.8+ppu2.0.0.oe`. Neither package was modified.

## Limitations

- This is the local 50-episode integration benchmark, not the complete 2,000-episode
  paper protocol.
- The local simulator uses MuJoCo 3.1.6 while the training data config names 3.3.2.
- No rollout videos were saved; accuracy, steps, and per-case outcomes are in JSON.
