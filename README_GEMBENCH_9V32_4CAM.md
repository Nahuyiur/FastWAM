# GEMBench 9V32 4cam FastWAM Branch Contract

This file describes the current production path on branch
`codex/gembench-wam-9v32`. It is intentionally narrower than
`README_GEMBENCH.md`, which still contains older GEMBench `keysteps_bbox` and
3-camera notes.

## Scope

Current training target:

```text
GEMBench train microsteps
  -> 9 visual anchors: t,t+4,t+8,t+12,t+16,t+20,t+24,t+28,t+32
  -> 32 executable 8D actions: t+1 ... t+32
  -> FastWAM 9V/32A training sample
```

Camera contract:

```text
cache_camera_order = left_shoulder,right_shoulder,wrist,front
camera_order       = left_shoulder,right_shoulder,wrist,front
video_size         = 224 x 896
```

This is a FastWAM-style WAM adaptation for GEMBench. GEMBench closed-loop eval
still receives one executable 8D action per simulator step; the evaluator may
use the first action from the predicted chunk, but training remains `9V/32A`.

## Current Files

- Data config: `configs/data/gembench_microsteps_9v32_4cam224.yaml`
- Task config: `configs/task/gembench_microsteps_9v32_4cam224_1e-4.yaml`
- Dataset: `src/fastwam/datasets/gembench/microsteps_9v32.py`
- Shard manifest builder: `scripts/build_gembench_microsteps_9v32_4cam224_shards.sh`
- RGB render shard wrapper: `scripts/render_gembench_microsteps_9v32_4cam224_shard.sh`
- RGB finalize/audit wrapper: `scripts/finalize_gembench_microsteps_9v32_4cam224_rgb_cache.sh`
- VAE latent precompute: `scripts/precompute_gembench_microsteps_9v32_4cam224_vae_latents_4gpu.sh`
- Training wrapper: `scripts/train_gembench_microsteps_9v32_4cam224_4gpu.sh`

Scripts/configs without `_4cam224` are legacy 3-camera/debug paths unless a run
explicitly opts into them.

## Required Pipeline

Run in this order:

```bash
cd /mnt/yuhan/FastWAM

# 1. Build balanced full/shard manifests.
NUM_SHARDS=24 bash scripts/build_gembench_microsteps_9v32_4cam224_shards.sh

# 2. Render a small visual smoke before full cache work.
bash scripts/render_gembench_microsteps_9v32_4cam224_smoke.sh

# 3. Render each shard. The shard wrapper intentionally uses
# GEMBENCH_9V32_4CAM_SHARD_MANIFEST, not GEMBENCH_9V32_4CAM_MANIFEST.
SHARD_ID=0 NUM_SHARDS=24 bash scripts/render_gembench_microsteps_9v32_4cam224_shard.sh

# 4. Finalize the train manifest only after rendered RGB files pass audit.
bash scripts/finalize_gembench_microsteps_9v32_4cam224_rgb_cache.sh

# 5. Precompute the matching 4cam 9V32 Wan VAE latent cache.
CUDA_VISIBLE_DEVICES=0,1,2,3 NUM_SHARDS=4 \
  bash scripts/precompute_gembench_microsteps_9v32_4cam224_vae_latents_4gpu.sh

# 6. Train only after RGB + VAE gates pass.
bash scripts/train_gembench_microsteps_9v32_4cam224_4gpu.sh
```

Default production paths:

```text
shards:   /mnt/yuhan/datasets/GEMBench/fastwam_cache/microsteps_9v32_4cam224_shards_24
RGB:      /mnt/yuhan/datasets/GEMBench/fastwam_cache/microsteps_9v32_4cam224_rgb
manifest: /mnt/yuhan/datasets/GEMBench/fastwam_cache/microsteps_9v32_4cam224_manifest.json
VAE:      /mnt/yuhan/datasets/GEMBench/fastwam_cache/vae_latents/microsteps_9v32_seed0_4cam224x896_t9_a32_v1
```

## Hard Gates

Do not launch full training unless all are true:

- The finalized manifest exists and points to rendered 4cam RGB cache files.
- RGB cache audit passes for the finalized manifest.
- VAE manifest exists, has `cache_version=gembench_microsteps_9v32_vae_latents_v1`,
  has `complete=true`, and records `video_size=[224,896]`.
- VAE manifest records both `camera_order` and `cache_camera_order` as
  `left_shoulder,right_shoulder,wrist,front`.
- Dataset samples have action target shape `[32,8]`.
- Missing RGB/VAE cache is treated as a hard failure; no fallback to sparse
  `keysteps_bbox` or legacy `9V/8A` is allowed.

## Runtime Artifacts

Do not commit generated files under `logs/`, `runs/`, `wandb/`,
`eval10_preview_downloads/`, rendered RGB cache, VAE cache, checkpoints, or
videos. These are intentionally ignored by `.gitignore`.
