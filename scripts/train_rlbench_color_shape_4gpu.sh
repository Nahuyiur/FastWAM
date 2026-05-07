#!/usr/bin/env bash
set -euo pipefail

cd /mnt/world_foundational_model/yuhan/FastWAM
exec bash scripts/train_rlbench_4gpu.sh rlbench_color_shape_3cam224_1e-4
