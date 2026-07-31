# Megatron Fast-WAM BF16 LIBERO 2k Work Log

Last updated: 2026-07-23

## Objective

用 Megatron BF16 DCP 在 Fast-WAM 论文的完整 LIBERO protocol 下评测四个 suite、
40 个 task、2,000 个 episode；保存可恢复入口、最终结果和交接文档。

## Final status

评测已完成，2,000 个原子 case 文件和最终 `summary.json` 均存在：

```text
outputs/fast_wam_megatron_dcp_bf16_libero_2k_20260723/
```

最终结果为 **1938/2000（96.90%）**。论文门槛是
1952/2000（97.60%），因此 `passed=false`，差 14 个成功 episode。

| Suite | Megatron | Paper | Delta |
| --- | ---: | ---: | ---: |
| `libero_spatial` | 485/500（97.0%） | 491/500（98.2%） | -6 |
| `libero_object` | 497/500（99.4%） | 500/500（100.0%） | -3 |
| `libero_goal` | 486/500（97.2%） | 485/500（97.0%） | +1 |
| `libero_10` | 470/500（94.0%） | 476/500（95.2%） | -6 |

同机 LeRobot 完整本地结果是 1922/2000（96.10%）；Megatron 高 16 个 episode，
但长时域 simulator 分叉意味着不能把 aggregate 差异解释为逐轨迹数值等价或模型提升。

## Per-task results

下表每格都是成功数 `/50`，列为 task 0–9：

| Suite | t0 | t1 | t2 | t3 | t4 | t5 | t6 | t7 | t8 | t9 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Spatial | 49 | 50 | 48 | 50 | 47 | 47 | 50 | 49 | 47 | 48 |
| Object | 49 | 50 | 50 | 50 | 50 | 48 | 50 | 50 | 50 | 50 |
| Goal | 50 | 47 | 50 | 48 | 46 | 50 | 49 | 50 | 50 | 46 |
| LIBERO-10 | 44 | 50 | 46 | 47 | 45 | 50 | 45 | 47 | 48 | 48 |

## Configuration

- source checkpoint:
  `/mnt/world_foundational_model/ruibin/checkpoints/Fast-WAM/lerobot/fastwam_libero_uncond_2cam224`
- Megatron DCP:
  `outputs/fast_wam_dcp_bf16_tp2_20260723/`
- load topology: TP1 + DP8 on 8 x PPU-ZW810E
- precision: BF16
- seed 42, reset no-op 30, denoising steps 10, action/replan interval 10
- horizon: 400 for Spatial/Object/Goal, 700 for LIBERO-10
- manifest: `fast_wam/eval/manifest_libero_full_2k.json`
- entry: `fast_wam/scripts/run_libero_full_2k_bf16.sh`

Command:

```bash
bash fast_wam/scripts/run_libero_full_2k_bf16.sh
```

The first case was written at 2026-07-23 05:22:46 UTC and the final missing
case at 09:49:21 UTC. This includes one resume after fixing final aggregation;
it is not a clean throughput benchmark.

## Aggregation recovery

The initial run wrote 1,999 complete case files, then a DP rank waiting in the
final NCCL barrier exceeded the 600-second watchdog because other ranks still
had long 700-step episodes. This was an aggregation-tail failure, not a model
inference failure.

`evaluate_libero.py` was changed to:

- keep atomic per-case JSON writes;
- let global rank 0 poll for all expected files with a configurable timeout;
- aggregate without a final distributed barrier/broadcast;
- atomically replace `summary.json`;
- omit the 2,000-result payload from console output while retaining it in the artifact.

Running the same script with `--resume` then evaluated only the missing
`long-t9-i47` case, which succeeded in 236 steps, and generated the final summary.
The resumed command exited 1 only because the 97.6% accuracy gate did not pass.

## Files changed for this evaluation

- `fast_wam/libero.py`
- `fast_wam/eval/acceptance.py`
- `fast_wam/eval/evaluate_libero.py`
- `fast_wam/eval/convert_to_dcp.py`
- `fast_wam/eval/manifest_libero_spatial_5trials.json`
- `fast_wam/eval/manifest_libero_full_2k.json`
- `fast_wam/scripts/run_libero_spatial_bf16.sh`
- `fast_wam/scripts/run_libero_full_2k_bf16.sh`
- `fast_wam/docs/libero_spatial_bf16_eval_zh.md`
- `fast_wam/docs/libero_full_2k_bf16_eval_zh.md`
- `fast_wam/README.md`
- `AGENTS.md`
- workspace `../AGENTS.md`

## Validation

- Full run: 2,000 unique case JSON files; summary has 2,000 ordered results.
- BF16 TP1+DP8 DCP smoke before the full run: 8/8.
- CPU tests: 5 passed.
- Full manifest expansion: 2,000 unique IDs.
- Static checks: `compileall`, Pyflakes, shell `bash -n`, and `git diff --check`.
- `git diff --name-only -- megatron` is empty.
- PyTorch `2.9.0+ali.10.ppu2.0.0.cu129` and Transformer Engine
  `2.8+ppu2.0.0.oe` were not changed.

## Limitations

- Local evaluation used MuJoCo 3.1.6, robosuite 1.4.0 and OSMesa. Fast-WAM's
  training data config names MuJoCo 3.3.2, so this is not a strict
  environment-matched paper reproduction.
- Overall, Spatial, Object and LIBERO-10 do not meet the paper targets; Goal does.
- Large raw outputs and the DCP remain ignored under `outputs/` and must not be committed.
