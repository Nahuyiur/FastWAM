# Megatron Fast-WAM BF16 LIBERO 2k 评测

Last updated: 2026-07-23

## 结论

Megatron DCP 已按 Fast-WAM 论文的完整 LIBERO protocol 跑完 2,000 个
episode，结果为 **1938/2000（96.90%）**。论文报告为
**1952/2000（97.60%）**，本地结果少 14 个成功 episode、低 0.70 个百分点，
因此严格的论文精度门槛 **未通过**。

同一机器和 simulator stack 上，LeRobot checkpoint 的完整本地结果为
1922/2000（96.10%）。Megatron 没有相对该本地基线回退，但闭环轨迹不能据此声称与
LeRobot 逐 episode 完全相同；固定 observation 的 action parity 才是确定性一致性
门禁。

| Suite | Megatron BF16 | Fast-WAM paper | 差值 |
| --- | ---: | ---: | ---: |
| LIBERO-Spatial | 485/500（97.0%） | 491/500（98.2%） | -6 |
| LIBERO-Object | 497/500（99.4%） | 500/500（100.0%） | -3 |
| LIBERO-Goal | 486/500（97.2%） | 485/500（97.0%） | +1 |
| LIBERO-10 | 470/500（94.0%） | 476/500（95.2%） | -6 |
| **Overall** | **1938/2000（96.9%）** | **1952/2000（97.6%）** | **-14** |

## Protocol

- suites：`libero_spatial`、`libero_object`、`libero_goal`、`libero_10`
- 每个 suite 10 个 task，每个 task 使用 official init state 0–49
- seed 42；reset 后 30 个 no-op
- BF16；10 个 FlowMatch denoising steps；每次执行 10 个 action 后 replan
- Spatial/Object/Goal 最多 400 policy steps；LIBERO-10 最多 700
- Megatron topology：TP1 + DP8，8 张 PPU-ZW810E
- checkpoint：BF16 Megatron DCP，由 TP2 转换并在 TP1 下 reshard 加载

Manifest：
`fast_wam/eval/manifest_libero_full_2k.json`。

## 复现

在仓库根目录执行：

```bash
bash fast_wam/scripts/run_libero_full_2k_bf16.sh
```

脚本默认离线运行并复用：

```text
outputs/fast_wam_dcp_bf16_tp2_20260723/
outputs/fast_wam_megatron_dcp_bf16_libero_2k_20260723/
```

每个 episode 原子写入 `cases/<case-id>.json`，`--resume` 会跳过已有结果。各 DP
rank 的 episode 时长差可能很大，因此 rank 0 通过原子 case 文件等待汇总，不在
rollout 结束时使用 NCCL barrier。默认最多等待 7200 秒，可用
`RESULT_WAIT_TIMEOUT` 覆盖。

Runner 把论文 overall 97.6% 作为硬门槛。本次 `summary.json` 正常生成且
`meets_target=false`，所以脚本退出码为 1；这是精度 gate 的预期结果，不是 checkpoint
加载或推理崩溃。

## 环境与限制

- PyTorch：`2.9.0+ali.10.ppu2.0.0.cu129`
- Transformer Engine：`2.8+ppu2.0.0.oe`
- MuJoCo：3.1.6
- robosuite：1.4.0
- rendering：OSMesa

PyTorch 和 Transformer Engine 均未修改。当前 simulator 是 MuJoCo 3.1.6，而
Fast-WAM training data config 标注 3.3.2，因此这是完整 protocol 的本地复现，
不是严格 environment-matched paper reproduction。

逐 task 数字、恢复过程和验证记录见
`fast_wam/log/2026-07-23-megatron-fastwam-bf16-libero-2k.md`；完整逐 episode
结果见 ignored artifact `summary.json`。
