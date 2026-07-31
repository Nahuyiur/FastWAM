# Megatron Fast-WAM BF16 LIBERO-Spatial 评测

更新时间：2026-07-23

## 结论

Megatron Fast-WAM 使用 BF16 TP2 DCP，在 8 张 PPU 上以 TP2+DP4 完成了
LIBERO-Spatial 10 个任务、每任务前 5 个官方初始状态的 50-episode 评测：

- Megatron：**48/50，96.00%**
- 官方仓库本地同协议结果：**47/50，94.00%**
- 结论：达到并超过 94% 对照标准，`summary.json` 中 `passed=true`

这是 50-episode 集成评测，不是论文的四 suite × 每任务 50 次、共 2,000 episodes
的完整协议。

## 分任务结果

| Task | Megatron BF16 | 官方本地 BF16 |
|---:|---:|---:|
| 0 | 5/5 | 5/5 |
| 1 | 5/5 | 5/5 |
| 2 | 5/5 | 5/5 |
| 3 | 5/5 | 5/5 |
| 4 | 5/5 | 5/5 |
| 5 | 5/5 | 5/5 |
| 6 | 5/5 | 5/5 |
| 7 | 5/5 | 4/5 |
| 8 | 5/5 | 4/5 |
| 9 | 3/5 | 4/5 |
| **总计** | **48/50 (96%)** | **47/50 (94%)** |

Megatron 的失败 case 是 `spatial-t9-i0` 和 `spatial-t9-i1`，均运行到 400 个
策略步。逐 episode 成败与官方不要求完全相同；TP reduction 和 BF16 舍入会在长闭环
中改变轨迹。本评测比较相同协议下的总成功率。

## 对齐协议

- suite：`libero_spatial`
- task：0–9
- init state：每任务 0–4
- simulator seed：42
- 模型精度：BF16
- diffusion steps：10
- action horizon：32
- 每次执行 10 个 action 后 replan
- reset 后执行 30 个 no-op
- 每 episode 最多 400 个策略步
- 两个 256×256 camera observation，resize 后横向拼成 224×448
- gripper action 二值化
- MuJoCo 3.1.6、robosuite 1.4.0、OSMesa

官方训练数据配置名为 MuJoCo 3.3.2，因此这里与官方本地 94% run 一样，属于当前
MuJoCo 3.1.6 环境中的对照，不应表述为严格环境匹配的论文复现。

## Megatron checkpoint 与并行方式

发布的 LeRobot safetensors 先转换为约 12 GiB 的 BF16 TP2 Megatron DCP：

```text
outputs/fast_wam_dcp_bf16_tp2_20260723/
├── __0_0.distcp
├── __1_0.distcp
├── .metadata
├── common.pt
└── metadata.json
```

正式评测用 8 个进程：

- TP=2：每个模型 replica 使用两张 PPU
- DP=4：四个 replica 按 manifest index 取模分片 50 个 episodes
- Wan VAE、UMT5 和 LIBERO env 仅存在于每个 TP group 的 rank 0

## 一键复现

从 `Megatron-Wan/` 运行：

```bash
bash fast_wam/scripts/run_libero_spatial_bf16.sh
```

脚本在 DCP 不存在时先用两张 PPU 转换 BF16 TP2 DCP，再用 8 张 PPU 运行评测。
常用覆盖变量：

```bash
OUTPUT=/path/to/output \
DCP=/path/to/megatron_dcp \
EVAL_DEVICES=0,1,2,3,4,5,6,7 \
bash fast_wam/scripts/run_libero_spatial_bf16.sh
```

脚本不会安装、升级或降级 PyTorch/Transformer Engine，也不会联网下载模型。

## Artifacts

- 最终汇总：
  `outputs/fast_wam_megatron_dcp_bf16_spatial_5trials_20260723/summary.json`
- 每个 DP replica 的增量结果：同目录 `dp_0.json` 至 `dp_3.json`
- BF16 TP2 DCP：`outputs/fast_wam_dcp_bf16_tp2_20260723/`
- 固定 manifest：`fast_wam/eval/manifest_libero_spatial_5trials.json`
- 详细工作日志：`fast_wam/log/2026-07-23-megatron-fastwam-bf16-libero.md`
