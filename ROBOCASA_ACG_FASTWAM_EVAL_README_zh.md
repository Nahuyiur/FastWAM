# RoboCasa-ACG FastWAM 在线评测同步说明

这个目录把 Jinshan FastWAM 的 RoboCasa eval 口径同步到 Ali pi0.5 的口径。核心原则是：环境 rollout、任务 bucket、success 指标、增强指标、视频保存逻辑保持一致，只替换 policy backend。

## 评测入口

正式入口：

```bash
cd /mnt/yuhan/FastWAM_robocasa_acg_8gpu
RUN_ID=robocasa_acg_v1_fastwam_8gpu_20260629_195226 \
bash scripts/run_robocasa_acg_fastwam_eval_after_training.sh
```

默认行为：

1. 等待对应 FastWAM 训练进程结束。
2. 找 `/mnt/yuhan/experiments/robocasa_acg_v1/fastwam/runs/<RUN_ID>/checkpoints/weights/step_*.pt` 的最新权重。
3. 默认要求 checkpoint step 达到 `50000`，避免半程 checkpoint 被误当正式结果。
4. 跑 `scripts/robocasa_acg_eval_plan_v1.json` 里的同一套 RoboCasa online rollout。
5. 保存 summary、episode jsonl、视频和 HTML 索引。

快速 smoke 可以临时指定：

```bash
WAIT_FOR_TRAINING=0 EXPECTED_MIN_STEP=1 BUCKET=target_id_sanity VIDEO_POLICY=none \
RUN_ID=<run_id> bash scripts/run_robocasa_acg_fastwam_eval_after_training.sh
```

## 与 Ali pi0.5 一致的部分

- `id_pretrain_online`
- `target_id_sanity`
- `ood_pair_strict`
- `ood_pair_probe`
- RoboCasa 官方 `info["success"]`
- Wilson 95% CI
- `summary_by_bucket.csv`
- `summary_metrics.json`
- `episode_results.jsonl`
- `per_task_metrics.csv`
- `cell_metrics.csv`
- `video_index.html`
- rollout MP4
- timing、trajectory、safety、wrong object / wrong target 诊断字段

## FastWAM 专属 adapter

FastWAM 训练时使用 2-camera 横向拼接输入：

- `robot0_agentview_left`
- `robot0_eye_in_hand`

在线 eval 中 adapter 会把 RoboCasa obs 转成 224x448 的 `[-1, 1]` 图像张量，使用同一份 train_id norm stats 归一化 16 维 proprio，并把 `infer_action` 的归一化 action 反归一化回 RoboCasa 原始 12 维 action，再交给 `convert_action`。

文本侧不加载 text encoder，而是使用训练时预计算的 text cache：

```text
/mnt/yuhan/experiments/robocasa_acg_v1/fastwam/cache/text_embeds/robocasa_acg_v1
```

## 输出路径

```text
/mnt/yuhan/experiments/robocasa_acg_v1/fastwam/eval/runs/<RUN_ID>_step_<STEP>_stage1_<timestamp>
```

最近一次 eval 输出路径会写入：

```text
/mnt/yuhan/experiments/robocasa_acg_v1/fastwam/eval/logs/latest_eval_output_dir.txt
```

## 注意

Jinshan 的 `motus` 环境原本只有 FastWAM 训练依赖，不一定有 RoboCasa online eval 依赖。第一次评测前先运行：

```bash
cd /mnt/yuhan/FastWAM_robocasa_acg_8gpu
bash scripts/setup_robocasa_eval_env.sh
```

这个脚本会安装 `robocasa`、`robosuite`、`mujoco==3.3.1`，并下载 RoboCasa kitchen assets。
