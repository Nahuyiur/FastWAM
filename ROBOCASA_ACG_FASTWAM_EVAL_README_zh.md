# RoboCasa-ACG FastWAM 评测同步说明

这个目录把 Jinshan FastWAM 的 RoboCasa eval 口径同步到 Ali pi0.5 的口径。这里有两类评测，不能混用：

1. `online task eval`：在 RoboCasa 环境中按 `task + split + seed` 重新采样场景，跑闭环 policy rollout，统计 success rate 和增强指标。这个评测没有一一对应的 RoboCasa365 GT demo。
2. `dataset-window GT-matched diagnostic`：从 RoboCasa365 manifest 里取具体 `episode_index/window_start`，让 FastWAM 预测同一个 dataset window 的视频/action，并和该 window 的 GT 视频/action 逐窗对比。这个评测有严格匹配的 GT，但不统计 RoboCasa 在线 success rate。

核心原则：online eval 看任务是否闭环成功；GT-matched diagnostic 看模型是否学会数据集窗口上的视觉/动作预测。不要把 online rollout 视频和任意 GT demo 拼在一起当作同场景证据。

## 入口 1：online task eval

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

输出协议字段：

- `eval_protocol=online_task_success_rate`
- `gt_matched=false`
- `gt_episode_index=null`
- `gt_window_start=null`

这类结果只能用于 success rate、timeout、safety、trajectory、timing 等在线指标。它没有对应的 GT demo 视频。

快速 smoke 可以临时指定：

```bash
WAIT_FOR_TRAINING=0 EXPECTED_MIN_STEP=1 BUCKET=target_id_sanity VIDEO_POLICY=none \
RUN_ID=<run_id> bash scripts/run_robocasa_acg_fastwam_eval_after_training.sh
```

## 入口 2：dataset-window GT-matched diagnostic

这个入口用于回答“模型预测和 GT 到底差在哪里”。它从 RoboCasa365 split 中采样真实 dataset window，每个样本都有严格对应的 GT。

```bash
cd /mnt/yuhan/FastWAM_robocasa_acg_8gpu
WAIT_FOR_TRAINING=0 \
RUN_ID=<run_id> \
SPLIT=val \
NUM_SAMPLES=5 \
DEVICE=cuda:0 \
bash scripts/run_robocasa_acg_gt_matched_wam_smoke.sh
```

如需固定某条 episode/window：

```bash
RUN_ID=<run_id> \
SPLIT=val \
EPISODE_INDEX=1234 \
WINDOW_START=0 \
bash scripts/run_robocasa_acg_gt_matched_wam_smoke.sh
```

输出路径：

```text
/mnt/yuhan/experiments/robocasa_acg_v1/fastwam/gt_matched_wam/<RUN_ID>_<split>_gtmatched_<timestamp>
```

关键输出：

- `summary.json`
- `eval_manifest.csv`
- `video_index.html`
- `sample_XX_pred.mp4`
- `sample_XX_pred_gt.mp4`

输出协议字段：

- `eval_protocol=dataset_window_gt_matched_wam`
- `gt_matched=true`
- `episode_index=<RoboCasa365 episode id>`
- `window_start=<dataset window start frame>`
- `gt_video_layout=two_camera_horizontal(robot0_agentview_left|robot0_eye_in_hand)`

注意：这个诊断是 open-loop WAM window 预测，通常使用 GT 第一帧、GT proprio/text context 作为条件；它不是 RoboCasa online success-rate rollout。

## 与 Ali pi0.5 一致的 online 指标

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
