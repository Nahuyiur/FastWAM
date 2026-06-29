# RoboCasa ACG v1 FastWAM 训练说明

本分支用于在 RoboCasa365 ACG v1 自定义划分上训练 FastWAM。训练数据只使用 manifest 里的 `train_id`，验证只使用 `val_id`，不会直接全量读取 `robocasa365-pretrain-atomic`。

## 关键路径

- 数据根目录：`/mnt/pub_dataset/RoboCasa365`
- split manifest：`/mnt/pub_dataset/RoboCasa365/splits/robocasa_acg_v1_episode_manifest.csv`
- 训练 repo：`/mnt/pub_dataset/RoboCasa365/repos/robocasa365-pretrain-atomic`
- FastWAM 工作区：`/mnt/yuhan/FastWAM_robocasa_acg_8gpu`
- 训练输出：`/mnt/yuhan/experiments/robocasa_acg_v1/fastwam/runs/<RUN_ID>`
- norm cache：`/mnt/yuhan/experiments/robocasa_acg_v1/fastwam/cache/norm_stats/robocasa_acg_v1_train_id_dataset_stats.json`
- text cache：`/mnt/yuhan/experiments/robocasa_acg_v1/fastwam/cache/text_embeds/robocasa_acg_v1`

## 数据划分使用方式

- train：`train_id`
- val：`val_id`
- 不能进入 train/norm 的 split：`excluded_pretrain_ood_pair`、`test_ood_pair_strict`、`test_ood_pair_probe`
- 训练脚本会先运行 `scripts/audit_robocasa_acg_split.py --strict-counts`，检查计数和 train/OOD overlap。

## 启动训练

```bash
cd /mnt/yuhan/FastWAM_robocasa_acg_8gpu
RUN_ID=robocasa_acg_v1_fastwam_8gpu_$(date +%Y%m%d_%H%M%S) \
nohup bash scripts/run_robocasa_acg_8gpu.sh \
  > logs/${RUN_ID}.launcher.log 2>&1 &
```

`scripts/run_robocasa_acg_8gpu.sh` 会依次执行：

1. split audit
2. `train_id` norm stats 预计算
3. `train_id,val_id` text embedding 预计算
4. dataset smoke
5. 8GPU FastWAM 训练，W&B project 默认 `robocasa-acg-fastwam`

## 状态检查

```bash
cd /mnt/yuhan/FastWAM_robocasa_acg_8gpu
bash scripts/check_robocasa_acg_fastwam_status.sh <RUN_ID>
```

## 当前配置

- 模型：`fastwam_joint`
- 视频：2 camera horizontal concat，`robot0_agentview_left + robot0_eye_in_hand`
- 时间结构：9 个视觉 anchor，offset 为 `0,4,8,12,16,20,24,28,32`
- 动作窗口：32 step
- action dim：12
- state/proprio dim：16
- 8GPU 设置：per-rank batch 1，gradient accumulation 4，有效 batch 32
