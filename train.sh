#!/usr/bin/env bash
# 一键训练流水线：划分数据集 -> 实验矩阵筛选 -> 排行榜 -> Top-K 长训练
#
# - 全程断点续训：--resume 让已完成的实验秒退、训练中的实验从 last.pth 继续，
#   脚本可反复执行而不会重复训练。
# - 日志与权重均落在数据盘：logs/、checkpoints/、results/ 为指向
#   /root/autodl-tmp/image_tme/ 的软链接，不占用系统盘。
set -euo pipefail

MATRIX=configs/experiment_matrix.yaml
mkdir -p logs/console
RUN=logs/console/run_$(date +%Y%m%d_%H%M).log

{
  # 按 ROI 划分本地训练/验证集；固定种子，重复执行结果一致（幂等）
  uv run main.py split --root data --val-ratio 0.15 --seed 42

  # 阶段一：短实验横向筛选（可断点续训，已完成实验不会重跑）
  uv run main.py experiments --matrix "$MATRIX" --stage screening --resume

  # 打印筛选阶段排行榜
  uv run main.py experiments --matrix "$MATRIX" --stage report

  # 阶段二：Top-K 长训练（可断点续训）
  uv run main.py experiments --matrix "$MATRIX" --stage full --resume
} 2>&1 | tee "$RUN"
