"""Checkpoint 管理：模型权重、优化器状态与训练进度的保存和恢复。

比赛复赛/半决赛要求提交模型文件并能正常加载推理，
因此 checkpoint 中同时保存模型结构参数与完整训练状态。
"""

from pathlib import Path
from typing import Any, Dict, Optional

import torch
import torch.nn as nn
from torch.optim import Optimizer


def save_checkpoint(
    model: nn.Module,
    optimizer: Optional[Optimizer],
    epoch: int,
    best_score: float,
    config: Dict[str, Any],
    save_path: str,
) -> None:
    """保存训练检查点。

    Args:
        model:      待保存的模型。
        optimizer:  优化器，纯推理用途的 checkpoint 可传 ``None``。
        epoch:      当前训练轮次。
        best_score: 目前最优验证指标（综合得分）。
        config:     实验配置快照，保证加载时可还原模型结构。
        save_path:  保存路径，通常位于 ``checkpoints/`` 目录。

    Returns:
        None
    """
    path = Path(save_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    checkpoint = {
        "epoch": epoch,
        "best_score": best_score,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": (
            optimizer.state_dict() if optimizer is not None else None
        ),
        "config": config,
    }
    torch.save(checkpoint, path)


def load_checkpoint(
    checkpoint_path: str,
    model: nn.Module,
    optimizer: Optional[Optimizer] = None,
    map_location: str = "cpu",
) -> Dict[str, Any]:
    """加载检查点并恢复模型（以及可选的优化器）状态。

    Args:
        checkpoint_path: checkpoint 文件路径。
        model:           已按相同结构实例化的模型。
        optimizer:       需要恢复状态的优化器，推理时传 ``None``。
        map_location:    权重映射设备，默认 ``"cpu"`` 以保证通用性。

    Returns:
        Dict[str, Any]: 包含 ``epoch``、``best_score``、``config`` 的元信息。
    """
    path = Path(checkpoint_path)
    if not path.is_file():
        raise FileNotFoundError(f"checkpoint 不存在: {checkpoint_path}")

    # torch>=2.6 默认 weights_only=True；本项目的 checkpoint 含配置字典，
    # 显式声明 weights_only=False 以保证跨版本行为一致（权重来源可信）。
    try:
        checkpoint = torch.load(
            path, map_location=map_location, weights_only=False
        )
    except TypeError:
        # 兼容 torch<2.0 无 weights_only 参数的情况。
        checkpoint = torch.load(path, map_location=map_location)
    model.load_state_dict(checkpoint["model_state_dict"])

    if optimizer is not None and checkpoint.get("optimizer_state_dict"):
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

    return {
        "epoch": checkpoint.get("epoch", 0),
        "best_score": checkpoint.get("best_score", 0.0),
        "config": checkpoint.get("config", {}),
    }
