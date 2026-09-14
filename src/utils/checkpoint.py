"""Checkpoint 管理。

模型权重、优化器状态与训练进度的保存和恢复。比赛复赛/半决赛
要求提交模型文件并能正常加载推理，因此 checkpoint 中同时保存
模型结构参数与完整训练状态。
"""

from pathlib import Path
from typing import Any, Dict, Optional

import torch
import torch.nn as nn
from torch.optim import Optimizer


class CheckpointManager:
    """训练检查点的保存与加载。"""

    @staticmethod
    def save(
        model: nn.Module,
        optimizer: Optional[Optimizer],
        epoch: int,
        best_score: float,
        config: Dict[str, Any],
        save_path: str,
        scheduler: Optional[Any] = None,
        include_optimizer: bool = True,
    ) -> None:
        """保存训练检查点。

        为控制 checkout 体积，仅当 ``include_optimizer=True`` 时才落盘优化器与
        调度器状态：推理用的 ``best.pth`` 只保留权重（约占完整状态的 1/3），
        续训用的 ``last.pth`` 才保存完整训练状态。

        Args:
            model:             待保存的模型。
            optimizer:         优化器，纯推理用途可传 ``None``。
            epoch:             当前训练轮次。
            best_score:        目前最优验证指标（综合得分）。
            config:            实验配置快照，保证加载时可还原模型结构。
            save_path:         保存路径，通常位于 ``checkpoints/`` 目录。
            scheduler:         学习率调度器，仅续训需要。
            include_optimizer: 是否保存优化器/调度器状态（``False`` 表示瘦身保存）。

        Returns:
            None
        """
        path = Path(save_path)
        path.parent.mkdir(parents=True, exist_ok=True)

        payload: Dict[str, Any] = {
            "epoch": epoch,
            "best_score": best_score,
            "model_state_dict": model.state_dict(),
            "config": config,
        }
        # 优化器状态（AdamW 含两组动量）体积约为参数量的 2 倍，仅在续训用途下保存。
        if include_optimizer and optimizer is not None:
            payload["optimizer_state_dict"] = optimizer.state_dict()
            if scheduler is not None:
                payload["scheduler_state_dict"] = scheduler.state_dict()

        torch.save(payload, path)

    @staticmethod
    def load(
        checkpoint_path: str,
        model: nn.Module,
        optimizer: Optional[Optimizer] = None,
        map_location: str = "cpu",
        scheduler: Optional[Any] = None,
    ) -> Dict[str, Any]:
        """加载检查点并恢复模型（以及可选的优化器/调度器）状态。

        Args:
            checkpoint_path: checkpoint 文件路径。
            model:           已按相同结构实例化的模型。
            optimizer:       需要恢复状态的优化器，推理时传 ``None``。
            map_location:    权重映射设备，默认 ``"cpu"`` 以保证通用性。
            scheduler:       需要恢复状态的学习率调度器，仅续训传入。

        Returns:
            Dict[str, Any]: 含 ``epoch`` / ``best_score`` / ``config`` /
            ``has_optimizer`` 的元信息。

        Raises:
            FileNotFoundError: checkpoint 不存在时抛出。
        """
        path = Path(checkpoint_path)
        if not path.is_file():
            raise FileNotFoundError(f"checkpoint 不存在: {checkpoint_path}")

        # torch>=2.6 默认 weights_only=True；本项目 checkpoint 含配置字典，
        # 显式声明 weights_only=False 以保证跨版本行为一致（权重来源可信）。
        try:
            checkpoint = torch.load(path, map_location=map_location, weights_only=False)
        except TypeError:
            # 兼容 torch<2.0 无 weights_only 参数的情况。
            checkpoint = torch.load(path, map_location=map_location)

        model.load_state_dict(checkpoint["model_state_dict"])
        if optimizer is not None and checkpoint.get("optimizer_state_dict"):
            optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        if scheduler is not None and checkpoint.get("scheduler_state_dict"):
            scheduler.load_state_dict(checkpoint["scheduler_state_dict"])

        return {
            "epoch": checkpoint.get("epoch", 0),
            "best_score": checkpoint.get("best_score", 0.0),
            "config": checkpoint.get("config", {}),
            # 瘦身保存（best.pth）不含优化器状态，续训前据此判断可否恢复。
            "has_optimizer": "optimizer_state_dict" in checkpoint,
        }
