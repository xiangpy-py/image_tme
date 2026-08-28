"""通用基础工具：随机种子、设备选择与指标统计容器。"""

import os
import random
from typing import Optional

import numpy as np
import torch


def seed_everything(seed: int = 42) -> None:
    """固定所有随机源，保证实验可复现（赛题明确要求可复现性）。

    Args:
        seed: 全局随机种子。

    Returns:
        None
    """
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    # 卷积算法固定，牺牲少量性能换取确定性。
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_device(device_name: Optional[str] = None) -> torch.device:
    """解析运行设备，未指定时优先使用 GPU。

    Args:
        device_name: 形如 ``"cuda"`` / ``"cuda:0"`` / ``"cpu"`` 的设备名，
            传入 ``None`` 表示自动选择。

    Returns:
        torch.device: 实际使用的计算设备。
    """
    if device_name is not None and device_name != "auto":
        return torch.device(device_name)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


class AverageMeter:
    """滑动平均值统计器，用于记录 loss / SSIM / PSNR 等标量指标。"""

    def __init__(self) -> None:
        """初始化计数与累加和。"""
        self.reset()

    def reset(self) -> None:
        """清空全部统计量。"""
        self.count: int = 0
        self.sum: float = 0.0

    def update(self, value: float, n: int = 1) -> None:
        """累加一个 batch 的指标值。

        Args:
            value: 当前 batch 的指标均值。
            n:     当前 batch 的样本数（用于按样本数加权）。

        Returns:
            None
        """
        self.sum += value * n
        self.count += n

    @property
    def avg(self) -> float:
        """float: 当前累计平均值，无样本时返回 0。"""
        if self.count == 0:
            return 0.0
        return self.sum / self.count
