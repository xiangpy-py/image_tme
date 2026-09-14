"""运行环境。

统一管理随机种子与计算设备的选择，保证实验可复现
（赛题明确要求可复现性）。
"""

import os
import random
from typing import Optional

import numpy as np
import torch


class Runtime:
    """运行环境：随机种子固定与设备解析。"""

    @staticmethod
    def seed_everything(seed: int = 42) -> None:
        """固定所有随机源。

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

    @staticmethod
    def get_device(device_name: Optional[str] = None) -> torch.device:
        """解析运行设备，未指定时优先使用 GPU。

        Args:
            device_name: 形如 ``"cuda"`` / ``"cuda:0"`` / ``"cpu"`` 的设备名，
                传入 ``None`` 或 ``"auto"`` 表示自动选择。

        Returns:
            torch.device: 实际使用的计算设备。
        """
        if device_name is not None and device_name != "auto":
            return torch.device(device_name)
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
