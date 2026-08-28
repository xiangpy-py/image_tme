"""可视化工具：生成「输入 - 预测 - 真值」对比图。

用于训练过程中的快速人工检查，以及技术报告中的定性结果展示。
"""

from pathlib import Path
from typing import Optional

import numpy as np
import torch


def _to_numpy_image(tensor: torch.Tensor) -> np.ndarray:
    """将单张 CHW 张量转为 HWC 的 uint8 图像数组。

    Args:
        tensor: 形状为 ``(C, H, W)``、取值 ``[0, 1]`` 的图像张量。

    Returns:
        np.ndarray: 形状 ``(H, W)`` 或 ``(H, W, 3)`` 的 uint8 数组。
    """
    array = tensor.detach().cpu().clamp(0.0, 1.0).numpy()
    array = np.transpose(array, (1, 2, 0))  # CHW -> HWC
    if array.shape[-1] == 1:
        array = array[..., 0]  # 单通道压掉通道维
    return (array * 255.0).round().astype(np.uint8)


def save_comparison_grid(
    inputs: torch.Tensor,
    predictions: torch.Tensor,
    targets: Optional[torch.Tensor],
    save_path: str,
    max_samples: int = 4,
) -> None:
    """将若干样本的 输入/预测/真值 纵向拼接保存为一张 JPG 对比图。

    Args:
        inputs:      输入 DAPI 批次，形状 ``(B, C, H, W)``。
        predictions: 模型预测批次，形状同 ``inputs``。
        targets:     真值批次；测试阶段没有真值时传 ``None``。
        save_path:   输出图片路径。
        max_samples: 最多展示的样本数。

    Returns:
        None
    """
    import cv2  # 延迟导入，避免纯指标场景强依赖 OpenCV

    batch_size = min(inputs.shape[0], max_samples)
    rows = []

    for index in range(batch_size):
        panels = [
            _to_numpy_image(inputs[index]),
            _to_numpy_image(predictions[index]),
        ]
        if targets is not None:
            panels.append(_to_numpy_image(targets[index]))

        # 统一成三通道便于灰度/彩色混合拼接。
        panels = [
            panel if panel.ndim == 3 else cv2.cvtColor(panel, cv2.COLOR_GRAY2BGR)
            for panel in panels
        ]
        rows.append(np.concatenate(panels, axis=1))

    grid = np.concatenate(rows, axis=0)

    path = Path(save_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(path), grid)
