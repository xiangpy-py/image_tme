"""多尺度输入策略。

源自 "Global Pixel Transformers"（Liu et al.）的 Multi-Scale Input：
以同一中心裁剪 3 个尺度的 patch（2H×2W、H×W、H/2×W/2），
统一 resize 到 H×W 后沿通道拼接，使网络同时获得全局上下文
与局部细节信息。

与本比赛的适配说明：赛题数据已是预先切好的 256x256 patch，
无法向 patch 外裁出 2H×2W 的真实大视野区域，因此做等价近似：

- scale > 1（如 2.0）：整图先降采样再升采样回原尺寸，
  以「低分辨率全图」近似更大感受野的全局上下文；
- scale = 1：原图，提供标准尺度信息；
- scale < 1（如 0.5）：中心裁剪后放大回原尺寸，
  放大局部细微区域，鼓励网络捕捉细节（对应论文的 X2）。

该变换只作用于模型输入，不影响真值；几何增强应先于其完成。
"""

from typing import List

import cv2
import numpy as np


def _resize_to(image: np.ndarray, size: int) -> np.ndarray:
    """将图像 resize 到 (size, size)，保持通道维。

    Args:
        image: HWC 布局的 float 数组。
        size:  目标边长。

    Returns:
        np.ndarray: 变换后的 float32 数组。
    """
    resized = cv2.resize(image, (size, size), interpolation=cv2.INTER_LINEAR)
    if resized.ndim == 2:  # 单通道图 resize 后通道维丢失，补回
        resized = resized[..., None]
    return resized.astype(np.float32)


def build_multiscale_input(
    image: np.ndarray,
    scales: List[float],
) -> np.ndarray:
    """由单张图像构建多尺度拼接输入。

    Args:
        image:  HWC 布局、float32、取值 ``[0, 1]`` 的输入图像。
        scales: 尺度因子列表，例如 ``[0.5, 1.0, 2.0]``。

    Returns:
        np.ndarray: 沿通道维拼接的多尺度输入，
        形状 ``(H, W, C * len(scales))``，float32。
    """
    height = image.shape[0]
    branches: List[np.ndarray] = []

    for scale in scales:
        if scale == 1.0:
            branch = image
        elif scale < 1.0:
            # 局部细节分支：中心裁剪后放大，等效论文中的 X2。
            crop = int(round(height * scale))
            start = (height - crop) // 2
            branch = _resize_to(image[start:start + crop, start:start + crop], height)
        else:
            # 全局上下文分支：降采样后再升采样（patch 外无真实大视野可用）。
            small = _resize_to(image, max(1, int(round(height / scale))))
            branch = _resize_to(small, height)
        branches.append(branch)

    return np.concatenate(branches, axis=-1).astype(np.float32)


class MultiScaleInput:
    """可调用的多尺度输入变换，挂在配对增强之后、转张量之前。"""

    def __init__(self, scales: List[float]) -> None:
        """初始化尺度配置。

        Args:
            scales: 尺度因子列表；要求必须包含 1.0 以保证原尺度分支存在。

        Raises:
            ValueError: 尺度列表为空或不含 1.0 时抛出。
        """
        if not scales or 1.0 not in scales:
            raise ValueError("多尺度配置必须包含 1.0 尺度分支")
        self.scales = sorted(scales)

    def __call__(self, image: np.ndarray) -> np.ndarray:
        """生成多尺度拼接输入。

        Args:
            image: HWC float32 输入图像。

        Returns:
            np.ndarray: ``(H, W, C * len(scales))`` 的多尺度输入。
        """
        return build_multiscale_input(image, self.scales)
