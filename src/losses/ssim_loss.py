"""SSIM 损失。

结构相似性（SSIM）是比赛的主评价指标之一（占综合分 70%），
将其设计为可微损失直接优化，可显著提升生成图像的结构一致性。
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


def _gaussian_kernel(window_size: int, sigma: float) -> torch.Tensor:
    """生成一维高斯核。

    Args:
        window_size: 核宽度，通常取 11。
        sigma:       高斯标准差，通常取 1.5。

    Returns:
        torch.Tensor: 归一化的一维核，形状 ``(window_size,)``。
    """
    coords = torch.arange(window_size, dtype=torch.float32) - window_size // 2
    kernel = torch.exp(-(coords ** 2) / (2.0 * sigma * sigma))
    return kernel / kernel.sum()


class SSIM(nn.Module):
    """可微 SSIM 模块，同时可用于损失与验证指标。"""

    def __init__(self, window_size: int = 11, sigma: float = 1.5) -> None:
        """预构建高斯窗。

        Args:
            window_size: 高斯窗尺寸。
            sigma:       高斯标准差。
        """
        super().__init__()
        self.window_size = window_size
        kernel_1d = _gaussian_kernel(window_size, sigma)
        kernel_2d = kernel_1d[:, None] @ kernel_1d[None, :]
        # 注册为 buffer，随模型自动迁移设备但不参与梯度更新。
        self.register_buffer("window", kernel_2d)

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """计算批量 SSIM 均值。

        Args:
            pred:   预测图像 ``(B, C, H, W)``，取值 ``[0, 1]``。
            target: 真值图像，形状与取值同 ``pred``。

        Returns:
            torch.Tensor: 标量，batch 平均 SSIM，越接近 1 越相似。
        """
        channels = pred.shape[1]
        window = self.window.expand(channels, 1, -1, -1).to(pred.dtype)
        padding = self.window_size // 2

        def _filter(x: torch.Tensor) -> torch.Tensor:
            """用分组卷积对每通道独立做高斯滤波。"""
            return F.conv2d(x, window, padding=padding, groups=channels)

        mu_pred = _filter(pred)
        mu_target = _filter(target)
        mu_pred_sq = mu_pred * mu_pred
        mu_target_sq = mu_target * mu_target
        mu_cross = mu_pred * mu_target

        sigma_pred_sq = _filter(pred * pred) - mu_pred_sq
        sigma_target_sq = _filter(target * target) - mu_target_sq
        sigma_cross = _filter(pred * target) - mu_cross

        # SSIM 稳定性常数，L=1（像素已归一化到 [0, 1]）。
        c1, c2 = 0.01 ** 2, 0.03 ** 2
        ssim_map = (
            (2.0 * mu_cross + c1) * (2.0 * sigma_cross + c2)
        ) / (
            (mu_pred_sq + mu_target_sq + c1)
            * (sigma_pred_sq + sigma_target_sq + c2)
        )
        return ssim_map.mean()


class SSIMLoss(nn.Module):
    """SSIM 损失：``1 - SSIM``，值越小结构越一致。"""

    def __init__(self, window_size: int = 11, sigma: float = 1.5) -> None:
        """内部包装一个 SSIM 模块。"""
        super().__init__()
        self.ssim = SSIM(window_size=window_size, sigma=sigma)

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """计算 SSIM 损失。

        Args:
            pred:   预测图像 ``(B, C, H, W)``。
            target: 真值图像。

        Returns:
            torch.Tensor: 标量损失，范围 ``[0, 2]``。
        """
        return 1.0 - self.ssim(pred, target)
