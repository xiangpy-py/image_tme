"""边缘损失 (Edge Loss)。

基于 Sobel 算子提取图像边缘，约束生成图像与真值在细胞边界、
组织边缘等关键结构位置的一致性。

病理图像中，细胞边界和组织结构的精确恢复至关重要。
L1/SSIM 可能使内部纹理模糊但边缘保持尚可，而 Edge Loss
直接惩罚边缘位置的差异，强制模型关注结构轮廓。

实现思路：
    1. 用 Sobel 算子分别提取 pred 和 target 的边缘强度图
    2. 计算边缘强度图的 L1 距离作为损失
    3. 可选：对边缘图做高斯平滑，降低噪声敏感度
"""

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class SobelEdgeLoss(nn.Module):
    """Sobel 边缘损失：约束生成图像与真值的边缘结构一致性。"""

    # Sobel 核：水平与垂直两个方向，注册为 buffer 自动迁移设备。
    SOBEL_X: torch.Tensor
    SOBEL_Y: torch.Tensor

    def __init__(
        self,
        kernel_size: int = 3,
        smooth_sigma: Optional[float] = 1.0,
    ) -> None:
        """构建 Sobel 边缘提取器。

        Args:
            kernel_size: Sobel 核尺寸，仅支持 3（默认）或 5。
            smooth_sigma: 边缘图高斯平滑的标准差；
                ``None`` 表示不做平滑，直接对原始边缘图计算损失。
        """
        super().__init__()
        self.kernel_size = kernel_size
        self.smooth_sigma = smooth_sigma

        # 构建 Sobel 核
        sobel_x, sobel_y = self._build_sobel_kernels(kernel_size)
        self.register_buffer("sobel_x", sobel_x)
        self.register_buffer("sobel_y", sobel_y)

        # 可选的高斯平滑核（边缘图去噪）
        if smooth_sigma is not None and smooth_sigma > 0:
            gaussian = self._build_gaussian_kernel(
                window_size=5, sigma=smooth_sigma
            )
            self.register_buffer("gaussian", gaussian)
        else:
            self.gaussian = None  # type: ignore[assignment]

    def _build_sobel_kernels(
        self, size: int
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """生成 Sobel 卷积核。

        Args:
            size: 核尺寸，仅支持 3 或 5。

        Returns:
            tuple[torch.Tensor, torch.Tensor]: (水平核, 垂直核)，
                形状均为 ``(1, 1, size, size)``。

        Raises:
            ValueError: 不支持的核尺寸时抛出。
        """
        if size == 3:
            # 标准 3x3 Sobel 核
            x_kernel = torch.tensor(
                [[-1.0, 0.0, 1.0],
                 [-2.0, 0.0, 2.0],
                 [-1.0, 0.0, 1.0]],
                dtype=torch.float32,
            )
            y_kernel = torch.tensor(
                [[-1.0, -2.0, -1.0],
                 [ 0.0,  0.0,  0.0],
                 [ 1.0,  2.0,  1.0]],
                dtype=torch.float32,
            )
        elif size == 5:
            # 5x5 Sobel 核，对噪声更鲁棒
            x_kernel = torch.tensor(
                [[-1.0, -2.0, 0.0, 2.0, 1.0],
                 [-2.0, -3.0, 0.0, 3.0, 2.0],
                 [-3.0, -4.0, 0.0, 4.0, 3.0],
                 [-2.0, -3.0, 0.0, 3.0, 2.0],
                 [-1.0, -2.0, 0.0, 2.0, 1.0]],
                dtype=torch.float32,
            )
            y_kernel = torch.tensor(
                [[-1.0, -2.0, -3.0, -2.0, -1.0],
                 [-2.0, -3.0, -4.0, -3.0, -2.0],
                 [ 0.0,  0.0,  0.0,  0.0,  0.0],
                 [ 2.0,  3.0,  4.0,  3.0,  2.0],
                 [ 1.0,  2.0,  3.0,  2.0,  1.0]],
                dtype=torch.float32,
            )
        else:
            raise ValueError(f"不支持的 Sobel 核尺寸: {size}，仅支持 3 或 5")

        # 扩展为 (1, 1, size, size)，适配 conv2d 的 weight 形状
        return (
            x_kernel.unsqueeze(0).unsqueeze(0),
            y_kernel.unsqueeze(0).unsqueeze(0),
        )

    @staticmethod
    def _build_gaussian_kernel(
        window_size: int, sigma: float
    ) -> torch.Tensor:
        """生成一维高斯核并扩展为 2D 卷积核。

        Args:
            window_size: 核宽度。
            sigma: 高斯标准差。

        Returns:
            torch.Tensor: 归一化 2D 高斯核，形状 ``(1, 1, W, W)``。
        """
        coords = torch.arange(window_size, dtype=torch.float32) - window_size // 2
        kernel_1d = torch.exp(-(coords ** 2) / (2.0 * sigma * sigma))
        kernel_1d = kernel_1d / kernel_1d.sum()
        kernel_2d = kernel_1d[:, None] @ kernel_1d[None, :]
        return kernel_2d.unsqueeze(0).unsqueeze(0)

    def _extract_edge(self, x: torch.Tensor) -> torch.Tensor:
        """用 Sobel 算子提取单张图像的边缘强度图。

        对多通道输入，逐通道提取后取平均，保证输出为单通道边缘图。

        Args:
            x: 输入图像 ``(B, C, H, W)``，取值 ``[0, 1]``。

        Returns:
            torch.Tensor: 边缘强度图 ``(B, 1, H, W)``，值域非负。
        """
        batch, channels, height, width = x.shape
        # 将多通道展平为 (B*C, 1, H, W)，统一做卷积
        x_flat = x.view(batch * channels, 1, height, width)

        # 水平与垂直边缘分量
        edge_x = F.conv2d(
            x_flat, self.sobel_x.to(x.dtype), padding=self.kernel_size // 2
        )
        edge_y = F.conv2d(
            x_flat, self.sobel_y.to(x.dtype), padding=self.kernel_size // 2
        )

        # 边缘强度 = sqrt(edge_x^2 + edge_y^2)
        magnitude = torch.sqrt(edge_x ** 2 + edge_y ** 2 + 1e-6)

        # 还原为 (B, C, H, W) 后沿通道取平均 -> (B, 1, H, W)
        magnitude = magnitude.view(batch, channels, height, width)
        return magnitude.mean(dim=1, keepdim=True)

    def _smooth(self, x: torch.Tensor) -> torch.Tensor:
        """对边缘图做可选的高斯平滑（降低噪声敏感度）。

        Args:
            x: 边缘图 ``(B, 1, H, W)``。

        Returns:
            torch.Tensor: 平滑后的边缘图。
        """
        if self.gaussian is None:
            return x
        return F.conv2d(
            x, self.gaussian.to(x.dtype), padding=2, groups=1
        )

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """计算边缘损失。

        Args:
            pred:   预测图像 ``(B, C, H, W)``，取值 ``[0, 1]``。
            target: 真值图像，形状与取值同 ``pred``。

        Returns:
            torch.Tensor: 标量损失，值域 ``[0, +inf)``。
        """
        pred_edge = self._smooth(self._extract_edge(pred))
        target_edge = self._smooth(self._extract_edge(target))
        return F.l1_loss(pred_edge, target_edge)
