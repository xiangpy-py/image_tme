"""损失函数。

组合损失::

    L = λ_l1·L1 + λ_mse·MSE + λ_ssim·SSIM + λ_edge·Edge + λ_cross·CrossMarker

- ``MSE``：均方误差，是评测指标 PSNR 的精确代理（PSNR 由全局 MSE 单调
  决定，占综合分 30%）；L1 优化中位数而 MSE 优化均值，二者互补；
- ``SSIM`` / ``SSIMLoss``：结构相似性，比赛主指标之一（占综合分 70%），
  设计为可微损失直接优化；
- ``SobelEdgeLoss``：约束细胞边界/组织边缘的结构一致性；
- ``CrossMarkerConsistencyLoss``：一对多建模时约束共享特征与标记无关；
- ``CombinedLoss``：按权重装配以上分项，权重为 0 即关闭。
"""

from typing import Any, Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


def _gaussian_kernel_1d(window_size: int, sigma: float) -> torch.Tensor:
    """生成归一化的一维高斯核。

    Args:
        window_size: 核宽度。
        sigma:       高斯标准差。

    Returns:
        torch.Tensor: 形状 ``(window_size,)`` 的一维核。
    """
    coords = torch.arange(window_size, dtype=torch.float32) - window_size // 2
    kernel = torch.exp(-(coords**2) / (2.0 * sigma * sigma))
    return kernel / kernel.sum()


def _sobel_kernels(size: int) -> Tuple[torch.Tensor, torch.Tensor]:
    """生成 Sobel 卷积核（水平/垂直两个方向）。

    Args:
        size: 核尺寸，仅支持 3 或 5。

    Returns:
        Tuple[torch.Tensor, torch.Tensor]: 形状均为 ``(1, 1, size, size)``。

    Raises:
        ValueError: 不支持的核尺寸时抛出。
    """
    if size == 3:
        x_kernel = torch.tensor(
            [[-1.0, 0.0, 1.0], [-2.0, 0.0, 2.0], [-1.0, 0.0, 1.0]],
            dtype=torch.float32,
        )
        y_kernel = torch.tensor(
            [[-1.0, -2.0, -1.0], [0.0, 0.0, 0.0], [1.0, 2.0, 1.0]],
            dtype=torch.float32,
        )
    elif size == 5:
        # 5x5 Sobel 核，对噪声更鲁棒。
        x_kernel = torch.tensor(
            [
                [-1.0, -2.0, 0.0, 2.0, 1.0],
                [-2.0, -3.0, 0.0, 3.0, 2.0],
                [-3.0, -4.0, 0.0, 4.0, 3.0],
                [-2.0, -3.0, 0.0, 3.0, 2.0],
                [-1.0, -2.0, 0.0, 2.0, 1.0],
            ],
            dtype=torch.float32,
        )
        y_kernel = torch.tensor(
            [
                [-1.0, -2.0, -3.0, -2.0, -1.0],
                [-2.0, -3.0, -4.0, -3.0, -2.0],
                [0.0, 0.0, 0.0, 0.0, 0.0],
                [2.0, 3.0, 4.0, 3.0, 2.0],
                [1.0, 2.0, 3.0, 2.0, 1.0],
            ],
            dtype=torch.float32,
        )
    else:
        raise ValueError(f"不支持的 Sobel 核尺寸: {size}，仅支持 3 或 5")

    # 扩展为 (1, 1, size, size)，适配 conv2d 的 weight 形状。
    return x_kernel.unsqueeze(0).unsqueeze(0), y_kernel.unsqueeze(0).unsqueeze(0)


def _gaussian_kernel_2d(window_size: int, sigma: float) -> torch.Tensor:
    """生成 2D 高斯卷积核。

    Args:
        window_size: 核宽度。
        sigma:       高斯标准差。

    Returns:
        torch.Tensor: 归一化 2D 高斯核，形状 ``(1, 1, W, W)``。
    """
    kernel_1d = _gaussian_kernel_1d(window_size, sigma)
    return (kernel_1d[:, None] @ kernel_1d[None, :]).unsqueeze(0).unsqueeze(0)


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
        kernel_1d = _gaussian_kernel_1d(window_size, sigma)
        # 注册为 buffer，随模型自动迁移设备但不参与梯度更新。
        self.register_buffer("window", kernel_1d[:, None] @ kernel_1d[None, :])

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
        mu_pred_sq, mu_target_sq = mu_pred * mu_pred, mu_target * mu_target
        mu_cross = mu_pred * mu_target

        sigma_pred_sq = _filter(pred * pred) - mu_pred_sq
        sigma_target_sq = _filter(target * target) - mu_target_sq
        sigma_cross = _filter(pred * target) - mu_cross

        # SSIM 稳定性常数，L=1（像素已归一化到 [0, 1]）。
        c1, c2 = 0.01**2, 0.03**2
        ssim_map = ((2.0 * mu_cross + c1) * (2.0 * sigma_cross + c2)) / (
            (mu_pred_sq + mu_target_sq + c1) * (sigma_pred_sq + sigma_target_sq + c2)
        )
        return ssim_map.mean()


class SSIMLoss(nn.Module):
    """SSIM 损失：``1 - SSIM``，值越小结构越一致。"""

    def __init__(self, window_size: int = 11, sigma: float = 1.5) -> None:
        """内部包装一个 SSIM 模块。

        Args:
            window_size: 高斯窗尺寸。
            sigma:       高斯标准差。
        """
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


class SobelEdgeLoss(nn.Module):
    """Sobel 边缘损失：约束生成图像与真值的边缘结构一致性。"""

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

        sobel_x, sobel_y = _sobel_kernels(kernel_size)
        self.register_buffer("sobel_x", sobel_x)
        self.register_buffer("sobel_y", sobel_y)

        # 可选的高斯平滑核（边缘图去噪）。
        if smooth_sigma is not None and smooth_sigma > 0:
            self.register_buffer("gaussian", _gaussian_kernel_2d(5, smooth_sigma))
        else:
            self.gaussian = None  # type: ignore[assignment]

    def _extract_edge(self, x: torch.Tensor) -> torch.Tensor:
        """用 Sobel 算子提取图像的边缘强度图（多通道取平均）。

        Args:
            x: 输入图像 ``(B, C, H, W)``，取值 ``[0, 1]``。

        Returns:
            torch.Tensor: 边缘强度图 ``(B, 1, H, W)``，值域非负。
        """
        batch, channels, height, width = x.shape
        # 将多通道展平为 (B*C, 1, H, W)，统一做卷积。
        x_flat = x.view(batch * channels, 1, height, width)
        padding = self.kernel_size // 2

        edge_x = F.conv2d(x_flat, self.sobel_x.to(x.dtype), padding=padding)
        edge_y = F.conv2d(x_flat, self.sobel_y.to(x.dtype), padding=padding)
        magnitude = torch.sqrt(edge_x**2 + edge_y**2 + 1e-6)

        # 还原为 (B, C, H, W) 后沿通道取平均 -> (B, 1, H, W)。
        return magnitude.view(batch, channels, height, width).mean(dim=1, keepdim=True)

    def _smooth(self, x: torch.Tensor) -> torch.Tensor:
        """对边缘图做可选的高斯平滑（降低噪声敏感度）。

        Args:
            x: 边缘图 ``(B, 1, H, W)``。

        Returns:
            torch.Tensor: 平滑后的边缘图。
        """
        if self.gaussian is None:
            return x
        return F.conv2d(x, self.gaussian.to(x.dtype), padding=2, groups=1)

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


class CrossMarkerConsistencyLoss(nn.Module):
    """跨标记一致性损失：强制共享特征独立于标记类型。

    对 AdapterUNet 的 bottleneck 共享特征，要求不同 marker 的适配前特征
    尽可能一致，即适配器只负责标记特异性变换，不破坏共享结构信息。
    """

    def __init__(self, margin: float = 0.0) -> None:
        """初始化损失。

        Args:
            margin: 预留的间隔参数，当前实现未使用（保持接口兼容）。
        """
        super().__init__()
        self.margin = margin

    def forward(
        self, shared_features: torch.Tensor, marker_idx: torch.Tensor
    ) -> torch.Tensor:
        """计算同一 batch 内不同标记样本的共享特征一致性。

        Args:
            shared_features: ``(B, C, H, W)``，适配器前的共享特征。
            marker_idx:      ``(B,)`` 整型，各样本的目标标记。

        Returns:
            torch.Tensor: 标量损失。
        """
        if shared_features.shape[0] < 2:
            return shared_features.new_zeros(())

        # 全局平均池化得到特征向量 (B, C)。
        pooled = F.adaptive_avg_pool2d(shared_features, (1, 1)).squeeze(-1).squeeze(-1)
        unique_markers = torch.unique(marker_idx)

        # 组内特征应彼此接近：惩罚到组均值的距离。
        loss = pooled.new_zeros(())
        for marker in unique_markers:
            group = pooled[marker_idx == marker]
            if group.shape[0] < 2:
                continue
            center = group.mean(dim=0, keepdim=True)
            loss = loss + F.mse_loss(group, center.expand_as(group))

        # 额外：不同 marker 的组均值也应接近（共享结构）。
        if len(unique_markers) > 1:
            centers = torch.stack(
                [
                    pooled[marker_idx == marker].mean(dim=0)
                    for marker in unique_markers
                    if (marker_idx == marker).sum() > 0
                ]
            )
            mean_center = centers.mean(dim=0, keepdim=True)
            loss = loss + F.mse_loss(centers, mean_center.expand_as(centers))

        return loss / max(len(unique_markers), 1)


class CombinedLoss(nn.Module):
    """加权组合损失，权重为 0 的分项自动关闭。"""

    def __init__(
        self,
        lambda_l1: float = 1.0,
        lambda_mse: float = 0.0,
        lambda_ssim: float = 1.0,
        lambda_edge: float = 0.0,
        lambda_cross: float = 0.0,
        edge_kernel_size: int = 3,
        edge_smooth_sigma: float = 1.0,
    ) -> None:
        """按权重装配各分项损失。

        Args:
            lambda_l1:         L1 损失权重。
            lambda_mse:        MSE 损失权重（PSNR 的直接代理）。
            lambda_ssim:       SSIM 损失权重。
            lambda_edge:       边缘损失权重。
            lambda_cross:      跨标记一致性损失权重（仅 adapter 模型有效）。
            edge_kernel_size:  Sobel 核尺寸，3 或 5。
            edge_smooth_sigma: 边缘图高斯平滑标准差，<=0 表示不平滑。
        """
        super().__init__()
        self.lambda_l1 = lambda_l1
        self.lambda_mse = lambda_mse
        self.lambda_ssim = lambda_ssim
        self.lambda_edge = lambda_edge
        self.lambda_cross = lambda_cross

        self.l1 = nn.L1Loss()
        self.mse = nn.MSELoss() if lambda_mse > 0 else None
        self.ssim = SSIMLoss() if lambda_ssim > 0 else None
        self.edge = (
            SobelEdgeLoss(
                kernel_size=edge_kernel_size,
                smooth_sigma=edge_smooth_sigma if edge_smooth_sigma > 0 else None,
            )
            if lambda_edge > 0
            else None
        )
        self.cross = CrossMarkerConsistencyLoss() if lambda_cross > 0 else None

    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> "CombinedLoss":
        """根据配置构建组合损失。

        Args:
            config: 全局配置，读取 ``loss`` 一节。

        Returns:
            CombinedLoss: 装配完成的组合损失。
        """
        loss_cfg = config.get("loss", {}) or {}
        return cls(
            lambda_l1=float(loss_cfg.get("lambda_l1", 1.0)),
            lambda_mse=float(loss_cfg.get("lambda_mse", 0.0)),
            lambda_ssim=float(loss_cfg.get("lambda_ssim", 1.0)),
            lambda_edge=float(loss_cfg.get("lambda_edge", 0.0)),
            lambda_cross=float(loss_cfg.get("lambda_cross", 0.0)),
            edge_kernel_size=int(loss_cfg.get("edge_kernel_size", 3)),
            edge_smooth_sigma=float(loss_cfg.get("edge_smooth_sigma", 1.0)),
        )

    def forward(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        aux: Optional[Dict[str, torch.Tensor]] = None,
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """计算加权总损失。

        Args:
            pred:   模型预测 ``(B, C, H, W)``。
            target: 真值图像。
            aux:    可选辅助信息（``shared_features`` / ``marker_idx``），
                供跨标记一致性损失使用。

        Returns:
            Tuple[torch.Tensor, Dict[str, float]]: (总损失, 各分项明细)。
        """
        total = pred.new_zeros(())
        details: Dict[str, float] = {}

        if self.lambda_l1 > 0:
            l1_value = self.l1(pred, target)
            total = total + self.lambda_l1 * l1_value
            details["l1"] = float(l1_value.detach())

        if self.mse is not None:
            mse_value = self.mse(pred, target)
            total = total + self.lambda_mse * mse_value
            details["mse"] = float(mse_value.detach())

        if self.ssim is not None:
            ssim_value = self.ssim(pred, target)
            total = total + self.lambda_ssim * ssim_value
            details["ssim"] = float(ssim_value.detach())

        if self.edge is not None:
            edge_value = self.edge(pred, target)
            total = total + self.lambda_edge * edge_value
            details["edge"] = float(edge_value.detach())

        if self.cross is not None and aux is not None:
            shared = aux.get("shared_features")
            marker_idx = aux.get("marker_idx")
            if shared is not None and marker_idx is not None:
                cross_value = self.cross(shared.float(), marker_idx)
                total = total + self.lambda_cross * cross_value
                details["cross"] = float(cross_value.detach())

        details["total"] = float(total.detach())
        return total, details
