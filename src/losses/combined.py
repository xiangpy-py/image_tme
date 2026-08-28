"""组合损失：L = λ1*L1 + λ2*SSIM + λ3*Edge + λ4*Perceptual + λ5*CrossMarker。

对应渐进式实验路线：
    Level 0: L1
    Level 1: L1 + SSIM
    Level 2: L1 + SSIM + Edge
    Level 3: L1 + SSIM + Edge + Perceptual
    Level 4: 一对多联合建模时追加 CrossMarker 一致性约束
"""

from typing import Any, Dict, Optional, Tuple

import torch
import torch.nn as nn

from .cross_marker_loss import CrossMarkerConsistencyLoss
from .edge_loss import SobelEdgeLoss
from .perceptual import PerceptualLoss
from .ssim_loss import SSIMLoss


class CombinedLoss(nn.Module):
    """加权组合损失，返回总损失与各分项便于日志记录。"""

    def __init__(
        self,
        lambda_l1: float = 1.0,
        lambda_ssim: float = 1.0,
        lambda_edge: float = 0.0,
        lambda_perceptual: float = 0.0,
        lambda_cross: float = 0.0,
        edge_kernel_size: int = 3,
        edge_smooth_sigma: float = 1.0,
        perceptual_pretrained: bool = True,
    ) -> None:
        super().__init__()
        self.lambda_l1 = lambda_l1
        self.lambda_ssim = lambda_ssim
        self.lambda_edge = lambda_edge
        self.lambda_perceptual = lambda_perceptual
        self.lambda_cross = lambda_cross

        self.l1 = nn.L1Loss()
        self.ssim = SSIMLoss() if lambda_ssim > 0 else None
        self.edge = (
            SobelEdgeLoss(
                kernel_size=edge_kernel_size,
                smooth_sigma=edge_smooth_sigma if edge_smooth_sigma > 0 else None,
            )
            if lambda_edge > 0
            else None
        )
        self.perceptual = (
            PerceptualLoss(pretrained=perceptual_pretrained)
            if lambda_perceptual > 0
            else None
        )
        # 跨标记一致性仅对能暴露共享特征的模型（如 AdapterUNet）生效。
        self.cross = CrossMarkerConsistencyLoss() if lambda_cross > 0 else None

    def forward(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        aux: Optional[Dict[str, torch.Tensor]] = None,
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """计算加权总损失。

        Args:
            pred:   模型预测 ``(B, C, H, W)``。
            target: 真值图像，形状与取值同 ``pred``。
            aux:    可选的辅助信息，跨标记一致性损失需要
                ``shared_features``（适配器前的共享特征）与
                ``marker_idx``（各样本的目标标记编号）。

        Returns:
            Tuple[torch.Tensor, Dict[str, float]]: (总损失, 各分项明细)。
        """
        total = pred.new_zeros(())
        details: Dict[str, float] = {}

        if self.lambda_l1 > 0:
            l1_value = self.l1(pred, target)
            total = total + self.lambda_l1 * l1_value
            details["l1"] = float(l1_value.detach())

        if self.ssim is not None:
            ssim_value = self.ssim(pred, target)
            total = total + self.lambda_ssim * ssim_value
            details["ssim"] = float(ssim_value.detach())

        if self.edge is not None:
            edge_value = self.edge(pred, target)
            total = total + self.lambda_edge * edge_value
            details["edge"] = float(edge_value.detach())

        if self.perceptual is not None:
            perceptual_value = self.perceptual(pred, target)
            total = total + self.lambda_perceptual * perceptual_value
            details["perceptual"] = float(perceptual_value.detach())

        # 模型未返回共享特征时跳过该项，保证对普通模型完全兼容。
        if self.cross is not None and aux is not None:
            shared = aux.get("shared_features")
            marker_idx = aux.get("marker_idx")
            if shared is not None and marker_idx is not None:
                cross_value = self.cross(shared.float(), marker_idx)
                total = total + self.lambda_cross * cross_value
                details["cross"] = float(cross_value.detach())

        details["total"] = float(total.detach())
        return total, details


def build_loss(config: Dict[str, Any]) -> CombinedLoss:
    """根据配置构建组合损失。

    Args:
        config: 全局配置，读取 ``loss`` 一节。

    Returns:
        CombinedLoss: 配置实例化后的组合损失。
    """
    loss_cfg = config.get("loss", {}) or {}
    return CombinedLoss(
        lambda_l1=float(loss_cfg.get("lambda_l1", 1.0)),
        lambda_ssim=float(loss_cfg.get("lambda_ssim", 1.0)),
        lambda_edge=float(loss_cfg.get("lambda_edge", 0.0)),
        lambda_perceptual=float(loss_cfg.get("lambda_perceptual", 0.0)),
        lambda_cross=float(loss_cfg.get("lambda_cross", 0.0)),
        edge_kernel_size=int(loss_cfg.get("edge_kernel_size", 3)),
        edge_smooth_sigma=float(loss_cfg.get("edge_smooth_sigma", 1.0)),
        perceptual_pretrained=bool(loss_cfg.get("perceptual_pretrained", True)),
    )
