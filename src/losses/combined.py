"""组合损失：L = λ1*L1 + λ2*SSIM + λ3*Edge + λ4*Perceptual。

对应渐进式实验路线：
    Level 0: L1
    Level 1: L1 + SSIM
    Level 2: L1 + SSIM + Edge
    Level 3: L1 + SSIM + Edge + Perceptual
"""

from typing import Any, Dict, Tuple

import torch
import torch.nn as nn

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
        edge_kernel_size: int = 3,
        edge_smooth_sigma: float = 1.0,
        perceptual_pretrained: bool = True,
    ) -> None:
        super().__init__()
        self.lambda_l1 = lambda_l1
        self.lambda_ssim = lambda_ssim
        self.lambda_edge = lambda_edge
        self.lambda_perceptual = lambda_perceptual

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

    def forward(
        self, pred: torch.Tensor, target: torch.Tensor
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
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

        details["total"] = float(total.detach())
        return total, details


def build_loss(config: Dict[str, Any]) -> CombinedLoss:
    loss_cfg = config.get("loss", {}) or {}
    return CombinedLoss(
        lambda_l1=float(loss_cfg.get("lambda_l1", 1.0)),
        lambda_ssim=float(loss_cfg.get("lambda_ssim", 1.0)),
        lambda_edge=float(loss_cfg.get("lambda_edge", 0.0)),
        lambda_perceptual=float(loss_cfg.get("lambda_perceptual", 0.0)),
        edge_kernel_size=int(loss_cfg.get("edge_kernel_size", 3)),
        edge_smooth_sigma=float(loss_cfg.get("edge_smooth_sigma", 1.0)),
        perceptual_pretrained=bool(loss_cfg.get("perceptual_pretrained", True)),
    )
