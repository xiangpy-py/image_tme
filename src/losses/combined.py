"""组合损失：L = λ1*L1 + λ2*SSIM + λ3*Perceptual。

对应 plan.md 第 8 节的 Loss 设计：权重通过 YAML 配置，
支持「先 L1+SSIM 基线，再逐步叠加感知损失」的实验路线。
"""

from typing import Any, Dict, Tuple

import torch
import torch.nn as nn

from .perceptual import PerceptualLoss
from .ssim_loss import SSIMLoss


class CombinedLoss(nn.Module):
    """加权组合损失，返回总损失与各分项便于日志记录。"""

    def __init__(
        self,
        lambda_l1: float = 1.0,
        lambda_ssim: float = 1.0,
        lambda_perceptual: float = 0.0,
        perceptual_pretrained: bool = True,
    ) -> None:
        """按权重构建各损失子模块。

        Args:
            lambda_l1:             L1 像素损失权重。
            lambda_ssim:           SSIM 结构损失权重。
            lambda_perceptual:     感知损失权重，0 表示不构建（节省显存）。
            perceptual_pretrained: 感知损失是否加载预训练 VGG。
        """
        super().__init__()
        self.lambda_l1 = lambda_l1
        self.lambda_ssim = lambda_ssim
        self.lambda_perceptual = lambda_perceptual

        self.l1 = nn.L1Loss()
        self.ssim = SSIMLoss() if lambda_ssim > 0 else None
        self.perceptual = (
            PerceptualLoss(pretrained=perceptual_pretrained)
            if lambda_perceptual > 0
            else None
        )

    def forward(
        self, pred: torch.Tensor, target: torch.Tensor
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """计算加权总损失。

        Args:
            pred:   预测图像 ``(B, C, H, W)``。
            target: 真值图像，形状与取值同 ``pred``。

        Returns:
            Tuple[torch.Tensor, Dict[str, float]]:
                - 标量总损失（带梯度，用于反向传播）；
                - 各分项损失的 float 值（仅用于日志，已 detach）。
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

        if self.perceptual is not None:
            perceptual_value = self.perceptual(pred, target)
            total = total + self.lambda_perceptual * perceptual_value
            details["perceptual"] = float(perceptual_value.detach())

        details["total"] = float(total.detach())
        return total, details


def build_loss(config: Dict[str, Any]) -> CombinedLoss:
    """根据配置构建组合损失。

    Args:
        config: 全局配置字典，读取 ``loss`` 一节。

    Returns:
        CombinedLoss: 组合损失实例。
    """
    loss_cfg = config.get("loss", {}) or {}
    return CombinedLoss(
        lambda_l1=float(loss_cfg.get("lambda_l1", 1.0)),
        lambda_ssim=float(loss_cfg.get("lambda_ssim", 1.0)),
        lambda_perceptual=float(loss_cfg.get("lambda_perceptual", 0.0)),
        perceptual_pretrained=bool(loss_cfg.get("perceptual_pretrained", True)),
    )
