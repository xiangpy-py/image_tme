"""损失子包：L1 / SSIM / 边缘 / 感知 / 跨标记一致性损失及加权组合。"""

from .ssim_loss import SSIM, SSIMLoss
from .edge_loss import SobelEdgeLoss
from .perceptual import PerceptualLoss
from .cross_marker_loss import CrossMarkerConsistencyLoss
from .combined import CombinedLoss, build_loss

__all__ = [
    "SSIM",
    "SSIMLoss",
    "SobelEdgeLoss",
    "PerceptualLoss",
    "CrossMarkerConsistencyLoss",
    "CombinedLoss",
    "build_loss",
]
