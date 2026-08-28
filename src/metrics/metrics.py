"""评测指标：SSIM、PSNR 与比赛综合得分。

比赛评分公式::

    Score = 70% * SSIM + 30% * Normalize(PSNR)

其中 Normalize(PSNR) 将 PSNR 线性裁剪归一化到 [0, 1]，
默认上界取 40 dB（可通过参数调整）。
"""

from typing import Dict

import torch


def ssim(pred: torch.Tensor, target: torch.Tensor) -> float:
    """计算批量平均 SSIM。

    与损失模块共用同一实现，保证训练目标与评测口径一致。

    Args:
        pred:   预测图像 ``(B, C, H, W)``，取值 ``[0, 1]``。
        target: 真值图像。

    Returns:
        float: batch 平均 SSIM，范围大致 ``[-1, 1]``。
    """
    from ..losses.ssim_loss import SSIM  # 延迟导入避免循环依赖

    metric = SSIM().to(pred.device)
    with torch.no_grad():
        return float(metric(pred, target))


def psnr(pred: torch.Tensor, target: torch.Tensor, max_value: float = 1.0) -> float:
    """计算批量平均 PSNR（单位 dB）。

    Args:
        pred:      预测图像 ``(B, C, H, W)``，取值 ``[0, 1]``。
        target:    真值图像。
        max_value: 像素最大值，归一化图像为 1.0。

    Returns:
        float: batch 平均 PSNR；MSE 为 0 时返回 ``inf``。
    """
    mse = torch.mean((pred - target) ** 2, dim=(1, 2, 3))
    mse = mse.clamp_min(1e-12)  # 避免 log(0)
    psnr_per_image = 10.0 * torch.log10(max_value ** 2 / mse)
    return float(psnr_per_image.mean())


def compute_competition_score(
    ssim_value: float,
    psnr_value: float,
    psnr_upper: float = 40.0,
) -> float:
    """按比赛公式计算综合得分。

    Args:
        ssim_value: SSIM 指标值。
        psnr_value: PSNR 指标值（dB）。
        psnr_upper: PSNR 归一化上界，超出部分截断。

    Returns:
        float: 综合得分，范围 ``[0, 1]``，越高越好。
    """
    normalized_psnr = min(max(psnr_value, 0.0), psnr_upper) / psnr_upper
    return 0.7 * ssim_value + 0.3 * normalized_psnr


class MetricAccumulator:
    """验证/评测阶段的指标累计器。

    逐 batch 累计 SSIM / PSNR（按样本数加权），
    结束时输出均值与比赛综合得分。
    """

    def __init__(self) -> None:
        """初始化累计量。"""
        self.reset()

    def reset(self) -> None:
        """清空累计状态，开始新一轮评测。"""
        self.count: int = 0
        self.ssim_sum: float = 0.0
        self.psnr_sum: float = 0.0

    def update(self, pred: torch.Tensor, target: torch.Tensor) -> None:
        """累计一个 batch 的指标。

        Args:
            pred:   预测图像 ``(B, C, H, W)``。
            target: 真值图像。

        Returns:
            None
        """
        batch_size = pred.shape[0]
        self.ssim_sum += ssim(pred, target) * batch_size
        self.psnr_sum += psnr(pred, target) * batch_size
        self.count += batch_size

    def compute(self) -> Dict[str, float]:
        """输出汇总指标。

        Returns:
            Dict[str, float]: 含 ``ssim`` / ``psnr`` / ``score`` 三项，
            无样本时全部返回 0。
        """
        if self.count == 0:
            return {"ssim": 0.0, "psnr": 0.0, "score": 0.0}

        mean_ssim = self.ssim_sum / self.count
        mean_psnr = self.psnr_sum / self.count
        return {
            "ssim": mean_ssim,
            "psnr": mean_psnr,
            "score": compute_competition_score(mean_ssim, mean_psnr),
        }
