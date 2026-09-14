"""评测指标与统计容器。

比赛评分公式::

    Score = 70% * SSIM + 30% * Normalize(PSNR)

对齐赛题口径：

- PSNR 的 MSE 是**全部像素**的均方误差 ``(1/N)·Σ(x_i-y_i)²``，
  再代入 ``10·log10(MAX²/MSE)`` 计算一次（``MAX`` 为 8-bit 图像的 255）；
  因此累计器需汇总误差平方和与像素总数，而非对逐图 PSNR 取平均。
- 图像在内部以 ``[0, 1]`` 浮点参与计算，此时 ``MAX=1``，
  与 8-bit 的 ``MAX=255`` 数值上完全等价（分子分母同缩放 255²）。
- SSIM 使用 11×11 高斯加权窗、K1=0.01、K2=0.03。

``Normalize(PSNR)`` 赛题未给出明确定义，取最常见做法：把 PSNR 线性裁剪
到 ``[0, 40] dB`` 再除以 40 归一化到 ``[0, 1]``。
"""

import math
from typing import Dict

import torch

# Normalize(PSNR) 的归一化上界（dB）。
PSNR_UPPER_BOUND: float = 40.0

# 复用常驻 SSIM 模块：验证阶段逐 batch 调用，重复构建会反复申请高斯窗，
# 按设备缓存可显著降低开销（延迟导入 SSIM 以避免循环依赖）。
_SSIM_CACHE: Dict[str, "SSIM"] = {}


class Metrics:
    """比赛指标计算：SSIM、PSNR 与综合得分。"""

    @staticmethod
    def ssim(pred: torch.Tensor, target: torch.Tensor) -> float:
        """计算批量平均 SSIM。

        与损失模块共用同一实现，保证训练目标与评测口径一致。

        Args:
            pred:   预测图像 ``(B, C, H, W)``，取值 ``[0, 1]``。
            target: 真值图像。

        Returns:
            float: batch 平均 SSIM，范围大致 ``[-1, 1]``。
        """
        from .losses import SSIM  # 延迟导入避免循环依赖

        device_key = str(pred.device)
        module = _SSIM_CACHE.get(device_key)
        if module is None:
            module = SSIM().to(pred.device)
            _SSIM_CACHE[device_key] = module

        with torch.no_grad():
            return float(module(pred, target))

    @staticmethod
    def psnr_from_mse(mse: float) -> float:
        """按赛题公式由 MSE 计算 PSNR（单位 dB）。

        ``PSNR = 10·log10(MAX² / MSE)``，其中 MSE 为全部像素的均方误差。
        内部图像取值 ``[0, 1]``，故 ``MAX=1``，与 8-bit 的 ``MAX=255`` 等价。

        Args:
            mse: 全局均方误差。

        Returns:
            float: PSNR；MSE 为 0 时通过下限截断返回极大值。
        """
        return float(10.0 * math.log10(1.0 / max(mse, 1e-12)))

    @staticmethod
    def score(ssim_value: float, psnr_value: float) -> float:
        """按比赛公式计算综合得分。

        Args:
            ssim_value: SSIM 指标值。
            psnr_value: PSNR 指标值（dB）。

        Returns:
            float: 综合得分，范围 ``[0, 1]``，越高越好。
        """
        normalized_psnr = min(max(psnr_value, 0.0), PSNR_UPPER_BOUND) / PSNR_UPPER_BOUND
        return 0.7 * ssim_value + 0.3 * normalized_psnr


class AverageMeter:
    """滑动平均值统计器，用于记录 loss 等标量指标。"""

    def __init__(self) -> None:
        """初始化计数与累加和。"""
        self.count: int = 0
        self.sum: float = 0.0

    def reset(self) -> None:
        """清空全部统计量。

        Returns:
            None
        """
        self.count = 0
        self.sum = 0.0

    def update(self, value: float, n: int = 1) -> None:
        """累加一个 batch 的指标值。

        Args:
            value: 当前 batch 的指标均值。
            n:     当前 batch 的样本数（用于按样本数加权）。

        Returns:
            None
        """
        self.sum += value * n
        self.count += n

    @property
    def avg(self) -> float:
        """float: 当前累计平均值，无样本时返回 0。"""
        return self.sum / self.count if self.count else 0.0


class MetricAccumulator:
    """验证/评测阶段的指标累计器。

    逐 batch 累计 SSIM / PSNR（按样本数加权），
    结束时输出均值与比赛综合得分。
    """

    def __init__(self) -> None:
        """初始化累计量。"""
        self.reset()

    def reset(self) -> None:
        """清空累计状态，开始新一轮评测。

        Returns:
            None
        """
        self.count: int = 0
        self.ssim_sum: float = 0.0
        self.sse_sum: float = 0.0
        self.pixel_count: int = 0

    def update(self, pred: torch.Tensor, target: torch.Tensor) -> None:
        """累计一个 batch 的指标。

        PSNR 要求基于全部像素的全局 MSE，因此这里累计误差平方和与像素总数，
        在 ``compute`` 阶段一次性换算，而不是对逐 batch 的 PSNR 取平均。

        Args:
            pred:   预测图像 ``(B, C, H, W)``。
            target: 真值图像。

        Returns:
            None
        """
        batch_size = pred.shape[0]
        self.ssim_sum += Metrics.ssim(pred, target) * batch_size
        self.count += batch_size
        self.sse_sum += float(((pred - target) ** 2).sum())
        self.pixel_count += pred.numel()

    def compute(self) -> Dict[str, float]:
        """输出汇总指标。

        Returns:
            Dict[str, float]: 含 ``ssim`` / ``psnr`` / ``score``，
            无样本时全部返回 0。
        """
        if self.count == 0 or self.pixel_count == 0:
            return {"ssim": 0.0, "psnr": 0.0, "score": 0.0}

        mean_ssim = self.ssim_sum / self.count
        mse = self.sse_sum / self.pixel_count
        mean_psnr = Metrics.psnr_from_mse(mse)
        return {
            "ssim": mean_ssim,
            "psnr": mean_psnr,
            "score": Metrics.score(mean_ssim, mean_psnr),
        }
