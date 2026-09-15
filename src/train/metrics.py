"""评测指标与统计容器。

比赛评分公式（已由赛方自动评分反馈反推确认）::

    Score = 70 × SSIM + 30 × Normalize(PSNR)        # 百分制
    Normalize(PSNR) = clip(PSNR, 0, 50) / 50

**反推依据**（两次真实提交的官方反馈）：

- 反馈一：SSIM = 0.804 / 0.790 / 0.784 / 0.762，PSNR = 25.393 / 22.462 /
  21.758 / 21.587（四标记均值 0.7850 / 22.8000），对应总分 **68.6337**。
  代入上式反解归一化分母得 ``D = 49.99``；按 ``D = 50`` 复算得 68.6300，
  与反馈仅差 0.0037（远小于指标三位小数的舍入量）。
- 因此归一化上界取 **50 dB**，而非早期按经验假定的 40 dB。用错上界会把
  本地读数系统性抬高约 3.5 分（同一模型 71.66 对实际的 68.63），
  这正是此前「本地分数与线上对不上」的全部来源。

**口径细节**：

- **PSNR 采用「逐图计算再跨图平均」**。官方反馈的 PSNR 一致高于本地按
  全局 MSE 计算的读数（四标记分别 +0.18 / +0.79 / +1.88 / +1.49 dB）。
  由于 ``mean_i[-10·log10(MSE_i)] ≥ -10·log10(mean_i[MSE_i])``，
  逐图平均是系统性偏高的，且偏差随逐图 MSE 的相对离散度放大——与观测到
  「不同标记偏差幅度不同」一致。故累计器以**逐图 PSNR 为主口径**，
  同时保留全局 MSE 口径 ``psnr_global`` 供诊断对照。
- **SSIM 采用逐图平均**：等尺寸图像下批内 SSIM 图均值等价于逐图平均。
- SSIM 使用 11×11 高斯加权窗、K1=0.01、K2=0.03，与训练损失同一实现。
- 图像内部以 ``[0, 1]`` 浮点参与计算，此时 ``MAX=1``，
  与 8-bit 的 ``MAX=255`` 数值上完全等价（分子分母同缩放 255²）。

> 若后续线上读数与本地仍不一致，先比较反馈值更接近 ``psnr`` 还是
> ``psnr_global``，据此判断赛方实际使用的 PSNR 聚合口径。
"""

import math
import re
from typing import Dict, List, Optional

import torch

from ..data.constants import MARKERS

# Normalize(PSNR) 的归一化上界（dB）：50 由官方反馈反推确认（见模块文档）。
PSNR_UPPER_BOUND: float = 50.0

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
            float: 综合得分，范围 ``[0, 1]``，越高越好（×100 即赛方百分制）。
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

    逐 batch 累计 SSIM 与 PSNR（按样本数加权），结束时输出均值与综合得分。

    PSNR 的主口径为**逐图计算再跨图平均**（与赛方反馈一致，见模块文档）；
    同时保留按全局 MSE 一次性换算的口径 ``psnr_global`` 供诊断对照——
    两者之差即为「PSNR 聚合方式」带来的系统性偏移。
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
        self.psnr_sum: float = 0.0
        self.sse_sum: float = 0.0
        self.pixel_count: int = 0

    def update(self, pred: torch.Tensor, target: torch.Tensor) -> None:
        """累计一个 batch 的指标。

        Args:
            pred:   预测图像 ``(B, C, H, W)``，取值 ``[0, 1]``。
            target: 真值图像，形状与取值同 ``pred``。

        Returns:
            None
        """
        batch_size = pred.shape[0]
        self.ssim_sum += Metrics.ssim(pred, target) * batch_size
        self.count += batch_size

        squared = (pred - target) ** 2
        # 主口径：逐图 MSE -> 逐图 PSNR -> 跨图平均（对齐赛方评分脚本）。
        per_image_mse = squared.flatten(1).mean(dim=1)
        per_image_psnr = 10.0 * torch.log10(1.0 / per_image_mse.clamp_min(1e-12))
        self.psnr_sum += float(per_image_psnr.sum())

        # 诊断口径：全部像素的全局 MSE，只在 compute 阶段换算一次 PSNR。
        self.sse_sum += float(squared.sum())
        self.pixel_count += pred.numel()

    def compute(self) -> Dict[str, float]:
        """输出汇总指标。

        Returns:
            Dict[str, float]: 含 ``ssim`` / ``psnr``（逐图平均，对齐赛方）/
            ``psnr_global``（全局 MSE）/ ``score``；无样本时全部返回 0。
        """
        if self.count == 0 or self.pixel_count == 0:
            return {"ssim": 0.0, "psnr": 0.0, "psnr_global": 0.0, "score": 0.0}

        mean_ssim = self.ssim_sum / self.count
        mean_psnr = self.psnr_sum / self.count
        psnr_global = Metrics.psnr_from_mse(self.sse_sum / self.pixel_count)
        return {
            "ssim": mean_ssim,
            "psnr": mean_psnr,
            "psnr_global": psnr_global,
            "score": Metrics.score(mean_ssim, mean_psnr),
        }


class OfficialFeedback:
    """赛方自动评分反馈的解析与本地对照。

    反馈文本形如::

        自动评分完成；输出类型=CD68,CD45RO,HLA-DR,Vimentin；输出数=4；
        缺失图=0；异常图=0；指标=CD68:SSIM=0.802,PSNR=25.344;CD45RO:...;

    仅做「文本 -> 结构化指标」的解析与「线上 vs 本地」对照，
    不参与训练；用于校验本地验证集能否预测赛方测试集得分。
    """

    # 逐标记指标片段：SSIM=..,PSNR=..，分隔符兼容中英文标点与任意空白。
    _MARKER_PATTERN = re.compile(
        r"(CD68|CD45RO|HLA-DR|Vimentin)\s*[:：]\s*"
        r"SSIM\s*[=＝]\s*([0-9.]+)\s*[;,，；\s]*\s*"
        r"PSNR\s*[=＝]\s*([0-9.]+)",
        re.IGNORECASE,
    )
    _SCORE_PATTERN = re.compile(
        r"(?:Score|score|得分|总分)\s*[=＝:：]\s*([0-9.]+)"
    )

    def __init__(
        self,
        metrics: Dict[str, Dict[str, float]],
        score: Optional[float] = None,
    ) -> None:
        """保存解析结果。

        Args:
            metrics: ``{标记名: {"ssim": 值, "psnr": 值}}``。
            score:   反馈中若含总分则记录（百分制），否则为 ``None``。
        """
        self.metrics = metrics
        self.score = score

    @classmethod
    def parse(cls, text: str) -> "OfficialFeedback":
        """把反馈文本解析为结构化指标。

        Args:
            text: 赛方反馈原文（可含任意前后缀修饰）。

        Returns:
            OfficialFeedback: 解析结果；未匹配到任何标记时 ``metrics`` 为空。

        Raises:
            ValueError: 文本中未解析出任何标记指标时抛出（避免静默通过）。
        """
        metrics: Dict[str, Dict[str, float]] = {}
        for marker, ssim_text, psnr_text in cls._MARKER_PATTERN.findall(text):
            # 归一化到 constants.MARKERS 的规范写法（忽略大小写差异）。
            canonical = next(
                (name for name in MARKERS if name.lower() == marker.lower()),
                marker.upper(),
            )
            metrics[canonical] = {
                "ssim": float(ssim_text),
                "psnr": float(psnr_text),
            }
        if not metrics:
            raise ValueError(
                "未能从反馈文本解析出任何标记指标，请检查格式，"
                "例如: 指标=CD68:SSIM=0.802,PSNR=25.344;CD45RO:..."
            )

        score_match = cls._SCORE_PATTERN.search(text)
        score = float(score_match.group(1)) if score_match else None
        return cls(metrics, score)

    def summarize(self) -> Dict[str, float]:
        """汇总线上四标记的均值指标。

        Returns:
            Dict[str, float]: 含 ``ssim`` / ``psnr`` / 按 70/30 公式复算的
            ``score``（百分制）以及线上总分 ``score_reported``
            （反馈未给出时等于复算值）。
        """
        count = len(self.metrics)
        mean_ssim = sum(item["ssim"] for item in self.metrics.values()) / count
        mean_psnr = sum(item["psnr"] for item in self.metrics.values()) / count
        derived = 100.0 * Metrics.score(mean_ssim, mean_psnr)
        return {
            "ssim": mean_ssim,
            "psnr": mean_psnr,
            "score": derived,
            "score_reported": self.score if self.score is not None else derived,
        }

    def compare(self, local: Dict[str, Dict[str, float]]) -> List[Dict[str, float]]:
        """逐标记对照线上与本地读数。

        同时给出与本地两种 PSNR 口径的差值，用于判断赛方实际采用的
        聚合方式（``psnr`` 为逐图平均口径，``psnr_global`` 为全局 MSE 口径）。

        Args:
            local: 本地评测结果 ``{标记名: {ssim, psnr, psnr_global, score}}``。

        Returns:
            List[Dict[str, float]]: 逐标记对照行，字段含
            ``marker`` / ``ssim_online`` / ``ssim_local`` / ``d_ssim``
            ``psnr_online`` / ``psnr_local`` / ``d_psnr``
            ``psnr_global_local`` / ``d_psnr_global``
            ``score_online`` / ``score_local``。
        """
        rows: List[Dict[str, float]] = []
        for marker, online in self.metrics.items():
            if marker not in local:
                continue
            item = local[marker]
            psnr_global_local = item.get("psnr_global", item["psnr"])
            rows.append(
                {
                    "marker": marker,
                    "ssim_online": online["ssim"],
                    "ssim_local": item["ssim"],
                    "d_ssim": online["ssim"] - item["ssim"],
                    "psnr_online": online["psnr"],
                    "psnr_local": item["psnr"],
                    "d_psnr": online["psnr"] - item["psnr"],
                    "psnr_global_local": psnr_global_local,
                    "d_psnr_global": online["psnr"] - psnr_global_local,
                    "score_online": 100.0 * Metrics.score(online["ssim"], online["psnr"]),
                    "score_local": 100.0 * item["score"],
                }
            )
        return rows
