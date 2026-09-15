"""单作业训练子系统：数据 / 模型 / 损失 / 指标 / 训练与推理引擎。

本子包只负责「一个训练作业」的完整实现，不感知并行调度与实验编排，
供上层操作器（Operator）与集成器（Ensembler）调用。

与训练无关的通用数据能力（标记常量、图像读取、ROI 划分、统计）位于
``src.data``。
"""

from .data import (
    DataLoaders,
    MultiMarkerDataset,
    PairedTransform,
    VirtualStainingDataset,
)
from .engine import MarkerEvaluator, Predictor, Trainer
from .losses import (
    CombinedLoss,
    CrossMarkerConsistencyLoss,
    SSIM,
    SSIMLoss,
    SobelEdgeLoss,
)
from .metrics import AverageMeter, MetricAccumulator, Metrics
from .models import (
    AdapterUNet,
    ConditionalResAttentionUNet,
    ConditionalUNet,
    ConditionalUNetV2,
    ModelRegistry,
    ResNetUNet,
    UNet,
)

__all__ = [
    # 数据
    "PairedTransform",
    "VirtualStainingDataset",
    "MultiMarkerDataset",
    "DataLoaders",
    # 模型
    "UNet",
    "ResNetUNet",
    "ConditionalUNet",
    "ConditionalUNetV2",
    "AdapterUNet",
    "ConditionalResAttentionUNet",
    "ModelRegistry",
    # 损失
    "SSIM",
    "SSIMLoss",
    "SobelEdgeLoss",
    "TVLoss",
    "CrossMarkerConsistencyLoss",
    "CombinedLoss",
    # 指标
    "Metrics",
    "MetricAccumulator",
    "AverageMeter",
    # 引擎
    "Trainer",
    "Predictor",
]
