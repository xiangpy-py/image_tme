"""感知损失（Perceptual Loss）。

利用 ImageNet 预训练 VGG16 的中间特征衡量生成图像与真值
在高层语义/纹理上的差异，弥补 L1 只约束像素级误差的不足。
属于 plan.md 中 Loss 设计的可选第三项（λ3 * Perceptual），
通过配置权重为 0 即可关闭。
"""

from typing import List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models


class VGGFeatureExtractor(nn.Module):
    """VGG16 特征提取器：输出多个中间层特征图。

    仅保留特征部分并冻结全部参数，作为固定的感知度量网络。
    """

    # 选取 relu1_2 / relu2_2 / relu3_3 三层，兼顾浅层纹理与中层结构。
    LAYER_INDICES: List[int] = [3, 8, 15]

    def __init__(self, pretrained: bool = True) -> None:
        """加载 VGG16 并冻结参数。

        Args:
            pretrained: 是否加载 ImageNet 预训练权重；
                离线评测环境可设为 ``False`` 退化为随机特征。
        """
        super().__init__()
        weights = models.VGG16_Weights.IMAGENET1K_V1 if pretrained else None
        vgg = models.vgg16(weights=weights).features
        self.blocks = vgg[: max(self.LAYER_INDICES) + 1]

        for param in self.parameters():
            param.requires_grad = False

        # ImageNet 归一化常数，注册为 buffer 自动迁移设备。
        self.register_buffer("mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))

    def forward(self, x: torch.Tensor) -> List[torch.Tensor]:
        """提取多层特征。

        Args:
            x: 输入图像 ``(B, C, H, W)``，取值 ``[0, 1]``。

        Returns:
            List[torch.Tensor]: 各选取层的特征图列表。
        """
        # 单通道图像复制为三通道以匹配 VGG 输入。
        if x.shape[1] == 1:
            x = x.repeat(1, 3, 1, 1)
        x = (x - self.mean) / self.std

        features = []
        for index, layer in enumerate(self.blocks):
            x = layer(x)
            if index in self.LAYER_INDICES:
                features.append(x)
        return features


class PerceptualLoss(nn.Module):
    """多层 VGG 特征的 L1 距离。"""

    def __init__(self, pretrained: bool = True) -> None:
        """构建特征提取器。"""
        super().__init__()
        self.extractor = VGGFeatureExtractor(pretrained=pretrained)

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """计算感知损失。

        Args:
            pred:   预测图像 ``(B, C, H, W)``，取值 ``[0, 1]``。
            target: 真值图像。

        Returns:
            torch.Tensor: 标量损失，各层特征 L1 距离之和。
        """
        pred_features = self.extractor(pred)
        target_features = self.extractor(target)

        loss = pred.new_zeros(())
        for pred_feat, target_feat in zip(pred_features, target_features):
            loss = loss + F.l1_loss(pred_feat, target_feat)
        return loss
