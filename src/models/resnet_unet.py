"""Version 2: ResNet 编码器 + U-Net 解码器。

结构::

    DAPI -> ResNet Encoder(可选 ImageNet 预训练) -> UNet Decoder -> IHC

利用预训练 ResNet 的强特征提取能力提升空间结构恢复能力。
赛题明确允许使用公开预训练权重。
"""

from typing import List

import torch
import torch.nn as nn
from torchvision import models

from .blocks import Up


class ResNetUNet(nn.Module):
    """以 torchvision ResNet 为编码器的 U 型生成网络。

    编码器取 ResNet 各 stage 输出作为多尺度特征，
    解码器与标准 U-Net 相同，通过跳跃连接逐级恢复分辨率。
    """

    def __init__(
        self,
        out_channels: int = 1,
        backbone: str = "resnet34",
        pretrained: bool = True,
    ) -> None:
        """加载 ResNet 编码器并搭建解码器。

        Args:
            out_channels: 输出通道数。
            backbone:     编码器名称，支持 ``resnet18`` / ``resnet34``。
            pretrained:   是否加载 ImageNet 预训练权重（赛题允许）。

        Raises:
            ValueError: 不支持的编码器名称时抛出。
        """
        super().__init__()

        if backbone == "resnet18":
            weights = models.ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
            resnet = models.resnet18(weights=weights)
            stage_channels = [64, 64, 128, 256, 512]
        elif backbone == "resnet34":
            weights = models.ResNet34_Weights.IMAGENET1K_V1 if pretrained else None
            resnet = models.resnet34(weights=weights)
            stage_channels = [64, 64, 128, 256, 512]
        else:
            raise ValueError(f"不支持的编码器: {backbone}")

        # ---- 编码器：拆解 ResNet 为若干 stage ----
        self.stage0 = nn.Sequential(resnet.conv1, resnet.bn1, resnet.relu)  # 1/2
        self.pool = resnet.maxpool
        self.stage1 = resnet.layer1  # 1/4
        self.stage2 = resnet.layer2  # 1/8
        self.stage3 = resnet.layer3  # 1/16
        self.stage4 = resnet.layer4  # 1/32

        # ---- 解码器：与 stage_channels 一一对应，自深向浅 ----
        self.up4 = Up(stage_channels[4], stage_channels[3], stage_channels[3])
        self.up3 = Up(stage_channels[3], stage_channels[2], stage_channels[2])
        self.up2 = Up(stage_channels[2], stage_channels[1], stage_channels[1])
        self.up1 = Up(stage_channels[1], stage_channels[0], stage_channels[0])
        # 回到原分辨率（stage0 输出为 1/2 尺度，需再升 2 倍）。
        self.up0 = nn.ConvTranspose2d(stage_channels[0], stage_channels[0], 2, stride=2)
        self.refine = nn.Sequential(
            nn.Conv2d(stage_channels[0], stage_channels[0], 3, padding=1, bias=False),
            nn.BatchNorm2d(stage_channels[0]),
            nn.ReLU(inplace=True),
        )
        self.head = nn.Conv2d(stage_channels[0], out_channels, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """前向传播。

        Args:
            x: 输入 DAPI 图像 ``(B, 3, H, W)``。

        Returns:
            torch.Tensor: 生成的目标标记图像 ``(B, C_out, H, W)``，
            取值范围 ``[0, 1]``。
        """
        f0 = self.stage0(x)            # 1/2
        f1 = self.stage1(self.pool(f0))  # 1/4
        f2 = self.stage2(f1)           # 1/8
        f3 = self.stage3(f2)           # 1/16
        f4 = self.stage4(f3)           # 1/32

        d3 = self.up4(f4, f3)
        d2 = self.up3(d3, f2)
        d1 = self.up2(d2, f1)
        d0 = self.up1(d1, f0)

        out = self.refine(self.up0(d0))
        return torch.sigmoid(self.head(out))
