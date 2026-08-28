"""Version 4: 多标记条件生成模型（一对多）。

结构::

    DAPI + Marker Token -> Unified U-Net -> 目标 IHC

单个模型通过 marker token 条件控制，生成四种目标标记中的任意一种。
对应赛题「一对多联合建模」创新加分项：利用跨标记关联信息，
在共享表示下联合生成多个目标标记。
"""

import torch
import torch.nn as nn

from ..datasets.constants import MARKERS
from .blocks import DoubleConv, Down, MarkerEmbedding, Up


class ConditionalUNet(nn.Module):
    """标记条件 U-Net：在瓶颈处注入目标标记嵌入。

    前向时除图像外还需提供 ``marker_idx``（batch 内每个样本
    要生成的目标标记编号），模型据此切换生成模式。
    """

    def __init__(
        self,
        in_channels: int = 3,
        out_channels: int = 1,
        base_channels: int = 64,
        depth: int = 4,
        num_markers: int = len(MARKERS),
    ) -> None:
        """搭建条件 U-Net。

        Args:
            in_channels:  输入通道数。
            out_channels: 输出通道数（各标记输出格式一致）。
            base_channels: 第一层特征宽度。
            depth:        下采样次数。
            num_markers:  标记类别数，决定嵌入表大小。
        """
        super().__init__()
        self.depth = depth
        bottleneck_channels = base_channels << depth

        self.stem = DoubleConv(in_channels, base_channels)
        self.encoders = nn.ModuleList([
            Down(base_channels << i, base_channels << (i + 1))
            for i in range(depth)
        ])

        # 标记条件嵌入：加到瓶颈特征上，广播到全部空间位置。
        self.marker_embedding = MarkerEmbedding(num_markers, bottleneck_channels)

        self.decoders = nn.ModuleList()
        for i in reversed(range(depth)):
            self.decoders.append(
                Up(
                    in_channels=base_channels << (i + 1),
                    skip_channels=base_channels << i,
                    out_channels=base_channels << i,
                )
            )
        self.head = nn.Conv2d(base_channels, out_channels, kernel_size=1)

    def forward(self, x: torch.Tensor, marker_idx: torch.Tensor) -> torch.Tensor:
        """条件前向传播。

        Args:
            x:          输入 DAPI 图像 ``(B, C_in, H, W)``。
            marker_idx: 目标标记编号 ``(B,)``，整型，
                编号顺序与 ``datasets.constants.MARKERS`` 一致。

        Returns:
            torch.Tensor: 指定标记的生成图像 ``(B, C_out, H, W)``，
            取值范围 ``[0, 1]``。
        """
        skips = []

        feature = self.stem(x)
        skips.append(feature)
        for encoder in self.encoders:
            feature = encoder(feature)
            skips.append(feature)

        # 瓶颈处注入标记条件：告诉模型生成哪一种标记。
        feature = skips.pop() + self.marker_embedding(marker_idx)

        for decoder in self.decoders:
            feature = decoder(feature, skips.pop())

        return torch.sigmoid(self.head(feature))
