"""Dense Block 模块。

实现自 "Global Pixel Transformers for Virtual Staining of
Microscopy Images"（Liu et al.）中的密集连接块：

- 每层生成 k 个新特征图（growth rate），与之前所有层特征沿通道拼接，
  促进层间特征复用并显著减少参数量；
- 每层内部按论文 Figure 2 的顺序：Conv -> BN -> ReLU -> Dropout；
- 块末尾以 1x1 卷积压缩通道数，使输出通道可控。
"""

from typing import Optional

import torch
import torch.nn as nn


class DenseLayer(nn.Module):
    """Dense Block 中的单层：Conv -> BN -> ReLU -> Dropout，输出与输入拼接。"""

    def __init__(self, in_channels: int, growth_rate: int, dropout: float = 0.0) -> None:
        """构建单层。

        Args:
            in_channels: 输入通道数（等于此前所有层输出的拼接通道数）。
            growth_rate: 本层新增的特征图数量 k。
            dropout:     Dropout 概率，0 表示关闭。
        """
        super().__init__()
        self.conv = nn.Conv2d(in_channels, growth_rate, kernel_size=3, padding=1)
        self.norm = nn.BatchNorm2d(growth_rate)
        self.activation = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout2d(dropout) if dropout > 0 else None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """前向传播：新特征与输入沿通道维拼接。

        Args:
            x: 输入特征图 ``(B, C, H, W)``。

        Returns:
            torch.Tensor: 拼接后特征 ``(B, C + k, H, W)``。
        """
        out = self.conv(x)
        out = self.norm(out)
        out = self.activation(out)
        if self.dropout is not None:
            out = self.dropout(out)
        return torch.cat([x, out], dim=1)


class DenseBlock(nn.Module):
    """由若干 DenseLayer 堆叠的密集连接块，末尾可选 1x1 卷积压缩通道。"""

    def __init__(
        self,
        in_channels: int,
        num_layers: int,
        growth_rate: int,
        dropout: float = 0.0,
        out_channels: Optional[int] = None,
    ) -> None:
        """构建 Dense Block。

        Args:
            in_channels:  输入通道数。
            num_layers:   Dense 层数 L。
            growth_rate:  每层新增通道数 k。
            dropout:      各层 Dropout 概率。
            out_channels: 末尾 1x1 卷积压缩到的通道数；``None`` 表示不压缩，
                此时输出通道数为 ``in_channels + L * k``。
        """
        super().__init__()
        layers = []
        current_channels = in_channels
        for _ in range(num_layers):
            layers.append(DenseLayer(current_channels, growth_rate, dropout))
            current_channels += growth_rate
        self.layers = nn.ModuleList(layers)

        self.bottleneck = (
            nn.Conv2d(current_channels, out_channels, kernel_size=1)
            if out_channels is not None
            else None
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """前向传播。

        Args:
            x: 输入特征图 ``(B, C_in, H, W)``。

        Returns:
            torch.Tensor: 输出特征图，空间尺寸不变；
            通道数为 ``out_channels``（压缩时）或 ``C_in + L * k``。
        """
        for layer in self.layers:
            x = layer(x)
        if self.bottleneck is not None:
            x = self.bottleneck(x)
        return x
