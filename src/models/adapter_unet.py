"""Version 4.2: Shared Encoder + Marker-specific Adapter。

核心创新：
    - Shared Encoder：提取与标记无关的组织结构特征
    - Marker Adapter：轻量级的标记专用分支，将共享特征转换为
      标记特定的表示
    - 共享 Decoder：统一解码恢复空间分辨率

Loss 中可配合 Cross-Marker Consistency，强制共享特征一致。

结构：
    DAPI → Shared Encoder → Shared Feature
                                ↓
                    ┌─────────┼─────────┐
                    ↓         ↓         ↓
                 Adapter   Adapter   Adapter  (per marker)
                 [CD68]   [CD45RO]  [Vimentin]...
                    │         │         │
                    └─────────┼─────────┘
                              ↓
                         Shared Decoder
                              ↓
                            IHC
"""

from typing import List

import torch
import torch.nn as nn

from ..datasets.constants import MARKERS
from .blocks import DoubleConv, Down, Up


class MarkerAdapter(nn.Module):
    """轻量标记适配器：1x1 Conv + BN + ReLU + 1x1 Conv。

    参数量极小，每个标记一个独立实例，学习标记特定的特征变换。
    """

    def __init__(self, channels: int, num_markers: int) -> None:
        super().__init__()
        # 使用条件选择而非并行卷积，节省前向计算
        self.adapters = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(channels, channels // 2, kernel_size=1, bias=False),
                nn.BatchNorm2d(channels // 2),
                nn.ReLU(inplace=True),
                nn.Conv2d(channels // 2, channels, kernel_size=1, bias=False),
                nn.BatchNorm2d(channels),
            )
            for _ in range(num_markers)
        ])
        # 残差缩放：初始时 adapter 输出接近 0，等价于恒等映射
        self.residual_scale = nn.Parameter(torch.zeros(1))

    def forward(
        self, feature: torch.Tensor, marker_idx: torch.Tensor
    ) -> torch.Tensor:
        """根据 marker_idx 选择对应的适配器分支。

        Args:
            feature:      (B, C, H, W)
            marker_idx:   (B,) 整型

        Returns:
            torch.Tensor: 适配后的特征 (B, C, H, W)
        """
        batch = feature.shape[0]
        output = torch.zeros_like(feature)

        # 按 marker 分组处理，避免 batch 内不同 marker 的串扰
        for m_idx in range(len(self.adapters)):
            mask = marker_idx == m_idx
            if mask.any():
                adapted = self.adapters[m_idx](feature[mask])
                output[mask] = feature[mask] + self.residual_scale * adapted

        return output


class AdapterUNet(nn.Module):
    """Shared Encoder + Marker Adapter + Shared Decoder。

    编码器和解码器全部标记共享，仅在 bottleneck 处插入
    标记专用适配器，实现高效的跨标记知识共享。
    """

    def __init__(
        self,
        in_channels: int = 3,
        out_channels: int = 1,
        base_channels: int = 64,
        depth: int = 4,
        num_markers: int = len(MARKERS),
    ) -> None:
        super().__init__()
        self.depth = depth

        # ---- Shared Encoder ----
        self.stem = DoubleConv(in_channels, base_channels)
        self.encoders = nn.ModuleList([
            Down(base_channels << i, base_channels << (i + 1))
            for i in range(depth)
        ])

        # ---- Marker-specific Adapter (bottleneck) ----
        bottleneck_ch = base_channels << depth
        self.adapter = MarkerAdapter(bottleneck_ch, num_markers)

        # ---- Shared Decoder ----
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

    def forward(
        self, x: torch.Tensor, marker_idx: torch.Tensor
    ) -> torch.Tensor:
        """前向传播。

        Args:
            x:          (B, C_in, H, W)
            marker_idx: (B,) 整型

        Returns:
            torch.Tensor: (B, C_out, H, W)，取值 [0, 1]
        """
        skips: List[torch.Tensor] = []

        feature = self.stem(x)
        skips.append(feature)
        for encoder in self.encoders:
            feature = encoder(feature)
            skips.append(feature)

        # Bottleneck：通过 marker adapter 注入标记特异性
        feature = self.adapter(skips.pop(), marker_idx)

        for decoder in self.decoders:
            skip = skips.pop()
            feature = decoder(feature, skip)

        return torch.sigmoid(self.head(feature))
