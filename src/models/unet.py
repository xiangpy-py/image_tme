"""Version 1: 标准 U-Net Baseline。

结构::

    DAPI -> Encoder(4 次下采样) -> Bottleneck -> Decoder(跳跃连接) -> IHC

作为比赛 pipeline 的基线模型，验证数据流、训练与指标计算是否正确。
"""

from typing import List

import torch
import torch.nn as nn

from .blocks import DoubleConv, Down, Up


class UNet(nn.Module):
    """经典 U 型编码器-解码器网络。

    支持输入/输出通道数与基础宽度可配置，
    解码器末端使用 Sigmoid 将输出约束到 ``[0, 1]`` 像素域。
    """

    def __init__(
        self,
        in_channels: int = 3,
        out_channels: int = 1,
        base_channels: int = 64,
        depth: int = 4,
    ) -> None:
        """搭建编码器与解码器。

        Args:
            in_channels:  输入通道数（DAPI 伪彩图为 3）。
            out_channels: 输出通道数（目标标记灰度图为 1，伪彩为 3）。
            base_channels: 第一层特征宽度，逐层翻倍。
            depth:        下采样次数，256x256 输入建议为 4。
        """
        super().__init__()
        self.depth = depth

        # ---- 编码器：逐层记录各尺度特征，供跳跃连接使用 ----
        self.stem = DoubleConv(in_channels, base_channels)
        self.encoders = nn.ModuleList([
            Down(base_channels << i, base_channels << (i + 1))
            for i in range(depth)
        ])

        # ---- 解码器：自底向上逐级融合编码器特征 ----
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

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """前向传播。

        Args:
            x: 输入 DAPI 图像 ``(B, C_in, H, W)``，H/W 需为 ``2**depth`` 的倍数。

        Returns:
            torch.Tensor: 生成的目标标记图像 ``(B, C_out, H, W)``，
            取值范围 ``[0, 1]``。
        """
        skips: List[torch.Tensor] = []

        feature = self.stem(x)
        skips.append(feature)
        for encoder in self.encoders:
            feature = encoder(feature)
            skips.append(feature)

        # 最深层为瓶颈，依次弹出作为跳跃连接。
        feature = skips.pop()
        for decoder in self.decoders:
            skip = skips.pop()
            feature = decoder(feature, skip)

        return torch.sigmoid(self.head(feature))
