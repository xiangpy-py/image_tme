"""GPTUNet：基于 Global Pixel Transformer 的 U 型虚拟染色网络。

实现自 Liu et al., "Global Pixel Transformers for Virtual Staining
of Microscopy Images"，并按本比赛 256x256 patch 与显存约束做适配。

论文结构（Table II）::

    Encoder: DenseBlock + GDT 逐级下采样
    Bottom:  DenseBlock + GST 全局传输
    Decoder: GUT + DenseBlock 逐级恢复，跳跃连接 Concat

本实现的适配（显存安全优先）：

- 注意力矩阵大小 = 输入位置数 x 查询位置数，因此 GPT 层只部署在
  <= 64x64 的特征图上；256 / 128 分辨率处保留 CNN 下采样与上采样；
- 各级 Dense Block 层数沿用论文的 2/4/8 + 瓶颈 8 + 解码 4/2/1；
- 跳跃连接采用论文的 Concat 方式；
- 支持多尺度输入（in_channels 直接设为 C x 尺度数）。
"""

from typing import List, Tuple

import torch
import torch.nn as nn

from .dense_block import DenseBlock
from .gpt_layer import build_gpt_layer


class ConvDown(nn.Module):
    """CNN 下采样：stride=2 卷积 + BN + ReLU（用于高分辨率浅层）。"""

    def __init__(self, in_channels: int, out_channels: int) -> None:
        """构建下采样路径。

        Args:
            in_channels:  输入通道数。
            out_channels: 输出通道数。
        """
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=2,
                      padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """空间尺寸减半。

        Args:
            x: 输入 ``(B, C_in, H, W)``。

        Returns:
            torch.Tensor: 输出 ``(B, C_out, H/2, W/2)``。
        """
        return self.block(x)


class ConvUp(nn.Module):
    """CNN 上采样：转置卷积 + BN + ReLU（用于高分辨率浅层）。"""

    def __init__(self, in_channels: int, out_channels: int) -> None:
        """构建上采样路径。

        Args:
            in_channels:  输入通道数。
            out_channels: 输出通道数。
        """
        super().__init__()
        self.block = nn.Sequential(
            nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2,
                               bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """空间尺寸翻倍。

        Args:
            x: 输入 ``(B, C_in, H, W)``。

        Returns:
            torch.Tensor: 输出 ``(B, C_out, 2H, 2W)``。
        """
        return self.block(x)


class GPTUNet(nn.Module):
    """混合 CNN + GPT 的 U 型网络（256x256 输入 -> 同尺寸输出）。

    信息流（以默认配置为例）::

        256px: stem(1x1) -> DB(2) ---------------> Concat <- DB(1) <- ConvUp
        128px: ConvDown -> DB(4) -----------> Concat <- DB(2) <- ConvUp
         64px: ConvDown -> DB(8) ------> Concat <- DB(4) <- GUT
         32px: GDT -> DB(8) -> GST ----(瓶颈, 全局建模)----

    GDT/GST/GUT 分别承担深层下采样、瓶颈全局传输与深层上采样，
    使输出像素直接融合输入特征图的全局上下文。
    """

    def __init__(
        self,
        in_channels: int = 3,
        out_channels: int = 1,
        stem_channels: int = 32,
        growth_rate: int = 16,
        enc_layers: Tuple[int, int, int] = (2, 4, 8),
        enc_channels: Tuple[int, int, int] = (64, 128, 256),
        bottleneck_layers: int = 8,
        bottleneck_channels: int = 384,
        dec_layers: Tuple[int, int, int] = (4, 2, 1),
        dec_channels: Tuple[int, int, int] = (288, 165, 90),
        dropout: float = 0.0,
    ) -> None:
        """搭建 GPTUNet。

        Args:
            in_channels:        输入通道数；启用多尺度输入时为 C x 尺度数。
            out_channels:       输出通道数（目标标记灰度图为 1）。
            stem_channels:      输入 1x1 卷积后的特征宽度（论文为 32）。
            growth_rate:        Dense Block 增长率 k（论文为 16）。
            enc_layers:         编码器三级 Dense Block 的层数，论文为 (2, 4, 8)。
            enc_channels:       编码器三级 Dense Block 的输出通道数。
            bottleneck_layers:  瓶颈 Dense Block 层数（论文为 8）。
            bottleneck_channels: 瓶颈输出通道数（论文为 384）。
            dec_layers:         解码器三级 Dense Block 层数，论文为 (4, 2, 1)。
            dec_channels:       解码器三级输出通道数，论文为 (288, 165, 90)。
            dropout:            Dense Block 内部 Dropout，论文为 0.5，
                本比赛数据量较大，默认 0，可按需开启。
        """
        super().__init__()

        # ---- 输入层：1x1 卷积压缩多尺度拼接输入（论文 Input Layer）----
        self.stem = nn.Conv2d(in_channels, stem_channels, kernel_size=1)

        # ---- 编码器：256 / 128 层 CNN，64 层 Dense，32 层 GDT ----
        self.enc_db1 = DenseBlock(stem_channels, enc_layers[0], growth_rate,
                                  dropout, out_channels=enc_channels[0])
        self.down1 = ConvDown(enc_channels[0], enc_channels[1])

        self.enc_db2 = DenseBlock(enc_channels[1], enc_layers[1], growth_rate,
                                  dropout, out_channels=enc_channels[1])
        self.down2 = ConvDown(enc_channels[1], enc_channels[2])

        self.enc_db3 = DenseBlock(enc_channels[2], enc_layers[2], growth_rate,
                                  dropout, out_channels=enc_channels[2])
        # 深层下采样交给 GDT：64x64 -> 32x32，输出位置融合全局信息。
        self.down3 = build_gpt_layer(
            "down", enc_channels[2], enc_channels[2],
            q_channels=enc_channels[2] // 2,
        )

        # ---- 瓶颈：Dense Block + GST 同尺寸全局传输 ----
        self.bottleneck_db = DenseBlock(enc_channels[2], bottleneck_layers,
                                        growth_rate, dropout,
                                        out_channels=bottleneck_channels)
        self.bottleneck_gst = build_gpt_layer(
            "same", bottleneck_channels, bottleneck_channels,
            q_channels=bottleneck_channels // 2,
        )

        # ---- 解码器：32 层 GUT 上采样，64/128/256 层恢复细节 ----
        self.up3 = build_gpt_layer(
            "up", bottleneck_channels, dec_channels[0],
            q_channels=bottleneck_channels // 2,
        )
        # 跳跃连接为 Concat：GUT 输出 + 编码器同级特征。
        self.dec_db3 = DenseBlock(dec_channels[0] + enc_channels[2],
                                  dec_layers[0], growth_rate, dropout,
                                  out_channels=dec_channels[0])

        self.up2 = ConvUp(dec_channels[0], dec_channels[1])
        self.dec_db2 = DenseBlock(dec_channels[1] + enc_channels[1],
                                  dec_layers[1], growth_rate, dropout,
                                  out_channels=dec_channels[1])

        self.up1 = ConvUp(dec_channels[1], dec_channels[2])
        self.dec_db1 = DenseBlock(dec_channels[2] + enc_channels[0],
                                  dec_layers[2], growth_rate, dropout,
                                  out_channels=dec_channels[2])

        # ---- 输出层：1x1 卷积映射到目标通道，Sigmoid 约束到 [0, 1] ----
        self.head = nn.Conv2d(dec_channels[2], out_channels, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """前向传播。

        Args:
            x: 输入 DAPI 图像 ``(B, C_in, 256, 256)``；
                启用多尺度输入时 C_in = C x 尺度数。

        Returns:
            torch.Tensor: 生成的目标标记图像 ``(B, C_out, 256, 256)``，
            取值范围 ``[0, 1]``。
        """
        feature = self.stem(x)

        # 编码器：逐级保存跳跃连接特征
        skip1 = self.enc_db1(feature)          # 256px
        skip2 = self.enc_db2(self.down1(skip1))  # 128px
        skip3 = self.enc_db3(self.down2(skip2))  # 64px

        # 瓶颈：GDT 下采样 + Dense + GST 全局传输（32px）
        feature = self.down3(skip3)
        feature = self.bottleneck_db(feature)
        feature = self.bottleneck_gst(feature)

        # 解码器：上采样 + 跳跃连接 Concat + Dense Block
        feature = self.up3(feature)                              # 64px
        feature = self.dec_db3(torch.cat([feature, skip3], dim=1))

        feature = self.up2(feature)                              # 128px
        feature = self.dec_db2(torch.cat([feature, skip2], dim=1))

        feature = self.up1(feature)                              # 256px
        feature = self.dec_db1(torch.cat([feature, skip1], dim=1))

        return torch.sigmoid(self.head(feature))
