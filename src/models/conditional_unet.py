"""多标记条件生成模型（一对多）。

包含两代实现：
    - ConditionalUNet   : 原版，仅在瓶颈处注入 marker 嵌入；
    - ConditionalUNetV2 : Multi-scale FiLM，在编码器/瓶颈/解码器
      各层持续注入 marker 条件，浅层也能感知目标标记类型。
"""

from typing import List

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


class FiLM(nn.Module):
    """Feature-wise Linear Modulation：用 marker 条件生成缩放与偏置。"""

    def __init__(self, feature_channels: int, embed_dim: int) -> None:
        super().__init__()
        # 从 marker 嵌入向量预测该层的 scale 和 shift
        self.scale_proj = nn.Linear(embed_dim, feature_channels)
        self.shift_proj = nn.Linear(embed_dim, feature_channels)
        nn.init.zeros_(self.scale_proj.weight)
        nn.init.ones_(self.scale_proj.bias)  # 默认 scale=1，不改变特征
        nn.init.zeros_(self.shift_proj.weight)
        nn.init.zeros_(self.shift_proj.bias)

    def forward(
        self, feature: torch.Tensor, marker_embed: torch.Tensor
    ) -> torch.Tensor:
        """将 marker 条件调制到空间特征上。

        Args:
            feature:      (B, C, H, W)
            marker_embed: (B, embed_dim)

        Returns:
            torch.Tensor: 调制后的特征 (B, C, H, W)
        """
        # (B, embed_dim) -> (B, C)
        scale = self.scale_proj(marker_embed)  # (B, C)
        shift = self.shift_proj(marker_embed)  # (B, C)
        # 广播到空间维度
        scale = scale[:, :, None, None]
        shift = shift[:, :, None, None]
        return feature * scale + shift


class ConditionalUNetV2(nn.Module):
    """Multi-scale Marker Conditioning U-Net。

    编码器每层、瓶颈、解码器每层均通过 FiLM 注入 marker 条件，
    实现从浅层到深层的全尺度条件控制。
    """

    def __init__(
        self,
        in_channels: int = 3,
        out_channels: int = 1,
        base_channels: int = 64,
        depth: int = 4,
        num_markers: int = len(MARKERS),
        embed_dim: int = 128,
    ) -> None:
        super().__init__()
        self.depth = depth
        self.embed_dim = embed_dim

        # Marker 嵌入表：将离散 marker_idx 映射为连续向量
        self.marker_embedding = nn.Embedding(num_markers, embed_dim)
        nn.init.normal_(self.marker_embedding.weight, std=0.02)

        # ---- 编码器：每层后接 FiLM 条件调制 ----
        self.stem = DoubleConv(in_channels, base_channels)
        self.stem_film = FiLM(base_channels, embed_dim)

        self.encoders = nn.ModuleList()
        self.encoder_films = nn.ModuleList()
        for i in range(depth):
            enc = Down(base_channels << i, base_channels << (i + 1))
            film = FiLM(base_channels << (i + 1), embed_dim)
            self.encoders.append(enc)
            self.encoder_films.append(film)

        # ---- 瓶颈条件调制 ----
        bottleneck_ch = base_channels << depth
        self.bottleneck_film = FiLM(bottleneck_ch, embed_dim)

        # ---- 解码器：每层前（上采样后）接 FiLM 条件调制 ----
        self.decoders = nn.ModuleList()
        self.decoder_films = nn.ModuleList()
        for i in reversed(range(depth)):
            up = Up(
                in_channels=base_channels << (i + 1),
                skip_channels=base_channels << i,
                out_channels=base_channels << i,
            )
            film = FiLM(base_channels << i, embed_dim)
            self.decoders.append(up)
            self.decoder_films.append(film)

        self.head = nn.Conv2d(base_channels, out_channels, kernel_size=1)

    def forward(
        self, x: torch.Tensor, marker_idx: torch.Tensor
    ) -> torch.Tensor:
        """条件前向传播。

        Args:
            x:          (B, C_in, H, W)
            marker_idx: (B,) 整型，目标标记编号

        Returns:
            torch.Tensor: (B, C_out, H, W)，取值 [0, 1]
        """
        # 查询 marker 嵌入向量，供各层 FiLM 共享使用
        marker_embed = self.marker_embedding(marker_idx)  # (B, embed_dim)

        # 编码器：逐层提取特征 + 条件调制
        feature = self.stem(x)
        feature = self.stem_film(feature, marker_embed)
        skips: List[torch.Tensor] = [feature]

        for encoder, film in zip(self.encoders, self.encoder_films):
            feature = encoder(feature)
            feature = film(feature, marker_embed)
            skips.append(feature)

        # 瓶颈：最深层的条件调制
        feature = skips.pop()
        feature = self.bottleneck_film(feature, marker_embed)

        # 解码器：上采样 + 跳跃连接 + 条件调制
        for decoder, film in zip(self.decoders, self.decoder_films):
            skip = skips.pop()
            feature = decoder(feature, skip)
            feature = film(feature, marker_embed)

        return torch.sigmoid(self.head(feature))
