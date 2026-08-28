"""Version 3: CNN 编码器 + Transformer 瓶颈 + U-Net 解码器。

结构::

    DAPI -> CNN Encoder -> Transformer Bottleneck -> Decoder -> IHC

CNN 负责局部纹理提取，Transformer 在最低分辨率处建模
长距离组织关系与细胞空间分布（plan.md Version 3 的目标）。
"""

import torch
import torch.nn as nn

from .blocks import DoubleConv, Down, TransformerBlock, Up


class TransUNet(nn.Module):
    """带 Transformer 瓶颈的 U 型网络。

    在编码器最深处把特征图展平为 token 序列，
    经多层 Transformer 编码后再还原为空间特征进入解码器。
    """

    def __init__(
        self,
        in_channels: int = 3,
        out_channels: int = 1,
        base_channels: int = 64,
        depth: int = 4,
        transformer_layers: int = 4,
        num_heads: int = 8,
        mlp_ratio: float = 4.0,
    ) -> None:
        """搭建 CNN 编解码器与 Transformer 瓶颈。

        Args:
            in_channels:       输入通道数。
            out_channels:      输出通道数。
            base_channels:     第一层特征宽度。
            depth:             下采样次数。
            transformer_layers: Transformer 编码块堆叠层数。
            num_heads:         多头注意力头数。
            mlp_ratio:         前馈网络扩展倍数。
        """
        super().__init__()
        self.depth = depth
        bottleneck_channels = base_channels << depth

        # ---- CNN 编码器 ----
        self.stem = DoubleConv(in_channels, base_channels)
        self.encoders = nn.ModuleList([
            Down(base_channels << i, base_channels << (i + 1))
            for i in range(depth)
        ])

        # ---- Transformer 瓶颈：token 维度 = 瓶颈通道数 ----
        self.transformer = nn.Sequential(*[
            TransformerBlock(
                embed_dim=bottleneck_channels,
                num_heads=num_heads,
                mlp_ratio=mlp_ratio,
            )
            for _ in range(transformer_layers)
        ])
        self.bottleneck_norm = nn.LayerNorm(bottleneck_channels)

        # ---- CNN 解码器 ----
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

    def _apply_transformer(self, feature: torch.Tensor) -> torch.Tensor:
        """在瓶颈特征上执行 Transformer 编码。

        空间维展平为 token 序列 -> 自注意力建模全局关系 -> 还原空间形状。

        Args:
            feature: 瓶颈特征 ``(B, C, H, W)``。

        Returns:
            torch.Tensor: 同形状的增强特征。
        """
        batch, channels, height, width = feature.shape

        # (B, C, H, W) -> (B, H*W, C)：每个空间位置是一个 token。
        tokens = feature.flatten(2).transpose(1, 2)
        tokens = self.transformer(tokens)
        tokens = self.bottleneck_norm(tokens)

        # 还原为 (B, C, H, W)。
        return tokens.transpose(1, 2).reshape(batch, channels, height, width)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """前向传播。

        Args:
            x: 输入 DAPI 图像 ``(B, C_in, H, W)``。

        Returns:
            torch.Tensor: 生成的目标标记图像 ``(B, C_out, H, W)``，
            取值范围 ``[0, 1]``。
        """
        skips = []

        feature = self.stem(x)
        skips.append(feature)
        for encoder in self.encoders:
            feature = encoder(feature)
            skips.append(feature)

        # 瓶颈处注入全局上下文。
        feature = self._apply_transformer(skips.pop())

        for decoder in self.decoders:
            feature = decoder(feature, skips.pop())

        return torch.sigmoid(self.head(feature))
