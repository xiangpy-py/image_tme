"""模型基础构件。

集中定义各模型复用的卷积块、残差块、下采样/上采样模块与
Transformer 编码块，遵循「由小到大」的组装思路：

- DoubleConv / ResidualBlock : 局部特征提取
- Down / Up                  : U 型结构的尺度变换
- TransformerBlock           : 全局依赖建模
- MarkerEmbedding            : 多标记条件生成的 token 嵌入
"""

from typing import List

import torch
import torch.nn as nn
import torch.nn.functional as F


class DoubleConv(nn.Module):
    """两次「卷积 + 归一化 + 激活」的标准卷积块。"""

    def __init__(self, in_channels: int, out_channels: int) -> None:
        """构建卷积序列。

        Args:
            in_channels:  输入通道数。
            out_channels: 输出通道数。
        """
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """前向传播。

        Args:
            x: 输入特征图，形状 ``(B, C_in, H, W)``。

        Returns:
            torch.Tensor: 输出特征图，形状 ``(B, C_out, H, W)``。
        """
        return self.block(x)


class ResidualBlock(nn.Module):
    """残差卷积块：缓解深层网络退化，增强特征提取能力。"""

    def __init__(self, in_channels: int, out_channels: int) -> None:
        """构建残差分支与快捷连接。

        Args:
            in_channels:  输入通道数。
            out_channels: 输出通道数。
        """
        super().__init__()
        self.conv_branch = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
        )
        # 通道数变化时用 1x1 卷积对齐快捷分支。
        self.shortcut = (
            nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
                nn.BatchNorm2d(out_channels),
            )
            if in_channels != out_channels
            else nn.Identity()
        )
        self.activation = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """前向传播：残差相加后激活。

        Args:
            x: 输入特征图。

        Returns:
            torch.Tensor: ``ReLU(conv_branch(x) + shortcut(x))``。
        """
        return self.activation(self.conv_branch(x) + self.shortcut(x))


class Down(nn.Module):
    """下采样模块：2 倍池化 + 卷积块。"""

    def __init__(self, in_channels: int, out_channels: int, residual: bool = False) -> None:
        """构建下采样路径。

        Args:
            in_channels:  输入通道数。
            out_channels: 输出通道数。
            residual:     是否使用残差块替代普通卷积块。
        """
        super().__init__()
        conv = ResidualBlock if residual else DoubleConv
        self.block = nn.Sequential(
            nn.MaxPool2d(kernel_size=2, stride=2),
            conv(in_channels, out_channels),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """空间尺寸减半、通道数调整。

        Args:
            x: 输入特征图 ``(B, C_in, H, W)``。

        Returns:
            torch.Tensor: 输出 ``(B, C_out, H/2, W/2)``。
        """
        return self.block(x)


class Up(nn.Module):
    """上采样模块：转置卷积升尺度 + 跳跃连接拼接 + 卷积块。"""

    def __init__(
        self,
        in_channels: int,
        skip_channels: int,
        out_channels: int,
        residual: bool = False,
    ) -> None:
        """构建上采样路径。

        Args:
            in_channels:   来自深层的特征通道数。
            skip_channels: 跳跃连接（编码器同层）的通道数。
            out_channels:  输出通道数。
            residual:      是否使用残差块。
        """
        super().__init__()
        self.up = nn.ConvTranspose2d(
            in_channels, in_channels // 2, kernel_size=2, stride=2
        )
        conv = ResidualBlock if residual else DoubleConv
        self.conv = conv(in_channels // 2 + skip_channels, out_channels)

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        """先升尺度再与跳跃连接融合。

        Args:
            x:    深层特征 ``(B, C_in, H, W)``。
            skip: 编码器同层特征 ``(B, C_skip, 2H, 2W)``。

        Returns:
            torch.Tensor: 融合后特征 ``(B, C_out, 2H, 2W)``。
        """
        x = self.up(x)

        # 处理奇数尺寸导致的 1 像素错位，保证可拼接。
        diff_y = skip.size(2) - x.size(2)
        diff_x = skip.size(3) - x.size(3)
        x = F.pad(x, [diff_x // 2, diff_x - diff_x // 2,
                      diff_y // 2, diff_y - diff_y // 2])

        x = torch.cat([skip, x], dim=1)
        return self.conv(x)


class TransformerBlock(nn.Module):
    """标准 Transformer 编码块：多头自注意力 + 前馈网络（Pre-LN）。

    插入 CNN 瓶颈处，用于建模组织图像中的长距离空间依赖。
    """

    def __init__(
        self,
        embed_dim: int,
        num_heads: int = 8,
        mlp_ratio: float = 4.0,
        dropout: float = 0.0,
    ) -> None:
        """构建注意力与前馈子层。

        Args:
            embed_dim: token 嵌入维度。
            num_heads: 注意力头数，需整除 ``embed_dim``。
            mlp_ratio: 前馈网络隐藏层扩展倍数。
            dropout:   Dropout 概率。
        """
        super().__init__()
        self.norm1 = nn.LayerNorm(embed_dim)
        self.attention = nn.MultiheadAttention(
            embed_dim, num_heads, dropout=dropout, batch_first=True
        )
        self.norm2 = nn.LayerNorm(embed_dim)
        hidden_dim = int(embed_dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, embed_dim),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """前向传播：两次残差连接。

        Args:
            x: token 序列 ``(B, N, C)``，N 为空间展平后的 token 数。

        Returns:
            torch.Tensor: 同形状输出。
        """
        h = self.norm1(x)
        attn_out, _ = self.attention(h, h, h, need_weights=False)
        x = x + attn_out
        x = x + self.mlp(self.norm2(x))
        return x


class MarkerEmbedding(nn.Module):
    """标记条件嵌入：把目标标记编号映射为通道维偏置向量。

    多标记联合建模时，将该嵌入加到瓶颈特征的空间位置上，
    告诉模型「本次要生成哪一种 IHC 标记」。
    """

    def __init__(self, num_markers: int, embed_dim: int) -> None:
        """创建嵌入表。

        Args:
            num_markers: 标记类别数（对应 constants.MARKERS 长度）。
            embed_dim:   嵌入维度，需与被加特征的通道数一致。
        """
        super().__init__()
        self.embedding = nn.Embedding(num_markers, embed_dim)
        nn.init.normal_(self.embedding.weight, std=0.02)

    def forward(self, marker_idx: torch.Tensor) -> torch.Tensor:
        """查询嵌入向量。

        Args:
            marker_idx: 标记编号张量，形状 ``(B,)``，整型。

        Returns:
            torch.Tensor: 嵌入向量 ``(B, embed_dim, 1, 1)``，
            可直接广播加到 ``(B, C, H, W)`` 特征上。
        """
        vector = self.embedding(marker_idx)  # (B, C)
        return vector[:, :, None, None]
