"""Global Pixel Transformer (GPT) 层。

实现自 Liu et al., "Global Pixel Transformers for Virtual Staining
of Microscopy Images"：用注意力算子替代 U-Net 中的局部
下采样/同尺寸/上采样算子，使输出每个位置融合输入的全局信息。

核心计算（论文 Eq.6）::

    Q = Generator(I)      # Query 生成器决定输出空间尺寸
    K = Conv1x1(I)        # CK == CQ
    V = Conv1x1(I)        # CV 决定输出通道数
    O = V @ Softmax(K^T @ Q)   # Softmax 沿输入空间位置（列方向）归一化

三种变体（仅 Query 生成器不同）：

- GDT: 3x3 Conv, stride=2 -> 空间减半（下采样）
- GST: 3x3 Conv, stride=1 -> 空间不变（同尺寸传输）
- GUT: 3x3 Deconv, stride=2 -> 空间翻倍（上采样）

显存安全约束：注意力矩阵大小为「输入位置数 x 查询位置数」，
仅在输入特征图空间尺寸 <= 64x64 的层使用本模块。
"""

from typing import Literal

import torch
import torch.nn as nn
import torch.nn.functional as F


class GPTLayer(nn.Module):
    """全局像素 Transformer 层：输出每个位置是输入全部位置的加权和。

    与自注意力不同，本层的输出空间尺寸由 Query 生成器决定，
    因此同一结构可承担下采样（GDT）、同尺寸传输（GST）与
    上采样（GUT）三种角色。
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        q_channels: int,
        q_generator: nn.Module,
        dropout: float = 0.0,
    ) -> None:
        """构建 GPT 层。

        Args:
            in_channels:  输入通道数 C。
            out_channels: 输出通道数，即 Value 张量通道数 CV。
            q_channels:   Query/Key 通道数 CQ=CK（注意力内积维度，两者必须相等）。
            q_generator:  Query 生成器模块，其输出空间尺寸即本层输出空间尺寸。
            dropout:      输出特征图上的 Dropout 概率，0 表示关闭。
        """
        super().__init__()
        self.q_channels = q_channels
        self.out_channels = out_channels

        # Query 由外部生成器产生（区分 GDT/GST/GUT 的唯一部件）。
        self.q_generator = q_generator

        # Key / Value 均为 1x1 卷积，保持输入空间尺寸不变。
        self.conv_k = nn.Conv2d(in_channels, q_channels, kernel_size=1)
        self.conv_v = nn.Conv2d(in_channels, out_channels, kernel_size=1)

        self.dropout = nn.Dropout2d(dropout) if dropout > 0 else None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """前向传播：O = V @ Softmax(K^T @ Q)。

        Args:
            x: 输入特征图 ``(B, C, H, W)``。

        Returns:
            torch.Tensor: 输出特征图 ``(B, CV, HQ, WQ)``，
            空间尺寸与 Query 张量一致。

        Raises:
            AssertionError: Query 通道数与设定不符（CK != CQ）时抛出。
        """
        batch = x.shape[0]

        # 1) 生成 Q / K / V
        query = self.q_generator(x)   # (B, CQ, HQ, WQ)
        key = self.conv_k(x)          # (B, CK, H, W)
        value = self.conv_v(x)        # (B, CV, H, W)
        assert query.shape[1] == key.shape[1], (
            f"注意力维度不匹配: CQ={query.shape[1]} vs CK={key.shape[1]}"
        )

        out_h, out_w = query.shape[2], query.shape[3]

        # 2) 空间维展平为矩阵（论文中的 mode-3 unfolding）
        query_mat = query.reshape(batch, self.q_channels, -1)      # (B, CQ, NQ)
        key_mat = key.reshape(batch, self.q_channels, -1)          # (B, CK, NK)
        value_mat = value.reshape(batch, self.out_channels, -1)    # (B, CV, NK)

        # 3) 注意力权重：K^T @ Q -> (B, NK, NQ)，沿输入位置（dim=1）做列归一化
        attn_scores = torch.bmm(key_mat.transpose(1, 2), query_mat)
        attn_weights = F.softmax(attn_scores, dim=1)

        # 4) 加权聚合 Value 并还原空间形状
        output_mat = torch.bmm(value_mat, attn_weights)            # (B, CV, NQ)
        output = output_mat.reshape(batch, self.out_channels, out_h, out_w)

        if self.dropout is not None:
            output = self.dropout(output)
        return output


def build_gpt_layer(
    mode: Literal["down", "same", "up"],
    in_channels: int,
    out_channels: int,
    q_channels: int,
    dropout: float = 0.0,
) -> GPTLayer:
    """按类型构建 GDT / GST / GUT 层（工厂函数）。

    Args:
        mode:         ``"down"``（GDT，尺寸减半）、``"same"``（GST，尺寸不变）
            或 ``"up"``（GUT，尺寸翻倍）。
        in_channels:  输入通道数。
        out_channels: 输出通道数（CV）。
        q_channels:   Query/Key 通道数（CQ=CK）。
        dropout:      Dropout 概率。

    Returns:
        GPTLayer: 配置好对应 Query 生成器的 GPT 层。

    Raises:
        ValueError: 未知的 mode 时抛出。
    """
    if mode == "down":
        # GDT: stride=2 卷积使 Query 空间尺寸减半 -> 输出尺寸减半。
        q_generator = nn.Conv2d(
            in_channels, q_channels, kernel_size=3, stride=2, padding=1
        )
    elif mode == "same":
        # GST: stride=1 卷积保持空间尺寸 -> 同尺寸全局传输。
        q_generator = nn.Conv2d(
            in_channels, q_channels, kernel_size=3, stride=1, padding=1
        )
    elif mode == "up":
        # GUT: 转置卷积使 Query 空间尺寸翻倍 -> 输出尺寸翻倍。
        q_generator = nn.ConvTranspose2d(
            in_channels, q_channels,
            kernel_size=3, stride=2, padding=1, output_padding=1,
        )
    else:
        raise ValueError(f"未知 GPT 类型: {mode}，可选: down / same / up")

    return GPTLayer(in_channels, out_channels, q_channels, q_generator, dropout)
