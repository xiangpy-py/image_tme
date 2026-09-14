"""模型定义与注册。

自小到大组织：

- 基础构件：``DoubleConv`` / ``ResidualBlock`` / ``CBAM`` / ``Down`` / ``Up`` /
  ``MarkerEmbedding`` / ``FiLM``；
- 单标记模型：``UNet`` / ``ResNetUNet``；
- 一对多条件模型：``ConditionalUNet`` / ``ConditionalUNetV2``（FiLM）/
  ``AdapterUNet``（共享编解码器 + 标记适配器）/
  ``ConditionalResAttentionUNet``（残差 + 深层 CBAM + 多尺度 FiLM）；
- 统一入口：``ModelRegistry``（按配置实例化、判断是否条件模型）。

所有模型输出均经 Sigmoid 约束到 ``[0, 1]`` 像素域。
"""

import inspect
from typing import Any, Callable, Dict, List, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models

from ..data.constants import MARKERS


# ------------------------------------------------------------------ #
# 基础构件
# ------------------------------------------------------------------ #
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
            x: 输入特征图 ``(B, C_in, H, W)``。

        Returns:
            torch.Tensor: 输出特征图 ``(B, C_out, H, W)``。
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


class ChannelAttention(nn.Module):
    """通道注意力（CBAM 前半）：对特征通道做重要性重标定。

    平均池化与最大池化分别捕获平滑统计与显著峰值，共享一个瓶颈 MLP
    融合后经 Sigmoid 输出各通道权重。
    """

    def __init__(self, channels: int, reduction: int = 16) -> None:
        """构建共享瓶颈 MLP。

        Args:
            channels:  输入通道数。
            reduction: 瓶颈压缩比，隐藏层宽度为 ``max(channels // reduction, 8)``。
        """
        super().__init__()
        hidden = max(channels // reduction, 8)
        self.mlp = nn.Sequential(
            nn.Linear(channels, hidden, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(hidden, channels, bias=False),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """计算通道权重并重标定特征。

        Args:
            x: 输入特征 ``(B, C, H, W)``。

        Returns:
            torch.Tensor: 重标定后的特征 ``(B, C, H, W)``。
        """
        # 两种全局池化共享同一 MLP：相加融合后映射为逐通道权重。
        weight = torch.sigmoid(
            self.mlp(x.mean(dim=(2, 3))) + self.mlp(x.amax(dim=(2, 3)))
        )
        return x * weight[:, :, None, None]


class SpatialAttention(nn.Module):
    """空间注意力（CBAM 后半）：对特征空间位置做重要性重标定。

    沿通道维取平均与最大两张空间图，拼接后经卷积 + Sigmoid 输出
    各位置权重，聚焦细胞区域、组织边界等高响应结构。
    """

    def __init__(self, kernel_size: int = 7) -> None:
        """构建空间权重卷积。

        Args:
            kernel_size: 卷积核尺寸（奇数），padding 取对称半宽。
        """
        super().__init__()
        self.conv = nn.Conv2d(2, 1, kernel_size, padding=kernel_size // 2, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """计算空间权重并重标定特征。

        Args:
            x: 输入特征 ``(B, C, H, W)``。

        Returns:
            torch.Tensor: 重标定后的特征 ``(B, C, H, W)``。
        """
        spatial = torch.cat(
            [x.mean(dim=1, keepdim=True), x.amax(dim=1, keepdim=True)], dim=1
        )
        weight = torch.sigmoid(self.conv(spatial))
        return x * weight


class CBAM(nn.Module):
    """CBAM 注意力：通道注意力 -> 空间注意力 串联。

    轻量设计（无全局自注意力大矩阵），适合插入深层特征，
    以极小开销换取对关键结构的选择性增强。
    """

    def __init__(self, channels: int, reduction: int = 16, kernel_size: int = 7) -> None:
        """串联通道与空间注意力。

        Args:
            channels:    特征通道数。
            reduction:   通道注意力瓶颈压缩比。
            kernel_size: 空间注意力卷积核尺寸。
        """
        super().__init__()
        self.channel = ChannelAttention(channels, reduction)
        self.spatial = SpatialAttention(kernel_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """先通道后空间依次重标定。

        Args:
            x: 输入特征 ``(B, C, H, W)``。

        Returns:
            torch.Tensor: 重标定后的特征 ``(B, C, H, W)``。
        """
        return self.spatial(self.channel(x))


class Down(nn.Module):
    """下采样模块：2 倍池化 + 卷积块（可选残差 / CBAM 注意力）。"""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        residual: bool = False,
        attention: bool = False,
    ) -> None:
        """构建下采样路径。

        Args:
            in_channels:  输入通道数。
            out_channels: 输出通道数。
            residual:     是否使用残差块替代普通卷积块。
            attention:    是否在卷积块后插入 CBAM（建议仅深层启用以控制开销）。
        """
        super().__init__()
        conv = ResidualBlock if residual else DoubleConv
        layers: List[nn.Module] = [
            nn.MaxPool2d(2, stride=2),
            conv(in_channels, out_channels),
        ]
        if attention:
            layers.append(CBAM(out_channels))
        self.block = nn.Sequential(*layers)

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
        attention: bool = False,
    ) -> None:
        """构建上采样路径。

        Args:
            in_channels:   来自深层的特征通道数。
            skip_channels: 跳跃连接（编码器同层）的通道数。
            out_channels:  输出通道数。
            residual:      是否使用残差块。
            attention:     是否在卷积块后插入 CBAM（建议仅深层启用）。
        """
        super().__init__()
        self.up = nn.ConvTranspose2d(
            in_channels, in_channels // 2, kernel_size=2, stride=2
        )
        conv = ResidualBlock if residual else DoubleConv
        conv_block: nn.Module = conv(in_channels // 2 + skip_channels, out_channels)
        # 深层解码器可选插入 CBAM：与编码器侧对称，聚焦高响应结构。
        self.conv = (
            nn.Sequential(conv_block, CBAM(out_channels)) if attention else conv_block
        )

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
        x = F.pad(
            x,
            [diff_x // 2, diff_x - diff_x // 2, diff_y // 2, diff_y - diff_y // 2],
        )
        return self.conv(torch.cat([skip, x], dim=1))


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
            marker_idx: 标记编号张量 ``(B,)``，整型。

        Returns:
            torch.Tensor: 嵌入向量 ``(B, embed_dim, 1, 1)``，
            可直接广播加到 ``(B, C, H, W)`` 特征上。
        """
        return self.embedding(marker_idx)[:, :, None, None]


# ------------------------------------------------------------------ #
# 单标记模型
# ------------------------------------------------------------------ #
class UNet(nn.Module):
    """经典 U 型编码器-解码器网络（基线）。"""

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
        self.encoders = nn.ModuleList(
            [Down(base_channels << i, base_channels << (i + 1)) for i in range(depth)]
        )

        # ---- 解码器：自底向上逐级融合编码器特征 ----
        self.decoders = nn.ModuleList(
            [
                Up(
                    in_channels=base_channels << (i + 1),
                    skip_channels=base_channels << i,
                    out_channels=base_channels << i,
                )
                for i in reversed(range(depth))
            ]
        )
        self.head = nn.Conv2d(base_channels, out_channels, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """前向传播。

        Args:
            x: 输入 DAPI 图像 ``(B, C_in, H, W)``，H/W 需为 ``2**depth`` 的倍数。

        Returns:
            torch.Tensor: 生成的目标标记图像 ``(B, C_out, H, W)``，取值 ``[0, 1]``。
        """
        skips: List[torch.Tensor] = [self.stem(x)]
        for encoder in self.encoders:
            skips.append(encoder(skips[-1]))

        # 最深层为瓶颈，依次弹出作为跳跃连接。
        feature = skips.pop()
        for decoder in self.decoders:
            feature = decoder(feature, skips.pop())
        return torch.sigmoid(self.head(feature))


class ResNetUNet(nn.Module):
    """ImageNet 预训练 ResNet 编码器 + U-Net 解码器。

    支持 ``in_channels != 3``：以 1x1 卷积升维后接入 ResNet，
    adapter 采用小方差初始化以免干扰预训练特征。
    """

    def __init__(
        self,
        in_channels: int = 3,
        out_channels: int = 1,
        backbone: str = "resnet34",
        pretrained: bool = True,
    ) -> None:
        """搭建编码器与解码器。

        Args:
            in_channels:  输入通道数（DAPI 伪彩图为 3）。
            out_channels: 输出通道数（目标标记灰度图为 1）。
            backbone:     编码器主干，可选 ``"resnet18"`` / ``"resnet34"``。
            pretrained:   是否加载 ImageNet 预训练权重。

        Raises:
            ValueError: 不支持的 backbone 时抛出。
        """
        super().__init__()

        if backbone == "resnet18":
            weights = models.ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
            resnet = models.resnet18(weights=weights)
        elif backbone == "resnet34":
            weights = models.ResNet34_Weights.IMAGENET1K_V1 if pretrained else None
            resnet = models.resnet34(weights=weights)
        else:
            raise ValueError(f"不支持的编码器: {backbone}")
        stage_channels = [64, 64, 128, 256, 512]

        # ---- 输入适配层：处理 in_channels != 3 的情况 ----
        if in_channels != 3:
            self.input_adapter = nn.Conv2d(in_channels, 3, kernel_size=1, bias=False)
            if pretrained:
                self.input_adapter.weight.normal_(std=0.01)
        else:
            self.input_adapter = nn.Identity()

        # ---- 编码器 ----
        self.stage0 = nn.Sequential(resnet.conv1, resnet.bn1, resnet.relu)
        self.pool = resnet.maxpool
        self.stage1, self.stage2, self.stage3, self.stage4 = (
            resnet.layer1,
            resnet.layer2,
            resnet.layer3,
            resnet.layer4,
        )

        # ---- 解码器 ----
        self.up4 = Up(stage_channels[4], stage_channels[3], stage_channels[3])
        self.up3 = Up(stage_channels[3], stage_channels[2], stage_channels[2])
        self.up2 = Up(stage_channels[2], stage_channels[1], stage_channels[1])
        self.up1 = Up(stage_channels[1], stage_channels[0], stage_channels[0])
        self.up0 = nn.ConvTranspose2d(
            stage_channels[0], stage_channels[0], 2, stride=2
        )
        self.refine = nn.Sequential(
            nn.Conv2d(stage_channels[0], stage_channels[0], 3, padding=1, bias=False),
            nn.BatchNorm2d(stage_channels[0]),
            nn.ReLU(inplace=True),
        )
        self.head = nn.Conv2d(stage_channels[0], out_channels, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """前向传播。

        Args:
            x: 输入 DAPI 图像 ``(B, C_in, H, W)``。

        Returns:
            torch.Tensor: 生成的目标标记图像 ``(B, C_out, H, W)``，取值 ``[0, 1]``。
        """
        f0 = self.stage0(self.input_adapter(x))
        f1 = self.stage1(self.pool(f0))
        f2 = self.stage2(f1)
        f3 = self.stage3(f2)
        f4 = self.stage4(f3)

        d3 = self.up4(f4, f3)
        d2 = self.up3(d3, f2)
        d1 = self.up2(d2, f1)
        d0 = self.up1(d1, f0)
        return torch.sigmoid(self.head(self.refine(self.up0(d0))))


# ------------------------------------------------------------------ #
# 一对多条件模型
# ------------------------------------------------------------------ #
class ConditionalUNet(nn.Module):
    """标记条件 U-Net：在瓶颈处注入目标标记嵌入。

    前向时除图像外还需提供 ``marker_idx``，模型据此切换生成模式。
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
            out_channels: 输出通道数。
            base_channels: 第一层特征宽度。
            depth:        下采样次数。
            num_markers:  标记类别数，决定嵌入表大小。
        """
        super().__init__()
        self.depth = depth
        bottleneck_channels = base_channels << depth

        self.stem = DoubleConv(in_channels, base_channels)
        self.encoders = nn.ModuleList(
            [Down(base_channels << i, base_channels << (i + 1)) for i in range(depth)]
        )
        # 标记条件嵌入：加到瓶颈特征上，广播到全部空间位置。
        self.marker_embedding = MarkerEmbedding(num_markers, bottleneck_channels)
        self.decoders = nn.ModuleList(
            [
                Up(
                    in_channels=base_channels << (i + 1),
                    skip_channels=base_channels << i,
                    out_channels=base_channels << i,
                )
                for i in reversed(range(depth))
            ]
        )
        self.head = nn.Conv2d(base_channels, out_channels, kernel_size=1)

    def forward(self, x: torch.Tensor, marker_idx: torch.Tensor) -> torch.Tensor:
        """条件前向传播。

        Args:
            x:          输入 DAPI 图像 ``(B, C_in, H, W)``。
            marker_idx: 目标标记编号 ``(B,)``，整型。

        Returns:
            torch.Tensor: 指定标记的生成图像 ``(B, C_out, H, W)``，取值 ``[0, 1]``。
        """
        skips: List[torch.Tensor] = [self.stem(x)]
        for encoder in self.encoders:
            skips.append(encoder(skips[-1]))

        # 瓶颈处注入标记条件：告诉模型生成哪一种标记。
        feature = skips.pop() + self.marker_embedding(marker_idx)
        for decoder in self.decoders:
            feature = decoder(feature, skips.pop())
        return torch.sigmoid(self.head(feature))


class FiLM(nn.Module):
    """Feature-wise Linear Modulation：用 marker 条件生成缩放与偏置。"""

    def __init__(self, feature_channels: int, embed_dim: int) -> None:
        """构建缩放/偏置投影层。

        Args:
            feature_channels: 待调制特征的通道数。
            embed_dim:        marker 嵌入维度。
        """
        super().__init__()
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
            feature:      ``(B, C, H, W)`` 待调制特征。
            marker_embed: ``(B, embed_dim)`` 标记嵌入向量。

        Returns:
            torch.Tensor: 调制后的特征 ``(B, C, H, W)``。
        """
        scale = self.scale_proj(marker_embed)[:, :, None, None]
        shift = self.shift_proj(marker_embed)[:, :, None, None]
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
        """搭建全尺度条件 U-Net。

        Args:
            in_channels:  输入通道数。
            out_channels: 输出通道数。
            base_channels: 第一层特征宽度。
            depth:        下采样次数。
            num_markers:  标记类别数。
            embed_dim:    marker 嵌入维度，供各层 FiLM 共享。
        """
        super().__init__()
        self.depth = depth
        self.embed_dim = embed_dim

        # Marker 嵌入表：将离散 marker_idx 映射为连续向量。
        self.marker_embedding = nn.Embedding(num_markers, embed_dim)
        nn.init.normal_(self.marker_embedding.weight, std=0.02)

        # ---- 编码器 / 瓶颈 / 解码器：各自后接 FiLM 条件调制 ----
        self.stem = DoubleConv(in_channels, base_channels)
        self.stem_film = FiLM(base_channels, embed_dim)

        self.encoders = nn.ModuleList()
        self.encoder_films = nn.ModuleList()
        for i in range(depth):
            self.encoders.append(Down(base_channels << i, base_channels << (i + 1)))
            self.encoder_films.append(FiLM(base_channels << (i + 1), embed_dim))

        self.bottleneck_film = FiLM(base_channels << depth, embed_dim)

        self.decoders = nn.ModuleList()
        self.decoder_films = nn.ModuleList()
        for i in reversed(range(depth)):
            self.decoders.append(
                Up(
                    in_channels=base_channels << (i + 1),
                    skip_channels=base_channels << i,
                    out_channels=base_channels << i,
                )
            )
            self.decoder_films.append(FiLM(base_channels << i, embed_dim))

        self.head = nn.Conv2d(base_channels, out_channels, kernel_size=1)

    def forward(self, x: torch.Tensor, marker_idx: torch.Tensor) -> torch.Tensor:
        """条件前向传播。

        Args:
            x:          输入 DAPI 图像 ``(B, C_in, H, W)``。
            marker_idx: 目标标记编号 ``(B,)``，整型。

        Returns:
            torch.Tensor: 指定标记的生成图像 ``(B, C_out, H, W)``，取值 ``[0, 1]``。
        """
        marker_embed = self.marker_embedding(marker_idx)

        feature = self.stem_film(self.stem(x), marker_embed)
        skips: List[torch.Tensor] = [feature]
        for encoder, film in zip(self.encoders, self.encoder_films):
            feature = film(encoder(feature), marker_embed)
            skips.append(feature)

        feature = self.bottleneck_film(skips.pop(), marker_embed)
        for decoder, film in zip(self.decoders, self.decoder_films):
            feature = film(decoder(feature, skips.pop()), marker_embed)
        return torch.sigmoid(self.head(feature))


class MarkerAdapter(nn.Module):
    """轻量标记适配器：1x1 Conv + BN + ReLU + 1x1 Conv。

    参数量极小，每个标记一个独立分支，学习标记特定的特征变换；
    残差缩放初始为 0，等价于恒等映射。
    """

    def __init__(self, channels: int, num_markers: int) -> None:
        """构建各标记分支。

        Args:
            channels:    特征通道数。
            num_markers: 标记类别数。
        """
        super().__init__()
        self.adapters = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Conv2d(channels, channels // 2, kernel_size=1, bias=False),
                    nn.BatchNorm2d(channels // 2),
                    nn.ReLU(inplace=True),
                    nn.Conv2d(channels // 2, channels, kernel_size=1, bias=False),
                    nn.BatchNorm2d(channels),
                )
                for _ in range(num_markers)
            ]
        )
        self.residual_scale = nn.Parameter(torch.zeros(1))

    def forward(self, feature: torch.Tensor, marker_idx: torch.Tensor) -> torch.Tensor:
        """根据 marker_idx 选择对应的适配器分支。

        Args:
            feature:    ``(B, C, H, W)`` 共享特征。
            marker_idx: ``(B,)`` 整型目标标记编号。

        Returns:
            torch.Tensor: 适配后的特征 ``(B, C, H, W)``。
        """
        output = torch.zeros_like(feature)
        # 按 marker 分组处理，避免 batch 内不同 marker 的串扰。
        for index, adapter in enumerate(self.adapters):
            mask = marker_idx == index
            if mask.any():
                adapted = adapter(feature[mask])
                output[mask] = (feature[mask] + self.residual_scale * adapted).to(
                    feature.dtype
                )
        return output


class AdapterUNet(nn.Module):
    """Shared Encoder + Marker Adapter + Shared Decoder。

    编码器和解码器全部标记共享，仅在 bottleneck 处插入标记专用适配器，
    实现高效的跨标记知识共享；``return_shared`` 打开时额外返回适配前的
    共享特征，供跨标记一致性损失使用。
    """

    def __init__(
        self,
        in_channels: int = 3,
        out_channels: int = 1,
        base_channels: int = 64,
        depth: int = 4,
        num_markers: int = len(MARKERS),
        return_shared: bool = False,
    ) -> None:
        """搭建共享编解码器与标记适配器。

        Args:
            in_channels:   输入通道数。
            out_channels:  输出通道数。
            base_channels: 第一层特征宽度。
            depth:         下采样次数。
            num_markers:   标记类别数。
            return_shared: 前向时是否额外返回适配器前的共享特征。
        """
        super().__init__()
        self.depth = depth
        self.return_shared = return_shared

        self.stem = DoubleConv(in_channels, base_channels)
        self.encoders = nn.ModuleList(
            [Down(base_channels << i, base_channels << (i + 1)) for i in range(depth)]
        )
        self.adapter = MarkerAdapter(base_channels << depth, num_markers)
        self.decoders = nn.ModuleList(
            [
                Up(
                    in_channels=base_channels << (i + 1),
                    skip_channels=base_channels << i,
                    out_channels=base_channels << i,
                )
                for i in reversed(range(depth))
            ]
        )
        self.head = nn.Conv2d(base_channels, out_channels, kernel_size=1)

    def forward(
        self, x: torch.Tensor, marker_idx: torch.Tensor
    ) -> Union[torch.Tensor, tuple]:
        """前向传播。

        Args:
            x:          ``(B, C_in, H, W)`` 输入 DAPI 图像。
            marker_idx: ``(B,)`` 整型目标标记编号。

        Returns:
            Union[torch.Tensor, tuple]: 生成图像 ``(B, C_out, H, W)``，取值
            ``[0, 1]``；``return_shared=True`` 时返回 ``(生成图像, 共享特征)``。
        """
        skips: List[torch.Tensor] = [self.stem(x)]
        for encoder in self.encoders:
            skips.append(encoder(skips[-1]))

        # Bottleneck：通过 marker adapter 注入标记特异性。
        shared_feature = skips.pop()
        feature = self.adapter(shared_feature, marker_idx)
        for decoder in self.decoders:
            feature = decoder(feature, skips.pop())

        output = torch.sigmoid(self.head(feature))
        return (output, shared_feature) if self.return_shared else output


class ConditionalResAttentionUNet(nn.Module):
    """多尺度残差注意力条件 U-Net（V3 主力模型）。

    在 ``ConditionalUNetV2`` 的「全尺度 FiLM 标记条件注入」基础上升级：

    - 编码器/解码器卷积块全部替换为残差块：任务本质是由 DAPI 恢复 IHC
      的配对图像重建，残差连接保留原始空间信息，利于 SSIM/PSNR 像素对齐；
    - 深层特征（通道数达到 ``base_channels << attn_start``）插入轻量
      CBAM 注意力（通道 + 空间），聚焦细胞区域、组织边界等高响应结构，
      浅层保持无注意力以控制计算与显存开销；
    - FiLM 条件注入保持 V2 口径：scale/shift 初始化为恒等映射，训练稳定。
    """

    def __init__(
        self,
        in_channels: int = 3,
        out_channels: int = 1,
        base_channels: int = 64,
        depth: int = 4,
        num_markers: int = len(MARKERS),
        embed_dim: int = 128,
        attn_start: int = 2,
        cbam_reduction: int = 16,
        cbam_kernel: int = 7,
    ) -> None:
        """搭建残差注意力条件 U-Net。

        Args:
            in_channels:   输入通道数（DAPI 为 3，启用上下文输入为 6）。
            out_channels:  输出通道数（目标标记灰度图为 1）。
            base_channels: 第一层特征宽度。
            depth:         下采样次数。
            num_markers:   标记类别数。
            embed_dim:     marker 嵌入维度，供各层 FiLM 共享。
            attn_start:    CBAM 启用阈值：特征通道数达到 ``base_channels << attn_start``
                的层起插入注意力（含瓶颈），浅层不插入。
            cbam_reduction: CBAM 通道注意力的瓶颈压缩比。
            cbam_kernel:   CBAM 空间注意力的卷积核尺寸。
        """
        super().__init__()
        self.depth = depth
        self.embed_dim = embed_dim

        # Marker 嵌入表：将离散 marker_idx 映射为连续向量（与 V2 一致）。
        self.marker_embedding = nn.Embedding(num_markers, embed_dim)
        nn.init.normal_(self.marker_embedding.weight, std=0.02)

        # ---- 编码器：残差卷积 + 深层 CBAM，每层后接 FiLM 条件调制 ----
        self.stem = DoubleConv(in_channels, base_channels)
        self.stem_film = FiLM(base_channels, embed_dim)

        self.encoders = nn.ModuleList()
        self.encoder_films = nn.ModuleList()
        for i in range(depth):
            self.encoders.append(
                Down(
                    base_channels << i,
                    base_channels << (i + 1),
                    residual=True,
                    attention=(i + 1) >= attn_start,
                )
            )
            self.encoder_films.append(FiLM(base_channels << (i + 1), embed_dim))

        # ---- 瓶颈：最深层 Down 已含残差与 CBAM，此处仅注入 FiLM 条件 ----
        self.bottleneck_film = FiLM(base_channels << depth, embed_dim)

        # ---- 解码器：与编码器对称的残差 + 深层 CBAM + FiLM ----
        self.decoders = nn.ModuleList()
        self.decoder_films = nn.ModuleList()
        for i in reversed(range(depth)):
            self.decoders.append(
                Up(
                    in_channels=base_channels << (i + 1),
                    skip_channels=base_channels << i,
                    out_channels=base_channels << i,
                    residual=True,
                    attention=i >= attn_start,
                )
            )
            self.decoder_films.append(FiLM(base_channels << i, embed_dim))

        self.head = nn.Conv2d(base_channels, out_channels, kernel_size=1)

    def forward(self, x: torch.Tensor, marker_idx: torch.Tensor) -> torch.Tensor:
        """条件前向传播。

        Args:
            x:          输入 DAPI 图像 ``(B, C_in, H, W)``。
            marker_idx: 目标标记编号 ``(B,)``，整型。

        Returns:
            torch.Tensor: 指定标记的生成图像 ``(B, C_out, H, W)``，取值 ``[0, 1]``。
        """
        marker_embed = self.marker_embedding(marker_idx)

        feature = self.stem_film(self.stem(x), marker_embed)
        skips: List[torch.Tensor] = [feature]
        for encoder, film in zip(self.encoders, self.encoder_films):
            feature = film(encoder(feature), marker_embed)
            skips.append(feature)

        # 瓶颈 FiLM 条件注入后，逐级上采样并与跳跃连接融合。
        feature = self.bottleneck_film(skips.pop(), marker_embed)
        for decoder, film in zip(self.decoders, self.decoder_films):
            feature = film(decoder(feature, skips.pop()), marker_embed)
        return torch.sigmoid(self.head(feature))


# ------------------------------------------------------------------ #
# 统一入口
# ------------------------------------------------------------------ #
class ModelRegistry:
    """模型注册表：按配置名统一实例化模型。"""

    REGISTRY: Dict[str, Callable[..., nn.Module]] = {
        "unet": UNet,
        "resnet_unet": ResNetUNet,
        "conditional_unet": ConditionalUNet,       # 原版：仅 bottleneck 条件
        "conditional_unet_v2": ConditionalUNetV2,  # V2：Multi-scale FiLM
        "adapter_unet": AdapterUNet,               # 共享编码器 + Marker Adapter
        # V3 主力：残差 + 深层 CBAM + Multi-scale FiLM
        "conditional_unet_v3": ConditionalResAttentionUNet,
    }

    # 需要 marker_idx 输入（一对多联合建模）的模型类型。
    CONDITIONAL_TYPES = {
        "conditional_unet",
        "conditional_unet_v2",
        "adapter_unet",
        "conditional_unet_v3",
    }

    @classmethod
    def build(cls, config: Dict[str, Any]) -> nn.Module:
        """按配置实例化模型。

        配置中目标模型不认识的参数键会被自动过滤——实验矩阵中
        「base 配置 + 覆盖模型类型」时，base 里残留的参数不会导致实例化失败。

        Args:
            config: 全局配置，读取 ``model`` 一节。

        Returns:
            nn.Module: 实例化后的模型。

        Raises:
            ValueError: 模型类型未注册时抛出。
        """
        model_cfg = dict(config.get("model", {}) or {})
        model_type = model_cfg.pop("type", "unet")

        if model_type not in cls.REGISTRY:
            raise ValueError(
                f"未知模型类型 '{model_type}'，可选: {list(cls.REGISTRY)}"
            )
        model_cls = cls.REGISTRY[model_type]

        # 仅保留构造函数声明了的参数；若构造函数接受 **kwargs 则原样传递。
        signature = inspect.signature(model_cls.__init__)
        accepts_kwargs = any(
            parameter.kind == inspect.Parameter.VAR_KEYWORD
            for parameter in signature.parameters.values()
        )
        if not accepts_kwargs:
            valid_keys = set(signature.parameters) - {"self"}
            model_cfg = {k: v for k, v in model_cfg.items() if k in valid_keys}
        return model_cls(**model_cfg)

    @classmethod
    def is_conditional(cls, config: Dict[str, Any]) -> bool:
        """判断模型是否需要 marker_idx 输入（含条件/适配器模型）。

        Args:
            config: 全局配置，读取 ``model.type``。

        Returns:
            bool: 一对多联合建模返回 ``True``。
        """
        return config.get("model", {}).get("type") in cls.CONDITIONAL_TYPES
