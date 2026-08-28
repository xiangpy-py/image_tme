"""Cross-Marker Consistency Loss。

约束：同一 DAPI 输入生成的不同标记图像，在共享特征空间应保持一致性。

实现：对 AdapterUNet 的 bottleneck 共享特征，要求不同 marker 的
      适配前特征（z_shared）尽可能一致，即适配器只负责标记特异性变换，
      不破坏共享结构信息。

公式：
    L_cross = || z_shared_marker_i - z_shared_marker_j ||_2
"""

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class CrossMarkerConsistencyLoss(nn.Module):
    """跨标记一致性损失：强制共享特征独立于标记类型。"""

    def __init__(self, margin: float = 0.0) -> None:
        super().__init__()
        self.margin = margin

    def forward(
        self,
        shared_features: torch.Tensor,
        marker_idx: torch.Tensor,
    ) -> torch.Tensor:
        """计算同一 batch 内不同标记样本的共享特征一致性。

        Args:
            shared_features: (B, C, H, W)，适配器前的共享特征
            marker_idx:      (B,) 整型，各样本的目标标记

        Returns:
            torch.Tensor: 标量损失
        """
        batch = shared_features.shape[0]
        if batch < 2:
            return shared_features.new_zeros(())

        # 全局平均池化得到特征向量 (B, C)
        pooled = F.adaptive_avg_pool2d(shared_features, (1, 1)).squeeze(-1).squeeze(-1)

        # 按 marker 分组计算组内方差（应小 = 一致）
        loss = pooled.new_zeros(())
        unique_markers = torch.unique(marker_idx)

        for m in unique_markers:
            mask = marker_idx == m
            if mask.sum() < 2:
                continue
            group = pooled[mask]  # (N, C)
            # 组内特征应彼此接近：惩罚到组均值的距离
            center = group.mean(dim=0, keepdim=True)
            loss = loss + F.mse_loss(group, center.expand_as(group))

        # 额外：不同 marker 的组均值也应接近（共享结构）
        if len(unique_markers) > 1:
            centers = torch.stack([
                pooled[marker_idx == m].mean(dim=0)
                for m in unique_markers
                if (marker_idx == m).sum() > 0
            ])
            # 惩罚不同 marker 中心之间的差异
            mean_center = centers.mean(dim=0, keepdim=True)
            loss = loss + F.mse_loss(centers, mean_center.expand_as(centers))

        return loss / max(len(unique_markers), 1)
