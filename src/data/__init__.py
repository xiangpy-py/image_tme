"""通用数据层。

集中放置与「数据本身」有关、但不涉及模型训练的通用能力：

- ``constants``:  赛题标记名、图像后缀与命名规范；
- ``DatasetSource``:  标记目录发现与图像读取；
- ``DatasetSplitter``: 按 ROI 划分训练/验证集并落盘；
- ``DatasetAnalyzer``: 数据集统计。

本子包只依赖标准库与 numpy / opencv，**不依赖 torch**，
因此数据准备（analyze / split）无需加载深度学习框架即可运行。
"""

from .constants import IMAGE_EXTENSIONS, MARKERS, SOURCE_MARKER, sanitize_marker_name
from .dataset import DatasetAnalyzer, DatasetSource, DatasetSplitter

__all__ = [
    "SOURCE_MARKER",
    "MARKERS",
    "IMAGE_EXTENSIONS",
    "sanitize_marker_name",
    "DatasetSource",
    "DatasetSplitter",
    "DatasetAnalyzer",
]
