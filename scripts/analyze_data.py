"""数据分析脚本。

扫描训练数据，统计图像数量、尺寸、格式、通道与像素分布，
结果写入 ``data/statistics.json``（对应 plan.md 数据分析阶段）。

用法::

    python scripts/analyze_data.py --root data
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict

# 允许脚本以 ``python scripts/xxx.py`` 方式直接运行。
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.datasets.constants import IMAGE_EXTENSIONS, MARKERS, SOURCE_MARKER


def analyze_marker_dir(marker_dir: Path, sample_limit: int = 200) -> Dict[str, Any]:
    """统计单个标记目录的图像属性。

    对规模较大的目录抽样统计像素分布，避免全量读图耗时过长。

    Args:
        marker_dir:   标记图像目录。
        sample_limit: 像素分布统计的最大抽样数。

    Returns:
        Dict[str, Any]: 含数量、尺寸、通道数、像素均值/标准差等统计项。
    """
    import cv2
    import numpy as np

    files = [
        p for p in marker_dir.iterdir()
        if p.suffix.lower() in IMAGE_EXTENSIONS
    ] if marker_dir.is_dir() else []

    sizes = set()
    channels = set()
    pixel_sum, pixel_sq_sum, pixel_count = 0.0, 0.0, 0

    for index, path in enumerate(sorted(files, key=lambda p: p.name)):
        image = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
        if image is None:
            continue
        sizes.add(f"{image.shape[1]}x{image.shape[0]}")
        channels.add(1 if image.ndim == 2 else image.shape[-1])

        if index < sample_limit:
            normalized = image.astype(np.float64) / 255.0
            pixel_sum += float(normalized.sum())
            pixel_sq_sum += float((normalized ** 2).sum())
            pixel_count += normalized.size

    mean = pixel_sum / max(pixel_count, 1)
    std = (pixel_sq_sum / max(pixel_count, 1) - mean ** 2) ** 0.5

    return {
        "count": len(files),
        "sizes": sorted(sizes),
        "channels": sorted(channels),
        "pixel_mean": round(mean, 4),
        "pixel_std": round(std, 4),
    }


def main() -> None:
    """扫描全部标记目录并汇总为 statistics.json。"""
    parser = argparse.ArgumentParser(description="数据集统计分析")
    parser.add_argument("--root", type=str, default="data", help="数据根目录")
    parser.add_argument(
        "--output", type=str, default="data/statistics.json", help="统计结果输出路径"
    )
    args = parser.parse_args()

    root = Path(args.root)
    statistics: Dict[str, Any] = {}

    # 同时兼容 data/train/<MARKER> 与 data/<organ>/train/<MARKER> 两种结构。
    split_dirs = [p for p in root.rglob("*") if p.is_dir() and p.name in ("train", "test")]
    for split_dir in sorted(split_dirs):
        group: Dict[str, Any] = {}
        for marker in [SOURCE_MARKER] + MARKERS:
            marker_dir = split_dir / marker
            if marker_dir.is_dir():
                group[marker] = analyze_marker_dir(marker_dir)
        if group:
            statistics[str(split_dir)] = group
            print(f"[统计] {split_dir}: " +
                  ", ".join(f"{k}={v['count']}张" for k, v in group.items()))

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as file:
        json.dump(statistics, file, ensure_ascii=False, indent=2)
    print(f"统计结果已保存: {output_path}")


if __name__ == "__main__":
    main()
