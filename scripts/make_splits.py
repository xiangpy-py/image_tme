"""数据划分脚本。

按 ROI 单位划分训练/验证集，结果保存到 ``data/splits/split.json``，
避免同一 ROI 的相邻 patch 同时出现在训练与验证两侧造成数据泄漏。

用法::

    python scripts/make_splits.py --root data --val-ratio 0.15
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.datasets.splits import save_splits, split_by_roi


def main() -> None:
    """执行 ROI 划分并落盘。"""
    parser = argparse.ArgumentParser(description="按 ROI 划分训练/验证集")
    parser.add_argument("--root", type=str, default="data", help="数据根目录")
    parser.add_argument("--val-ratio", type=float, default=0.15, help="验证集 ROI 占比")
    parser.add_argument("--seed", type=int, default=42, help="随机种子")
    parser.add_argument(
        "--save-dir", type=str, default="data/splits", help="划分结果保存目录"
    )
    args = parser.parse_args()

    train_names, val_names = split_by_roi(
        root=args.root, val_ratio=args.val_ratio, seed=args.seed
    )
    save_splits(train_names, val_names, args.save_dir)

    print(f"训练集: {len(train_names)} 张 | 验证集: {len(val_names)} 张")
    print(f"划分文件已保存: {Path(args.save_dir) / 'split.json'}")


if __name__ == "__main__":
    main()
