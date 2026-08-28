"""数据划分模块：按 ROI 划分训练/验证集。

同一 ROI 切出的相邻 patch 在空间上高度相关，
若随机按 patch 划分会造成验证集信息泄漏（赛题注意事项第 1 条）。
因此以文件名前缀 ``ROIxxx`` 为单位划分，
保证同一 ROI 的所有 patch 只出现在同一侧。
"""

import json
import random
import re
from pathlib import Path
from typing import Dict, List, Tuple

from .constants import SOURCE_MARKER
from .virtual_staining_dataset import discover_marker_dirs, list_image_files


def extract_roi_id(filename: str) -> str:
    """从 patch 文件名中提取 ROI 标识。

    赛题命名约定为 ``ROI000_00_01.jpg``，提取第一个下划线前的
    ``ROI000`` 作为划分单位；不符合该命名时退化为整个文件名前缀。

    Args:
        filename: 文件名（可带后缀）。

    Returns:
        str: ROI 标识字符串。
    """
    match = re.match(r"(ROI\d+)", filename)
    if match:
        return match.group(1)
    return Path(filename).stem.split("_")[0]


def split_by_roi(
    root: str,
    split: str = "train",
    val_ratio: float = 0.15,
    seed: int = 42,
) -> Tuple[List[str], List[str]]:
    """按 ROI 将配对样本划分为训练集与验证集文件名列表。

    Args:
        root:      数据根目录。
        split:     待划分的目录名，通常为 ``"train"``。
        val_ratio: 验证集 ROI 占比。
        seed:      随机种子，保证划分可复现。

    Returns:
        Tuple[List[str], List[str]]: (训练集文件名列表, 验证集文件名列表)，
        元素为不含后缀的文件名。

    Raises:
        ValueError: 未找到任何样本时抛出。
    """
    all_names: List[str] = []
    for _split_dir, marker_dirs in discover_marker_dirs(root, split):
        for path in list_image_files(marker_dirs[SOURCE_MARKER]):
            all_names.append(path.stem)

    if not all_names:
        raise ValueError(f"未找到样本: root={root}, split={split}")

    # 收集全部 ROI 并打乱，再按比例切分。
    roi_ids = sorted({extract_roi_id(name) for name in all_names})
    rng = random.Random(seed)
    rng.shuffle(roi_ids)

    val_count = max(1, int(round(len(roi_ids) * val_ratio)))
    val_rois = set(roi_ids[:val_count])

    train_names = [n for n in all_names if extract_roi_id(n) not in val_rois]
    val_names = [n for n in all_names if extract_roi_id(n) in val_rois]
    return train_names, val_names


def save_splits(
    train_names: List[str],
    val_names: List[str],
    save_dir: str,
) -> None:
    """将划分结果落盘为 JSON，供训练脚本加载复用。

    Args:
        train_names: 训练集文件名列表。
        val_names:   验证集文件名列表。
        save_dir:    保存目录，约定为 ``data/splits/``。

    Returns:
        None
    """
    path = Path(save_dir)
    path.mkdir(parents=True, exist_ok=True)

    payload: Dict[str, List[str]] = {
        "train": sorted(train_names),
        "val": sorted(val_names),
    }
    with open(path / "split.json", "w", encoding="utf-8") as file:
        json.dump(payload, file, ensure_ascii=False, indent=2)


def load_splits(save_dir: str) -> Tuple[List[str], List[str]]:
    """读取已保存的划分文件。

    Args:
        save_dir: 划分文件所在目录。

    Returns:
        Tuple[List[str], List[str]]: (训练集文件名, 验证集文件名)。

    Raises:
        FileNotFoundError: 划分文件不存在时抛出。
    """
    path = Path(save_dir) / "split.json"
    if not path.is_file():
        raise FileNotFoundError(f"划分文件不存在: {path}，请先运行 split 命令")

    with open(path, "r", encoding="utf-8") as file:
        payload = json.load(file)
    return payload["train"], payload["val"]
