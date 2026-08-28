"""DataLoader 构建模块。

根据配置统一创建训练/验证/测试 DataLoader，
把「数据集选择（单标记 or 多标记）」与「ROI 划分」逻辑封装在一处，
供训练与推理脚本直接调用。
"""

from pathlib import Path
from typing import Any, Dict, Optional, Tuple

from torch.utils.data import DataLoader, Dataset

from .multiscale import MultiScaleInput
from .splits import load_splits
from .transforms import build_transforms
from .virtual_staining_dataset import (
    MultiMarkerDataset,
    VirtualStainingDataset,
)


def _build_multiscale(config: Dict[str, Any]) -> Optional[MultiScaleInput]:
    """根据配置构建多尺度输入变换（论文 GPTs 的输入策略）。

    Args:
        config: 全局配置，读取 ``data.multiscale`` 一节，
            形如 ``{enabled: true, scales: [0.5, 1.0, 2.0]}``。

    Returns:
        Optional[MultiScaleInput]: 启用时返回变换实例，否则返回 ``None``。
    """
    ms_cfg = config.get("data", {}).get("multiscale", {}) or {}
    if not ms_cfg.get("enabled", False):
        return None
    return MultiScaleInput(scales=list(ms_cfg.get("scales", [0.5, 1.0, 2.0])))


def _build_train_val_datasets(config: Dict[str, Any]) -> Tuple[Dataset, Dataset]:
    """根据配置选择数据集类型并构建训练/验证数据集。

    Args:
        config: 全局配置，读取 ``data`` 与 ``model`` 两节。

    Returns:
        Tuple[Dataset, Dataset]: (训练数据集, 验证数据集)。
    """
    data_cfg = config.get("data", {})
    model_cfg = config.get("model", {})

    root = data_cfg.get("root", "data")
    marker = data_cfg.get("marker", "CD68")
    split_dir = data_cfg.get("split_dir", "data/splits")
    multi_marker = model_cfg.get("type") == "conditional_unet"
    multiscale = _build_multiscale(config)

    # 优先复用已保存的 ROI 划分；不存在时退化为数据集内部全量+验证复制。
    split_file = Path(split_dir) / "split.json"
    train_list: Optional[list] = None
    val_list: Optional[list] = None
    if split_file.is_file():
        train_list, val_list = load_splits(split_dir)

    if multi_marker:
        train_dataset = MultiMarkerDataset(
            root=root, split="train",
            transform=build_transforms(config, train=True),
            file_list=train_list, multiscale=multiscale,
        )
        val_dataset = MultiMarkerDataset(
            root=root, split="train",
            transform=build_transforms(config, train=False),
            file_list=val_list, multiscale=multiscale,
        )
    else:
        train_dataset = VirtualStainingDataset(
            root=root, marker=marker, split="train",
            transform=build_transforms(config, train=True),
            file_list=train_list, multiscale=multiscale,
        )
        val_dataset = VirtualStainingDataset(
            root=root, marker=marker, split="train",
            transform=build_transforms(config, train=False),
            file_list=val_list, multiscale=multiscale,
        )

    return train_dataset, val_dataset


def build_dataloaders(config: Dict[str, Any]) -> Dict[str, DataLoader]:
    """构建训练与验证 DataLoader。

    Args:
        config: 全局配置，读取 ``data`` / ``training`` / ``model`` 节。

    Returns:
        Dict[str, DataLoader]: 含 ``"train"`` 与 ``"val"`` 两个字典项。
    """
    train_cfg = config.get("training", {})
    batch_size = int(train_cfg.get("batch_size", 16))
    num_workers = int(config.get("data", {}).get("num_workers", 4))

    train_dataset, val_dataset = _build_train_val_datasets(config)

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True,
        persistent_workers=num_workers > 0,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )
    return {"train": train_loader, "val": val_loader}


def build_test_loader(config: Dict[str, Any]) -> DataLoader:
    """构建测试集 DataLoader（仅 DAPI 输入，无真值）。

    Args:
        config: 全局配置，读取 ``data.root`` / ``data.marker`` 等。

    Returns:
        DataLoader: 顺序遍历、不 shuffle 的测试加载器。
    """
    data_cfg = config.get("data", {})
    dataset = VirtualStainingDataset(
        root=data_cfg.get("root", "data"),
        marker=data_cfg.get("marker", "CD68"),
        split="test",
        transform=build_transforms(config, train=False),
        multiscale=_build_multiscale(config),
    )
    return DataLoader(
        dataset,
        batch_size=int(config.get("training", {}).get("batch_size", 16)),
        shuffle=False,
        num_workers=int(data_cfg.get("num_workers", 4)),
        pin_memory=True,
    )
