"""虚拟染色配对数据集。

目录约定（与赛题数据组织方式一致）::

    data/<organ>/train/DAPI/xxx.jpg        # 源图像
    data/<organ>/train/<MARKER>/xxx.jpg    # 目标真值（同名配对）
    data/<organ>/test/DAPI/xxx.jpg         # 测试输入（无真值）

也兼容无器官层级的扁平结构（data/train/...），
通过 ``discover_marker_dirs`` 递归定位各标记目录。
"""

from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

from .constants import IMAGE_EXTENSIONS, MARKERS, SOURCE_MARKER
from .multiscale import MultiScaleInput


def list_image_files(directory: Path) -> List[Path]:
    """列出目录下全部图像文件，按文件名排序保证顺序稳定。

    Args:
        directory: 图像所在目录。

    Returns:
        List[Path]: 排序后的图像路径列表，目录不存在时返回空列表。
    """
    if not directory.is_dir():
        return []
    files = [
        p for p in directory.iterdir()
        if p.suffix.lower() in IMAGE_EXTENSIONS
    ]
    return sorted(files, key=lambda p: p.name)


def discover_marker_dirs(root: str, split: str) -> List[Tuple[Path, Dict[str, Path]]]:
    """在数据根目录下定位所有「标记目录组」。

    兼容两种组织方式：

    - 多器官: ``root/<organ>/<split>/<MARKER>/``
    - 单器官: ``root/<split>/<MARKER>/`` 或 ``root/<MARKER>/``（root 直接指向 split）

    Args:
        root:  数据根目录，例如 ``"data"``。
        split: 划分名称，例如 ``"train"`` 或 ``"test"``。

    Returns:
        List[Tuple[Path, Dict[str, Path]]]:
            每个元素为 ``(split目录, {标记名: 标记目录路径})``，
            仅收录至少包含 DAPI 目录的组。
    """
    root_path = Path(root)
    groups: List[Tuple[Path, Dict[str, Path]]] = []

    def _collect(split_dir: Path) -> Optional[Dict[str, Path]]:
        """检查某目录是否直接包含标记子目录，是则返回映射。"""
        marker_dirs = {
            marker: split_dir / marker
            for marker in [SOURCE_MARKER] + MARKERS
            if (split_dir / marker).is_dir()
        }
        return marker_dirs if SOURCE_MARKER in marker_dirs else None

    # 情形 1：root 本身就是 split 目录（如 root=data/train）。
    direct = _collect(root_path)
    if direct is not None and root_path.name == split:
        groups.append((root_path, direct))
        return groups

    # 情形 2：root/split/<MARKER>/（无器官层级）。
    plain_split = root_path / split
    if plain_split.is_dir():
        collected = _collect(plain_split)
        if collected is not None:
            groups.append((plain_split, collected))

    # 情形 3：root/<organ>/<split>/<MARKER>/（多器官层级）。
    for child in sorted(root_path.iterdir()):
        if not child.is_dir() or child.name == split:
            continue
        organ_split = child / split
        if not organ_split.is_dir():
            continue
        collected = _collect(organ_split)
        if collected is not None:
            groups.append((organ_split, collected))

    return groups


def read_image_as_float(path: Path, grayscale: bool = False) -> np.ndarray:
    """读取图像并转换为 ``[0, 1]`` 范围的 float32 数组（HWC）。

    DAPI 源图按 3 通道读取；目标标记图统一按单通道灰度读取
    （赛事发布的目标图为灰度强度图，即使以 3 通道 JPG 存储，
    三个通道也完全相同），与模型 ``out_channels=1`` 保持一致。

    Args:
        path:      图像文件路径。
        grayscale: 是否按单通道灰度读取，目标标记图应传 ``True``。

    Returns:
        np.ndarray: HWC 布局、float32、取值 ``[0, 1]`` 的图像数组。
    """
    flag = cv2.IMREAD_GRAYSCALE if grayscale else cv2.IMREAD_UNCHANGED
    image = cv2.imread(str(path), flag)
    if image is None:
        raise IOError(f"图像读取失败: {path}")

    # OpenCV 默认 BGR，这里转为 RGB 保持语义一致。
    if image.ndim == 3 and image.shape[-1] == 3:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    elif image.ndim == 2:
        image = image[..., None]  # 灰度图补通道维 -> (H, W, 1)

    return (image.astype(np.float32) / 255.0)


def _to_tensor(image: np.ndarray) -> torch.Tensor:
    """将 HWC float 数组转换为 CHW 的 torch 张量。"""
    return torch.from_numpy(np.ascontiguousarray(image.transpose(2, 0, 1)))


class VirtualStainingDataset(Dataset):
    """单标记配对数据集：DAPI -> 指定 IHC 标记。

    每个样本返回::

        {
            "input":  (C_in, H, W)  张量,   # DAPI
            "target": (C_out, H, W) 张量,   # 目标标记（测试模式无此键）
            "name":   文件名（不含后缀）,   # 用于结果命名对应
        }
    """

    def __init__(
        self,
        root: str,
        marker: str,
        split: str = "train",
        transform: Optional[Callable] = None,
        file_list: Optional[List[str]] = None,
        multiscale: Optional[MultiScaleInput] = None,
    ) -> None:
        """收集 DAPI 与目标标记的同名配对样本。

        Args:
            root:       数据根目录（支持多器官/单器官两种组织方式）。
            marker:     目标标记名，必须是 ``MARKERS`` 之一。
            split:      ``"train"`` 或 ``"test"``。
            transform:  增强流水线，签名为 ``(image, target) -> (image, target)``。
            file_list:  可选的文件名白名单（来自 ROI 划分文件），
                为 ``None`` 时使用目录下全部样本。
            multiscale: 可选的多尺度输入变换（论文 GPTs 的输入策略），
                仅作用于输入图像；启用后输入通道数变为 ``C * len(scales)``。

        Raises:
            ValueError: 标记名非法或未找到任何配对样本时抛出。
        """
        super().__init__()
        if marker not in MARKERS:
            raise ValueError(f"未知标记 '{marker}'，可选: {MARKERS}")

        self.marker = marker
        self.transform = transform
        self.multiscale = multiscale
        self.samples: List[Tuple[Path, Optional[Path]]] = []

        # 跨器官聚合所有配对样本，文件名作为配对键。
        for _split_dir, marker_dirs in discover_marker_dirs(root, split):
            source_dir = marker_dirs[SOURCE_MARKER]
            target_dir = marker_dirs.get(marker) if split == "train" else None

            for source_path in list_image_files(source_dir):
                if file_list is not None and source_path.stem not in file_list:
                    continue

                target_path: Optional[Path] = None
                if target_dir is not None:
                    candidate = target_dir / source_path.name
                    if not candidate.is_file():
                        continue  # 缺失配对真值的样本直接跳过
                    target_path = candidate

                self.samples.append((source_path, target_path))

        if not self.samples:
            raise ValueError(
                f"未找到配对样本: root={root}, marker={marker}, split={split}"
            )

    def __len__(self) -> int:
        """返回配对样本总数。"""
        return len(self.samples)

    def __getitem__(self, index: int) -> Dict[str, Any]:
        """读取并增强一个样本。

        Args:
            index: 样本下标。

        Returns:
            Dict[str, Any]: 包含 ``input``/``name``，训练模式另含 ``target``。
        """
        source_path, target_path = self.samples[index]

        image = read_image_as_float(source_path)
        target = read_image_as_float(target_path, grayscale=True) if target_path else None

        if self.transform is not None:
            image, target = self.transform(image, target)

        # 多尺度输入只作用于源图像，须在配对增强之后执行，
        # 保证真值始终对应 1.0 尺度分支的空间位置。
        if self.multiscale is not None:
            image = self.multiscale(image)

        sample: Dict[str, Any] = {
            "input": _to_tensor(image),
            "name": source_path.stem,
        }
        if target is not None:
            sample["target"] = _to_tensor(target)
        return sample


class MultiMarkerDataset(Dataset):
    """多标记条件数据集：一个 DAPI 样本随机配对一种目标标记。

    用于一对多联合建模（Version 4）：每次取样本时随机采样
    一个目标标记并返回其编号，供模型的 marker token 使用。

    每个样本返回::

        {
            "input":       (C, H, W) 张量,
            "target":      (C, H, W) 张量,
            "marker_idx":  目标标记编号（对应 constants.MARKERS 顺序）,
            "name":        文件名,
        }
    """

    def __init__(
        self,
        root: str,
        split: str = "train",
        markers: Optional[List[str]] = None,
        transform: Optional[Callable] = None,
        file_list: Optional[List[str]] = None,
        multiscale: Optional[MultiScaleInput] = None,
    ) -> None:
        """收集同时存在全部目标标记真值的样本。

        只有四类标记真值齐全的 patch 才纳入，保证任意随机采样
        标记时都有监督信号。

        Args:
            root:       数据根目录。
            split:      划分名称（该数据集仅用于训练/验证）。
            markers:    参与联合建模的标记列表，默认全部四类。
            transform:  增强流水线。
            file_list:  ROI 划分文件名白名单。
            multiscale: 可选的多尺度输入变换，仅作用于输入图像。

        Raises:
            ValueError: 未找到任何全配对样本时抛出。
        """
        super().__init__()
        self.markers = markers or list(MARKERS)
        self.transform = transform
        self.multiscale = multiscale
        # 文件名 -> (DAPI路径, {标记: 路径})
        self.index: Dict[str, Tuple[Path, Dict[str, Path]]] = {}

        for _split_dir, marker_dirs in discover_marker_dirs(root, split):
            if SOURCE_MARKER not in marker_dirs:
                continue
            if not all(m in marker_dirs for m in self.markers):
                continue  # 该组标记不全，无法用于联合建模

            for source_path in list_image_files(marker_dirs[SOURCE_MARKER]):
                if file_list is not None and source_path.stem not in file_list:
                    continue

                target_map: Dict[str, Path] = {}
                complete = True
                for marker in self.markers:
                    candidate = marker_dirs[marker] / source_path.name
                    if not candidate.is_file():
                        complete = False
                        break
                    target_map[marker] = candidate

                if complete:
                    self.index[source_path.stem] = (source_path, target_map)

        if not self.index:
            raise ValueError(f"未找到全标记配对样本: root={root}, split={split}")

        self.names = sorted(self.index.keys())

    def __len__(self) -> int:
        """返回样本总数（每样本每 epoch 随机配对一种标记）。"""
        return len(self.names)

    def __getitem__(self, index: int) -> Dict[str, Any]:
        """读取样本并随机采样一个目标标记。

        Args:
            index: 样本下标。

        Returns:
            Dict[str, Any]: 含 ``input``/``target``/``marker_idx``/``name``。
        """
        name = self.names[index]
        source_path, target_map = self.index[name]

        # 随机选择本步监督的目标标记，实现一对多训练。
        marker_idx = np.random.randint(0, len(self.markers))
        marker = self.markers[marker_idx]

        image = read_image_as_float(source_path)
        target = read_image_as_float(target_map[marker], grayscale=True)

        if self.transform is not None:
            image, target = self.transform(image, target)

        # 多尺度输入只作用于源图像，须在配对增强之后执行。
        if self.multiscale is not None:
            image = self.multiscale(image)

        return {
            "input": _to_tensor(image),
            "target": _to_tensor(target),
            "marker_idx": marker_idx,
            "name": name,
        }
