"""数据集操作：目录定位与图像读取、ROI 划分、数据统计。

目录约定（与赛题数据组织方式一致）::

    data/<organ>/train/DAPI/xxx.jpg        # 源图像
    data/<organ>/train/<MARKER>/xxx.jpg    # 目标真值（同名配对）
    data/<organ>/test/DAPI/xxx.jpg         # 测试输入（无真值）

也兼容无器官层级的扁平结构（data/train/...）。
本模块只依赖 numpy / opencv，不涉及模型训练。
"""

import json
import random
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np

from .constants import IMAGE_EXTENSIONS, MARKERS, SOURCE_MARKER


class DatasetSource:
    """数据源定位与读取：找到标记目录并把图像读成数组。"""

    @staticmethod
    def list_images(directory: Path) -> List[Path]:
        """列出目录下全部图像文件，按文件名排序保证顺序稳定。

        Args:
            directory: 图像所在目录。

        Returns:
            List[Path]: 排序后的图像路径列表，目录不存在时返回空列表。
        """
        if not directory.is_dir():
            return []
        files = [p for p in directory.iterdir() if p.suffix.lower() in IMAGE_EXTENSIONS]
        return sorted(files, key=lambda p: p.name)

    @staticmethod
    def discover_marker_dirs(
        root: str, split: str
    ) -> List[Tuple[Path, Dict[str, Path]]]:
        """在数据根目录下定位所有「标记目录组」。

        兼容两种组织方式：

        - 多器官: ``root/<organ>/<split>/<MARKER>/``
        - 单器官: ``root/<split>/<MARKER>/`` 或 ``root/<MARKER>/``

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

    @staticmethod
    def read_uint8(path: Path, grayscale: bool = False) -> np.ndarray:
        """读取图像并返回 HWC 布局的 uint8 数组（保持原始像素值）。

        供内存缓存使用：解码结果原样保存，取用时再归一化，
        避免每个 epoch 重复 JPG 解码的开销。

        Args:
            path:      图像文件路径。
            grayscale: 是否按单通道灰度读取，目标标记图应传 ``True``。

        Returns:
            np.ndarray: HWC 布局、uint8 的图像数组。

        Raises:
            IOError: 图像解码失败时抛出。
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
        return image

    @staticmethod
    def uint8_to_float(image: np.ndarray) -> np.ndarray:
        """将 HWC uint8 图像数组归一化到 ``[0, 1]`` 的 float32 数组。

        Args:
            image: HWC uint8 图像数组。

        Returns:
            np.ndarray: 归一化后的 float32 数组。
        """
        return image.astype(np.float32) / 255.0

    @classmethod
    def read_float(cls, path: Path, grayscale: bool = False) -> np.ndarray:
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
        return cls.uint8_to_float(cls.read_uint8(path, grayscale=grayscale))


class DatasetSplitter:
    """按 ROI 划分训练/验证集并落盘。

    同一 ROI 切出的相邻 patch 在空间上高度相关，若随机按 patch 划分
    会造成验证集信息泄漏（赛题注意事项第 1 条）。因此以文件名前缀
    ``ROIxxx`` 为单位划分，保证同一 ROI 的所有 patch 只出现在同一侧。
    """

    @staticmethod
    def extract_roi_id(filename: str) -> str:
        """从 patch 文件名中提取 ROI 标识。

        赛题命名约定为 ``ROI000_00_01.jpg``，提取第一个下划线前的
        ``ROI000`` 作为划分单位；不符合该命名时退化为文件名前缀。

        Args:
            filename: 文件名（可带后缀）。

        Returns:
            str: ROI 标识字符串。
        """
        match = re.match(r"(ROI\d+)", filename)
        if match:
            return match.group(1)
        return Path(filename).stem.split("_")[0]

    @classmethod
    def split(
        cls,
        root: str,
        split: str = "train",
        val_ratio: float = 0.15,
        seed: int = 42,
    ) -> Tuple[List[str], List[str]]:
        """按 ROI 把配对样本划分为训练集与验证集文件名列表。

        Args:
            root:      数据根目录。
            split:     待划分的目录名，通常为 ``"train"``。
            val_ratio: 验证集 ROI 占比。
            seed:      随机种子，保证划分可复现。

        Returns:
            Tuple[List[str], List[str]]: (训练集文件名, 验证集文件名)。

        Raises:
            ValueError: 未找到任何样本时抛出。
        """
        all_names: List[str] = []
        for _split_dir, marker_dirs in DatasetSource.discover_marker_dirs(root, split):
            for path in DatasetSource.list_images(marker_dirs[SOURCE_MARKER]):
                all_names.append(path.stem)

        if not all_names:
            raise ValueError(f"未找到样本: root={root}, split={split}")

        # 收集全部 ROI 并打乱，再按比例切分。
        roi_ids = sorted({cls.extract_roi_id(name) for name in all_names})
        rng = random.Random(seed)
        rng.shuffle(roi_ids)

        val_count = max(1, int(round(len(roi_ids) * val_ratio)))
        val_rois = set(roi_ids[:val_count])

        train_names = [n for n in all_names if cls.extract_roi_id(n) not in val_rois]
        val_names = [n for n in all_names if cls.extract_roi_id(n) in val_rois]
        return train_names, val_names

    @staticmethod
    def save(train_names: List[str], val_names: List[str], save_dir: str) -> None:
        """把划分结果落盘为 JSON，供训练加载复用。

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

    @staticmethod
    def load(save_dir: str) -> Tuple[List[str], List[str]]:
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
            raise FileNotFoundError(f"划分文件不存在: {path}，请先执行 split 命令")

        with open(path, "r", encoding="utf-8") as file:
            payload = json.load(file)
        return payload["train"], payload["val"]

    @classmethod
    def create(
        cls,
        root: str,
        val_ratio: float = 0.15,
        seed: int = 42,
        save_dir: str = "data/splits",
    ) -> Tuple[List[str], List[str]]:
        """执行「按 ROI 划分 + 落盘」的完整流程（对应 ``split`` 命令）。

        Args:
            root:      数据根目录。
            val_ratio: 验证集 ROI 占比。
            seed:      随机种子。
            save_dir:  划分结果保存目录。

        Returns:
            Tuple[List[str], List[str]]: (训练集文件名, 验证集文件名)。
        """
        train_names, val_names = cls.split(
            root=root, val_ratio=val_ratio, seed=seed
        )
        cls.save(train_names, val_names, save_dir)
        return train_names, val_names


class DatasetAnalyzer:
    """数据集统计：图像数量、尺寸、通道与像素分布。"""

    @staticmethod
    def analyze_marker_dir(
        marker_dir: Path, sample_limit: int = 200
    ) -> Dict[str, Any]:
        """统计单个标记目录的图像属性。

        对规模较大的目录抽样统计像素分布，避免全量读图耗时过长。

        Args:
            marker_dir:   标记图像目录。
            sample_limit: 像素分布统计的最大抽样数。

        Returns:
            Dict[str, Any]: 含数量、尺寸、通道数、像素均值/标准差等统计项。
        """
        files = (
            [p for p in marker_dir.iterdir() if p.suffix.lower() in IMAGE_EXTENSIONS]
            if marker_dir.is_dir()
            else []
        )

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
                pixel_sq_sum += float((normalized**2).sum())
                pixel_count += normalized.size

        mean = pixel_sum / max(pixel_count, 1)
        std = (pixel_sq_sum / max(pixel_count, 1) - mean**2) ** 0.5

        return {
            "count": len(files),
            "sizes": sorted(sizes),
            "channels": sorted(channels),
            "pixel_mean": round(mean, 4),
            "pixel_std": round(std, 4),
        }

    @classmethod
    def analyze(cls, root: str, output: str = "data/statistics.json") -> Dict[str, Any]:
        """扫描全部标记目录并汇总为统计 JSON（对应 ``analyze`` 命令）。

        同时兼容 ``data/train/<MARKER>`` 与 ``data/<organ>/train/<MARKER>``。

        Args:
            root:   数据根目录。
            output: 统计结果输出路径。

        Returns:
            Dict[str, Any]: ``{split目录: {标记: 统计项}}`` 的统计结果。
        """
        root_path = Path(root)
        statistics: Dict[str, Any] = {}

        split_dirs = [
            p for p in root_path.rglob("*") if p.is_dir() and p.name in ("train", "test")
        ]
        for split_dir in sorted(split_dirs):
            group: Dict[str, Any] = {}
            for marker in [SOURCE_MARKER] + MARKERS:
                marker_dir = split_dir / marker
                if marker_dir.is_dir():
                    group[marker] = cls.analyze_marker_dir(marker_dir)
            if group:
                statistics[str(split_dir)] = group
                print(
                    f"[统计] {split_dir}: "
                    + ", ".join(f"{k}={v['count']}张" for k, v in group.items())
                )

        output_path = Path(output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as file:
            json.dump(statistics, file, ensure_ascii=False, indent=2)
        print(f"统计结果已保存: {output_path}")
        return statistics
