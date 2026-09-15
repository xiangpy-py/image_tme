"""训练侧数据：配对增强、数据集与 DataLoader 构建。

只放「服务于模型训练」的数据组件；通用的图像发现/读取、ROI 划分与
数据统计等与训练无关的能力位于 ``src.data``。

- ``PairedTransform``：几何变换对输入与真值同步施加，光度扰动仅作用于输入；
- ``VirtualStainingDataset``：单标记配对数据集（DAPI -> 指定 IHC 标记）；
- ``MultiMarkerDataset``：一对多数据集，每次随机采样一个目标标记；
- ``DataLoaders``：训练/验证/测试 DataLoader 的调优构建。
"""

import os
import random
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from ..data.alignment import AlignmentMap
from ..data.constants import MARKERS, SOURCE_MARKER
from ..data.dataset import DatasetSource, DatasetSplitter
from .models import ModelRegistry


def _derive_context_path(
    context_root: Optional[Path],
    root_path: Path,
    source_path: Path,
) -> Optional[Path]:
    """由源图路径派生上下文图路径（输出与源目录同构、后缀为 .png）。

    上下文图缺失时快速失败并提示生成命令，避免静默退化为
    无上下文训练造成训练/推理口径不一致。

    Args:
        context_root: 上下文图根目录，``None`` 表示未启用。
        root_path:    数据根目录。
        source_path:  DAPI 源图路径。

    Returns:
        Optional[Path]: 上下文图路径；未启用上下文时为 ``None``。

    Raises:
        FileNotFoundError: 上下文图缺失时抛出。
    """
    if context_root is None:
        return None
    path = (context_root / source_path.relative_to(root_path)).with_suffix(".png")
    if not path.is_file():
        raise FileNotFoundError(
            f"上下文图缺失: {path}，请先执行 "
            "`uv run main.py context --root <数据根目录>`"
        )
    return path


# ------------------------------------------------------------------ #
# 数据增强
# ------------------------------------------------------------------ #
class PairedTransform:
    """对 (input, target[, context]) 图像组执行同步随机增强。

    输入约定为 HWC 布局的 float32 数组，取值范围 ``[0, 1]``。
    验证/测试阶段使用 ``train=False``，不做任何随机变换。

    context 为多尺度上下文图（同属模型输入）：几何变换与光度扰动的
    随机因子与 input 完全同步，保证三者在空间与强度上始终对齐。
    """

    def __init__(
        self,
        train: bool = True,
        hflip_prob: float = 0.5,
        vflip_prob: float = 0.5,
        rotate90: bool = True,
        brightness: float = 0.1,
        contrast: float = 0.1,
        noise_std: float = 0.01,
    ) -> None:
        """初始化增强参数。

        Args:
            train:       是否启用随机增强（验证/测试时为 ``False``）。
            hflip_prob:  水平翻转概率。
            vflip_prob:  垂直翻转概率。
            rotate90:    是否启用随机 90 度倍数旋转（病理图像无方向先验）。
            brightness:  亮度扰动幅度，0 表示关闭。
            contrast:    对比度扰动幅度，0 表示关闭。
            noise_std:   高斯噪声标准差，0 表示关闭。
        """
        self.train = train
        self.hflip_prob = hflip_prob
        self.vflip_prob = vflip_prob
        self.rotate90 = rotate90
        self.brightness = brightness
        self.contrast = contrast
        self.noise_std = noise_std

    @classmethod
    def from_config(cls, config: Dict[str, Any], train: bool) -> "PairedTransform":
        """根据配置构建增强流水线。

        Args:
            config: 全局配置字典，读取 ``augmentation`` 一节。
            train:  是否为训练阶段。

        Returns:
            PairedTransform: 可调用对象，接受 (image, target) 并返回增强结果。
        """
        if not train:
            # 验证/测试阶段一律使用确定性变换，保证指标可复现。
            return cls(train=False)

        aug_cfg = config.get("augmentation", {}) or {}
        return cls(
            train=True,
            hflip_prob=float(aug_cfg.get("hflip_prob", 0.5)),
            vflip_prob=float(aug_cfg.get("vflip_prob", 0.5)),
            rotate90=bool(aug_cfg.get("rotate90", True)),
            brightness=float(aug_cfg.get("brightness", 0.1)),
            contrast=float(aug_cfg.get("contrast", 0.1)),
            noise_std=float(aug_cfg.get("noise_std", 0.01)),
        )

    def _apply_geometric(
        self,
        image: np.ndarray,
        target: Optional[np.ndarray],
        context: Optional[np.ndarray] = None,
    ) -> Tuple[np.ndarray, Optional[np.ndarray], Optional[np.ndarray]]:
        """同步执行翻转与 90 度倍数旋转（对全部在场图像施加同一变换）。

        Args:
            image:   输入 DAPI 图像，HWC 布局。
            target:  配对真值图像，无真值时为 ``None``。
            context: 多尺度上下文图，未启用时为 ``None``。

        Returns:
            Tuple[np.ndarray, Optional[np.ndarray], Optional[np.ndarray]]:
                变换后的 (输入, 真值, 上下文)。
        """
        images = [item for item in (image, target, context) if item is not None]

        if random.random() < self.hflip_prob:
            images = [np.ascontiguousarray(item[:, ::-1]) for item in images]
        if random.random() < self.vflip_prob:
            images = [np.ascontiguousarray(item[::-1, :]) for item in images]
        if self.rotate90:
            k = random.randint(0, 3)  # 0/90/180/270 度
            if k > 0:
                images = [np.ascontiguousarray(np.rot90(item, k)) for item in images]

        # 按原位回填，保持 (image, target, context) 的返回结构。
        result: List[Optional[np.ndarray]] = []
        iterator = iter(images)
        for item in (image, target, context):
            result.append(next(iterator) if item is not None else None)
        return result[0], result[1], result[2]  # type: ignore[return-value]

    def _apply_photometric(
        self,
        image: np.ndarray,
        brightness_delta: float,
        contrast_factor: float,
    ) -> np.ndarray:
        """对单张输入图像施加亮度、对比度与高斯噪声扰动。

        亮度/对比度因子由调用方统一样本（输入与上下文共用同一组因子，
        保持两者强度语义一致）；噪声逐图独立采样。

        Args:
            image:            输入图像，取值 ``[0, 1]``。
            brightness_delta: 亮度偏移量（已随机采样）。
            contrast_factor:  对比度缩放因子（已随机采样）。

        Returns:
            np.ndarray: 扰动后的图像，仍裁剪在 ``[0, 1]`` 内。
        """
        if brightness_delta != 0.0:
            image = image + brightness_delta

        if contrast_factor != 1.0:
            mean = float(image.mean())
            image = (image - mean) * contrast_factor + mean

        if self.noise_std > 0:
            noise = np.random.normal(0.0, self.noise_std, image.shape)
            image = image + noise.astype(np.float32)

        return np.clip(image, 0.0, 1.0).astype(np.float32)

    def __call__(
        self,
        image: np.ndarray,
        target: Optional[np.ndarray] = None,
        context: Optional[np.ndarray] = None,
    ) -> Tuple[np.ndarray, Optional[np.ndarray], Optional[np.ndarray]]:
        """执行增强流水线。

        Args:
            image:   输入图像，HWC float32，取值 ``[0, 1]``。
            target:  配对真值，可为 ``None``。
            context: 多尺度上下文图，可为 ``None``。

        Returns:
            Tuple[np.ndarray, Optional[np.ndarray], Optional[np.ndarray]]:
                (增强后输入, 增强后真值, 增强后上下文)。
        """
        if not self.train:
            return image, target, context

        image, target, context = self._apply_geometric(image, target, context)
        # 光度因子统一采样一次：输入与上下文共用，保证强度语义对齐。
        brightness_delta = (
            random.uniform(-self.brightness, self.brightness)
            if self.brightness > 0
            else 0.0
        )
        contrast_factor = (
            1.0 + random.uniform(-self.contrast, self.contrast)
            if self.contrast > 0
            else 1.0
        )
        image = self._apply_photometric(image, brightness_delta, contrast_factor)
        if context is not None:
            context = self._apply_photometric(context, brightness_delta, contrast_factor)
        return image, target, context


# ------------------------------------------------------------------ #
# 数据集
# ------------------------------------------------------------------ #
class VirtualStainingDataset(Dataset):
    """单标记配对数据集：DAPI -> 指定 IHC 标记。

    每个样本返回::

        {
            "input":  (C_in, H, W)  张量,   # DAPI（启用上下文时拼接 context）
            "target": (C_out, H, W) 张量,   # 目标标记（测试模式无此键）
            "name":   文件名（不含后缀）,   # 用于结果命名对应
        }

    启用 ``context_root`` 时，输入为 ``[DAPI(3ch), context(3ch)]`` 的
    通道维拼接（共 6 通道），模型侧只需把 ``in_channels`` 改为 6。
    """

    def __init__(
        self,
        root: str,
        marker: str,
        split: str = "train",
        transform: Optional[Callable] = None,
        file_list: Optional[List[str]] = None,
        cache: bool = False,
        context_root: Optional[str] = None,
        alignment: Optional[AlignmentMap] = None,
    ) -> None:
        """收集 DAPI 与目标标记的同名配对样本。

        Args:
            root:         数据根目录（支持多器官/单器官两种组织方式）。
            marker:       目标标记名，必须是 ``MARKERS`` 之一。
            split:        ``"train"`` 或 ``"test"``。
            transform:    增强流水线，签名为
                ``(image, target, context) -> (image, target, context)``。
            file_list:    可选的文件名白名单（来自 ROI 划分文件）。
            cache:        是否在内存中缓存全部解码图像（大内存机器建议开启）。
            context_root: 多尺度上下文图根目录（由 ContextGenerator 生成），
                ``None`` 表示不使用上下文输入。
            alignment:    配准校正表（由 ``align`` 命令生成），``None`` 表示
                不做目标图平移校正。

        Raises:
            ValueError:       标记名非法或未找到任何配对样本时抛出。
            FileNotFoundError: 启用上下文但存在缺失的上下文图时抛出。
        """
        super().__init__()
        if marker not in MARKERS:
            raise ValueError(f"未知标记 '{marker}'，可选: {MARKERS}")

        self.marker = marker
        self.transform = transform
        self.alignment = alignment
        self._root_path = Path(root)
        self.context_root = Path(context_root) if context_root else None
        # (源图路径, 真值路径或 None[测试模式], 上下文路径或 None[未启用])，
        # 跨器官聚合配对样本。
        self.samples: List[Tuple[Path, Optional[Path], Optional[Path]]] = []

        for _split_dir, marker_dirs in DatasetSource.discover_marker_dirs(root, split):
            # 测试集无真值目录，此时真值置 None 仅保留输入。
            target_dir = marker_dirs.get(marker) if split == "train" else None
            for source_path in DatasetSource.list_images(marker_dirs[SOURCE_MARKER]):
                if file_list is not None and source_path.stem not in file_list:
                    continue

                target_path: Optional[Path] = None
                if target_dir is not None:
                    candidate = target_dir / source_path.name
                    if not candidate.is_file():
                        continue  # 缺失配对真值的样本直接跳过
                    target_path = candidate
                self.samples.append(
                    (source_path, target_path, self._context_path(source_path))
                )

        if not self.samples:
            raise ValueError(
                f"未找到配对样本: root={root}, marker={marker}, split={split}"
            )

        # 可选的内存缓存：一次性解码全部图像为 uint8，取用时再归一化，
        # 避免每个 epoch 重复 JPG 解码的开销。
        self._cache: Optional[Dict[Path, np.ndarray]] = None
        if cache:
            self._cache = {}
            for source_path, target_path, context_path in self.samples:
                self._cache[source_path] = DatasetSource.read_uint8(source_path)
                if target_path is not None:
                    self._cache[target_path] = DatasetSource.read_uint8(
                        target_path, grayscale=True
                    )
                if context_path is not None:
                    self._cache[context_path] = DatasetSource.read_uint8(context_path)

    def _context_path(self, source_path: Path) -> Optional[Path]:
        """由源图路径派生上下文图路径（输出与源目录同构、后缀为 .png）。

        Args:
            source_path: DAPI 源图路径。

        Returns:
            Optional[Path]: 上下文图路径；未启用上下文时为 ``None``。

        Raises:
            FileNotFoundError: 上下文图缺失时抛出（提示先生成），
                避免静默退化为无上下文训练造成口径不一致。
        """
        if self.context_root is None:
            return None
        path = (
            self.context_root / source_path.relative_to(self._root_path)
        ).with_suffix(".png")
        if not path.is_file():
            raise FileNotFoundError(
                f"上下文图缺失: {path}，请先执行 "
                "`uv run main.py context --root <数据根目录>`"
            )
        return path

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
        source_path, target_path, context_path = self.samples[index]

        if self._cache is not None:
            # 命中缓存时直接复用 uint8 数据，避免重复 JPG 解码。
            image = DatasetSource.uint8_to_float(self._cache[source_path])
            target = (
                DatasetSource.uint8_to_float(self._cache[target_path])
                if target_path is not None
                else None
            )
            context = (
                DatasetSource.uint8_to_float(self._cache[context_path])
                if context_path is not None
                else None
            )
        else:
            image = DatasetSource.read_float(source_path)
            target = (
                DatasetSource.read_float(target_path, grayscale=True)
                if target_path is not None
                else None
            )
            context = (
                DatasetSource.read_float(context_path)
                if context_path is not None
                else None
            )

        # 配准校正必须在增强之前：把目标图平移回与 DAPI 对齐，
        # 之后的几何增强对两者同步施加，不会破坏对齐关系。
        if self.alignment is not None and target is not None:
            shift = self.alignment.get(source_path.stem, self.marker)
            if shift is not None:
                target = AlignmentMap.apply(target, shift[0], shift[1])

        if self.transform is not None:
            image, target, context = self.transform(image, target, context)

        # 启用上下文时沿通道维拼接：DAPI(3ch) + context(3ch)。
        if context is not None:
            image = np.concatenate([image, context], axis=-1)

        sample: Dict[str, Any] = {
            "input": self._to_tensor(image),
            "name": source_path.stem,
        }
        if target is not None:
            sample["target"] = self._to_tensor(target)
        return sample

    @staticmethod
    def _to_tensor(image: np.ndarray) -> torch.Tensor:
        """将 HWC float 数组转换为 CHW 的 torch 张量。

        Args:
            image: HWC 布局的 float 数组。

        Returns:
            torch.Tensor: CHW 张量。
        """
        return torch.from_numpy(np.ascontiguousarray(image.transpose(2, 0, 1)))


class MultiMarkerDataset(Dataset):
    """多标记条件数据集：一个 DAPI 样本随机配对一种目标标记。

    用于一对多联合建模的训练策略：每次取样本时随机采样一个目标标记
    并返回其编号，供模型的 marker token 使用。

    每个样本返回::

        {
            "input":       (C, H, W) 张量,
            "target":      (C, H, W) 张量,
            "marker_idx":  目标标记编号（对应 src.data.constants.MARKERS 顺序）,
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
        cache: bool = False,
        random_marker: bool = True,
        marker_weights: Optional[Dict[str, float]] = None,
        context_root: Optional[str] = None,
        alignment: Optional[AlignmentMap] = None,
    ) -> None:
        """收集同时存在全部目标标记真值的样本。

        只有四类标记真值齐全的 patch 才纳入，保证任意随机采样
        标记时都有监督信号。

        Args:
            root:      数据根目录。
            split:     划分名称（该数据集仅用于训练/验证）。
            markers:   参与联合建模的标记列表，默认全部四类。
            transform: 增强流水线。
            file_list: ROI 划分文件名白名单。
            cache:     是否在内存中缓存全部解码图像（大内存机器建议开启）。
            random_marker: 是否随机采样目标标记；训练置 ``True``，
                验证置 ``False`` 以按样本下标确定性轮转标记，稳定指标。
            marker_weights: ``{标记名: 采样权重}``，用于标记均衡采样；
                未列出的标记取 1.0，``None`` 表示等概率采样。四标记取平均
                计分的赛制下，对弱势标记加权可提升其梯度占比。
            context_root: 多尺度上下文图根目录，``None`` 表示不使用。
            alignment:    配准校正表（由 ``align`` 命令生成），``None`` 表示
                不做目标图平移校正。

        Raises:
            ValueError:       未找到任何全配对样本时抛出。
            FileNotFoundError: 启用上下文但存在缺失的上下文图时抛出。
        """
        super().__init__()
        self.markers = markers or list(MARKERS)
        self.transform = transform
        self.alignment = alignment
        self.random_marker = random_marker
        self.marker_probs = self._resolve_marker_probs(marker_weights)
        root_path = Path(root)
        context_root_path = Path(context_root) if context_root else None
        # 文件名 -> (DAPI路径, {标记: 路径}, 上下文路径或 None)
        self.index: Dict[str, Tuple[Path, Dict[str, Path], Optional[Path]]] = {}

        for _split_dir, marker_dirs in DatasetSource.discover_marker_dirs(root, split):
            if SOURCE_MARKER not in marker_dirs:
                continue
            if not all(marker in marker_dirs for marker in self.markers):
                continue  # 该组标记不全，无法用于联合建模

            for source_path in DatasetSource.list_images(marker_dirs[SOURCE_MARKER]):
                if file_list is not None and source_path.stem not in file_list:
                    continue

                target_map: Dict[str, Path] = {}
                for marker in self.markers:
                    candidate = marker_dirs[marker] / source_path.name
                    if not candidate.is_file():
                        break
                    target_map[marker] = candidate
                else:
                    context_path = _derive_context_path(
                        context_root_path, root_path, source_path
                    )
                    self.index[source_path.stem] = (
                        source_path,
                        target_map,
                        context_path,
                    )

        if not self.index:
            raise ValueError(f"未找到全标记配对样本: root={root}, split={split}")
        self.names = sorted(self.index.keys())

        # 可选的内存缓存：一次性解码全部源图、各标记真值与上下文图，
        # 取用时再归一化。
        self._cache: Optional[
            Dict[str, Tuple[np.ndarray, Dict[str, np.ndarray], Optional[np.ndarray]]]
        ] = None
        if cache:
            self._cache = {
                name: (
                    DatasetSource.read_uint8(source_path),
                    {
                        marker: DatasetSource.read_uint8(path, grayscale=True)
                        for marker, path in target_map.items()
                    },
                    (
                        DatasetSource.read_uint8(context_path)
                        if context_path is not None
                        else None
                    ),
                )
                for name, (source_path, target_map, context_path) in self.index.items()
            }

    def _resolve_marker_probs(
        self, marker_weights: Optional[Dict[str, float]]
    ) -> Optional[np.ndarray]:
        """把配置中的标记采样权重解析为与 ``self.markers`` 对齐的概率分布。

        未配置、权重全为 0 或全为负时返回 ``None``，表示按标记等概率采样，
        保持与历史实验一致的行为。

        Args:
            marker_weights: ``{标记名: 权重}``，未列出的标记取权重 1.0。

        Returns:
            Optional[np.ndarray]: 归一化概率数组；等概率采样时为 ``None``。
        """
        if not marker_weights:
            return None
        weights = np.array(
            [max(float(marker_weights.get(name, 1.0)), 0.0) for name in self.markers],
            dtype=np.float64,
        )
        total = float(weights.sum())
        if total <= 0:
            return None
        return weights / total

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
        source_path, target_map, context_path = self.index[name]

        # 训练阶段随机采样目标标记实现一对多监督；验证阶段按样本下标
        # 确定性轮转标记，避免每轮验证的 marker 随机导致指标抖动、
        # 早停选点不稳定。
        if self.random_marker:
            if self.marker_probs is None:
                marker_idx = int(np.random.randint(0, len(self.markers)))
            else:
                # 标记均衡采样：按配置权重抽取，提升弱势标记的梯度占比。
                marker_idx = int(
                    np.random.choice(len(self.markers), p=self.marker_probs)
                )
        else:
            marker_idx = index % len(self.markers)
        marker = self.markers[marker_idx]

        if self._cache is not None:
            source_uint8, cached_targets, cached_context = self._cache[name]
            image = DatasetSource.uint8_to_float(source_uint8)
            target = DatasetSource.uint8_to_float(cached_targets[marker])
            context = (
                DatasetSource.uint8_to_float(cached_context)
                if cached_context is not None
                else None
            )
        else:
            image = DatasetSource.read_float(source_path)
            target = DatasetSource.read_float(target_map[marker], grayscale=True)
            context = (
                DatasetSource.read_float(context_path)
                if context_path is not None
                else None
            )

        # 配准校正必须在增强之前（与单标记数据集同一口径）。
        if self.alignment is not None:
            shift = self.alignment.get(name, marker)
            if shift is not None:
                target = AlignmentMap.apply(target, shift[0], shift[1])

        if self.transform is not None:
            image, target, context = self.transform(image, target, context)

        # 启用上下文时沿通道维拼接：DAPI(3ch) + context(3ch)。
        if context is not None:
            image = np.concatenate([image, context], axis=-1)

        return {
            "input": VirtualStainingDataset._to_tensor(image),
            "target": VirtualStainingDataset._to_tensor(target),
            "marker_idx": marker_idx,
            "name": name,
        }


# ------------------------------------------------------------------ #
# DataLoader
# ------------------------------------------------------------------ #
class DataLoaders:
    """训练/验证/测试 DataLoader 的构建与并行调优。

    关键参数：自动探测 ``num_workers``、增大 ``prefetch_factor``、
    常开 ``persistent_workers``，并让每个 worker 拥有独立随机种子。
    """

    @staticmethod
    def optimal_workers(data_cfg: Dict[str, Any]) -> int:
        """计算最优 num_workers：物理核心数 - 2，留余量给主进程。

        Args:
            data_cfg: 配置中的 ``data`` 一节。

        Returns:
            int: worker 进程数，取值 ``[1, 16]``。
        """
        configured = data_cfg.get("num_workers")
        if configured is not None and int(configured) > 0:
            return int(configured)
        # 留 2 个核心给主进程和其他任务，避免数据预处理与训练争抢 CPU。
        return max(1, min((os.cpu_count() or 4) - 2, 16))

    @staticmethod
    def _worker_init(worker_id: int) -> None:
        """多进程随机种子独立，避免增强同质化。

        Args:
            worker_id: DataLoader 分配的 worker 序号。

        Returns:
            None
        """
        seed = torch.initial_seed() % (2**32)
        np.random.seed(seed + worker_id)

    @classmethod
    def _datasets(
        cls, config: Dict[str, Any]
    ) -> Tuple[Dataset, Dataset]:
        """构建训练/验证数据集（条件模型用一对多数据集）。

        Args:
            config: 全局配置字典。

        Returns:
            Tuple[Dataset, Dataset]: (训练集, 验证集)。
        """
        data_cfg = config.get("data", {})
        root = data_cfg.get("root", "data")
        split_dir = data_cfg.get("split_dir", "data/splits")
        cache = bool(data_cfg.get("cache", False))
        # 多尺度上下文：非空路径即启用，训练/验证/测试共用同一配置。
        context_root = data_cfg.get("context_dir") or None
        # 标记采样权重：仅训练集生效，验证集按样本下标确定性轮转标记。
        marker_weights = data_cfg.get("marker_weights") or None
        # 配准校正：仅作用于训练集。验证集保持与赛方一致的原始真值，
        # 使验证指标仍反映线上口径（模型选择不被校正口径带偏）。
        alignment_file = data_cfg.get("alignment_file") or None
        alignment = AlignmentMap.load(alignment_file) if alignment_file else None

        # ROI 划分文件必须存在：否则训练集与验证集会退化为同一份数据，
        # 造成严重的信息泄漏（验证指标虚高）。此处快速失败并提示修复方式。
        split_file = Path(split_dir) / "split.json"
        if not split_file.is_file():
            raise FileNotFoundError(
                f"未找到 ROI 划分文件: {split_file}，请先执行 "
                "`uv run main.py split --root <数据根目录>`（按 ROI 划分以避免数据泄漏）"
            )
        train_list, val_list = DatasetSplitter.load(split_dir)

        if ModelRegistry.is_conditional(config):
            return (
                MultiMarkerDataset(
                    root=root,
                    transform=PairedTransform.from_config(config, train=True),
                    file_list=train_list,
                    cache=cache,
                    marker_weights=marker_weights,
                    context_root=context_root,
                    alignment=alignment,
                ),
                MultiMarkerDataset(
                    root=root,
                    transform=PairedTransform.from_config(config, train=False),
                    file_list=val_list,
                    cache=cache,
                    random_marker=False,
                    context_root=context_root,
                ),
            )

        marker = data_cfg.get("marker", "CD68")
        return (
            VirtualStainingDataset(
                root=root,
                marker=marker,
                transform=PairedTransform.from_config(config, train=True),
                file_list=train_list,
                cache=cache,
                context_root=context_root,
                alignment=alignment,
            ),
            VirtualStainingDataset(
                root=root,
                marker=marker,
                transform=PairedTransform.from_config(config, train=False),
                file_list=val_list,
                cache=cache,
                context_root=context_root,
            ),
        )

    @classmethod
    def build(cls, config: Dict[str, Any]) -> Dict[str, DataLoader]:
        """构建训练与验证 DataLoader。

        Args:
            config: 全局配置字典。

        Returns:
            Dict[str, DataLoader]: 键为 ``"train"`` / ``"val"``。
        """
        data_cfg = config.get("data", {})
        train_dataset, val_dataset = cls._datasets(config)

        common_kwargs: Dict[str, Any] = {
            "num_workers": cls.optimal_workers(data_cfg),
            "pin_memory": True,
            "persistent_workers": True,
            # prefetch_factor: 每个 worker 预取 batch 数，GPU 越快该值应越大。
            "prefetch_factor": max(2, int(data_cfg.get("prefetch_factor", 4))),
            "worker_init_fn": cls._worker_init,
        }
        batch_size = int(config.get("training", {}).get("batch_size", 16))
        return {
            "train": DataLoader(
                train_dataset,
                batch_size=batch_size,
                shuffle=True,
                drop_last=True,
                **common_kwargs,
            ),
            "val": DataLoader(
                val_dataset, batch_size=batch_size, shuffle=False, **common_kwargs
            ),
        }

    @classmethod
    def for_validation_marker(cls, config: Dict[str, Any], marker: str) -> DataLoader:
        """构建指定标记的验证集 DataLoader（用于逐标记短板分析）。

        无论配置是单标记还是条件模型，均按 ROI 划分文件中的验证名单构建
        该标记的配对数据，保证四种标记的指标在同一验证集上可横向对比。

        Args:
            config: 全局配置字典。
            marker: 目标标记名，必须是 ``MARKERS`` 之一。

        Returns:
            DataLoader: 该标记的验证集加载器（确定性变换、不打乱顺序）。

        Raises:
            FileNotFoundError: ROI 划分文件不存在时抛出。
        """
        data_cfg = config.get("data", {})
        split_dir = str(data_cfg.get("split_dir", "data/splits"))
        split_file = Path(split_dir) / "split.json"
        if not split_file.is_file():
            raise FileNotFoundError(
                f"未找到 ROI 划分文件: {split_file}，请先执行 "
                "`uv run main.py split --root <数据根目录>`"
            )
        _train_list, val_list = DatasetSplitter.load(split_dir)

        dataset = VirtualStainingDataset(
            root=str(data_cfg.get("root", "data")),
            marker=marker,
            split="train",
            transform=PairedTransform.from_config(config, train=False),
            file_list=val_list,
            cache=bool(data_cfg.get("cache", False)),
            context_root=data_cfg.get("context_dir") or None,
        )
        return DataLoader(
            dataset,
            batch_size=int(config.get("training", {}).get("batch_size", 16)),
            shuffle=False,
            num_workers=cls.optimal_workers(data_cfg),
            pin_memory=True,
            persistent_workers=True,
            prefetch_factor=4,
        )

    @classmethod
    def for_test(cls, config: Dict[str, Any]) -> DataLoader:
        """构建测试集 DataLoader（仅有 DAPI 输入，无真值）。

        Args:
            config: 全局配置字典。

        Returns:
            DataLoader: 保持文件名顺序的测试集加载器。
        """
        data_cfg = config.get("data", {})
        dataset = VirtualStainingDataset(
            root=data_cfg.get("root", "data"),
            marker=data_cfg.get("marker", "CD68"),
            split="test",
            transform=PairedTransform.from_config(config, train=False),
            context_root=data_cfg.get("context_dir") or None,
        )
        return DataLoader(
            dataset,
            batch_size=int(config.get("training", {}).get("batch_size", 16)),
            shuffle=False,
            num_workers=cls.optimal_workers(data_cfg),
            pin_memory=True,
            persistent_workers=True,
            prefetch_factor=4,
        )
