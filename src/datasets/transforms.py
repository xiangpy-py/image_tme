"""数据增强模块。

虚拟染色是像素级配对任务，几何变换（翻转/旋转/平移/缩放）
必须对输入与真值「同步」施加；光度扰动（亮度/对比度/噪声）
只作用于输入 DAPI 图像，模拟染色强度差异、增强泛化能力。

不依赖 albumentations 等第三方库，基于 NumPy 实现，
保证在任意比赛评测环境中可直接运行。
"""

import numbers
import random
from typing import Any, Dict, List, Optional, Tuple

import numpy as np


class PairedTransform:
    """对 (input, target) 图像对执行同步随机增强。

    输入约定为 HWC 布局的 float32 数组，取值范围 ``[0, 1]``。
    测试/验证阶段传入 ``train=False``，则仅执行确定性的尺寸对齐。
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

    # ------------------------------------------------------------------ #
    # 几何变换：对输入与真值同步施加
    # ------------------------------------------------------------------ #
    def _apply_geometric(
        self,
        image: np.ndarray,
        target: Optional[np.ndarray],
    ) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """同步执行翻转与 90 度倍数旋转。

        Args:
            image:  输入 DAPI 图像，HWC 布局。
            target: 配对真值图像，无真值时为 ``None``。

        Returns:
            Tuple[np.ndarray, Optional[np.ndarray]]: 变换后的 (输入, 真值)。
        """
        if random.random() < self.hflip_prob:
            image = np.ascontiguousarray(image[:, ::-1])
            if target is not None:
                target = np.ascontiguousarray(target[:, ::-1])

        if random.random() < self.vflip_prob:
            image = np.ascontiguousarray(image[::-1, :])
            if target is not None:
                target = np.ascontiguousarray(target[::-1, :])

        if self.rotate90:
            k = random.randint(0, 3)  # 0/90/180/270 度
            if k > 0:
                image = np.ascontiguousarray(np.rot90(image, k))
                if target is not None:
                    target = np.ascontiguousarray(np.rot90(target, k))

        return image, target

    # ------------------------------------------------------------------ #
    # 光度扰动：仅作用于输入，模拟不同染色/成像条件
    # ------------------------------------------------------------------ #
    def _apply_photometric(self, image: np.ndarray) -> np.ndarray:
        """对输入图像施加亮度、对比度与高斯噪声扰动。

        Args:
            image: 输入 DAPI 图像，取值 ``[0, 1]``。

        Returns:
            np.ndarray: 扰动后的图像，仍裁剪在 ``[0, 1]`` 内。
        """
        if self.brightness > 0:
            image = image + random.uniform(-self.brightness, self.brightness)

        if self.contrast > 0:
            factor = 1.0 + random.uniform(-self.contrast, self.contrast)
            mean = float(image.mean())
            image = (image - mean) * factor + mean

        if self.noise_std > 0:
            noise = np.random.normal(0.0, self.noise_std, image.shape)
            image = image + noise.astype(np.float32)

        return np.clip(image, 0.0, 1.0).astype(np.float32)

    def __call__(
        self,
        image: np.ndarray,
        target: Optional[np.ndarray] = None,
    ) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """执行增强流水线。

        Args:
            image:  输入图像，HWC float32，取值 ``[0, 1]``。
            target: 配对真值，可为 ``None``。

        Returns:
            Tuple[np.ndarray, Optional[np.ndarray]]: (增强后输入, 增强后真值)。
        """
        if self.train:
            image, target = self._apply_geometric(image, target)
            image = self._apply_photometric(image)
        return image, target


def build_transforms(config: Dict[str, Any], train: bool) -> PairedTransform:
    """根据配置构建增强流水线。

    Args:
        config: 全局配置字典，读取 ``augmentation`` 一节。
        train:  是否为训练阶段。

    Returns:
        PairedTransform: 可调用对象，接受 (image, target) 并返回增强结果。
    """
    aug_cfg = config.get("augmentation", {}) or {}

    if not train:
        # 验证/测试阶段一律使用确定性变换，保证指标可复现。
        return PairedTransform(train=False)

    return PairedTransform(
        train=True,
        hflip_prob=float(aug_cfg.get("hflip_prob", 0.5)),
        vflip_prob=float(aug_cfg.get("vflip_prob", 0.5)),
        rotate90=bool(aug_cfg.get("rotate90", True)),
        brightness=float(aug_cfg.get("brightness", 0.1)),
        contrast=float(aug_cfg.get("contrast", 0.1)),
        noise_std=float(aug_cfg.get("noise_std", 0.01)),
    )
