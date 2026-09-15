"""DAPI 与目标标记图像的配准误差估计与校正。

动机：mIHC 多轮染色/扫描常存在 1~3 像素的通道间平移。像素级评测
（SSIM/PSNR）对平移极其敏感——即使模型输出与真值内容完全一致，
1 像素的系统性错位也会给指标设一个硬上限，且模型学不到比错位
更准的输出。

方案：

- ``AlignmentAnalyzer``：用相位相关（phase correlation）逐样本估计
  DAPI 与各目标标记之间的平移向量，汇总统计并落盘为 JSON；
- ``AlignmentMap``：加载估计结果，训练取图时把目标图按整数像素
  平移回与 DAPI 对齐（``np.roll`` 循环平移，1~3 像素误差的边缘
  卷绕影响可忽略），从数据侧消除系统性错位。

本模块只依赖 numpy / opencv，不涉及模型训练。
"""

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np

from .constants import MARKERS, SOURCE_MARKER
from .dataset import DatasetSource


class AlignmentAnalyzer:
    """配准误差估计器：相位相关逐样本估计 DAPI -> 各标记的平移量。"""

    @staticmethod
    def _to_gray_float(image: np.ndarray) -> np.ndarray:
        """把 HWC 图像转为单通道 float32（相位相关输入要求）。

        Args:
            image: HWC 布局的 uint8 图像（1 或 3 通道）。

        Returns:
            np.ndarray: ``(H, W)`` 的 float32 灰度图，取值 ``[0, 1]``。
        """
        if image.ndim == 3 and image.shape[-1] == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        else:
            gray = image[..., 0]
        return gray.astype(np.float32) / 255.0

    @classmethod
    def estimate_pair(
        cls, source: np.ndarray, target: np.ndarray
    ) -> Tuple[float, float, float]:
        """估计一对图像的平移量（target 相对 source 的 dy, dx）。

        相位相关对亮度差异与噪声鲁棒，适合跨染色通道；加 Hann 窗
        抑制边缘不连续引起的频谱泄漏。

        Args:
            source: DAPI 灰度图 ``(H, W)``，float32。
            target: 目标标记灰度图 ``(H, W)``，float32。

        Returns:
            Tuple[float, float, float]: (dy, dx, 相关响应强度)。
            响应强度 ∈ [0, 1]，越低说明两图内容差异越大、估计越不可信。
        """
        window = cv2.createHanningWindow(source.shape[::-1], cv2.CV_32F)
        (dx, dy), response = cv2.phaseCorrelate(source, target, window)
        return dy, dx, float(response)

    @classmethod
    def analyze(
        cls,
        root: str,
        split: str = "train",
        markers: Optional[List[str]] = None,
        sample_limit: int = 500,
        max_shift: float = 8.0,
        output: str = "data/alignment.json",
    ) -> Dict[str, Any]:
        """抽样估计全部配对样本的平移量并落盘（对应 ``align`` 命令）。

        仅保留「相关响应足够高且平移量不超过 ``max_shift``」的估计，
        其余视为内容差异过大（标记表达与细胞核分布本就不同源），
        不做校正——相位相关在这类样本上的输出没有意义。

        Args:
            root:         数据根目录。
            split:        划分名称，通常为 ``"train"``。
            markers:      参与估计的标记列表，默认全部四类。
            sample_limit: 每个标记最多抽样的样本数。
            max_shift:    可信平移量上限（像素），超过则判定为无效估计。
            output:       结果 JSON 输出路径。

        Returns:
            Dict[str, Any]: 含各标记统计量与逐样本平移向量的结果。
        """
        markers = markers or list(MARKERS)
        result: Dict[str, Any] = {"shifts": {}, "stats": {}}

        for _split_dir, marker_dirs in DatasetSource.discover_marker_dirs(root, split):
            source_paths = DatasetSource.list_images(marker_dirs[SOURCE_MARKER])[
                :sample_limit
            ]
            for marker in markers:
                if marker not in marker_dirs:
                    continue
                shifts: Dict[str, List[float]] = {}
                dys, dxs, responses = [], [], []
                for source_path in source_paths:
                    target_path = marker_dirs[marker] / source_path.name
                    if not target_path.is_file():
                        continue
                    source = cls._to_gray_float(
                        DatasetSource.read_uint8(source_path)
                    )
                    target = cls._to_gray_float(
                        DatasetSource.read_uint8(target_path, grayscale=True)
                    )
                    dy, dx, response = cls.estimate_pair(source, target)
                    responses.append(response)
                    # 仅收录可信估计：响应高且平移量在合理范围内。
                    if response >= 0.05 and abs(dy) <= max_shift and abs(dx) <= max_shift:
                        shifts[source_path.stem] = [round(dy, 3), round(dx, 3)]
                        dys.append(dy)
                        dxs.append(dx)

                marker_stats = result["stats"].setdefault(marker, {})
                for key, values in (("dy", dys), ("dx", dxs)):
                    if values:
                        marker_stats[f"{key}_mean"] = round(float(np.mean(values)), 3)
                        marker_stats[f"{key}_std"] = round(float(np.std(values)), 3)
                        marker_stats[f"{key}_abs_p95"] = round(
                            float(np.percentile(np.abs(values), 95)), 3
                        )
                marker_stats["estimated"] = len(shifts)
                marker_stats["skipped"] = len(source_paths) - len(shifts)
                marker_stats["response_mean"] = (
                    round(float(np.mean(responses)), 4) if responses else 0.0
                )
                # 逐样本平移量按标记归档：{样本名: {标记: [dy, dx]}}。
                for name, shift in shifts.items():
                    result["shifts"].setdefault(name, {})[marker] = shift

        output_path = Path(output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as file:
            json.dump(result, file, ensure_ascii=False, indent=2)
        return result


class AlignmentMap:
    """配准校正表：加载估计结果并对目标图施加整数像素平移。"""

    def __init__(self, shifts: Dict[str, Dict[str, List[float]]]) -> None:
        """构建校正表。

        Args:
            shifts: ``{样本名: {标记: [dy, dx]}}`` 的逐样本平移量。
        """
        self.shifts = shifts

    @classmethod
    def load(cls, path: str) -> "AlignmentMap":
        """从 JSON 文件加载校正表。

        Args:
            path: ``align`` 命令生成的 JSON 路径。

        Returns:
            AlignmentMap: 校正表实例。

        Raises:
            FileNotFoundError: 文件不存在时抛出（提示先生成）。
        """
        file_path = Path(path)
        if not file_path.is_file():
            raise FileNotFoundError(
                f"配准校正文件不存在: {path}，请先执行 "
                "`uv run main.py align --root <数据根目录>`"
            )
        with open(file_path, "r", encoding="utf-8") as file:
            payload = json.load(file)
        return cls(payload.get("shifts", {}))

    def get(self, name: str, marker: str) -> Optional[Tuple[int, int]]:
        """查询某样本某标记的整数平移量。

        Args:
            name:   样本文件名（不含后缀）。
            marker: 目标标记名。

        Returns:
            Optional[Tuple[int, int]]: (dy, dx) 整数平移量；无估计时返回
            ``None``（表示不校正该样本，保持原图）。
        """
        shift = self.shifts.get(name, {}).get(marker)
        if shift is None:
            return None
        return int(round(shift[0])), int(round(shift[1]))

    @staticmethod
    def apply(image: np.ndarray, dy: int, dx: int) -> np.ndarray:
        """对图像施加循环平移（np.roll），把目标图移回与 DAPI 对齐。

        估计的是「target 相对 source 的位移」，因此校正方向为取负。
        1~3 像素的边缘卷绕对 256×256 训练的影响可忽略。

        Args:
            image: HWC 布局图像数组。
            dy:    垂直平移量（像素，向下为正）。
            dx:    水平平移量（像素，向右为正）。

        Returns:
            np.ndarray: 平移后的图像。
        """
        if dy == 0 and dx == 0:
            return image
        return np.roll(image, shift=(-dy, -dx), axis=(0, 1))
