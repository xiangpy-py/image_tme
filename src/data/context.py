"""多尺度上下文生成。

核心思路（Global Pixel Transformers 论文的多尺度输入）：IHC 标记表达与
组织区域强相关，而 256×256 的单 patch 视野往往不足以判断所处结构。
本模块利用 patch 文件名 ``ROIxxx_rr_cc`` 中的网格坐标，把同一 ROI 的
相邻 DAPI patch 拼回大图，为每个 patch 生成一张「上下文图」::

    以 patch 中心为锚，取 patch_size × scale 的邻域窗口
    -> 重采样回 patch 尺寸 -> 与原始输入在通道维拼接

- ``scale > 1``：更大视野的组织级上下文（如 scale=2 即 512×512 邻域）；
- ``scale < 1``：中心区域的细节放大视图；
- 输出与源目录同构（仅后缀改为 .png），数据集按源图路径直接派生
  上下文路径，训练/测试管线完全一致。

本模块只依赖 numpy / opencv，不涉及模型训练。
"""

import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np

from .constants import CONTEXT_DIR_MARKER, SOURCE_MARKER
from .dataset import DatasetSource

# patch 命名约定：ROI000_00_01.jpg -> (ROI000, 行=0, 列=1)。
GRID_PATTERN = re.compile(r"^(ROI\d+)_(\d+)_(\d+)$")


class ContextGenerator:
    """上下文图生成器：ROI 网格拼接 -> 邻域窗口重采样 -> 落盘。"""

    # ------------------------------------------------------------------ #
    # 网格解析与拼接
    # ------------------------------------------------------------------ #
    @staticmethod
    def parse_grid_name(stem: str) -> Tuple[str, int, int]:
        """从 patch 文件名解析 (ROI 标识, 行号, 列号)。

        符合 ``ROIxxx_rr_cc`` 约定时按网格解析；不符合时退化为
        「单格 ROI」（整个 patch 即全部上下文来源），保证任意命名可用。

        Args:
            stem: 文件名（不含后缀）。

        Returns:
            Tuple[str, int, int]: (ROI 标识, 行号, 列号)。
        """
        match = GRID_PATTERN.match(stem)
        if match:
            return match.group(1), int(match.group(2)), int(match.group(3))
        return stem, 0, 0

    @classmethod
    def group_by_roi(
        cls, image_paths: List[Path]
    ) -> Dict[str, Dict[Tuple[int, int], Path]]:
        """把同一 ROI 的 patch 按网格坐标分组。

        Args:
            image_paths: DAPI patch 路径列表。

        Returns:
            Dict[str, Dict[Tuple[int, int], Path]]:
                ``{ROI标识: {(行, 列): 路径}}``。
        """
        groups: Dict[str, Dict[Tuple[int, int], Path]] = {}
        for path in image_paths:
            roi, row, col = cls.parse_grid_name(path.stem)
            groups.setdefault(roi, {})[(row, col)] = path
        return groups

    @staticmethod
    def build_canvas(
        cells: Dict[Tuple[int, int], np.ndarray],
        patch_shape: Tuple[int, int],
    ) -> Tuple[np.ndarray, int]:
        """把同 ROI 的 patch 按网格坐标拼成一张大图（画布）。

        网格中缺失的位置用「曼哈顿距离最近的有效格」原样填充，
        避免缺口在反射填充时传播；实际赛题数据网格通常完整，该分支
        只是兜底。

        Args:
            cells:       ``{(行, 列): HWC uint8 图像}``。
            patch_shape: 统一的 patch 尺寸 ``(高, 宽)``。

        Returns:
            Tuple[np.ndarray, int]: (拼接画布, 填充的缺口数)。
        """
        patch_h, patch_w = patch_shape
        max_row = max(row for row, _col in cells)
        max_col = max(col for _row, col in cells)
        channels = next(iter(cells.values())).shape[-1]

        canvas = np.zeros(
            ((max_row + 1) * patch_h, (max_col + 1) * patch_w, channels),
            dtype=np.uint8,
        )
        holes = 0
        occupied = sorted(cells)
        for row in range(max_row + 1):
            for col in range(max_col + 1):
                cell = cells.get((row, col))
                if cell is None:
                    # 缺口：取曼哈顿距离最近的有效格内容填充。
                    nearest = min(
                        occupied, key=lambda rc: abs(rc[0] - row) + abs(rc[1] - col)
                    )
                    cell = cells[nearest]
                    holes += 1
                if cell.shape[:2] != patch_shape:
                    # 防御：个别 patch 尺寸不一致时先缩放到统一尺寸。
                    cell = cv2.resize(cell, (patch_w, patch_h), interpolation=cv2.INTER_AREA)
                canvas[
                    row * patch_h : (row + 1) * patch_h,
                    col * patch_w : (col + 1) * patch_w,
                ] = cell
        return canvas, holes

    # ------------------------------------------------------------------ #
    # 上下文窗口提取
    # ------------------------------------------------------------------ #
    @staticmethod
    def extract_context(
        canvas: np.ndarray,
        row: int,
        col: int,
        patch_shape: Tuple[int, int],
        scale: float,
    ) -> np.ndarray:
        """从画布上提取某 patch 的多尺度上下文图。

        先对整幅画布做反射填充（反射量为窗口半径），随后无论 patch
        位于网格何处（含边角），窗口都完整落在画布内，无需分支处理。

        Args:
            canvas:      拼接画布 ``(H', W', C)``。
            row / col:   目标 patch 的网格坐标。
            patch_shape: patch 尺寸 ``(高, 宽)``。
            scale:       视野倍率；>1 为下采样上下文，<1 为中心放大。

        Returns:
            np.ndarray: 与 patch 同尺寸的上下文图（HWC uint8）。
        """
        patch_h, patch_w = patch_shape
        window_h = max(1, int(round(patch_h * scale)))
        window_w = max(1, int(round(patch_w * scale)))

        # 反射填充整个画布，保证任意位置的窗口都完整可取。
        pad = max(window_h, window_w) // 2 + 1
        padded = cv2.copyMakeBorder(
            canvas, pad, pad, pad, pad, cv2.BORDER_REFLECT
        )

        # patch 中心在填充后画布上的坐标。
        center_y = row * patch_h + patch_h // 2 + pad
        center_x = col * patch_w + patch_w // 2 + pad
        top = center_y - window_h // 2
        left = center_x - window_w // 2
        window = padded[top : top + window_h, left : left + window_w]

        # 重采样回 patch 尺寸：缩视野用 AREA（保均值），放大用 CUBIC（保细节）。
        interpolation = cv2.INTER_AREA if scale > 1 else cv2.INTER_CUBIC
        return cv2.resize(window, (patch_w, patch_h), interpolation=interpolation)

    # ------------------------------------------------------------------ #
    # 生成入口
    # ------------------------------------------------------------------ #
    @classmethod
    def _generate_group(
        cls,
        cells: Dict[Tuple[int, int], np.ndarray],
        scale: float,
    ) -> Tuple[Dict[Tuple[int, int], np.ndarray], int]:
        """为一个 ROI 生成全部 patch 的上下文图。

        Args:
            cells: ``{(行, 列): HWC uint8 图像}``。
            scale: 视野倍率。

        Returns:
            Tuple[Dict[Tuple[int, int], np.ndarray], int]:
                (``{(行, 列): 上下文图}``, 画布缺口数)。
        """
        patch_shape = next(iter(cells.values())).shape[:2]
        canvas, holes = cls.build_canvas(cells, patch_shape)
        contexts = {
            (row, col): cls.extract_context(canvas, row, col, patch_shape, scale)
            for row, col in cells
        }
        return contexts, holes

    @classmethod
    def generate(
        cls,
        root: str,
        output_root: str,
        scale: float = 2.0,
        splits: Tuple[str, ...] = ("train", "test"),
    ) -> Dict[str, Any]:
        """为数据集生成全部上下文图（对应 ``context`` 命令）。

        输出目录与源数据同构：``<输出根>/<源图相对root的路径>.png``，
        数据集侧据此按源图路径直接派生上下文路径。

        Args:
            root:        数据根目录。
            output_root: 上下文图输出根目录，约定为 ``data/context``。
            scale:       视野倍率（默认 2，即 2× 邻域）。
            splits:      需要生成的划分，测试集同样需要（推理输入）。

        Returns:
            Dict[str, Any]: 各划分的生成统计（图像数 / ROI 数 / 缺口数）。

        Raises:
            ValueError: 未找到任何 DAPI 图像时抛出。
        """
        root_path = Path(root)
        statistics: Dict[str, Any] = {}

        # 写入标记文件：目录发现逻辑据此跳过上下文目录（其结构与源数据
        # 同构，若不标记会被误认为一个新器官而重复计数）。
        output_path_root = Path(output_root)
        output_path_root.mkdir(parents=True, exist_ok=True)
        (output_path_root / CONTEXT_DIR_MARKER).touch()

        for split in splits:
            for split_dir, marker_dirs in DatasetSource.discover_marker_dirs(root, split):
                images = DatasetSource.list_images(marker_dirs[SOURCE_MARKER])
                if not images:
                    continue

                split_stats = {"images": 0, "rois": 0, "holes": 0}
                for _roi, cells_paths in cls.group_by_roi(images).items():
                    # 同 ROI 的 patch 全部读入内存再拼接（单 ROI 规模可控）。
                    cells = {
                        rc: DatasetSource.read_uint8(path)
                        for rc, path in cells_paths.items()
                    }
                    contexts, holes = cls._generate_group(cells, scale)
                    split_stats["rois"] += 1
                    split_stats["holes"] += holes

                    for rc, context in contexts.items():
                        source_path = cells_paths[rc]
                        output_path = (
                            Path(output_root)
                            / source_path.relative_to(root_path)
                        ).with_suffix(".png")
                        output_path.parent.mkdir(parents=True, exist_ok=True)
                        # read_uint8 返回 RGB，OpenCV 写盘需转回 BGR。
                        cv2.imwrite(
                            str(output_path),
                            cv2.cvtColor(context, cv2.COLOR_RGB2BGR),
                        )
                        split_stats["images"] += 1

                statistics[str(split_dir)] = split_stats
                print(
                    f"[上下文] {split_dir}: {split_stats['images']} 张 / "
                    f"{split_stats['rois']} 个 ROI / 缺口 {split_stats['holes']} 处"
                )

        if not statistics:
            raise ValueError(f"未找到任何 DAPI 图像: root={root}")
        return statistics
