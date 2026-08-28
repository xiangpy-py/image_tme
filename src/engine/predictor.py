"""推理引擎：在测试集上自动生成虚拟染色结果。

输出组织方式与比赛提交要求一致::

    results/test/<MARKER>/<原名>_fake.jpg

单标记模型逐标记加载各自 checkpoint 推理；
多标记条件模型加载一次 checkpoint，按 marker token 批量生成四种标记。
"""

from pathlib import Path
from typing import Any, Dict, List, Optional

import cv2
import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm

from ..datasets import MARKERS, build_test_loader
from ..models import build_model, is_conditional_model
from ..utils import get_device, load_checkpoint, setup_logger


class Predictor:
    """预测器：加载 checkpoint 并批量生成测试集结果。"""

    def __init__(self, config: Dict[str, Any]) -> None:
        """初始化设备、模型与数据加载器。

        Args:
            config: 全局配置字典。
        """
        self.config = config
        self.device = get_device(config.get("runtime", {}).get("device"))
        self.logger = setup_logger("predictor")
        self.conditional = is_conditional_model(config)

        self.model: nn.Module = build_model(config).to(self.device)
        self.test_loader = build_test_loader(config)

        self.output_root = Path(config.get("inference", {}).get("output_dir", "results"))
        self.suffix = config.get("inference", {}).get("suffix", "_fake")

    def _load_weights(self, checkpoint_path: str) -> None:
        """加载模型权重并切换为评估模式。

        Args:
            checkpoint_path: checkpoint 文件路径。

        Returns:
            None
        """
        info = load_checkpoint(checkpoint_path, self.model, map_location="cpu")
        self.model.to(self.device)
        self.model.eval()
        self.logger.info(
            f"已加载 {checkpoint_path} (epoch={info['epoch']}, "
            f"best_score={info['best_score']:.4f})"
        )

    def _save_batch(
        self,
        predictions: torch.Tensor,
        names: List[str],
        marker: str,
    ) -> None:
        """将一个 batch 的预测结果按比赛命名规范保存为 JPG。

        Args:
            predictions: 预测张量 ``(B, C, H, W)``，取值 ``[0, 1]``。
            names:       每个样本的文件名（不含后缀）。
            marker:      目标标记名，决定输出子目录。

        Returns:
            None
        """
        output_dir = self.output_root / "test" / marker
        output_dir.mkdir(parents=True, exist_ok=True)

        for prediction, name in zip(predictions, names):
            array = prediction.detach().cpu().clamp(0.0, 1.0).numpy()
            array = np.transpose(array, (1, 2, 0))  # CHW -> HWC
            array = (array * 255.0).round().astype(np.uint8)

            if array.shape[-1] == 1:
                image = array[..., 0]  # 灰度直接保存
            else:
                image = cv2.cvtColor(array, cv2.COLOR_RGB2BGR)

            cv2.imwrite(str(output_dir / f"{name}{self.suffix}.jpg"), image)

    @staticmethod
    def _unpack_predictions(output: Any) -> torch.Tensor:
        """解包模型输出，兼容返回 (预测, 共享特征) 元组的模型。

        Args:
            output: 模型前向输出，张量或元组。

        Returns:
            torch.Tensor: 预测图像 ``(B, C, H, W)``。
        """
        return output[0] if isinstance(output, tuple) else output

    @torch.no_grad()
    def run_single_marker(self, marker: str, checkpoint_path: str) -> None:
        """用单标记模型生成一种标记的全部测试结果。

        Args:
            marker:          目标标记名。
            checkpoint_path: 该标记对应的模型 checkpoint。

        Returns:
            None
        """
        self._load_weights(checkpoint_path)

        for batch in tqdm(self.test_loader, desc=f"infer [{marker}]"):
            inputs = batch["input"].to(self.device)
            predictions = self._unpack_predictions(self.model(inputs))
            self._save_batch(predictions, batch["name"], marker)

    @torch.no_grad()
    def run_multi_marker(self, checkpoint_path: str) -> None:
        """用多标记条件模型一次性生成全部四种标记的结果。

        Args:
            checkpoint_path: 条件模型 checkpoint。

        Returns:
            None
        """
        self._load_weights(checkpoint_path)

        for marker_idx, marker in enumerate(MARKERS):
            for batch in tqdm(self.test_loader, desc=f"infer [{marker}]"):
                inputs = batch["input"].to(self.device)
                idx_tensor = torch.full(
                    (inputs.shape[0],), marker_idx,
                    dtype=torch.long, device=self.device,
                )
                predictions = self._unpack_predictions(
                    self.model(inputs, idx_tensor)
                )
                self._save_batch(predictions, batch["name"], marker)

    def run(self, checkpoint_paths: Dict[str, str]) -> None:
        """推理入口：按模型类型分发到单标记或多标记流程。

        Args:
            checkpoint_paths: ``{标记名: checkpoint路径}``；
                多标记模式约定使用键 ``"all"``。

        Returns:
            None

        Raises:
            KeyError: 多标记模式缺少 ``"all"`` 键，
                或单标记模式缺少某标记 checkpoint 时抛出。
        """
        if self.conditional:
            if "all" not in checkpoint_paths:
                raise KeyError("多标记模式需要提供 {'all': checkpoint路径}")
            self.run_multi_marker(checkpoint_paths["all"])
        else:
            for marker in MARKERS:
                if marker not in checkpoint_paths:
                    raise KeyError(f"缺少标记 {marker} 的 checkpoint 路径")
                self.run_single_marker(marker, checkpoint_paths[marker])

        self.logger.info(f"推理完成，结果保存于 {self.output_root}/test/")
