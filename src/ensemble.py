"""作为集成器。

根据多个并行训练任务作业的结果进行集成：把同一标记下多个实验
（例如实验矩阵筛选出的 Top-K）的预测结果逐像素取平均，以更稳定的
结果作为最终提交，对应赛题「集成提升指标」的做法。
"""

from typing import Any, Dict, List, Optional

import torch
import torch.nn as nn
from tqdm import tqdm

from .data.constants import MARKERS
from .train import DataLoaders, ModelRegistry, Predictor
from .utils import CheckpointManager, LoggerFactory, Runtime


class Ensembler:
    """集成器：多实验预测结果逐像素平均。"""

    def __init__(
        self,
        config: Dict[str, Any],
        output_root: Optional[str] = None,
        suffix: Optional[str] = None,
    ) -> None:
        """初始化推理设置。

        Args:
            config:      全局配置（决定模型结构与推理设置）。
            output_root: 结果根目录；``None`` 时取 ``inference.output_dir``。
            suffix:      输出文件名后缀；``None`` 时取 ``inference.suffix``。
        """
        inference_cfg = config.get("inference", {})
        self.config = config
        self.conditional = ModelRegistry.is_conditional(config)
        self.output_root = output_root or str(
            inference_cfg.get("output_dir", "results")
        )
        self.suffix = suffix or str(inference_cfg.get("suffix", "_fake"))
        self.device = Runtime.get_device(config.get("runtime", {}).get("device"))
        self.logger = LoggerFactory.create("ensemble")

    def _load_models(self, checkpoint_paths: List[str]) -> List[nn.Module]:
        """按同一配置实例化并加载多个 checkpoint 的模型。

        Args:
            checkpoint_paths: 参与集成的 checkpoint 路径列表。

        Returns:
            List[nn.Module]: 已加载权重并切换为评估模式的模型列表。
        """
        models: List[nn.Module] = []
        for path in checkpoint_paths:
            model = ModelRegistry.build(self.config)
            info = CheckpointManager.load(path, model, map_location="cpu")
            model.to(self.device).eval()
            self.logger.info(f"已加载集成成员 {path} (epoch={info['epoch']})")
            models.append(model)
        return models

    def _ensemble_marker(self, marker: str, checkpoint_paths: List[str]) -> None:
        """对单个标记集成多个模型并保存平均后的预测结果。

        Args:
            marker:           目标标记名。
            checkpoint_paths: 参与集成的 checkpoint 路径列表。

        Returns:
            None
        """
        models = self._load_models(checkpoint_paths)
        test_loader = DataLoaders.for_test(self.config)
        marker_idx = torch.full((1,), MARKERS.index(marker), dtype=torch.long)

        with torch.no_grad():
            for batch in tqdm(test_loader, desc=f"ensemble [{marker}]"):
                inputs = batch["input"].to(self.device)
                predictions: List[torch.Tensor] = []
                for model in models:
                    if self.conditional:
                        idx = marker_idx.expand(inputs.shape[0]).to(self.device)
                        output = model(inputs, idx)
                    else:
                        output = model(inputs)
                    predictions.append(Predictor.unpack(output))

                # 逐像素平均后按比赛规范保存。
                Predictor.save_batch(
                    torch.stack(predictions, dim=0).mean(dim=0),
                    batch["name"],
                    marker,
                    self.output_root,
                    self.suffix,
                )

        # 释放集成成员，避免多标记循环时显存累积。
        del models

    def run(
        self, experiments: List[str], markers: Optional[List[str]] = None
    ) -> None:
        """集成多个实验的预测结果并输出到比赛提交目录。

        - 单标记模型：对每个标记分别收集各实验的权重并取平均；
        - 一对多条件模型：每个实验一个权重，逐标记推理后取平均。

        Args:
            experiments: 参与集成的实验名列表。
            markers:     需要生成的标记；``None`` 表示全部四类。

        Returns:
            None
        """
        # 每个实验对应一份 checkpoint 映射（条件模型用键 "all"）。
        checkpoint_map = {
            experiment: Predictor.find_checkpoints(experiment, self.conditional)
            for experiment in experiments
        }

        for marker in list(markers) if markers is not None else list(MARKERS):
            key = "all" if self.conditional else marker
            self._ensemble_marker(
                marker, [checkpoint_map[experiment][key] for experiment in experiments]
            )

        self.logger.info(
            f"集成完成（{len(experiments)} 个实验），"
            f"结果保存于 {self.output_root}/test/"
        )
