"""作为人为的规划布局（布局器）。

把传入的超参数按布局规则展开为可供分配器调度的训练作业列表：

- 单实验训练：一个配置 + 目标标记 -> 1~4 个作业；
- 实验矩阵：以「base 配置 + overrides 覆盖」描述候选实验，按阶段
  （筛选 / 长训练）展开为作业；筛选阶段单标记模型统一使用代理标记
  以保证横向可比，条件模型一次训练覆盖全部标记。
"""

import copy
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from .data.constants import MARKERS, sanitize_marker_name
from .train.models import ModelRegistry
from .utils import ConfigManager


@dataclass
class JobSpec:
    """一个训练作业的完整描述。

    Attributes:
        name:   实验名，决定 ``logs/``、``checkpoints/`` 下的目录名。
        config: 可直接交给 ``Trainer`` 的完整配置。
        stage:  所属阶段（``train`` / ``screening`` / ``full``）。
        marker: 目标标记；一对多条件模型统一记为 ``"all"``。
    """

    name: str
    config: Dict[str, Any]
    stage: str = "train"
    marker: str = "all"


class Layout:
    """布局器：超参数 -> 训练作业列表。"""

    # ------------------------------------------------------------------ #
    # 单实验布局
    # ------------------------------------------------------------------ #
    @staticmethod
    def marker_config(base_config: Dict[str, Any], marker: str) -> Dict[str, Any]:
        """为指定标记生成独立配置：写入目标标记并派生实验名。

        实验名约定为 ``<原实验名>_<标记名小写>``，例如
        ``exp001_unet_baseline_cd68``，与推理阶段的权重查找约定一致。

        Args:
            base_config: 基础配置（不修改传入对象）。
            marker:      目标标记名。

        Returns:
            Dict[str, Any]: 该标记的独立配置副本。
        """
        marker_config = copy.deepcopy(base_config)
        marker_config.setdefault("data", {})["marker"] = marker

        base_name = marker_config.get("experiment", {}).get("name", "exp")
        marker_config.setdefault("experiment", {})["name"] = (
            f"{base_name}_{sanitize_marker_name(marker)}"
        )
        return marker_config

    @staticmethod
    def resolve_markers(marker: Optional[str], config: Dict[str, Any]) -> List[str]:
        """确定本次要训练的标记列表。

        Args:
            marker: 命令行指定的标记；``None`` 表示沿用配置，``"all"`` 表示全部。
            config: 全局配置。

        Returns:
            List[str]: 待训练的标记列表。

        Raises:
            ValueError: 标记名非法时抛出。
        """
        if marker is None:
            return [config.get("data", {}).get("marker", "CD68")]
        if marker.lower() == "all":
            return list(MARKERS)
        if marker not in MARKERS:
            raise ValueError(f"未知标记: {marker}，可选: {MARKERS} 或 all")
        return [marker]

    @classmethod
    def train_jobs(
        cls, config: Dict[str, Any], marker: Optional[str] = None
    ) -> List[JobSpec]:
        """由单实验配置构建作业列表（对应 ``train`` 命令）。

        一对多条件模型一次训练覆盖全部标记，只产生一个作业；
        单标记模型按目标标记展开为多个作业。

        Args:
            config: 全局配置（已合并命令行覆盖项）。
            marker: 命令行指定的标记。

        Returns:
            List[JobSpec]: 待调度作业列表。
        """
        if ModelRegistry.is_conditional(config):
            name = config.get("experiment", {}).get("name", "exp")
            return [JobSpec(name=name, config=config, stage="train", marker="all")]

        jobs: List[JobSpec] = []
        for target in cls.resolve_markers(marker, config):
            job_config = cls.marker_config(config, target)
            jobs.append(
                JobSpec(
                    name=job_config["experiment"]["name"],
                    config=job_config,
                    stage="train",
                    marker=target,
                )
            )
        return jobs

    # ------------------------------------------------------------------ #
    # 实验矩阵布局
    # ------------------------------------------------------------------ #
    @staticmethod
    def job_config(
        matrix: Dict[str, Any], experiment: Dict[str, Any], stage: str, resume: bool = False
    ) -> Dict[str, Any]:
        """由矩阵条目合成一份完整训练配置。

        Args:
            matrix:     实验矩阵（含 screening / full 两阶段参数）。
            experiment: 矩阵中的单个实验条目（name / base / overrides）。
            stage:      ``"screening"`` 或 ``"full"``。
            resume:     是否开启断点续训（注入 ``training.resume``）。

        Returns:
            Dict[str, Any]: 可直接交给 ``Trainer`` 的配置。
        """
        config = ConfigManager.deep_merge(
            ConfigManager.load(experiment["base"]), experiment.get("overrides", {})
        )

        stage_cfg = matrix.get(stage, {}) or {}
        config.setdefault("training", {})["epochs"] = int(
            stage_cfg.get("epochs", 15 if stage == "screening" else 100)
        )

        # 续训开关：矩阵级统一注入，作业内 Trainer 据此从 last.pth 恢复进度。
        if resume:
            config.setdefault("training", {})["resume"] = True

        # 筛选阶段实验名加后缀，避免覆盖长训练的 checkpoint 与日志。
        name = experiment["name"]
        if stage == "screening":
            name = f"{name}_screen"
        config.setdefault("experiment", {})["name"] = name

        # 筛选阶段单标记模型只训代理标记，保证横向可比。
        if stage == "screening" and not ModelRegistry.is_conditional(config):
            config.setdefault("data", {})["marker"] = str(
                stage_cfg.get("marker", "CD68")
            )
        return config

    @classmethod
    def stage_jobs(
        cls, matrix: Dict[str, Any], stage: str, resume: bool = False
    ) -> List[JobSpec]:
        """把实验矩阵的某一阶段展开为作业列表。

        条件模型一次训练覆盖全部标记（1 个作业）；单标记模型在筛选阶段
        只训代理标记，长训练阶段训全部四种标记。

        Args:
            matrix: 实验矩阵。
            stage:  ``"screening"`` 或 ``"full"``。
            resume: 是否开启断点续训。

        Returns:
            List[JobSpec]: 该阶段全部待调度作业。
        """
        jobs: List[JobSpec] = []
        for experiment in matrix.get("experiments", []):
            config = cls.job_config(matrix, experiment, stage, resume=resume)

            if ModelRegistry.is_conditional(config):
                jobs.append(
                    JobSpec(
                        name=config["experiment"]["name"],
                        config=config,
                        stage=stage,
                        marker="all",
                    )
                )
                continue

            # 筛选阶段只训代理标记，长训练阶段展开为全部四种标记。
            markers = (
                [str(matrix.get("screening", {}).get("marker", "CD68"))]
                if stage == "screening"
                else list(MARKERS)
            )
            for target in markers:
                job_config = cls.marker_config(config, target)
                jobs.append(
                    JobSpec(
                        name=job_config["experiment"]["name"],
                        config=job_config,
                        stage=stage,
                        marker=target,
                    )
                )
        return jobs

    @staticmethod
    def select_experiments(
        matrix: Dict[str, Any], names: List[str]
    ) -> Dict[str, Any]:
        """裁剪实验矩阵，仅保留指定实验（用于 Top-K 长训练阶段）。

        Args:
            matrix: 实验矩阵。
            names:  需要保留的实验名集合。

        Returns:
            Dict[str, Any]: 仅含选中实验的新矩阵（不修改传入对象）。
        """
        selected_names = set(names)
        reduced = copy.deepcopy(matrix)
        reduced["experiments"] = [
            experiment
            for experiment in matrix.get("experiments", [])
            if experiment["name"] in selected_names
        ]
        return reduced
