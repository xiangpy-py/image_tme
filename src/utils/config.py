"""配置管理。

负责 YAML 实验配置的读取、与命令行参数的合并以及配置落盘，
保证实验参数集中管理、结果可复现。
"""

import argparse
import copy
from pathlib import Path
from typing import Any, Dict

import yaml


class ConfigManager:
    """实验配置的读取、覆盖合并与落盘。"""

    @staticmethod
    def load(config_path: str) -> Dict[str, Any]:
        """读取 YAML 配置文件并返回字典形式的配置。

        Args:
            config_path: YAML 配置文件路径，例如 ``configs/baseline.yaml``。

        Returns:
            Dict[str, Any]: 解析后的配置字典。

        Raises:
            FileNotFoundError: 配置文件不存在时抛出。
        """
        path = Path(config_path)
        if not path.is_file():
            raise FileNotFoundError(f"配置文件不存在: {config_path}")

        with open(path, "r", encoding="utf-8") as file:
            config = yaml.safe_load(file)
        return config if config is not None else {}

    @staticmethod
    def save(config: Dict[str, Any], save_path: str) -> None:
        """把运行时实际生效的配置落盘，便于实验追溯与复现。

        Args:
            config:    需要保存的配置字典。
            save_path: 目标 YAML 文件路径。

        Returns:
            None
        """
        path = Path(save_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as file:
            yaml.safe_dump(config, file, allow_unicode=True, sort_keys=False)

    @staticmethod
    def merge_with_args(
        config: Dict[str, Any], args: argparse.Namespace
    ) -> Dict[str, Any]:
        """把命令行显式指定的超参覆盖到 YAML 配置上。

        仅当命令行取值不为 ``None`` 时才覆盖，
        避免 argparse 默认值抹掉配置文件中的设定。

        Args:
            config: 从 YAML 读取的原始配置。
            args:   命令行解析结果。

        Returns:
            Dict[str, Any]: 合并后的新配置（不修改原始字典）。
        """
        merged = copy.deepcopy(config)

        # 约定：命令行中的 --xxx 对应配置中 training.xxx 或 runtime.device。
        override_map = {
            "epochs": ("training", "epochs"),
            "batch_size": ("training", "batch_size"),
            "lr": ("training", "lr"),
            "device": ("runtime", "device"),
            "resume": ("training", "resume"),
        }
        for arg_name, (section, key) in override_map.items():
            value = getattr(args, arg_name, None)
            if value is not None:
                merged.setdefault(section, {})[key] = value
        return merged

    @staticmethod
    def deep_merge(base: Dict[str, Any], overrides: Dict[str, Any]) -> Dict[str, Any]:
        """递归合并两个字典，``overrides`` 中的值优先。

        用于实验矩阵中「基础配置 + 局部覆盖」的组合方式，
        嵌套字典逐层合并而非整体替换。

        Args:
            base:      基础配置（不修改传入对象）。
            overrides: 覆盖项。

        Returns:
            Dict[str, Any]: 合并后的新配置字典。
        """
        merged = copy.deepcopy(base)
        for key, value in overrides.items():
            if isinstance(value, dict) and isinstance(merged.get(key), dict):
                merged[key] = ConfigManager.deep_merge(merged[key], value)
            else:
                merged[key] = copy.deepcopy(value)
        return merged
