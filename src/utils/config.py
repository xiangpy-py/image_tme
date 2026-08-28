"""配置管理模块。

负责 YAML 实验配置的读取、与命令行参数的合并以及配置落盘，
保证实验参数集中管理、结果可复现。
"""

import argparse
import copy
from pathlib import Path
from typing import Any, Dict, Optional

import yaml


def load_config(config_path: str) -> Dict[str, Any]:
    """读取 YAML 配置文件并返回字典形式的配置。

    Args:
        config_path: YAML 配置文件路径，例如 ``configs/baseline.yaml``。

    Returns:
        Dict[str, Any]: 解析后的配置字典，键结构与原 YAML 保持一致。

    Raises:
        FileNotFoundError: 当配置文件不存在时抛出。
    """
    path = Path(config_path)
    if not path.is_file():
        raise FileNotFoundError(f"配置文件不存在: {config_path}")

    with open(path, "r", encoding="utf-8") as file:
        config = yaml.safe_load(file)

    return config if config is not None else {}


def save_config(config: Dict[str, Any], save_path: str) -> None:
    """将运行时的实际配置保存到日志目录，便于实验追溯与复现。

    Args:
        config: 需要保存的配置字典。
        save_path: 目标 YAML 文件路径。

    Returns:
        None
    """
    path = Path(save_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    with open(path, "w", encoding="utf-8") as file:
        yaml.safe_dump(config, file, allow_unicode=True, sort_keys=False)


def merge_config_with_args(
    config: Dict[str, Any], args: argparse.Namespace
) -> Dict[str, Any]:
    """将命令行显式指定的参数覆盖到 YAML 配置上。

    覆盖规则：仅当命令行参数取值不为 ``None`` 时才覆盖，
    避免 argparse 默认值意外抹掉配置文件中的设定。

    Args:
        config: 从 YAML 读取的原始配置。
        args:   命令行解析结果。

    Returns:
        Dict[str, Any]: 合并后的新配置（不修改原始字典）。
    """
    merged = copy.deepcopy(config)

    # 约定：命令行中的 --xxx 对应配置中 training.xxx 或顶层字段。
    override_map = {
        "epochs": ("training", "epochs"),
        "batch_size": ("training", "batch_size"),
        "lr": ("training", "lr"),
        "device": ("runtime", "device"),
    }

    for arg_name, (section, key) in override_map.items():
        value = getattr(args, arg_name, None)
        if value is None:
            continue
        merged.setdefault(section, {})[key] = value

    return merged


def get_nested(config: Dict[str, Any], path: str, default: Optional[Any] = None) -> Any:
    """按点分路径安全地读取嵌套配置值。

    Args:
        config: 配置字典。
        path:   点分键路径，例如 ``"training.batch_size"``。
        default: 路径不存在时返回的默认值。

    Returns:
        Any: 配置值或默认值。
    """
    current: Any = config
    for key in path.split("."):
        if not isinstance(current, dict) or key not in current:
            return default
        current = current[key]
    return current


def deep_merge(base: Dict[str, Any], overrides: Dict[str, Any]) -> Dict[str, Any]:
    """递归合并两个字典，overrides 中的值优先。

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
            merged[key] = deep_merge(merged[key], value)
        else:
            merged[key] = copy.deepcopy(value)
    return merged


def build_marker_config(
    config: Dict[str, Any], marker: str
) -> Dict[str, Any]:
    """为指定标记生成独立配置：写入目标标记并派生实验名。

    实验名约定为 ``<原实验名>_<标记名小写>``，例如
    ``exp001_unet_baseline_cd68``，与推理阶段的权重查找约定一致。

    Args:
        config: 基础配置（不修改传入对象）。
        marker: 目标标记名。

    Returns:
        Dict[str, Any]: 该标记的独立配置副本。
    """
    # 延迟导入，避免与 datasets 包形成循环依赖。
    from ..datasets.constants import sanitize_marker_name

    marker_config = copy.deepcopy(config)
    marker_config.setdefault("data", {})["marker"] = marker

    base_name = marker_config.get("experiment", {}).get("name", "exp")
    marker_config.setdefault("experiment", {})["name"] = (
        f"{base_name}_{sanitize_marker_name(marker)}"
    )
    return marker_config
