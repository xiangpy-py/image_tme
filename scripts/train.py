"""训练入口脚本。

用法::

    # 训练配置文件中指定的单个标记
    python scripts/train.py --config configs/baseline.yaml

    # 命令行指定标记（无需改 YAML）
    python scripts/train.py --config configs/baseline.yaml --marker CD68

    # 一条命令依次训练全部四种标记（推荐）
    python scripts/train.py --config configs/baseline.yaml --marker all

    # 多标记条件模型（一个模型生成四种标记，无需 --marker）
    python scripts/train.py --config configs/multi_marker.yaml

「--marker all」模式下，每个标记自动以 ``<experiment.name>_<标记名小写>``
为实验名独立训练，checkpoint 保存在 ``checkpoints/<实验名>/best.pth``，
推理脚本可按同一约定自动找到它们。
"""

import argparse
import copy
import sys
from pathlib import Path
from typing import Any, Dict, List

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.datasets import MARKERS
from src.datasets.constants import sanitize_marker_name
from src.engine import Trainer
from src.models import is_conditional_model
from src.utils import load_config, merge_config_with_args, save_config


def train_one(config: Dict[str, Any]) -> float:
    """按给定配置执行一次完整训练。

    Args:
        config: 已确定目标标记与实验名的配置。

    Returns:
        float: 本次训练的最优验证综合得分。
    """
    trainer = Trainer(config)

    # 保存实际生效的配置，保证实验可复现。
    save_config(config, str(trainer.log_dir / "config.yaml"))
    return trainer.fit()


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
    marker_config = copy.deepcopy(config)
    marker_config.setdefault("data", {})["marker"] = marker

    base_name = marker_config.get("experiment", {}).get("name", "exp")
    marker_config.setdefault("experiment", {})["name"] = (
        f"{base_name}_{sanitize_marker_name(marker)}"
    )
    return marker_config


def parse_args() -> argparse.Namespace:
    """解析命令行参数。

    Returns:
        argparse.Namespace: 含配置路径、目标标记与可覆盖的超参项。
    """
    parser = argparse.ArgumentParser(description="虚拟染色模型训练")
    parser.add_argument("--config", type=str, required=True, help="YAML 配置文件路径")
    parser.add_argument(
        "--marker", type=str, default=None,
        help="目标标记名；传 'all' 依次训练全部四种标记；不传则使用配置文件中的标记",
    )
    # 以下为可选覆盖项，传 None 时不生效，以配置文件为准。
    parser.add_argument("--epochs", type=int, default=None, help="覆盖训练轮数")
    parser.add_argument("--batch-size", type=int, default=None, help="覆盖批大小")
    parser.add_argument("--lr", type=float, default=None, help="覆盖学习率")
    parser.add_argument("--device", type=str, default=None, help="覆盖运行设备")
    return parser.parse_args()


def main() -> None:
    """训练主流程：解析配置 -> 确定标记列表 -> 逐个训练。"""
    args = parse_args()
    config = merge_config_with_args(load_config(args.config), args)

    # 多标记条件模型一次训练覆盖全部标记，不需要 --marker。
    if is_conditional_model(config):
        best_score = train_one(config)
        print(f"训练结束，最优验证 Score = {best_score:.4f}")
        return

    # 确定本次要训练的标记列表。
    if args.marker is None:
        markers: List[str] = [config.get("data", {}).get("marker", "CD68")]
    elif args.marker.lower() == "all":
        markers = list(MARKERS)
    else:
        if args.marker not in MARKERS:
            raise SystemExit(f"未知标记: {args.marker}，可选: {MARKERS} 或 all")
        markers = [args.marker]

    results: Dict[str, float] = {}
    for marker in markers:
        print(f"\n===== 开始训练标记 [{marker}] =====")
        results[marker] = train_one(build_marker_config(config, marker))

    print("\n===== 训练汇总 =====")
    for marker, score in results.items():
        print(f"  {marker}: 最优验证 Score = {score:.4f}")


if __name__ == "__main__":
    main()
