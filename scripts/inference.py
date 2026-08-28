"""推理入口脚本。

按比赛提交要求，在测试集 DAPI 上自动生成各目标标记的虚拟染色图像，
输出到 ``results/test/<MARKER>/<原名>_fake.jpg``。

用法::

    # 推荐：按实验名自动查找全部标记的权重（与训练脚本的命名约定对应）
    python scripts/inference.py --config configs/baseline.yaml --exp exp001_unet

    # 多标记条件模型：一个权重生成全部标记
    python scripts/inference.py --config configs/multi_marker.yaml --exp exp004_multi_marker

    # 手动指定（高级用法，可与 --exp 混用作覆盖）
    python scripts/inference.py --config configs/baseline.yaml \
        --ckpt-CD68 checkpoints/exp001_unet_cd68/best.pth ...

权重查找约定（与训练脚本一致）：

- 单标记模型: ``checkpoints/<实验名>_<标记名小写>/best.pth``
- 条件模型:   ``checkpoints/<实验名>/best.pth``
"""

import argparse
import sys
from pathlib import Path
from typing import Dict

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.datasets import MARKERS
from src.datasets.constants import sanitize_marker_name  # 与训练脚本同一命名约定
from src.engine import Predictor
from src.models import is_conditional_model
from src.utils import load_config


def find_checkpoints_by_experiment(
    experiment: str,
    conditional: bool,
    checkpoint_root: str = "checkpoints",
) -> Dict[str, str]:
    """按命名约定自动收集实验对应的全部 checkpoint。

    Args:
        experiment:      实验名（训练配置中的 ``experiment.name``）。
        conditional:     是否为多标记条件模型。
        checkpoint_root: checkpoint 根目录。

    Returns:
        Dict[str, str]: ``{标记名: checkpoint路径}``；条件模型使用键 ``"all"``。

    Raises:
        FileNotFoundError: 任一必需的 checkpoint 不存在时抛出，
            错误信息中列出缺失的路径。
    """
    root = Path(checkpoint_root)

    if conditional:
        path = root / experiment / "best.pth"
        if not path.is_file():
            raise FileNotFoundError(f"未找到条件模型 checkpoint: {path}")
        return {"all": str(path)}

    checkpoints: Dict[str, str] = {}
    missing = []
    for marker in MARKERS:
        path = root / f"{experiment}_{sanitize_marker_name(marker)}" / "best.pth"
        if path.is_file():
            checkpoints[marker] = str(path)
        else:
            missing.append(str(path))

    if missing:
        raise FileNotFoundError(
            "以下标记的 checkpoint 缺失，请先完成对应训练:\n  " + "\n  ".join(missing)
        )
    return checkpoints


def parse_args() -> argparse.Namespace:
    """解析命令行参数。

    Returns:
        argparse.Namespace: 配置路径、实验名与各标记的 checkpoint 路径。
    """
    parser = argparse.ArgumentParser(description="虚拟染色测试集推理")
    parser.add_argument("--config", type=str, required=True, help="YAML 配置文件路径")
    parser.add_argument(
        "--exp", type=str, default=None,
        help="实验名，按命名约定自动查找全部标记的 checkpoint",
    )
    parser.add_argument(
        "--ckpt-all", type=str, default=None,
        help="多标记条件模型的统一 checkpoint",
    )
    for marker in MARKERS:
        parser.add_argument(
            f"--ckpt-{marker}", type=str, default=None,
            help=f"标记 {marker} 的单模型 checkpoint",
        )
    return parser.parse_args()


def main() -> None:
    """推理主流程：收集 checkpoint 映射并执行批量生成。

    优先级：手动指定的 ``--ckpt-*`` 覆盖 ``--exp`` 自动查找的结果。
    """
    args = parse_args()
    config = load_config(args.config)
    conditional = is_conditional_model(config)

    # 1) 按实验名自动查找（若提供）。
    checkpoint_paths: Dict[str, str] = {}
    if args.exp is not None:
        checkpoint_paths = find_checkpoints_by_experiment(args.exp, conditional)

    # 2) 手动指定的 checkpoint 覆盖/补充。
    if args.ckpt_all is not None:
        checkpoint_paths["all"] = args.ckpt_all
    for marker in MARKERS:
        path = getattr(args, f"ckpt_{marker.replace('-', '_')}", None)
        if path is not None:
            checkpoint_paths[marker] = path

    if not checkpoint_paths:
        raise SystemExit(
            "错误: 请提供 --exp <实验名> 或至少一个 --ckpt-<标记> 路径"
        )

    predictor = Predictor(config)
    predictor.run(checkpoint_paths)


if __name__ == "__main__":
    main()
