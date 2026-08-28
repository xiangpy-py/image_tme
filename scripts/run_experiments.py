"""实验矩阵执行脚本：赛马制「短实验筛选 -> 长训练冲刺」。

用法::

    # 第一阶段：全部实验短跑，生成排行榜
    python scripts/run_experiments.py --matrix configs/experiment_matrix.yaml --stage screening

    # 查看排行榜（也可单独执行）
    python scripts/run_experiments.py --matrix configs/experiment_matrix.yaml --stage report

    # 第二阶段：排行榜 Top-K 进入完整训练
    python scripts/run_experiments.py --matrix configs/experiment_matrix.yaml --stage full

    # 一键跑完两阶段
    python scripts/run_experiments.py --matrix configs/experiment_matrix.yaml --stage all

排行榜持久化在 ``logs/leaderboard.csv``，筛选阶段的实验目录
以 ``_screen`` 后缀与正式长训练区分。
"""

import argparse
import csv
import sys
import time
from pathlib import Path
from typing import Any, Dict, List

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.datasets import MARKERS
from src.models import is_conditional_model
from src.utils import (
    build_marker_config,
    deep_merge,
    load_config,
    save_config,
    setup_logger,
)

LEADERBOARD_PATH = Path("logs") / "leaderboard.csv"
LEADERBOARD_FIELDS = [
    "stage", "experiment", "marker", "model_type",
    "epochs", "best_score", "elapsed_min", "timestamp",
]

logger = setup_logger("experiments")


# ------------------------------------------------------------------ #
# 矩阵解析与配置合成
# ------------------------------------------------------------------ #
def build_experiment_config(
    matrix: Dict[str, Any],
    experiment: Dict[str, Any],
    stage: str,
) -> Dict[str, Any]:
    """由矩阵条目合成一份完整训练配置。

    Args:
        matrix:     实验矩阵（含 screening / full 两阶段参数）。
        experiment: 矩阵中的单个实验条目（name / base / overrides）。
        stage:      ``"screening"`` 或 ``"full"``。

    Returns:
        Dict[str, Any]: 可直接交给 ``Trainer`` 的配置。
    """
    base_config = load_config(experiment["base"])
    config = deep_merge(base_config, experiment.get("overrides", {}))

    stage_cfg = matrix.get(stage, {}) or {}
    config.setdefault("training", {})["epochs"] = int(
        stage_cfg.get("epochs", 15 if stage == "screening" else 100)
    )

    # 筛选阶段实验名加后缀，避免覆盖长训练的 checkpoint 与日志。
    name = experiment["name"]
    if stage == "screening":
        name = f"{name}_screen"
    config.setdefault("experiment", {})["name"] = name

    # 筛选阶段单标记模型只训代理标记，保证横向可比。
    if stage == "screening" and not is_conditional_model(config):
        proxy = str(stage_cfg.get("marker", "CD68"))
        config.setdefault("data", {})["marker"] = proxy

    return config


def experiment_markers(config: Dict[str, Any], stage: str, matrix: Dict[str, Any]) -> List[str]:
    """确定一个实验本次要训练的标记列表。

    条件模型一次训练覆盖全部标记（返回空列表表示无需逐标记展开）；
    单标记模型在筛选阶段只训代理标记，长训练阶段训全部四种。

    Args:
        config: 已合成的实验配置。
        stage:  ``"screening"`` 或 ``"full"``。
        matrix: 实验矩阵。

    Returns:
        List[str]: 待训练的标记列表；条件模型返回空列表。
    """
    if is_conditional_model(config):
        return []
    if stage == "screening":
        proxy = str(matrix.get("screening", {}).get("marker", "CD68"))
        return [proxy]
    return list(MARKERS)


# ------------------------------------------------------------------ #
# 排行榜
# ------------------------------------------------------------------ #
def append_leaderboard(rows: List[Dict[str, Any]]) -> None:
    """将一批实验结果追加到排行榜 CSV。

    Args:
        rows: 每个元素为一次训练的汇总记录。

    Returns:
        None
    """
    LEADERBOARD_PATH.parent.mkdir(parents=True, exist_ok=True)
    write_header = not LEADERBOARD_PATH.is_file()

    with open(LEADERBOARD_PATH, "a", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=LEADERBOARD_FIELDS)
        if write_header:
            writer.writeheader()
        writer.writerows(rows)


def read_leaderboard() -> List[Dict[str, Any]]:
    """读取排行榜全部记录。

    Returns:
        List[Dict[str, Any]]: 排行榜行列表，文件不存在时返回空列表。
    """
    if not LEADERBOARD_PATH.is_file():
        return []
    with open(LEADERBOARD_PATH, "r", encoding="utf-8") as file:
        return list(csv.DictReader(file))


def print_leaderboard(stage: str = "") -> None:
    """按综合得分降序打印排行榜。

    Args:
        stage: 只展示某一阶段（``screening`` / ``full``），空串表示全部。

    Returns:
        None
    """
    rows = read_leaderboard()
    if stage:
        rows = [row for row in rows if row["stage"] == stage]
    if not rows:
        print("排行榜为空，请先运行 --stage screening")
        return

    # 同一实验可能跑过多次，取每 (stage, experiment, marker) 的最高分。
    best_rows: Dict[tuple, Dict[str, Any]] = {}
    for row in rows:
        key = (row["stage"], row["experiment"], row["marker"])
        if key not in best_rows or float(row["best_score"]) > float(best_rows[key]["best_score"]):
            best_rows[key] = row

    ordered = sorted(best_rows.values(), key=lambda r: float(r["best_score"]), reverse=True)

    header = f"{'排名':<4}{'阶段':<11}{'实验':<34}{'标记':<10}{'模型':<20}{'Score':<8}{'耗时(min)':<10}"
    print("\n" + header)
    print("-" * len(header))
    for rank, row in enumerate(ordered, start=1):
        print(
            f"{rank:<4}{row['stage']:<11}{row['experiment']:<34}{row['marker']:<10}"
            f"{row['model_type']:<20}{float(row['best_score']):.4f}  {row['elapsed_min']:<10}"
        )


# ------------------------------------------------------------------ #
# 训练执行
# ------------------------------------------------------------------ #
def run_single_experiment(config: Dict[str, Any], stage: str, marker: str) -> Dict[str, Any]:
    """执行一次训练并返回排行榜记录。

    Args:
        config: 完整训练配置（实验名与标记已确定）。
        stage:  阶段名，用于排行榜归类。
        marker: 本次训练的标记（条件模型记为 ``all``）。

    Returns:
        Dict[str, Any]: 排行榜行记录。
    """
    # 延迟导入：避免脚本启动即加载 torch，report 等只读操作更轻。
    from src.engine import Trainer

    start = time.time()
    trainer = Trainer(config)
    save_config(config, str(trainer.log_dir / "config.yaml"))
    best_score = trainer.fit()
    elapsed_min = (time.time() - start) / 60.0

    return {
        "stage": stage,
        "experiment": config.get("experiment", {}).get("name", "exp"),
        "marker": marker,
        "model_type": config.get("model", {}).get("type", "unknown"),
        "epochs": config.get("training", {}).get("epochs", 0),
        "best_score": f"{best_score:.6f}",
        "elapsed_min": f"{elapsed_min:.1f}",
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
    }


def run_stage(matrix: Dict[str, Any], stage: str) -> List[Dict[str, Any]]:
    """执行矩阵中全部实验的某一阶段。

    Args:
        matrix: 实验矩阵。
        stage:  ``"screening"`` 或 ``"full"``。

    Returns:
        List[Dict[str, Any]]: 本阶段全部排行榜记录。
    """
    experiments = matrix.get("experiments", [])
    logger.info(f"===== 阶段 [{stage}]：共 {len(experiments)} 组实验 =====")

    rows: List[Dict[str, Any]] = []
    for index, experiment in enumerate(experiments, start=1):
        config = build_experiment_config(matrix, experiment, stage)
        exp_name = config["experiment"]["name"]
        logger.info(f"----- [{index}/{len(experiments)}] {exp_name} -----")

        markers = experiment_markers(config, stage, matrix)
        if not markers:
            # 条件模型：一次训练覆盖全部标记。
            rows.append(run_single_experiment(config, stage, marker="all"))
            continue

        for marker in markers:
            marker_config = (
                build_marker_config(config, marker) if len(markers) > 1 or marker
                else config
            )
            rows.append(run_single_experiment(marker_config, stage, marker))

    append_leaderboard(rows)
    return rows


def run_full_stage(matrix: Dict[str, Any]) -> None:
    """长训练阶段：按筛选排行榜取 Top-K 实验进行完整训练。

    Args:
        matrix: 实验矩阵。

    Returns:
        None
    """
    top_k = int(matrix.get("full", {}).get("top_k", 3))
    screening_rows = [
        row for row in read_leaderboard() if row["stage"] == "screening"
    ]
    if not screening_rows:
        raise SystemExit("筛选阶段排行榜为空，请先运行 --stage screening")

    # 按实验聚合：取该实验在筛选阶段的最高分。
    best_by_experiment: Dict[str, float] = {}
    for row in screening_rows:
        name = row["experiment"].removesuffix("_screen")
        best_by_experiment[name] = max(
            best_by_experiment.get(name, 0.0), float(row["best_score"])
        )

    top_names = {
        name
        for name, _score in sorted(
            best_by_experiment.items(), key=lambda item: item[1], reverse=True
        )[:top_k]
    }
    logger.info(f"筛选 Top-{top_k} 进入长训练: {sorted(top_names)}")

    selected = [
        exp for exp in matrix.get("experiments", []) if exp["name"] in top_names
    ]
    reduced = dict(matrix)
    reduced["experiments"] = selected
    run_stage(reduced, "full")


# ------------------------------------------------------------------ #
# 入口
# ------------------------------------------------------------------ #
def parse_args() -> argparse.Namespace:
    """解析命令行参数。

    Returns:
        argparse.Namespace: 含矩阵路径与执行阶段。
    """
    parser = argparse.ArgumentParser(description="实验矩阵：短实验筛选 + 长训练")
    parser.add_argument("--matrix", type=str, required=True, help="实验矩阵 YAML 路径")
    parser.add_argument(
        "--stage", type=str, required=True,
        choices=["screening", "full", "report", "all"],
        help="screening=短实验筛选; full=Top-K长训练; report=只看排行榜; all=两阶段连跑",
    )
    return parser.parse_args()


def main() -> None:
    """主流程：读取矩阵 -> 按阶段执行 -> 输出排行榜。"""
    args = parse_args()
    matrix = load_config(args.matrix)

    if args.stage in ("screening", "all"):
        run_stage(matrix, "screening")
        print_leaderboard("screening")

    if args.stage in ("full", "all"):
        run_full_stage(matrix)
        print_leaderboard("full")

    if args.stage == "report":
        print_leaderboard()


if __name__ == "__main__":
    main()
