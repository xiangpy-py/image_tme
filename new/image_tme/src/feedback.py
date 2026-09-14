"""作为反馈器。

根据操作器传入的作业结果进行汇总与修正（写入排行榜、筛选更优实验），
然后反馈给分配器：输出「下一轮优先调度哪些实验」的提示与 Top-K 实验名单。
"""

import csv
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Sequence

from .data.constants import MARKERS, sanitize_marker_name
from .utils import LoggerFactory

LEADERBOARD_FIELDS = [
    "stage",
    "experiment",
    "marker",
    "model_type",
    "epochs",
    "best_score",
    "elapsed_min",
    "timestamp",
]


@dataclass
class JobResult:
    """一次训练作业的运行结果（同时是排行榜的一条记录）。

    Attributes:
        experiment:  实验名。
        marker:      目标标记；条件模型为 ``"all"``。
        stage:       所属阶段。
        model_type:  模型类型（配置中的 ``model.type``）。
        epochs:      训练轮数。
        best_score:  最优验证综合得分。
        elapsed_min: 训练耗时（分钟）。
        timestamp:   完成时间。
    """

    experiment: str
    marker: str
    stage: str
    model_type: str
    epochs: int
    best_score: float
    elapsed_min: float
    timestamp: str

    def to_row(self) -> Dict[str, Any]:
        """转换为排行榜 CSV 行（浮点保留固定精度便于阅读）。

        Returns:
            Dict[str, Any]: CSV 行字典。
        """
        row = asdict(self)
        row["best_score"] = f"{self.best_score:.6f}"
        row["elapsed_min"] = f"{self.elapsed_min:.1f}"
        return row


class Feedback:
    """反馈器：结果汇总、排行榜与再分配提示。"""

    def __init__(self, leaderboard_path: str = "logs/leaderboard.csv") -> None:
        """指定排行榜文件位置。

        Args:
            leaderboard_path: 排行榜 CSV 路径。
        """
        self.leaderboard_path = Path(leaderboard_path)
        self.logger = LoggerFactory.create("feedback")

    # ------------------------------------------------------------------ #
    # 结果落盘与读取
    # ------------------------------------------------------------------ #
    def record(self, results: Sequence[JobResult]) -> None:
        """把一批作业结果追加到排行榜 CSV。

        Args:
            results: 作业结果列表。

        Returns:
            None
        """
        if not results:
            return

        self.leaderboard_path.parent.mkdir(parents=True, exist_ok=True)
        write_header = not self.leaderboard_path.is_file()
        with open(self.leaderboard_path, "a", newline="", encoding="utf-8") as file:
            writer = csv.DictWriter(file, fieldnames=LEADERBOARD_FIELDS)
            if write_header:
                writer.writeheader()
            writer.writerows(result.to_row() for result in results)

    def read(self) -> List[Dict[str, str]]:
        """读取排行榜全部记录。

        Returns:
            List[Dict[str, str]]: 排行榜行列表，文件不存在时返回空列表。
        """
        if not self.leaderboard_path.is_file():
            return []
        with open(self.leaderboard_path, "r", encoding="utf-8") as file:
            return list(csv.DictReader(file))

    # ------------------------------------------------------------------ #
    # 汇总与筛选
    # ------------------------------------------------------------------ #
    @staticmethod
    def strip_marker_suffix(name: str) -> str:
        """剥离实验名末尾的 ``_<标记名>`` 与 ``_screen`` 后缀。

        用于把「单标记模型按标记展开的实验名」还原为矩阵中的原始实验名，
        例如 ``s04_adapter_baseline_screen_cd68`` -> ``s04_adapter_baseline``。

        Args:
            name: 排行榜中的实验名。

        Returns:
            str: 还原后的矩阵实验名。
        """
        for marker in MARKERS:
            suffix = f"_{sanitize_marker_name(marker)}"
            if name.endswith(suffix):
                name = name[: -len(suffix)]
                break
        return name.removesuffix("_screen")

    def aggregate(self, stage: str = "") -> Dict[str, float]:
        """按原始实验名聚合最高分。

        同一实验可能跑过多次（或按多个标记展开），取最高分代表该实验。

        Args:
            stage: 只统计某一阶段，空串表示全部。

        Returns:
            Dict[str, float]: ``{原始实验名: 最高分}``。
        """
        scores: Dict[str, float] = {}
        for row in self.read():
            if stage and row["stage"] != stage:
                continue
            name = self.strip_marker_suffix(row["experiment"])
            scores[name] = max(scores.get(name, 0.0), float(row["best_score"]))
        return scores

    @staticmethod
    def top_k(scores: Dict[str, float], k: int) -> List[str]:
        """按得分降序选出 Top-K 实验名。

        Args:
            scores: ``{实验名: 得分}``。
            k:      选取数量。

        Returns:
            List[str]: 选中的实验名列表（按得分降序）。
        """
        ordered = sorted(scores.items(), key=lambda item: item[1], reverse=True)
        return [name for name, _score in ordered[:k]]

    @staticmethod
    def hints(top_names: Sequence[str]) -> Dict[str, Any]:
        """生成给分配器的再分配提示。

        Args:
            top_names: 优先调度的实验名列表。

        Returns:
            Dict[str, Any]: 可传给 ``Allocator.schedule(prioritize=...)`` 的提示。
        """
        return {"prioritize": list(top_names)}

    # ------------------------------------------------------------------ #
    # 展示
    # ------------------------------------------------------------------ #
    def _best_rows(self, stage: str = "") -> List[Dict[str, str]]:
        """取每个 (阶段, 实验, 标记) 的最高分记录并按得分降序排列。

        Args:
            stage: 只保留某一阶段，空串表示全部。

        Returns:
            List[Dict[str, str]]: 排行后的记录列表。
        """
        best: Dict[tuple, Dict[str, str]] = {}
        for row in self.read():
            if stage and row["stage"] != stage:
                continue
            key = (row["stage"], row["experiment"], row["marker"])
            if key not in best or float(row["best_score"]) > float(
                best[key]["best_score"]
            ):
                best[key] = row
        return sorted(
            best.values(), key=lambda row: float(row["best_score"]), reverse=True
        )

    def print_leaderboard(self, stage: str = "") -> None:
        """按综合得分降序打印排行榜。

        Args:
            stage: 只展示某一阶段（``screening`` / ``full``），空串表示全部。

        Returns:
            None
        """
        rows = self._best_rows(stage)
        if not rows:
            print("排行榜为空，请先运行 experiments --stage screening")
            return

        header = (
            f"{'排名':<4}{'阶段':<11}{'实验':<34}{'标记':<10}"
            f"{'模型':<20}{'Score':<8}{'耗时(min)':<10}"
        )
        print("\n" + header)
        print("-" * len(header))
        for rank, row in enumerate(rows, start=1):
            print(
                f"{rank:<4}{row['stage']:<11}{row['experiment']:<34}{row['marker']:<10}"
                f"{row['model_type']:<20}{float(row['best_score']):.4f}  "
                f"{row['elapsed_min']:<10}"
            )
