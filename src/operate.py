"""作为操作器。

根据分配器传入的调度计划执行多个并行训练任务作业：单作业波次直接在
当前进程内训练；多作业波次使用 ``spawn`` 子进程（每作业一进程、绑定
各自设备）真正并行，最后汇总各作业结果。

同时提供上层用例（把「布局 -> 分配 -> 操作 -> 反馈」串起来），
供组织器直接调用：单实验训练、矩阵某阶段、矩阵 Top-K 长训练。
"""

import copy
import time
from concurrent.futures import ProcessPoolExecutor
from multiprocessing import get_context
from typing import Any, Callable, Dict, List, Optional, Sequence

from .allocate import Allocator, Schedule, Wave
from .feedback import Feedback, JobResult
from .layout import JobSpec, Layout
from .utils import ConfigManager, LoggerFactory


class Operator:
    """操作器：执行作业，并串联布局 / 分配 / 反馈三个角色。"""

    def __init__(self, device: Optional[str] = None) -> None:
        """初始化分配器与反馈器。

        Args:
            device: 显式指定设备；``None`` 表示自动探测。
        """
        self.allocator = Allocator(device)
        self.feedback = Feedback()
        self.logger = LoggerFactory.create("operate")

    # ------------------------------------------------------------------ #
    # 单个作业与波次
    # ------------------------------------------------------------------ #
    @staticmethod
    def run_job(job: JobSpec, device: str) -> JobResult:
        """执行单个训练作业，返回其运行结果。

        Args:
            job:    作业描述（含完整训练配置）。
            device: 本作业绑定的运行设备。

        Returns:
            JobResult: 该作业的最优验证得分与耗时等信息。
        """
        # 延迟导入：子进程内才加载 torch 与训练子系统。
        from .train import Trainer

        config = copy.deepcopy(job.config)
        config.setdefault("runtime", {})["device"] = device

        start = time.time()
        trainer = Trainer(config)
        # 保存实际生效的配置，保证实验可复现。
        ConfigManager.save(config, str(trainer.log_dir / "config.yaml"))
        best_score = trainer.fit()

        return JobResult(
            experiment=config.get("experiment", {}).get("name", job.name),
            marker=job.marker,
            stage=job.stage,
            model_type=config.get("model", {}).get("type", "unknown"),
            epochs=int(config.get("training", {}).get("epochs", 0)),
            best_score=float(best_score),
            elapsed_min=(time.time() - start) / 60.0,
            timestamp=time.strftime("%Y-%m-%d %H:%M:%S"),
        )

    @staticmethod
    def run_wave(wave: Wave) -> List[JobResult]:
        """执行一个波次内的全部作业。

        Args:
            wave: 波次内 ``(作业, 设备)`` 列表。

        Returns:
            List[JobResult]: 本波次各作业的运行结果。
        """
        if not wave:
            return []
        if len(wave) == 1:
            job, device = wave[0]
            return [Operator.run_job(job, device)]

        # 多设备并行：spawn 子进程避免 fork 复制 CUDA 上下文。
        context = get_context("spawn")
        with ProcessPoolExecutor(max_workers=len(wave), mp_context=context) as executor:
            futures = [
                executor.submit(Operator.run_job, job, device) for job, device in wave
            ]
            return [future.result() for future in futures]

    @staticmethod
    def execute(
        schedule: Schedule,
        on_wave: Optional[Callable[[List[JobResult]], None]] = None,
    ) -> List[JobResult]:
        """按调度计划依次执行各波次，汇总全部作业结果。

        每完成一个波次即回调 ``on_wave``：排行榜等产物边跑边落盘，
        即使后续波次崩溃（如磁盘写满）也不会丢失已完成作业的成绩。

        Args:
            schedule: 分配器产出的波次列表。
            on_wave:  每完成一个波次后的回调（接收该波次结果）。

        Returns:
            List[JobResult]: 全部作业的运行结果（按执行顺序）。
        """
        results: List[JobResult] = []
        for index, wave in enumerate(schedule, start=1):
            print(
                f"[operate] 波次 {index}/{len(schedule)}："
                + ", ".join(job.name for job, _device in wave)
            )
            wave_results = Operator.run_wave(wave)
            results.extend(wave_results)
            if on_wave is not None and wave_results:
                on_wave(wave_results)
        return results

    # ------------------------------------------------------------------ #
    # 上层用例
    # ------------------------------------------------------------------ #
    def run_schedule(
        self, jobs: Sequence[JobSpec], prioritize: Optional[Sequence[str]] = None
    ) -> List[JobResult]:
        """把作业交给分配器排期后并行执行（布局 -> 分配 -> 操作）。

        Args:
            jobs:       布局器产出的作业列表。
            prioritize: 反馈器给出的优先调度实验名。

        Returns:
            List[JobResult]: 各作业运行结果。
        """
        schedule = self.allocator.schedule(jobs, prioritize=prioritize)
        self.logger.info(
            f"作业数={len(jobs)} | {self.allocator.description} | "
            f"波次数={len(schedule)}"
        )
        # 排行榜随每个波次增量落盘，避免整阶段崩溃导致成绩全部丢失。
        return self.execute(schedule, on_wave=self.feedback.record)

    def train_experiment(
        self, config: Dict[str, Any], marker: Optional[str] = None
    ) -> List[JobResult]:
        """单实验训练：按目标标记展开作业并执行。

        Args:
            config: 全局配置（已合并命令行覆盖项）。
            marker: 目标标记；``"all"`` 表示全部四类，``None`` 表示沿用配置。

        Returns:
            List[JobResult]: 各作业运行结果。
        """
        # 排行榜已由 run_schedule 逐波次落盘，此处无需重复写入。
        return self.run_schedule(Layout.train_jobs(config, marker))

    def run_stage(
        self,
        matrix: Dict[str, Any],
        stage: str,
        prioritize: Optional[Sequence[str]] = None,
        resume: bool = False,
    ) -> List[JobResult]:
        """执行实验矩阵的某一阶段（screening / full）。

        Args:
            matrix:     实验矩阵。
            stage:      阶段名。
            prioritize: 优先调度的实验名（来自反馈器的再分配提示）。
            resume:     是否开启断点续训（作业从各自 ``last.pth`` 恢复）。

        Returns:
            List[JobResult]: 本阶段各作业运行结果。
        """
        jobs = Layout.stage_jobs(matrix, stage, resume=resume)
        self.logger.info(
            f"===== 阶段 [{stage}]：{len(jobs)} 个作业"
            + ("（断点续训）=====" if resume else "=====")
        )
        results = self.run_schedule(jobs, prioritize=prioritize)
        return results

    def run_full_stage(
        self, matrix: Dict[str, Any], resume: bool = False
    ) -> List[JobResult]:
        """长训练阶段：按筛选排行榜取 Top-K 实验进行完整训练。

        这是「反馈器 -> 分配器」闭环的落地点：Top-K 由反馈器选出，
        并以再分配提示的形式交给分配器优先调度。

        Args:
            matrix: 实验矩阵（读取 ``full.top_k``）。
            resume: 是否开启断点续训。

        Returns:
            List[JobResult]: 本阶段各作业运行结果。

        Raises:
            ValueError: 筛选阶段排行榜为空时抛出。
        """
        scores = self.feedback.aggregate(stage="screening")
        if not scores:
            raise ValueError("筛选阶段排行榜为空，请先执行 experiments --stage screening")

        top_names = self.feedback.top_k(scores, int(matrix.get("full", {}).get("top_k", 3)))
        self.logger.info(f"筛选 Top-{len(top_names)} 进入长训练: {top_names}")

        return self.run_stage(
            Layout.select_experiments(matrix, top_names),
            "full",
            prioritize=self.feedback.hints(top_names)["prioritize"],
            resume=resume,
        )

    def run_experiments(
        self, matrix: Dict[str, Any], stage: str, resume: bool = False
    ) -> None:
        """实验矩阵总入口：赛马制「短实验筛选 -> 排行榜 -> Top-K 长训练」。

        Args:
            matrix: 实验矩阵。
            stage:  ``"screening"`` / ``"full"`` / ``"report"`` / ``"all"``。
            resume: 是否开启断点续训（沿用各作业已有的 ``last.pth``）。

        Returns:
            None
        """
        if stage == "report":
            self.feedback.print_leaderboard()
            return

        if stage in ("screening", "all"):
            self.run_stage(matrix, "screening", resume=resume)
            self.feedback.print_leaderboard("screening")

        if stage in ("full", "all"):
            self.run_full_stage(matrix, resume=resume)
            self.feedback.print_leaderboard("full")
