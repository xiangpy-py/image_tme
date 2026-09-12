"""作为分配器。

根据布局器传入的作业列表，结合当前电脑配置（GPU 数量 / 设备类型）
生成并行任务调度计划：把作业切分为若干「波次」，同一波次内的作业
分配到不同设备上并行执行。每轮作业过后，可依据反馈器给出的提示
重新分配（把表现更优的实验提前调度）。
"""

from dataclasses import dataclass, field
from typing import List, Optional, Sequence, Tuple

import torch

from .layout import JobSpec

# 一个波次内各作业及其绑定的运行设备。
Wave = List[Tuple[JobSpec, str]]
# 完整调度计划：按波次顺序依次执行，波次内并行。
Schedule = List[Wave]


@dataclass
class ResourceProfile:
    """本机可用于并行训练的资源画像。

    Attributes:
        devices: 可用设备列表，如 ``["cuda:0", "cuda:1"]`` 或 ``["cpu"]``。
        slots:   并行槽位数（同时运行的作业数上限）。
    """

    devices: List[str] = field(default_factory=lambda: ["cpu"])
    slots: int = 1


class Allocator:
    """分配器：按本机资源把作业切分为并行波次。"""

    def __init__(self, device: Optional[str] = None) -> None:
        """探测并持有本机资源画像。

        Args:
            device: 配置中显式指定的设备（``"auto"`` / ``"cpu"`` / ``"cuda:1"``）。
        """
        self.resources = self.detect_resources(device)

    @staticmethod
    def detect_resources(device: Optional[str] = None) -> ResourceProfile:
        """探测本机可用的训练设备与并行槽位。

        有 GPU 时按 GPU 数量并行（每卡一个作业）；仅 CPU 时串行执行，
        避免多进程争抢算力反而拖慢整体训练。

        Args:
            device: 显式指定的设备。

        Returns:
            ResourceProfile: 资源画像。
        """
        # 显式指定了带序号的单卡（如 ``cuda:1``）或 CPU 时只用一个槽位，
        # 尊重用户的设备选择，避免仍然铺满全部 GPU。
        if device is not None and device != "auto" and device != "cuda":
            return ResourceProfile(devices=[device], slots=1)

        # 未指定设备，或只写 ``cuda`` 表示使用全部可见 GPU。
        if torch.cuda.is_available():
            count = max(1, torch.cuda.device_count())
            return ResourceProfile(
                devices=[f"cuda:{index}" for index in range(count)], slots=count
            )
        return ResourceProfile(devices=["cpu"], slots=1)

    @property
    def description(self) -> str:
        """str: 资源画像的可读描述，用于日志与命令行输出。"""
        return (
            f"设备={','.join(self.resources.devices)} | "
            f"并行槽位={self.resources.slots}"
        )

    def schedule(
        self,
        jobs: Sequence[JobSpec],
        prioritize: Optional[Sequence[str]] = None,
    ) -> Schedule:
        """把作业列表切分为可并行的调度计划。

        Args:
            jobs:       布局器产出的作业列表。
            prioritize: 优先调度的实验名（来自反馈器的再分配提示）。
                命中规则为「实验名相同」或「作业名以 ``<实验名>_`` 开头」，
                因此单标记模型按标记展开的作业（如
                ``exp001_unet_baseline_cd68``）也能被 ``exp001_unet_baseline``
                命中；未命中的作业保持原相对顺序。

        Returns:
            Schedule: 波次列表；同一波次内作业数量不超过 ``slots``。
        """
        ordered = list(jobs)
        if prioritize:
            priority_names = list(prioritize)

            def _rank(key: str) -> int:
                for index, name in enumerate(priority_names):
                    if key == name or key.startswith(f"{name}_"):
                        return index
                return len(priority_names)

            ordered.sort(key=lambda job: _rank(job.name))

        devices = self.resources.devices
        return [
            [
                (job, devices[index % len(devices)])
                for index, job in enumerate(ordered[start : start + self.resources.slots])
            ]
            for start in range(0, len(ordered), self.resources.slots)
        ]
