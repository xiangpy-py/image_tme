"""image_tme 核心包。

按「包 -> 类 -> 方法」三级组织，自底向上分层：

- ``utils``:    通用工具链（``ConfigManager`` / ``Runtime`` /
  ``LoggerFactory`` / ``ExperimentLogger`` / ``CheckpointManager`` /
  ``ModelEMA``），不依赖其它子包；
- ``data``:     通用数据层（``DatasetSource`` / ``DatasetSplitter`` /
  ``DatasetAnalyzer``），只依赖 numpy / opencv，不依赖 torch；
- ``train``:    单作业训练子系统（数据集、模型、损失、指标、训练与推理引擎）；
- ``layout``:   布局器，把超参数展开为训练作业列表（``JobSpec``）；
- ``allocate``: 分配器，按本机硬件生成并行调度计划；
- ``operate``:  操作器，执行并行训练作业并串联完整编排链路；
- ``feedback``: 反馈器，汇总结果、维护排行榜并给出再分配提示；
- ``ensemble``: 集成器，对多实验预测结果取平均。
"""

__version__ = "0.1.0"
