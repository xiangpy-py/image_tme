"""日志模块：控制台日志 + 实验记录（CSV / JSON）。

比赛要求提交结果可复现，因此每次实验都会把
训练超参数与逐 epoch 指标持久化到 ``logs/`` 目录。
"""

import csv
import json
import logging
import sys
from pathlib import Path
from typing import Any, Dict, List


def setup_logger(name: str, log_file: str = "") -> logging.Logger:
    """创建同时输出到控制台与文件的 logger。

    Args:
        name:     logger 名称，通常使用入口脚本名。
        log_file: 日志文件路径，空字符串表示仅输出到控制台。

    Returns:
        logging.Logger: 配置完成的 logger 实例。
    """
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    logger.propagate = False

    # 避免重复添加 handler（例如在交互式环境中多次调用）。
    if logger.handlers:
        return logger

    formatter = logging.Formatter(
        fmt="[%(asctime)s][%(levelname)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    if log_file:
        path = Path(log_file)
        path.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(path, encoding="utf-8")
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    return logger


class ExperimentLogger:
    """实验记录器：逐 epoch 追加 CSV，并在结束时汇总为 JSON。

    使用方式::

        recorder = ExperimentLogger("logs/exp001", fieldnames=["epoch", "loss"])
        recorder.log({"epoch": 1, "loss": 0.12})
        recorder.finish()
    """

    def __init__(self, log_dir: str, fieldnames: List[str]) -> None:
        """初始化实验记录器并创建 CSV 表头。

        Args:
            log_dir:    本次实验的日志目录。
            fieldnames: CSV 列名，例如 ``["epoch", "train_loss", "val_ssim"]``。
        """
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)

        self.csv_path = self.log_dir / "history.csv"
        self.json_path = self.log_dir / "history.json"
        self._fieldnames = fieldnames
        self._records: List[Dict[str, Any]] = []

        with open(self.csv_path, "w", newline="", encoding="utf-8") as file:
            writer = csv.DictWriter(file, fieldnames=fieldnames)
            writer.writeheader()

    def log(self, record: Dict[str, Any]) -> None:
        """追加一条记录（通常对应一个 epoch）。

        Args:
            record: 键需与初始化时的 ``fieldnames`` 对应。

        Returns:
            None
        """
        row = {key: record.get(key, "") for key in self._fieldnames}
        self._records.append(row)

        with open(self.csv_path, "a", newline="", encoding="utf-8") as file:
            writer = csv.DictWriter(file, fieldnames=self._fieldnames)
            writer.writerow(row)

    def finish(self) -> None:
        """将全部记录汇总写入 JSON 文件，便于后续分析脚本读取。"""
        with open(self.json_path, "w", encoding="utf-8") as file:
            json.dump(self._records, file, ensure_ascii=False, indent=2)
