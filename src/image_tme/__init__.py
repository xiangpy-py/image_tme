"""项目统一 CLI 入口。

将各执行脚本聚合为子命令形式::

    image-tme analyze  --root data
    image-tme split    --root data --val-ratio 0.15
    image-tme train    --config configs/baseline.yaml
    image-tme infer    --config configs/multi_marker.yaml --ckpt-all ...
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

__version__ = "0.1.0"


def main() -> None:
    if len(sys.argv) < 2:
        print("用法: image-tme <analyze|split|train|infer> [参数...]")
        print("  analyze  数据统计分析")
        print("  split    按 ROI 划分训练/验证集")
        print("  train    模型训练")
        print("  infer    测试集推理并生成提交结果")
        raise SystemExit(1)

    command = sys.argv[1]
    script_map = {
        "analyze": "analyze_data.py",
        "split": "make_splits.py",
        "train": "train.py",
        "infer": "inference.py",
    }
    if command not in script_map:
        print(f"未知命令: {command}，可选: {list(script_map)}")
        raise SystemExit(1)

    script_path = Path(__file__).resolve().parent.parent.parent / "scripts" / script_map[command]
    result = subprocess.run([sys.executable, str(script_path)] + sys.argv[2:], check=False)
    raise SystemExit(result.returncode)

