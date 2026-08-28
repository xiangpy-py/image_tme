"""项目统一 CLI 入口。

将各执行脚本聚合为子命令形式::

    python main.py analyze  --root data
    python main.py split    --root data --val-ratio 0.15
    python main.py train    --config configs/baseline.yaml
    python main.py infer    --config configs/multi_marker.yaml --ckpt-all ...

与直接运行 ``scripts/`` 下的脚本等价，便于比赛评审统一调用。
"""

import sys
from pathlib import Path


def main() -> None:
    """根据第一个参数分发到对应的 scripts 入口。

    Returns:
        None
    """
    if len(sys.argv) < 2:
        print("用法: python main.py <analyze|split|train|infer> [参数...]")
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

    # 以子进程方式转发剩余参数到对应脚本。
    import subprocess

    script_path = Path(__file__).parent / "scripts" / script_map[command]
    result = subprocess.run(
        [sys.executable, str(script_path)] + sys.argv[2:],
        check=False,
    )
    raise SystemExit(result.returncode)


if __name__ == "__main__":
    main()
