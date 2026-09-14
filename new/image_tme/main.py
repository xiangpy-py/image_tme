"""组织器：项目统一 CLI 入口。

本文件只负责**解析运行指令与超参数**，具体功能全部由 ``src`` 承担：

    analyze       数据统计分析        -> DatasetAnalyzer.analyze
    split         按 ROI 划分数据集   -> DatasetSplitter.create
    context       多尺度上下文生成    -> ContextGenerator.generate
    train         单实验训练          -> Operator.train_experiment
    experiments   实验矩阵赛马        -> Operator.run_experiments
    infer         测试集推理          -> Predictor.run_experiment
    ensemble      多实验结果集成      -> Ensembler.run
    marker-report 逐标记短板分析      -> MarkerEvaluator.report

其中训练类指令统一走「布局器 -> 分配器 -> 操作器（-> 反馈器）」编排链路，
配置的读取与合并统一由 ``ConfigManager`` 完成。
"""

import argparse
from typing import Dict

from src.data import ContextGenerator, DatasetAnalyzer, DatasetSplitter
from src.data.constants import MARKERS
from src.ensemble import Ensembler
from src.operate import Operator
from src.train import Predictor
from src.utils import ConfigManager

# 保留换行的帮助格式，便于在 epilog 中书写多行示例。
FORMATTER = argparse.RawDescriptionHelpFormatter


def collect_ckpt_overrides(args: argparse.Namespace) -> Dict[str, str]:
    """从命令行参数收集 ``--ckpt-*`` 覆盖项。

    ``infer`` 与 ``marker-report`` 共用同一套权重指定方式，统一在此整理。

    Args:
        args: 含 ``ckpt_all`` 与各标记 ``ckpt_<标记名>`` 属性。

    Returns:
        Dict[str, str]: ``{标记名: checkpoint路径}``，未指定的项不出现。
    """
    overrides: Dict[str, str] = {}
    if getattr(args, "ckpt_all", None) is not None:
        overrides["all"] = args.ckpt_all
    for marker in MARKERS:
        value = getattr(args, f"ckpt_{marker.replace('-', '_')}", None)
        if value is not None:
            overrides[marker] = value
    return overrides


# ------------------------------------------------------------------ #
# 子命令处理：只做「参数 -> 功能调用 -> 结果展示」的转发
# ------------------------------------------------------------------ #
def cmd_analyze(args: argparse.Namespace) -> None:
    """数据统计分析。

    Args:
        args: 含 ``root`` / ``output``。

    Returns:
        None
    """
    DatasetAnalyzer.analyze(args.root, args.output)


def cmd_split(args: argparse.Namespace) -> None:
    """按 ROI 划分训练/验证集并落盘。

    Args:
        args: 含 ``root`` / ``val_ratio`` / ``seed`` / ``save_dir``。

    Returns:
        None
    """
    train_names, val_names = DatasetSplitter.create(
        args.root, args.val_ratio, args.seed, args.save_dir
    )
    print(f"训练集: {len(train_names)} 张 | 验证集: {len(val_names)} 张")
    print(f"划分文件已保存: {args.save_dir}/split.json")


def cmd_context(args: argparse.Namespace) -> None:
    """多尺度上下文生成：按 ROI 网格拼接并输出各 patch 的上下文图。

    Args:
        args: 含 ``root`` / ``output`` / ``scale``。

    Returns:
        None
    """
    statistics = ContextGenerator.generate(args.root, args.output, args.scale)
    total = sum(item["images"] for item in statistics.values())
    print(f"上下文图生成完成：共 {total} 张，输出目录 {args.output}")
    print("训练/推理配置中设置 data.context_dir 为该目录即可启用。")


def cmd_train(args: argparse.Namespace) -> None:
    """单实验训练：解析超参覆盖项后交给操作器执行。

    Args:
        args: 含 ``config`` / ``marker`` 与可选超参覆盖项。

    Returns:
        None
    """
    config = ConfigManager.merge_with_args(ConfigManager.load(args.config), args)
    results = Operator(args.device).train_experiment(config, args.marker)

    print("\n===== 训练汇总 =====")
    for result in results:
        print(
            f"  {result.experiment} | 标记={result.marker} | "
            f"最优 Score={result.best_score:.4f} | 耗时={result.elapsed_min:.1f}min"
        )


def cmd_experiments(args: argparse.Namespace) -> None:
    """实验矩阵：赛马制「短实验筛选 -> 排行榜 -> Top-K 长训练」。

    Args:
        args: 含 ``matrix`` / ``stage`` / ``device`` / ``resume``。

    Returns:
        None
    """
    Operator(args.device).run_experiments(
        ConfigManager.load(args.matrix), args.stage, resume=bool(args.resume)
    )


def cmd_infer(args: argparse.Namespace) -> None:
    """测试集推理：把 ``--ckpt-*`` 覆盖项整理后交给推理入口。

    Args:
        args: 含 ``config`` / ``exp`` / 各标记 checkpoint 覆盖项。

    Returns:
        None
    """
    Predictor.run_experiment(
        ConfigManager.load(args.config), args.exp, collect_ckpt_overrides(args)
    )


def cmd_marker_report(args: argparse.Namespace) -> None:
    """逐标记短板分析：把 ``--ckpt-*`` 覆盖项整理后交给逐标记评估器。

    Args:
        args: 含 ``config`` / ``exp`` / 各标记 checkpoint 覆盖项。

    Returns:
        None
    """
    MarkerEvaluator(ConfigManager.load(args.config)).report(
        args.exp, collect_ckpt_overrides(args)
    )


def cmd_ensemble(args: argparse.Namespace) -> None:
    """模型集成：对多个实验的预测结果取平均。

    Args:
        args: 含 ``config`` / ``exps``。

    Returns:
        None
    """
    Ensembler(ConfigManager.load(args.config)).run(args.exps)


# ------------------------------------------------------------------ #
# 命令行定义
# ------------------------------------------------------------------ #
def parse_args() -> argparse.Namespace:
    """定义并解析全部子命令。

    Returns:
        argparse.Namespace: 解析结果。
    """
    parser = argparse.ArgumentParser(
        prog="image-tme",
        description=(
            "虚拟染色：以 DAPI 图像为输入生成目标 IHC 标记图像"
            "（CD68 / CD45RO / HLA-DR / Vimentin）。"
        ),
        epilog=(
            "示例:\n"
            "  uv run main.py analyze     --root data\n"
            "  uv run main.py split       --root data --val-ratio 0.15\n"
            "  uv run main.py train       --config configs/baseline.yaml --marker all\n"
            "  uv run main.py experiments --matrix configs/experiment_matrix.yaml --stage all\n"
            "  uv run main.py infer       --config configs/baseline.yaml --exp exp001_unet_baseline\n"
            "  uv run main.py ensemble    --config configs/adapter_unet.yaml "
            "--exps exp007_adapter_unet_cd68 exp007_adapter_unet_cd45ro\n"
            "\n查看某条指令的全部超参: uv run main.py <指令> --help"
        ),
        formatter_class=FORMATTER,
    )
    subparsers = parser.add_subparsers(
        title="可用指令", dest="command", required=True, metavar="<指令>"
    )

    # ---- analyze ----
    analyze_parser = subparsers.add_parser(
        "analyze",
        help="数据统计分析",
        description=(
            "扫描数据目录，统计各标记的图像数量、尺寸、通道与像素分布，"
            "结果落盘为 JSON，用于赛题数据分析阶段。"
        ),
        epilog="示例:\n  uv run main.py analyze --root data --output data/statistics.json",
        formatter_class=FORMATTER,
    )
    analyze_parser.add_argument(
        "--root",
        type=str,
        default="data",
        metavar="DIR",
        help="数据根目录，兼容 data/<organ>/train 与 data/train 两种结构（默认: data）",
    )
    analyze_parser.add_argument(
        "--output",
        type=str,
        default="data/statistics.json",
        metavar="FILE",
        help="统计结果输出路径（默认: data/statistics.json）",
    )
    analyze_parser.set_defaults(func=cmd_analyze)

    # ---- split ----
    split_parser = subparsers.add_parser(
        "split",
        help="按 ROI 划分训练/验证集",
        description=(
            "按 ROI 前缀（ROIxxx）划分训练/验证集：同一 ROI 切出的相邻 patch "
            "高度相关，必须整体划分以避免验证集信息泄漏。"
        ),
        epilog=(
            "示例:\n  uv run main.py split --root data --val-ratio 0.15 --seed 42\n"
            "输出: <save-dir>/split.json（训练时自动加载）"
        ),
        formatter_class=FORMATTER,
    )
    split_parser.add_argument(
        "--root", type=str, default="data", metavar="DIR", help="数据根目录（默认: data）"
    )
    split_parser.add_argument(
        "--val-ratio",
        type=float,
        default=0.15,
        metavar="P",
        help="验证集 ROI 占比，取值 (0, 1)（默认: 0.15）",
    )
    split_parser.add_argument(
        "--seed",
        type=int,
        default=42,
        metavar="N",
        help="随机种子，保证划分可复现（默认: 42）",
    )
    split_parser.add_argument(
        "--save-dir",
        type=str,
        default="data/splits",
        metavar="DIR",
        help="划分结果保存目录（默认: data/splits）",
    )
    split_parser.set_defaults(func=cmd_split)

    # ---- context ----
    context_parser = subparsers.add_parser(
        "context",
        help="多尺度上下文生成",
        description=(
            "利用 patch 文件名 ROIxxx_rr_cc 中的网格坐标把同一 ROI 的相邻 "
            "DAPI patch 拼回大图，为每个 patch 生成一张「邻域重采样」上下文图；"
            "训练/推理时在配置中设置 data.context_dir 即可把输入从 3 通道 "
            "扩展为 6 通道（DAPI + 上下文）。"
        ),
        epilog=(
            "示例:\n"
            "  uv run main.py context --root data --output data/context --scale 2\n"
            "注意: 数据集变动后需重新生成；测试集会一并处理。"
        ),
        formatter_class=FORMATTER,
    )
    context_parser.add_argument(
        "--root",
        type=str,
        default="data",
        metavar="DIR",
        help="数据根目录，兼容 data/<organ>/train 与 data/train 两种结构（默认: data）",
    )
    context_parser.add_argument(
        "--output",
        type=str,
        default="data/context",
        metavar="DIR",
        help="上下文图输出根目录（默认: data/context）",
    )
    context_parser.add_argument(
        "--scale",
        type=float,
        default=2.0,
        metavar="S",
        help="视野倍率：>1 为组织级上下文（默认: 2），<1 为中心细节放大",
    )
    context_parser.set_defaults(func=cmd_context)

    # ---- train ----
    train_parser = subparsers.add_parser(
        "train",
        help="单实验模型训练",
        description=(
            "单实验训练：按目标标记展开为训练作业，走「布局器 -> 分配器 -> 操作器」"
            "编排链路；产物写入 checkpoints/<实验名>/ 与 logs/<实验名>/，"
            "并把结果汇总进 logs/leaderboard.csv。"
        ),
        epilog=(
            "示例:\n"
            "  # 训练配置中的单个标记\n"
            "  uv run main.py train --config configs/baseline.yaml\n"
            "  # 一条命令训练四种标记（实验名自动追加 _<标记> 后缀）\n"
            "  uv run main.py train --config configs/baseline.yaml --marker all\n"
            "  # 一对多条件模型：一次训练覆盖全部标记\n"
            "  uv run main.py train --config configs/conditional_v2.yaml"
        ),
        formatter_class=FORMATTER,
    )
    train_parser.add_argument(
        "--config", type=str, required=True, metavar="FILE", help="训练配置 YAML 路径（必填）"
    )
    train_parser.add_argument(
        "--marker",
        type=str,
        default=None,
        metavar="NAME",
        help=(
            "目标标记：CD68 / CD45RO / HLA-DR / Vimentin，或 all 训练全部四类；"
            "条件模型可省略（默认: 用配置中的 data.marker）"
        ),
    )
    train_parser.add_argument(
        "--epochs", type=int, default=None, metavar="N", help="覆盖 training.epochs"
    )
    train_parser.add_argument(
        "--batch-size", type=int, default=None, metavar="N", help="覆盖 training.batch_size"
    )
    train_parser.add_argument(
        "--lr", type=float, default=None, metavar="LR", help="覆盖 training.lr"
    )
    train_parser.add_argument(
        "--device",
        type=str,
        default=None,
        metavar="DEV",
        help="覆盖 runtime.device，如 auto / cpu / cuda / cuda:0",
    )
    train_parser.add_argument(
        "--resume",
        action="store_true",
        default=None,
        help="从 checkpoints/<实验名>/last.pth 断点续训（不传则以配置为准）",
    )
    train_parser.add_argument(
        "--init-from",
        type=str,
        default=None,
        metavar="FILE",
        help=(
            "两阶段微调：从指定 checkpoint 加载模型权重作为训练起点"
            "（只迁移权重，不迁移优化器状态；与 --resume 互斥）"
        ),
    )
    train_parser.set_defaults(func=cmd_train)

    # ---- experiments ----
    exp_parser = subparsers.add_parser(
        "experiments",
        help="实验矩阵：筛选 + 长训练",
        description=(
            "赛马制实验矩阵：先用少量 epoch 横向筛选全部候选实验并生成排行榜，"
            "再取 Top-K 进入完整长训练；筛选阶段的实验目录带 _screen 后缀。"
        ),
        epilog=(
            "示例:\n"
            "  uv run main.py experiments --matrix configs/experiment_matrix.yaml --stage screening\n"
            "  uv run main.py experiments --matrix configs/experiment_matrix.yaml --stage report\n"
            "  uv run main.py experiments --matrix configs/experiment_matrix.yaml --stage full\n"
            "  uv run main.py experiments --matrix configs/experiment_matrix.yaml --stage all\n"
            "排行榜: logs/leaderboard.csv"
        ),
        formatter_class=FORMATTER,
    )
    exp_parser.add_argument(
        "--matrix",
        type=str,
        required=True,
        metavar="FILE",
        help="实验矩阵 YAML 路径（含 screening / full / experiments 三节，必填）",
    )
    exp_parser.add_argument(
        "--stage",
        type=str,
        required=True,
        choices=["screening", "full", "report", "all"],
        help=(
            "screening=短实验筛选; full=Top-K 长训练; "
            "report=仅打印排行榜; all=两阶段连跑"
        ),
    )
    exp_parser.add_argument(
        "--device",
        type=str,
        default=None,
        metavar="DEV",
        help="覆盖运行设备（默认: 自动探测）",
    )
    exp_parser.add_argument(
        "--resume",
        action="store_true",
        default=False,
        help="断点续训：各作业从各自 last.pth 恢复进度，已完成实验不重跑",
    )
    exp_parser.set_defaults(func=cmd_experiments)

    # ---- infer ----
    infer_parser = subparsers.add_parser(
        "infer",
        help="测试集推理并生成提交结果",
        description=(
            "在测试集 DAPI 上生成目标标记图像，按比赛要求输出到 "
            "results/test/<MARKER>/<原名>_fake.jpg（三通道 JPG，全自动无人工后处理）。"
        ),
        epilog=(
            "权重查找约定:\n"
            "  单标记模型: checkpoints/<实验名>_<标记名小写>/best.pth\n"
            "  条件模型:   checkpoints/<实验名>/best.pth\n"
            "\n示例:\n"
            "  uv run main.py infer --config configs/baseline.yaml --exp exp001_unet_baseline\n"
            "  uv run main.py infer --config configs/conditional_v2.yaml --exp exp006_conditional_v2\n"
            "  uv run main.py infer --config configs/baseline.yaml --ckpt-CD68 checkpoints/exp001_unet_baseline_cd68/best.pth"
        ),
        formatter_class=FORMATTER,
    )
    infer_parser.add_argument(
        "--config",
        type=str,
        required=True,
        metavar="FILE",
        help="推理配置 YAML 路径（模型结构需与训练时一致，必填）",
    )
    infer_parser.add_argument(
        "--exp",
        type=str,
        default=None,
        metavar="NAME",
        help="实验名，按命名约定自动查找全部标记的 checkpoint",
    )
    infer_parser.add_argument(
        "--ckpt-all",
        type=str,
        default=None,
        metavar="FILE",
        help="多标记条件模型的统一 checkpoint（优先于 --exp）",
    )
    for marker in MARKERS:
        infer_parser.add_argument(
            f"--ckpt-{marker}",
            type=str,
            default=None,
            metavar="FILE",
            help=f"标记 {marker} 的单模型 checkpoint（优先于 --exp）",
        )
    infer_parser.set_defaults(func=cmd_infer)

    # ---- marker-report ----
    report_parser = subparsers.add_parser(
        "marker-report",
        help="逐标记短板分析",
        description=(
            "在同一验证集上分别评估四种标记（SSIM/PSNR/Score），"
            "按赛题「多输出取平均分」口径标出短板标记，"
            "供损失权重或采样策略调整参考。"
        ),
        epilog=(
            "示例:\n"
            "  uv run main.py marker-report --config configs/baseline.yaml --exp exp001_unet_baseline\n"
            "  uv run main.py marker-report --config configs/conditional_v2.yaml --exp exp006_conditional_v2"
        ),
        formatter_class=FORMATTER,
    )
    report_parser.add_argument(
        "--config",
        type=str,
        required=True,
        metavar="FILE",
        help="评估配置 YAML 路径（模型结构需与 checkpoint 一致，必填）",
    )
    report_parser.add_argument(
        "--exp",
        type=str,
        default=None,
        metavar="NAME",
        help="实验名，按命名约定自动查找全部标记的 checkpoint",
    )
    report_parser.add_argument(
        "--ckpt-all",
        type=str,
        default=None,
        metavar="FILE",
        help="多标记条件模型的统一 checkpoint（优先于 --exp）",
    )
    for marker in MARKERS:
        report_parser.add_argument(
            f"--ckpt-{marker}",
            type=str,
            default=None,
            metavar="FILE",
            help=f"标记 {marker} 的单模型 checkpoint（优先于 --exp）",
        )
    report_parser.set_defaults(func=cmd_marker_report)

    # ---- ensemble ----
    ensemble_parser = subparsers.add_parser(
        "ensemble",
        help="多实验预测结果集成",
        description=(
            "把多个实验对同一标记的预测结果逐像素取平均后输出，"
            "通常用于把筛选出的 Top-K 模型集成以提升 SSIM/PSNR 稳定性。"
        ),
        epilog=(
            "示例:\n"
            "  uv run main.py ensemble --config configs/adapter_unet.yaml \\\n"
            "      --exps exp007_adapter_unet_cd68 exp007_adapter_unet_cd45ro"
        ),
        formatter_class=FORMATTER,
    )
    ensemble_parser.add_argument(
        "--config",
        type=str,
        required=True,
        metavar="FILE",
        help="推理配置 YAML 路径（模型结构需与训练时一致，必填）",
    )
    ensemble_parser.add_argument(
        "--exps",
        type=str,
        nargs="+",
        required=True,
        metavar="NAME",
        help="参与集成的实验名，可多个（空格分隔）",
    )
    ensemble_parser.set_defaults(func=cmd_ensemble)

    return parser.parse_args()


def main() -> None:
    """解析指令并转发给对应子命令。

    Returns:
        None
    """
    args = parse_args()
    try:
        args.func(args)
    except (ValueError, FileNotFoundError, KeyError) as error:
        raise SystemExit(f"错误: {error}") from error


if __name__ == "__main__":
    main()
