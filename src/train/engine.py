"""训练与推理引擎。

- ``Trainer``：单配置实验的完整生命周期（数据 -> 模型 -> 训练 -> 验证 ->
  保存 checkpoint），支持 AMP 混合精度、EMA 权重滑动平均、torch.compile
  与两阶段微调（``training.init_from`` 加载已有权重作为初始化）；
- ``Predictor``：checkpoint 查找/加载与测试集批量推理，支持测试时增强
  （TTA，8 种几何等价视图平均），输出按比赛提交规范组织为
  ``results/test/<MARKER>/<原名>_fake.jpg``；
- ``MarkerEvaluator``：逐标记验证集评估，输出各标记的 SSIM/PSNR/Score，
  用于定位平均分口径下的短板标记。

兼容单标记模型与多标记条件模型（按模型类型决定是否传入 ``marker_idx``）。
"""

import time
import warnings
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np
import torch
import torch.nn as nn
from torch.optim import Optimizer
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader
from tqdm import tqdm

from ..data.constants import MARKERS, sanitize_marker_name
from ..utils import (
    CheckpointManager,
    ExperimentLogger,
    LoggerFactory,
    ModelEMA,
    Runtime,
)
from .data import DataLoaders
from .losses import CombinedLoss
from .metrics import AverageMeter, MetricAccumulator
from .models import ModelRegistry


class Trainer:
    """训练器：驱动单配置实验的完整生命周期。"""

    def __init__(self, config: Dict[str, Any]) -> None:
        """按配置初始化训练所需的全部组件。

        Args:
            config: 全局配置字典（已合并命令行覆盖项）。
        """
        self.config = config
        train_cfg = config.get("training", {})
        runtime_cfg = config.get("runtime", {})

        Runtime.seed_everything(int(runtime_cfg.get("seed", 42)))
        self.device = Runtime.get_device(runtime_cfg.get("device"))

        # ---- 计算加速配置 ----
        self.use_amp = bool(train_cfg.get("amp", False)) and self.device.type == "cuda"
        self.scaler = (
            torch.amp.GradScaler("cuda", enabled=self.use_amp)
            if self.device.type == "cuda"
            else None
        )
        if self.device.type == "cpu":
            num_threads = int(runtime_cfg.get("num_threads", 0))
            if num_threads > 0:
                torch.set_num_threads(num_threads)
        elif not bool(runtime_cfg.get("deterministic", True)):
            torch.backends.cudnn.deterministic = False
            torch.backends.cudnn.benchmark = True

        self.experiment_name = config.get("experiment", {}).get("name", "exp")
        self.log_dir = Path("logs") / self.experiment_name
        self.checkpoint_dir = Path("checkpoints") / self.experiment_name
        # logger 名称带上实验名：同一进程顺序训练多个作业时，避免复用首个
        # logger 的文件 handler，导致后续实验日志写入前一个实验的 train.log。
        self.logger = LoggerFactory.create(
            f"trainer.{self.experiment_name}", str(self.log_dir / "train.log")
        )

        self.epochs = int(train_cfg.get("epochs", 100))
        self.conditional = ModelRegistry.is_conditional(config)
        # 验证指标的 SSIM 口径：默认与训练损失一致；置 skimage 则与赛方
        # 评测（skimage 默认参数）完全同口径，选出的 best.pth 更贴近线上分。
        self.ssim_impl = str(
            config.get("evaluation", {}).get("ssim_impl", "gaussian")
        )

        # ---- 数据 ----
        loaders = DataLoaders.build(config)
        self.train_loader: DataLoader = loaders["train"]
        self.val_loader: DataLoader = loaders["val"]

        # ---- 模型 / 损失 / 优化器 ----
        # base_model: 原始模型，用于 checkpoint 与 EMA（避免 compile 的 _orig_mod 前缀）。
        # model:      实际前向用的模型，可能是 torch.compile 包装后的版本。
        self.base_model: nn.Module = ModelRegistry.build(config).to(self.device)

        self.compile_mode: Optional[str] = None
        if bool(train_cfg.get("compile", False)) and self.device.type == "cuda":
            self.compile_mode = str(train_cfg.get("compile_mode", "default"))
            self.logger.info(f"torch.compile 模式: {self.compile_mode}")
            # 屏蔽 Inductor 首轮内核调优刷屏与弃用警告，仅影响日志输出。
            from torch._inductor import config as inductor_config

            inductor_config.autotune_num_choices_displayed = 0
            inductor_config.max_autotune_report_choices_stats = False
            warnings.filterwarnings("ignore", message="TypedStorage is deprecated")
            self.model: nn.Module = torch.compile(
                self.base_model,
                mode=self.compile_mode,
                fullgraph=False,
                dynamic=False,
            )
        else:
            self.model = self.base_model

        self.criterion = CombinedLoss.from_config(config).to(self.device)
        self.optimizer: Optimizer = torch.optim.AdamW(
            self.base_model.parameters(),
            lr=float(train_cfg.get("lr", 1e-4)),
            weight_decay=float(train_cfg.get("weight_decay", 1e-4)),
        )

        # ---- 学习率调度器 ----
        self.scheduler_type = str(train_cfg.get("scheduler", "cosine"))
        if self.scheduler_type == "onecycle":
            self.scheduler = torch.optim.lr_scheduler.OneCycleLR(
                self.optimizer,
                max_lr=float(train_cfg.get("lr", 1e-4)),
                total_steps=self.epochs * len(self.train_loader),
                pct_start=0.3,
                div_factor=25.0,
                final_div_factor=1e4,
            )
        else:
            self.scheduler = CosineAnnealingLR(
                self.optimizer,
                T_max=self.epochs,
                eta_min=float(train_cfg.get("min_lr", 1e-6)),
            )

        # ---- 训练状态与断点续训 / 两阶段微调 ----
        self.best_score = 0.0
        self.start_epoch = 0
        self.resume = bool(train_cfg.get("resume", False))
        # init_from: 加载已有实验权重作为初始化（微调第二阶段用），
        # 与 resume 互斥——resume 恢复完整训练状态，优先级更高。
        self.init_from = str(train_cfg.get("init_from", "") or "")
        if self.resume:
            # 必须早于 EMA 初始化：让 EMA 阴影以续训后的权重为起点。
            self._resume_from_last()
        elif self.init_from:
            self._init_from_checkpoint()

        # ---- EMA ----
        self.use_ema = bool(train_cfg.get("ema", False))
        self.ema = (
            ModelEMA(self.base_model, decay=0.9999, warmup_steps=100)
            if self.use_ema
            else None
        )
        # EMA 每隔 N 个 epoch 才额外评估一次，验证集较大时可显著降低开销。
        self.ema_eval_every = max(1, int(train_cfg.get("ema_eval_every", 1)))

        # ---- 逐 epoch 记录器 ----
        self.recorder = ExperimentLogger(
            str(self.log_dir),
            fieldnames=[
                "epoch",
                "train_loss",
                "val_loss",
                "val_ssim",
                "val_psnr",
                "val_score",
                "lr",
            ],
            append=self.resume,
        )

        self.logger.info(
            f"实验 [{self.experiment_name}] 初始化完成 | "
            f"设备: {self.device} | 训练样本: {len(self.train_loader.dataset)} | "
            f"验证样本: {len(self.val_loader.dataset)}"
            + (" | AMP: 开" if self.use_amp else "")
            + (" | EMA: 开" if self.use_ema else "")
            + (f" | Compile: {self.compile_mode}" if self.compile_mode else "")
            + (f" | 续训自 epoch {self.start_epoch}" if self.start_epoch else "")
        )

    # ------------------------------------------------------------------ #
    # 断点续训
    # ------------------------------------------------------------------ #
    def _resume_from_last(self) -> None:
        """从 ``last.pth`` 恢复模型 / 优化器 / 调度器状态与训练进度。

        找不到 ``last.pth`` 时仅告警并从头训练，不中断流程。

        Returns:
            None
        """
        last_path = self.checkpoint_dir / "last.pth"
        if not last_path.is_file():
            self.logger.warning(f"续训已开启但未找到 {last_path}，将从头开始训练")
            return

        # 权重与优化器状态同设备加载，避免 CPU -> GPU 的额外拷贝。
        info = CheckpointManager.load(
            str(last_path),
            self.base_model,
            self.optimizer,
            map_location=str(self.device),
            scheduler=self.scheduler,
        )
        if not info["has_optimizer"]:
            # last.pth 被瘦身保存时无法精确恢复优化器动量，仍可续训但需知晓。
            self.logger.warning("last.pth 不含优化器状态，优化器动量将从零重新累计")

        self.start_epoch = int(info["epoch"])
        self.best_score = float(info["best_score"])
        self.logger.info(
            f"续训：已恢复 {self.start_epoch} 个 epoch，"
            f"历史最优 Score={self.best_score:.4f}"
        )

    def _init_from_checkpoint(self) -> None:
        """从已有 checkpoint 加载权重作为训练起点（两阶段微调）。

        只迁移模型权重、不迁移优化器状态与训练进度：微调阶段使用新的
        学习率与损失组合，优化器动量从零累计，best_score 从零重新竞争。

        Returns:
            None
        """
        info = CheckpointManager.load(
            self.init_from, self.base_model, map_location=str(self.device)
        )
        self.logger.info(
            f"微调初始化：已加载 {self.init_from} 的模型权重 "
            f"(源 epoch={info['epoch']}, 源 best_score={info['best_score']:.4f})"
        )

    # ------------------------------------------------------------------ #
    # 前向与验证
    # ------------------------------------------------------------------ #
    def _forward(
        self, batch: Dict[str, Any]
    ) -> Tuple[torch.Tensor, Optional[Dict[str, torch.Tensor]]]:
        """根据模型类型执行前向传播。

        Args:
            batch: DataLoader 输出的批次字典。

        Returns:
            Tuple[torch.Tensor, Optional[Dict[str, torch.Tensor]]]:
                (预测, 辅助信息)。模型返回元组时打包进 aux 供损失使用。
        """
        inputs = batch["input"].to(self.device, non_blocking=True)
        if not self.conditional:
            return self.model(inputs), None

        marker_idx = batch["marker_idx"].to(self.device, non_blocking=True)
        output = self.model(inputs, marker_idx)
        if isinstance(output, tuple):
            return output[0], {"shared_features": output[1], "marker_idx": marker_idx}
        return output, None

    def train_one_epoch(self, epoch: int) -> float:
        """训练一个 epoch。

        Args:
            epoch: 当前轮次（从 0 开始）。

        Returns:
            float: 本 epoch 的训练平均损失。
        """
        self.model.train()
        loss_meter = AverageMeter()

        progress = tqdm(
            self.train_loader,
            desc=f"Epoch {epoch + 1}/{self.epochs} [train]",
            leave=False,
        )
        for batch in progress:
            targets = batch["target"].to(self.device, non_blocking=True)

            self.optimizer.zero_grad(set_to_none=True)
            with torch.autocast(
                device_type=self.device.type, dtype=torch.float16, enabled=self.use_amp
            ):
                predictions, aux = self._forward(batch)

            # 损失计算放到 autocast 之外并强制 fp32：
            # SSIM 中的相近数相减在 fp16 下会灾难性抵消，导致 loss=NaN。
            loss, _details = self.criterion(predictions.float(), targets, aux=aux)

            if self.scaler is not None:
                self.scaler.scale(loss).backward()
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                loss.backward()
                self.optimizer.step()

            # EMA 更新必须基于 base_model（与 compile 包装无关）。
            if self.ema is not None:
                self.ema.update(self.base_model)
            if self.scheduler_type == "onecycle":
                self.scheduler.step()

            loss_meter.update(float(loss.detach()), predictions.shape[0])
            progress.set_postfix(loss=f"{loss_meter.avg:.4f}")
        return loss_meter.avg

    def validate(self, use_ema: bool = False) -> Dict[str, float]:
        """在验证集上评估模型，可选择临时切换为 EMA 权重。

        Args:
            use_ema: 是否用 EMA 阴影权重评估（评估后自动恢复训练权重）。

        Returns:
            Dict[str, float]: 含 ``loss`` / ``ssim`` / ``psnr`` / ``score``。
        """
        if use_ema and self.ema is not None:
            # 阴影权重必须施加到 base_model：compile 包装后的模型参数名带
            # _orig_mod. 前缀，与 EMA 的 shadow 键（基于 base_model 建立）不匹配，
            # 会导致切换静默失效。base_model 与 compile 版本共享同一批参数张量，
            # 原地写入即可被前向传播感知。
            self.ema.apply_shadow(self.base_model)
            metrics = self._validate_impl()
            self.ema.restore(self.base_model)
            return metrics
        return self._validate_impl()

    @torch.no_grad()
    def _validate_impl(self) -> Dict[str, float]:
        """验证实现：遍历验证集累计指标（不用 tqdm，避免 CPU 阻塞 GPU）。

        Returns:
            Dict[str, float]: 含 ``loss`` / ``ssim`` / ``psnr`` / ``score``。
        """
        self.model.eval()
        loss_meter = AverageMeter()
        # SSIM 口径可选：gaussian（训练同款，快）/ skimage（赛方口径，用于选模）。
        accumulator = MetricAccumulator(ssim_impl=self.ssim_impl)

        for batch in self.val_loader:
            targets = batch["target"].to(self.device, non_blocking=True)
            with torch.autocast(
                device_type=self.device.type, dtype=torch.float16, enabled=self.use_amp
            ):
                predictions, aux = self._forward(batch)

            # 与训练一致：损失在 fp32 下计算，避免 fp16 数值不稳定。
            loss, _details = self.criterion(predictions.float(), targets, aux=aux)
            loss_meter.update(float(loss), predictions.shape[0])
            accumulator.update(predictions.float(), targets.float())

        metrics = accumulator.compute()
        metrics["loss"] = loss_meter.avg
        return metrics

    # ------------------------------------------------------------------ #
    # 主流程
    # ------------------------------------------------------------------ #
    def fit(self) -> float:
        """执行完整训练流程，逐 epoch 训练、验证并保存最优模型。

        Returns:
            float: 训练过程中的最优验证综合得分。
        """
        if self.start_epoch > 0:
            self.logger.info(
                f"续训开始：从第 {self.start_epoch + 1} 个 epoch 继续，"
                f"共 {self.epochs} 个 epoch"
            )
        else:
            self.logger.info(f"开始训练，共 {self.epochs} 个 epoch")
        start_time = time.time()

        for epoch in range(self.start_epoch, self.epochs):
            train_loss = self.train_one_epoch(epoch)

            # 每个 epoch 验证原始权重；EMA 按 ema_eval_every 间隔评估。
            val_metrics = self.validate(use_ema=False)
            weight_source = "raw"
            should_eval_ema = self.ema is not None and (
                (epoch + 1) % self.ema_eval_every == 0 or epoch + 1 == self.epochs
            )
            if should_eval_ema:
                ema_metrics = self.validate(use_ema=True)
                if ema_metrics["score"] > val_metrics["score"]:
                    val_metrics = ema_metrics
                    weight_source = "ema"

            if self.scheduler_type != "onecycle":
                self.scheduler.step()

            self.recorder.log(
                {
                    "epoch": epoch + 1,
                    "train_loss": f"{train_loss:.6f}",
                    "val_loss": f"{val_metrics['loss']:.6f}",
                    "val_ssim": f"{val_metrics['ssim']:.6f}",
                    "val_psnr": f"{val_metrics['psnr']:.4f}",
                    "val_score": f"{val_metrics['score']:.6f}",
                    "lr": f"{self.optimizer.param_groups[0]['lr']:.2e}",
                }
            )
            self.logger.info(
                f"Epoch {epoch + 1}/{self.epochs} | "
                f"train_loss={train_loss:.4f} | val_loss={val_metrics['loss']:.4f} | "
                f"SSIM={val_metrics['ssim']:.4f} | PSNR={val_metrics['psnr']:.2f} | "
                f"Score={val_metrics['score']:.4f} | 权重来源={weight_source}"
            )

            if val_metrics["score"] > self.best_score:
                self.best_score = val_metrics["score"]
                self._save_best(epoch, weight_source)

            # last.pth 保存完整训练状态（含优化器/调度器），供断点续训。
            # 必须在 best_score 更新之后保存，续训时才能读到最新的历史最优分，
            # 否则续训会把此前更优的 best.pth 覆盖成较差的权重。
            CheckpointManager.save(
                self.base_model,
                self.optimizer,
                epoch + 1,
                self.best_score,
                self.config,
                str(self.checkpoint_dir / "last.pth"),
                scheduler=self.scheduler,
            )

        self.recorder.finish()
        self.logger.info(
            f"训练完成，耗时 {(time.time() - start_time) / 60:.1f} 分钟，"
            f"最优 Score={self.best_score:.4f}"
        )
        return self.best_score

    def _save_best(self, epoch: int, weight_source: str) -> None:
        """保存最优 checkpoint，EMA 更优时保存 EMA 阴影权重。

        Args:
            epoch:         当前轮次（从 0 开始）。
            weight_source: 最优权重来源，``"raw"`` 或 ``"ema"``。

        Returns:
            None
        """
        restore_after = weight_source == "ema" and self.ema is not None
        if restore_after:
            # 同上：EMA 阴影施加/还原统一作用于 base_model。
            self.ema.apply_shadow(self.base_model)

        CheckpointManager.save(
            self.base_model,
            None,
            epoch + 1,
            self.best_score,
            self.config,
            str(self.checkpoint_dir / "best.pth"),
            # 推理只需权重，瘦身保存（不含优化器状态），体积约为 1/3。
            include_optimizer=False,
        )

        if restore_after:
            self.ema.restore(self.base_model)
        self.logger.info(
            f"  -> 新的最优模型 (Score={self.best_score:.4f}, 来源={weight_source})"
        )


class Predictor:
    """预测器：checkpoint 查找/加载与测试集结果生成。"""

    def __init__(self, config: Dict[str, Any]) -> None:
        """初始化设备、模型与测试集加载器。

        Args:
            config: 全局配置字典。
        """
        self.config = config
        self.device = Runtime.get_device(config.get("runtime", {}).get("device"))
        self.logger = LoggerFactory.create("predictor")
        self.conditional = ModelRegistry.is_conditional(config)

        self.model: nn.Module = ModelRegistry.build(config).to(self.device)
        self.test_loader = DataLoaders.for_test(config)

        inference_cfg = config.get("inference", {})
        self.output_root = str(inference_cfg.get("output_dir", "results"))
        self.suffix = str(inference_cfg.get("suffix", "_fake"))
        # TTA（测试时增强）：对 4 旋转 × 2 翻转共 8 种几何等价视图分别推理
        # 并逐像素平均。病理图像无方向先验，训练增强已覆盖同组变换，
        # 因此 TTA 可稳定降低预测方差；属模型推理环节，全自动、
        # 不涉及赛题禁止的人工作后处理。
        self.tta = bool(inference_cfg.get("tta", False))

    # ------------------------------------------------------------------ #
    # 测试时增强（TTA）
    # ------------------------------------------------------------------ #
    @staticmethod
    def _tta_variants(inputs: torch.Tensor) -> List[Tuple[int, bool, torch.Tensor]]:
        """生成输入的 8 种几何等价视图（4 种 90° 旋转 × 是否水平翻转）。

        Args:
            inputs: 输入张量 ``(B, C, H, W)``。

        Returns:
            List[Tuple[int, bool, torch.Tensor]]: ``(旋转次数k, 是否翻转, 视图)``。
        """
        variants: List[Tuple[int, bool, torch.Tensor]] = []
        for k in range(4):
            rotated = torch.rot90(inputs, k, dims=(-2, -1))
            variants.append((k, False, rotated))
            variants.append((k, True, torch.flip(rotated, dims=(-1,))))
        return variants

    @staticmethod
    def _tta_invert(prediction: torch.Tensor, k: int, flip: bool) -> torch.Tensor:
        """把某视图上的预测逆变换回原图方向（翻转与旋转的逆操作）。

        Args:
            prediction: 视图上的预测 ``(B, C, H, W)``。
            k:          视图生成时的旋转次数。
            flip:       视图生成时是否水平翻转。

        Returns:
            torch.Tensor: 方向还原后的预测。
        """
        if flip:
            prediction = torch.flip(prediction, dims=(-1,))
        if k:
            prediction = torch.rot90(prediction, -k, dims=(-2, -1))
        return prediction

    @staticmethod
    def tta_forward(
        model: nn.Module,
        inputs: torch.Tensor,
        marker_idx: Optional[torch.Tensor] = None,
        enabled: bool = True,
    ) -> torch.Tensor:
        """执行（可选 TTA 的）前向推理，返回逐像素平均后的预测。

        TTA 关闭时退化为单次前向，与原有行为一致。供 ``Predictor`` 与
        ``Ensembler`` 共用，避免两处维护同一套变换逻辑。

        Args:
            model:      已加载权重的模型（评估模式）。
            inputs:     输入张量 ``(B, C, H, W)``。
            marker_idx: 条件模型的标记编号张量；单标记模型传 ``None``。
            enabled:    是否启用 TTA。

        Returns:
            torch.Tensor: 预测图像 ``(B, C, H, W)``。
        """
        if not enabled:
            output = model(inputs, marker_idx) if marker_idx is not None else model(inputs)
            return Predictor.unpack(output)

        total: Optional[torch.Tensor] = None
        count = 0
        for k, flip, view in Predictor._tta_variants(inputs):
            output = model(view, marker_idx) if marker_idx is not None else model(view)
            prediction = Predictor._tta_invert(Predictor.unpack(output), k, flip)
            total = prediction if total is None else total + prediction
            count += 1
        assert total is not None and count > 0
        return total / count

    # ------------------------------------------------------------------ #
    # 静态工具
    # ------------------------------------------------------------------ #
    @staticmethod
    def unpack(output: Any) -> torch.Tensor:
        """解包模型输出，兼容返回 (预测, 共享特征) 元组的模型。

        Args:
            output: 模型前向输出，张量或元组。

        Returns:
            torch.Tensor: 预测图像 ``(B, C, H, W)``。
        """
        return output[0] if isinstance(output, tuple) else output

    @staticmethod
    def save_batch(
        predictions: torch.Tensor,
        names: List[str],
        marker: str,
        output_root: str = "results",
        suffix: str = "_fake",
    ) -> None:
        """把一个 batch 的预测按比赛命名规范保存为 JPG。

        输出路径为 ``<output_root>/test/<MARKER>/<原名><suffix>.jpg``。

        Args:
            predictions: 预测张量 ``(B, C, H, W)``，取值 ``[0, 1]``。
            names:       每个样本的文件名（不含后缀）。
            marker:      目标标记名，决定输出子目录。
            output_root: 结果根目录。
            suffix:      输出文件名后缀。

        Returns:
            None
        """
        output_dir = Path(output_root) / "test" / marker
        output_dir.mkdir(parents=True, exist_ok=True)

        for prediction, name in zip(predictions, names):
            array = prediction.detach().cpu().clamp(0.0, 1.0).numpy()
            array = np.transpose(array, (1, 2, 0))  # CHW -> HWC
            array = (array * 255.0).round().astype(np.uint8)

            # 统一输出三通道 JPG：目标真值为灰度强度图但以 3 通道 JPG 存储
            # （三通道完全相同）。提交三通道可同时兼容按灰度或按 RGB 读取的
            # 评测脚本，且 SSIM/PSNR 数值与单通道完全一致。
            if array.shape[-1] == 1:
                array = np.repeat(array, 3, axis=-1)
            # JPEG 编码口径：quality=100 + 4:4:4 不子采样。
            # 默认 quality=95 且 4:2:0 色度子采样会让 R/G/B 三通道解码后不再
            # 相等，评测脚本若按灰度加权读取会引入额外噪声，直接损失 PSNR。
            cv2.imwrite(
                str(output_dir / f"{name}{suffix}.jpg"),
                cv2.cvtColor(array, cv2.COLOR_RGB2BGR),
                [
                    cv2.IMWRITE_JPEG_QUALITY,
                    100,
                    cv2.IMWRITE_JPEG_SAMPLING_FACTOR,
                    cv2.IMWRITE_JPEG_SAMPLING_FACTOR_444,
                ],
            )

    @staticmethod
    def find_checkpoints(
        experiment: str, conditional: bool, checkpoint_root: str = "checkpoints"
    ) -> Dict[str, str]:
        """按命名约定自动收集实验对应的全部 checkpoint。

        权重查找约定（与训练命名规则一致）：

        - 单标记模型: ``checkpoints/<实验名>_<标记名小写>/best.pth``
        - 条件模型:   ``checkpoints/<实验名>/best.pth``

        Args:
            experiment:      实验名（训练配置中的 ``experiment.name``）。
            conditional:     是否为多标记条件模型。
            checkpoint_root: checkpoint 根目录。

        Returns:
            Dict[str, str]: ``{标记名: checkpoint路径}``；条件模型用键 ``"all"``。

        Raises:
            FileNotFoundError: 任一必需的 checkpoint 不存在时抛出。
        """
        root = Path(checkpoint_root)
        if conditional:
            path = root / experiment / "best.pth"
            if not path.is_file():
                raise FileNotFoundError(f"未找到条件模型 checkpoint: {path}")
            return {"all": str(path)}

        checkpoints: Dict[str, str] = {}
        missing: List[str] = []
        for marker in MARKERS:
            path = root / f"{experiment}_{sanitize_marker_name(marker)}" / "best.pth"
            if path.is_file():
                checkpoints[marker] = str(path)
            else:
                missing.append(str(path))

        if missing:
            raise FileNotFoundError(
                "以下标记的 checkpoint 缺失，请先完成对应训练:\n  "
                + "\n  ".join(missing)
            )
        return checkpoints

    @classmethod
    def resolve_checkpoints(
        cls,
        experiment: Optional[str],
        conditional: bool,
        overrides: Optional[Dict[str, str]] = None,
        checkpoint_root: str = "checkpoints",
    ) -> Dict[str, str]:
        """汇总推理所需的 checkpoint 路径。

        ``overrides`` 中手动指定的路径优先于按实验名自动查找的结果，
        因此可与 ``experiment`` 混用作为补充或覆盖。

        Args:
            experiment:      实验名；``None`` 表示不按实验名自动查找。
            conditional:     是否为多标记条件模型。
            overrides:       ``{标记名: checkpoint路径}`` 手动覆盖项。
            checkpoint_root: checkpoint 根目录。

        Returns:
            Dict[str, str]: ``{标记名: checkpoint路径}``；条件模型用键 ``"all"``。

        Raises:
            ValueError: 既没有实验名也没有任何手动覆盖项时抛出。
        """
        checkpoint_paths: Dict[str, str] = {}
        if experiment is not None:
            checkpoint_paths = cls.find_checkpoints(
                experiment, conditional, checkpoint_root
            )
        checkpoint_paths.update(overrides or {})

        if not checkpoint_paths:
            raise ValueError("请提供实验名或至少一个 checkpoint 覆盖项")
        return checkpoint_paths

    @classmethod
    def run_experiment(
        cls,
        config: Dict[str, Any],
        experiment: Optional[str] = None,
        overrides: Optional[Dict[str, str]] = None,
        checkpoint_root: str = "checkpoints",
    ) -> None:
        """推理入口：汇总 checkpoint 并在测试集上批量生成提交结果。

        Args:
            config:          全局配置字典。
            experiment:      实验名，按命名约定自动查找全部标记的权重。
            overrides:       ``{标记名: checkpoint路径}`` 手动覆盖项。
            checkpoint_root: checkpoint 根目录。

        Returns:
            None
        """
        checkpoint_paths = cls.resolve_checkpoints(
            experiment, ModelRegistry.is_conditional(config), overrides, checkpoint_root
        )
        cls(config).run(checkpoint_paths)

    # ------------------------------------------------------------------ #
    # 推理流程
    # ------------------------------------------------------------------ #
    def _load_weights(self, checkpoint_path: str) -> None:
        """加载模型权重并切换为评估模式。

        Args:
            checkpoint_path: checkpoint 文件路径。

        Returns:
            None
        """
        info = CheckpointManager.load(checkpoint_path, self.model, map_location="cpu")
        self.model.to(self.device).eval()
        self.logger.info(
            f"已加载 {checkpoint_path} (epoch={info['epoch']}, "
            f"best_score={info['best_score']:.4f})"
        )

    @torch.no_grad()
    def run_single_marker(self, marker: str, checkpoint_path: str) -> None:
        """用单标记模型生成一种标记的全部测试结果。

        Args:
            marker:          目标标记名。
            checkpoint_path: 该标记对应的模型 checkpoint。

        Returns:
            None
        """
        self._load_weights(checkpoint_path)
        for batch in tqdm(self.test_loader, desc=f"infer [{marker}]"):
            predictions = self.tta_forward(
                self.model, batch["input"].to(self.device), enabled=self.tta
            )
            self.save_batch(
                predictions, batch["name"], marker, self.output_root, self.suffix
            )

    @torch.no_grad()
    def run_multi_marker(self, checkpoint_path: str) -> None:
        """用多标记条件模型一次性生成全部四种标记的结果。

        推理前向统一走 ``tta_forward``：与单标记推理、逐标记评估及集成保持
        同一口径，TTA 开启时对 8 种几何等价视图取平均，避免条件模型静默
        退化为单次前向。

        Args:
            checkpoint_path: 条件模型 checkpoint。

        Returns:
            None
        """
        self._load_weights(checkpoint_path)
        for marker_idx, marker in enumerate(MARKERS):
            for batch in tqdm(self.test_loader, desc=f"infer [{marker}]"):
                inputs = batch["input"].to(self.device)
                idx_tensor = torch.full(
                    (inputs.shape[0],), marker_idx, dtype=torch.long, device=self.device
                )
                # 条件模型按标记编号传入 marker_idx；TTA 开关由配置决定。
                predictions = self.tta_forward(
                    self.model, inputs, idx_tensor, enabled=self.tta
                )
                self.save_batch(
                    predictions, batch["name"], marker, self.output_root, self.suffix
                )

    def run(self, checkpoint_paths: Dict[str, str]) -> None:
        """推理分发：按模型类型走单标记或多标记流程。

        Args:
            checkpoint_paths: ``{标记名: checkpoint路径}``；
                多标记模式约定使用键 ``"all"``。

        Returns:
            None

        Raises:
            KeyError: 缺少所需 checkpoint 键时抛出。
        """
        if self.conditional:
            if "all" not in checkpoint_paths:
                raise KeyError("多标记模式需要提供 {'all': checkpoint路径}")
            self.run_multi_marker(checkpoint_paths["all"])
        else:
            for marker in MARKERS:
                if marker not in checkpoint_paths:
                    raise KeyError(f"缺少标记 {marker} 的 checkpoint 路径")
                self.run_single_marker(marker, checkpoint_paths[marker])

        self.logger.info(f"推理完成，结果保存于 {self.output_root}/test/")


class MarkerEvaluator:
    """逐标记评估器：在同一验证集上分别评估四种标记，定位短板标记。

    赛题对多输出取平均分，因此最差标记的边际收益最大（HEMIT 论文亦指出
    模型易系统性忽视弱势标记）。本类对每个标记独立累计指标，输出
    SSIM/PSNR/Score 对照表并标出短板，供损失权重或采样策略调整参考。
    """

    def __init__(self, config: Dict[str, Any]) -> None:
        """初始化设备、模型骨架与推理设置。

        Args:
            config: 全局配置字典（模型结构需与待评估 checkpoint 一致）。
        """
        self.config = config
        self.device = Runtime.get_device(config.get("runtime", {}).get("device"))
        self.logger = LoggerFactory.create("marker_evaluator")
        self.conditional = ModelRegistry.is_conditional(config)
        self.tta = bool(config.get("inference", {}).get("tta", False))
        # 与 Trainer 一致的 SSIM 口径开关，保证逐标记短板分析同样可对齐赛方。
        self.ssim_impl = str(
            config.get("evaluation", {}).get("ssim_impl", "gaussian")
        )

    @torch.no_grad()
    def _evaluate_loader(
        self,
        model: nn.Module,
        loader: DataLoader,
        marker_idx: Optional[int] = None,
        desc: str = "eval",
    ) -> Dict[str, float]:
        """在一个标记的验证集上累计指标。

        Args:
            model:      已加载权重的模型。
            loader:     该标记的验证集 DataLoader。
            marker_idx: 条件模型的标记编号；单标记模型传 ``None``。
            desc:       进度条描述文字，用于区分当前评估的标记。

        Returns:
            Dict[str, float]: 含 ``ssim`` / ``psnr`` / ``score``。
        """
        model.eval()
        accumulator = MetricAccumulator(ssim_impl=self.ssim_impl)
        # 全量验证集叠加 TTA 后单个标记即需一分多钟，必须给出进度反馈，
        # 否则终端长时间静默会被误判为进程卡死。
        for batch in tqdm(loader, desc=desc):
            inputs = batch["input"].to(self.device, non_blocking=True)
            targets = batch["target"].to(self.device, non_blocking=True)
            idx_tensor = (
                torch.full(
                    (inputs.shape[0],), marker_idx, dtype=torch.long, device=self.device
                )
                if marker_idx is not None
                else None
            )
            predictions = Predictor.tta_forward(
                model, inputs, idx_tensor, enabled=self.tta
            )
            accumulator.update(predictions.float(), targets)
        return accumulator.compute()

    def evaluate(
        self,
        experiment: Optional[str] = None,
        overrides: Optional[Dict[str, str]] = None,
        checkpoint_root: str = "checkpoints",
    ) -> Dict[str, Dict[str, float]]:
        """逐标记评估指定实验的全部四种标记。

        权重查找复用 ``Predictor.resolve_checkpoints`` 的约定：单标记模型
        每个标记各自加载一次权重；条件模型只加载一次，逐标记切换条件。

        Args:
            experiment:      实验名，按命名约定自动查找 checkpoint。
            overrides:       ``{标记名: checkpoint路径}`` 手动覆盖项。
            checkpoint_root: checkpoint 根目录。

        Returns:
            Dict[str, Dict[str, float]]: ``{标记名: {ssim, psnr, score}}``。
        """
        checkpoint_paths = Predictor.resolve_checkpoints(
            experiment, self.conditional, overrides, checkpoint_root
        )
        # 条件模型四个标记共享一份权重，只实例化/加载一次。
        shared_model: Optional[nn.Module] = None
        report: Dict[str, Dict[str, float]] = {}

        for marker_idx, marker in enumerate(MARKERS):
            if self.conditional:
                if shared_model is None:
                    shared_model = ModelRegistry.build(self.config).to(self.device)
                    CheckpointManager.load(
                        checkpoint_paths["all"], shared_model, map_location="cpu"
                    )
                    shared_model.to(self.device).eval()
                model = shared_model
                idx: Optional[int] = marker_idx
            else:
                model = ModelRegistry.build(self.config).to(self.device)
                CheckpointManager.load(checkpoint_paths[marker], model, map_location="cpu")
                model.to(self.device).eval()
                idx = None

            loader = DataLoaders.for_validation_marker(self.config, marker)
            report[marker] = self._evaluate_loader(
                model, loader, marker_idx=idx, desc=f"eval [{marker}]"
            )
            self.logger.info(
                f"[{marker}] SSIM={report[marker]['ssim']:.4f} | "
                f"PSNR={report[marker]['psnr']:.2f} | Score={report[marker]['score']:.4f}"
            )

        # 释放模型占用的显存/内存。
        del shared_model
        return report

    def report(
        self,
        experiment: Optional[str] = None,
        overrides: Optional[Dict[str, str]] = None,
        checkpoint_root: str = "checkpoints",
    ) -> Dict[str, Dict[str, float]]:
        """执行逐标记评估并打印对照表，末尾标出短板标记与平均分。

        Args:
            experiment:      实验名。
            overrides:       checkpoint 手动覆盖项。
            checkpoint_root: checkpoint 根目录。

        Returns:
            Dict[str, Dict[str, float]]: 各标记指标明细。
        """
        report = self.evaluate(experiment, overrides, checkpoint_root)

        # 赛题口径：多输出取平均分。
        mean_score = sum(item["score"] for item in report.values()) / len(report)
        worst = min(report, key=lambda marker: report[marker]["score"])

        print(f"\n{'标记':<12}{'SSIM':<10}{'PSNR(dB)':<12}{'Score':<10}")
        print("-" * 44)
        for marker, item in report.items():
            flag = "  <- 短板" if marker == worst else ""
            print(
                f"{marker:<12}{item['ssim']:<10.4f}{item['psnr']:<12.2f}"
                f"{item['score']:<10.4f}{flag}"
            )
        print("-" * 44)
        print(f"四标记平均 Score: {mean_score:.4f}")
        print(f"短板标记: {worst}（优先为其调整损失权重或采样比例）")
        return report
