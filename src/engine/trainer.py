"""训练引擎：封装完整的训练-验证-保存流程。

对应 plan.md 第 5 节流程::

    读取配置 -> 创建 Dataset -> 创建 Model -> 训练 -> Validation -> 保存 checkpoint

兼容单标记模型与多标记条件模型（自动按模型类型决定是否传入 marker_idx）。
"""

import time
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import torch
import torch.nn as nn
from torch.optim import Optimizer
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader
from tqdm import tqdm

from ..datasets import build_dataloaders
from ..losses import build_loss
from ..metrics import MetricAccumulator
from ..models import build_model, is_conditional_model
from ..utils import (
    AverageMeter,
    ExperimentLogger,
    get_device,
    save_checkpoint,
    seed_everything,
    setup_logger,
)
from ..utils.ema import ModelEMA


class Trainer:
    """训练器：驱动单配置实验的完整生命周期。"""

    def __init__(self, config: Dict[str, Any]) -> None:
        """按配置初始化训练所需的全部组件。

        初始化顺序遵循依赖关系：设备 -> 数据 -> 模型 -> 损失 ->
        优化器 -> 调度器 -> EMA，保证后构建的组件总能引用到先构建的。

        Args:
            config: 全局配置字典（已合并命令行覆盖项）。
        """
        self.config = config
        train_cfg = config.get("training", {})
        runtime_cfg = config.get("runtime", {})

        seed_everything(int(runtime_cfg.get("seed", 42)))
        self.device = get_device(runtime_cfg.get("device"))

        # ---- 计算加速配置 ----
        # 混合精度（仅 CUDA 生效），通过 training.amp 开关。
        self.use_amp = bool(train_cfg.get("amp", False)) and self.device.type == "cuda"
        self.scaler = (
            torch.amp.GradScaler("cuda", enabled=self.use_amp)
            if self.device.type == "cuda"
            else None
        )
        if self.device.type == "cpu":
            # CPU 训练时可按需调整线程数；未显式配置（<=0）时保留 PyTorch 默认，
            # 避免盲目用满全部核导致 OpenMP 线程竞争反而变慢。
            num_threads = int(runtime_cfg.get("num_threads", 0))
            if num_threads > 0:
                torch.set_num_threads(num_threads)
        elif not bool(runtime_cfg.get("deterministic", True)):
            # 关闭确定性卷积以启用 cuDNN 自动调优（仅当用户显式关闭确定性）。
            torch.backends.cudnn.deterministic = False
            torch.backends.cudnn.benchmark = True

        self.experiment_name = config.get("experiment", {}).get("name", "exp")
        self.log_dir = Path("logs") / self.experiment_name
        self.checkpoint_dir = Path("checkpoints") / self.experiment_name
        self.logger = setup_logger(
            "trainer", str(self.log_dir / "train.log")
        )

        self.epochs = int(train_cfg.get("epochs", 100))
        self.conditional = is_conditional_model(config)

        # ---- 数据（先构建，OneCycleLR 需要 train_loader 的长度） ----
        loaders = build_dataloaders(config)
        self.train_loader: DataLoader = loaders["train"]
        self.val_loader: DataLoader = loaders["val"]

        # ---- 模型 / 损失 / 优化器 ----
        # base_model 始终为原始模型；启用 torch.compile 时 self.model
        # 为编译后的包装，二者共享参数。checkpoint 与 EMA 一律作用于
        # base_model，避免 state_dict 键带 _orig_mod 前缀导致推理加载失败。
        self.base_model: nn.Module = build_model(config).to(self.device)
        if bool(train_cfg.get("compile", False)) and self.device.type == "cuda":
            self.model: nn.Module = torch.compile(self.base_model)
        else:
            self.model = self.base_model
        self.criterion = build_loss(config).to(self.device)
        self.optimizer: Optimizer = torch.optim.AdamW(
            self.base_model.parameters(),
            lr=float(train_cfg.get("lr", 1e-4)),
            weight_decay=float(train_cfg.get("weight_decay", 1e-4)),
        )

        # ---- 学习率调度器 ----
        # cosine 按 epoch 步进；onecycle 按 batch 步进（在 train_one_epoch 内）。
        self.scheduler_type = str(train_cfg.get("scheduler", "cosine"))
        if self.scheduler_type == "onecycle":
            self.scheduler = torch.optim.lr_scheduler.OneCycleLR(
                self.optimizer,
                max_lr=float(train_cfg.get("lr", 1e-4)),
                total_steps=self.epochs * len(self.train_loader),
                pct_start=0.3,  # 30% 时间用于 warmup
                div_factor=25.0,
                final_div_factor=1e4,
            )
        else:
            self.scheduler = CosineAnnealingLR(
                self.optimizer,
                T_max=self.epochs,
                eta_min=float(train_cfg.get("min_lr", 1e-6)),
            )

        # ---- EMA（依赖已创建的模型，须在模型之后初始化） ----
        self.use_ema = bool(train_cfg.get("ema", False))
        self.ema = (
            ModelEMA(self.base_model, decay=0.9999, warmup_steps=100)
            if self.use_ema
            else None
        )
        # EMA 权重每隔 N 个 epoch 才额外评估一次（默认 1 = 每轮都评），
        # 验证集较大时可显著降低验证开销；最后一轮必定评估。
        self.ema_eval_every = max(1, int(train_cfg.get("ema_eval_every", 1)))

        # ---- 训练状态 ----
        self.start_epoch = 0
        self.best_score = 0.0
        self.recorder = ExperimentLogger(
            str(self.log_dir),
            fieldnames=[
                "epoch", "train_loss", "val_loss",
                "val_ssim", "val_psnr", "val_score", "lr",
            ],
        )

        self.logger.info(
            f"实验 [{self.experiment_name}] 初始化完成 | "
            f"设备: {self.device} | 训练样本: {len(self.train_loader.dataset)} | "
            f"验证样本: {len(self.val_loader.dataset)}"
            + (" | AMP: 开" if self.use_amp else "")
            + (" | EMA: 开" if self.use_ema else "")
        )

    # ------------------------------------------------------------------ #
    # 单步前向：统一处理单标记 / 多标记条件两种模型
    # ------------------------------------------------------------------ #
    def _forward(
        self, batch: Dict[str, Any]
    ) -> Tuple[torch.Tensor, Optional[Dict[str, torch.Tensor]]]:
        """根据模型类型执行前向传播。

        Args:
            batch: DataLoader 输出的批次字典。

        Returns:
            Tuple[torch.Tensor, Optional[Dict[str, torch.Tensor]]]:
                (预测 ``(B, C, H, W)``, 辅助信息)。模型返回元组时
                （如 AdapterUNet 的共享特征）打包进 ``aux`` 供损失使用。
        """
        inputs = batch["input"].to(self.device, non_blocking=True)
        if self.conditional:
            marker_idx = batch["marker_idx"].to(self.device, non_blocking=True)
            output = self.model(inputs, marker_idx)
            # 模型返回 (预测, 共享特征) 时，附加 marker_idx 供跨标记损失使用。
            if isinstance(output, tuple):
                predictions, shared = output
                return predictions, {
                    "shared_features": shared,
                    "marker_idx": marker_idx,
                }
            return output, None
        return self.model(inputs), None

    def train_one_epoch(self, epoch: int) -> float:
        """训练一个 epoch。

        Args:
            epoch: 当前轮次（从 0 开始），仅用于日志展示。

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

            # EMA 必须在每个优化 step 后更新，与是否启用 AMP 无关。
            if self.ema is not None:
                self.ema.update(self.base_model)

            # OneCycleLR 按 batch 步进；cosine 在 fit() 中按 epoch 步进。
            if self.scheduler_type == "onecycle":
                self.scheduler.step()

            loss_meter.update(float(loss.detach()), predictions.shape[0])
            progress.set_postfix(loss=f"{loss_meter.avg:.4f}")

        return loss_meter.avg

    # ------------------------------------------------------------------ #
    # 验证：支持原始权重与 EMA 权重两种评估
    # ------------------------------------------------------------------ #
    def validate(self, use_ema: bool = False) -> Dict[str, float]:
        """在验证集上评估模型，可选择临时切换为 EMA 权重。

        Args:
            use_ema: 是否用 EMA 阴影权重评估（评估后自动恢复训练权重）。

        Returns:
            Dict[str, float]: 含 ``loss`` / ``ssim`` / ``psnr`` / ``score``。
        """
        if use_ema and self.ema is not None:
            self.ema.apply_shadow(self.model)
            metrics = self._validate_impl()
            self.ema.restore(self.model)
            return metrics
        return self._validate_impl()

    @torch.no_grad()
    def _validate_impl(self) -> Dict[str, float]:
        """验证实现：遍历验证集累计指标。

        Returns:
            Dict[str, float]: 含 ``loss`` / ``ssim`` / ``psnr`` / ``score``。
        """
        self.model.eval()
        loss_meter = AverageMeter()
        accumulator = MetricAccumulator()

        for batch in tqdm(self.val_loader, desc="validate", leave=False):
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

        启用 EMA 时，每个 epoch 同时评估原始权重与 EMA 权重，
        取分数更高者参与最优判定；若 EMA 更优，则 checkpoint
        保存的是 EMA 权重（推理直接使用即可）。

        Returns:
            float: 训练过程中的最优验证综合得分。
        """
        self.logger.info(f"开始训练，共 {self.epochs} 个 epoch")
        start_time = time.time()

        for epoch in range(self.start_epoch, self.epochs):
            train_loss = self.train_one_epoch(epoch)

            # 每个 epoch 只验证一次原始权重；EMA 按 ema_eval_every 间隔评估。
            val_metrics = self.validate(use_ema=False)
            weight_source = "raw"
            should_eval_ema = (
                self.ema is not None
                and (
                    (epoch + 1) % self.ema_eval_every == 0
                    or epoch + 1 == self.epochs
                )
            )
            if should_eval_ema:
                ema_metrics = self.validate(use_ema=True)
                if ema_metrics["score"] > val_metrics["score"]:
                    val_metrics = ema_metrics
                    weight_source = "ema"

            if self.scheduler_type != "onecycle":
                self.scheduler.step()

            lr = self.optimizer.param_groups[0]["lr"]
            self.recorder.log({
                "epoch": epoch + 1,
                "train_loss": f"{train_loss:.6f}",
                "val_loss": f"{val_metrics['loss']:.6f}",
                "val_ssim": f"{val_metrics['ssim']:.6f}",
                "val_psnr": f"{val_metrics['psnr']:.4f}",
                "val_score": f"{val_metrics['score']:.6f}",
                "lr": f"{lr:.2e}",
            })
            self.logger.info(
                f"Epoch {epoch + 1}/{self.epochs} | "
                f"train_loss={train_loss:.4f} | val_loss={val_metrics['loss']:.4f} | "
                f"SSIM={val_metrics['ssim']:.4f} | PSNR={val_metrics['psnr']:.2f} | "
                f"Score={val_metrics['score']:.4f} | 权重来源={weight_source}"
            )

            # 保存最新与最优两份 checkpoint，最优按比赛综合得分判定。
            save_checkpoint(
                self.base_model, self.optimizer, epoch + 1, self.best_score,
                self.config, str(self.checkpoint_dir / "last.pth"),
            )
            if val_metrics["score"] > self.best_score:
                self.best_score = val_metrics["score"]
                self._save_best(epoch, weight_source)

        self.recorder.finish()
        elapsed = time.time() - start_time
        self.logger.info(
            f"训练完成，耗时 {elapsed / 60:.1f} 分钟，最优 Score={self.best_score:.4f}"
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
        best_path = str(self.checkpoint_dir / "best.pth")
        if weight_source == "ema" and self.ema is not None:
            # 临时切入 EMA 权重保存，保存后恢复训练权重。
            self.ema.apply_shadow(self.model)
            save_checkpoint(
                self.model, self.optimizer, epoch + 1, self.best_score,
                self.config, best_path,
            )
            self.ema.restore(self.model)
        else:
            save_checkpoint(
                self.model, self.optimizer, epoch + 1, self.best_score,
                self.config, best_path,
            )
        self.logger.info(
            f"  -> 新的最优模型 (Score={self.best_score:.4f}, 来源={weight_source})"
        )
