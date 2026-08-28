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
from ..utils.config import get_nested

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

        # ---- EMA ----
        self.use_ema = bool(train_cfg.get("ema", False))
        self.ema = ModelEMA(self.model, decay=0.9999, warmup_steps=100) if self.use_ema else None

        # ---- 学习率调度器选择 ----
        scheduler_type = train_cfg.get("scheduler", "cosine")
        if scheduler_type == "onecycle":
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

        # ---- 数据 ----
        loaders = build_dataloaders(config)
        self.train_loader: DataLoader = loaders["train"]
        self.val_loader: DataLoader = loaders["val"]

        # ---- 模型 / 损失 / 优化器 ----
        self.model: nn.Module = build_model(config).to(self.device)
        self.criterion = build_loss(config).to(self.device)
        self.optimizer: Optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=float(train_cfg.get("lr", 1e-4)),
            weight_decay=float(train_cfg.get("weight_decay", 1e-4)),
        )
        self.scheduler = CosineAnnealingLR(
            self.optimizer,
            T_max=self.epochs,
            eta_min=float(train_cfg.get("min_lr", 1e-6)),
        )

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
        )

    # ------------------------------------------------------------------ #
    # 单步前向：统一处理单标记 / 多标记条件两种模型
    # ------------------------------------------------------------------ #
    def _forward(self, batch: Dict[str, Any]) -> torch.Tensor:
        """根据模型类型执行前向传播。

        Args:
            batch: DataLoader 输出的批次字典。

        Returns:
            torch.Tensor: 模型预测 ``(B, C, H, W)``。
        """
        inputs = batch["input"].to(self.device, non_blocking=True)
        if self.conditional:
            marker_idx = batch["marker_idx"].to(self.device)
            return self.model(inputs, marker_idx)
        return self.model(inputs)

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
                predictions = self._forward(batch)

            # 损失计算放到 autocast 之外并强制 fp32：
            # SSIM 中的相近数相减在 fp16 下会灾难性抵消，导致 loss=NaN。
            loss, _details = self.criterion(predictions.float(), targets)

            if self.scaler is not None:
                self.scaler.scale(loss).backward()
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                loss.backward()
                self.optimizer.step()
                if self.ema is not None:
                    self.ema.update(self.model)
            loss_meter.update(float(loss.detach()), predictions.shape[0])
            progress.set_postfix(loss=f"{loss_meter.avg:.4f}")

        return loss_meter.avg


    @torch.no_grad()
    def validate(self, use_ema: bool = False) -> Dict[str, float]:
        if use_ema and self.ema is not None:
            self.ema.apply_shadow(self.model)
            metrics = self._validate_impl()
            self.ema.restore(self.model)
            return metrics
        return self._validate_impl()

    def _validate_impl(self) -> Dict[str, float]:
        """在验证集上评估模型。

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
                predictions = self._forward(batch)

            # 与训练一致：损失在 fp32 下计算，避免 fp16 数值不稳定。
            loss, _details = self.criterion(predictions.float(), targets)

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
        self.logger.info(f"开始训练，共 {self.epochs} 个 epoch")
        start_time = time.time()

        for epoch in range(self.start_epoch, self.epochs):
            train_loss = self.train_one_epoch(epoch)
            val_metrics = self.validate()
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
                f"Score={val_metrics['score']:.4f}"
            )

            # 加入 EMA 状态
            ckpt_dict = {
                "epoch": epoch + 1,
                "best_score": self.best_score,
                "model_state_dict": self.model.state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict(),
                "config": self.config,
            }
            if self.ema is not None:
                ckpt_dict["ema_state_dict"] = self.ema.state_dict()

            # 验证时同时评估普通权重和 EMA 权重，取最优
            val_metrics = self.validate(use_ema=False)
            if self.ema is not None:
                ema_metrics = self.validate(use_ema=True)
                if ema_metrics["score"] > val_metrics["score"]:
                    val_metrics = ema_metrics
                    val_metrics["source"] = "ema"

            # 保存最新与最优两份 checkpoint，最优按比赛综合得分判定。
            save_checkpoint(
                self.model, self.optimizer, epoch + 1, self.best_score,
                self.config, str(self.checkpoint_dir / "last.pth"),
            )
            if val_metrics["score"] > self.best_score:
                self.best_score = val_metrics["score"]
                save_checkpoint(
                    self.model, self.optimizer, epoch + 1, self.best_score,
                    self.config, str(self.checkpoint_dir / "best.pth"),
                )
                self.logger.info(f"  -> 新的最优模型 (Score={self.best_score:.4f})")

        self.recorder.finish()
        elapsed = time.time() - start_time
        self.logger.info(
            f"训练完成，耗时 {elapsed / 60:.1f} 分钟，最优 Score={self.best_score:.4f}"
        )
        return self.best_score
