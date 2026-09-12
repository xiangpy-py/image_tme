"""指数滑动平均 (EMA, Exponential Moving Average)。

训练过程中维护模型权重的滑动平均副本，
推理时使用 EMA 权重通常能获得更稳定、更泛化的结果。

使用方式：
    1. 训练前用 ``ema = ModelEMA(model, decay=0.9999)`` 创建；
    2. 每个训练 step 后调用 ``ema.update(model)``；
    3. 验证/推理前调用 ``ema.apply_shadow(model)`` 将 EMA 权重加载到模型；
    4. 验证/推理后调用 ``ema.restore(model)`` 恢复原始训练权重。
"""

from typing import Dict

import torch
import torch.nn as nn


class ModelEMA:
    """模型权重的指数滑动平均管理器。

    内部维护一份与模型结构对应的 ``state_dict`` 副本，
    通过 ``update()`` 方法在每个训练 step 后同步更新。
    """

    def __init__(
        self,
        model: nn.Module,
        decay: float = 0.9999,
        warmup_steps: int = 0,
    ) -> None:
        """初始化 EMA 状态字典。

        Args:
            model: 被跟踪的模型，EMA 副本与其结构一致。
            decay: 衰减系数，越接近 1 历史权重占比越高，平滑效果越强。
                常用值：0.999 ~ 0.9999。
            warmup_steps: 前 N 个 step 使用线性增长的 decay，
                避免训练初期参数变化剧烈时 EMA 滞后过多。
        """
        self.decay = decay
        self.warmup_steps = warmup_steps
        self.step_count = 0

        # 仅对可训练参数做 EMA，buffer（如 BN running_mean）不参与。
        self.shadow: Dict[str, torch.Tensor] = {}
        self.backup: Dict[str, torch.Tensor] = {}

        for name, param in model.named_parameters():
            if param.requires_grad:
                # 深拷贝并脱离计算图，EMA 不参与梯度。
                self.shadow[name] = param.data.clone().detach()

    def _get_current_decay(self) -> float:
        """获取当前 step 的有效 decay 值。

        warmup 阶段线性增长：从 0 逐步过渡到目标 decay，
        避免训练初期参数剧烈变化时 EMA 过于滞后。

        Returns:
            float: 当前 step 的有效 decay 系数。
        """
        if self.step_count < self.warmup_steps:
            return self.decay * (self.step_count / max(self.warmup_steps, 1))
        return self.decay

    def update(self, model: nn.Module) -> None:
        """用当前模型参数更新 EMA 阴影权重。

        应在每个优化 step 完成后调用（即 ``optimizer.step()`` 之后）。

        Args:
            model: 当前训练中的模型。

        Returns:
            None
        """
        self.step_count += 1
        decay = self._get_current_decay()

        with torch.no_grad():
            for name, param in model.named_parameters():
                if param.requires_grad and name in self.shadow:
                    # EMA 更新公式：shadow = decay * shadow + (1-decay) * param
                    self.shadow[name].mul_(decay).add_(param.data, alpha=1.0 - decay)

    def apply_shadow(self, model: nn.Module) -> None:
        """将 EMA 阴影权重加载到模型，用于验证/推理。

        调用前会自动备份当前训练权重，以便后续 ``restore()`` 恢复。

        Args:
            model: 目标模型。

        Returns:
            None
        """
        self.backup.clear()
        with torch.no_grad():
            for name, param in model.named_parameters():
                if param.requires_grad and name in self.shadow:
                    self.backup[name] = param.data.clone()
                    param.data.copy_(self.shadow[name])

    def restore(self, model: nn.Module) -> None:
        """恢复模型的原始训练权重。

        在验证/推理结束后调用，保证训练继续时模型参数正确。

        Args:
            model: 目标模型。

        Returns:
            None
        """
        with torch.no_grad():
            for name, param in model.named_parameters():
                if param.requires_grad and name in self.backup:
                    param.data.copy_(self.backup[name])
        self.backup.clear()

    def state_dict(self) -> Dict[str, torch.Tensor]:
        """导出 EMA 状态字典，用于 checkpoint 保存。

        Returns:
            Dict[str, torch.Tensor]: EMA 阴影权重的深拷贝。
        """
        return {key: value.clone() for key, value in self.shadow.items()}

    def load_state_dict(self, state_dict: Dict[str, torch.Tensor]) -> None:
        """从 checkpoint 恢复 EMA 状态。

        Args:
            state_dict: 由 ``state_dict()`` 导出的字典。

        Returns:
            None
        """
        self.shadow = {key: value.clone() for key, value in state_dict.items()}
