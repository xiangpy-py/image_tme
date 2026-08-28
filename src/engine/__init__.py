"""训练引擎子包：训练循环、验证与推理执行逻辑。"""

from .trainer import Trainer
from .predictor import Predictor

__all__ = ["Trainer", "Predictor"]
