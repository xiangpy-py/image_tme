"""模型注册表：按配置名统一实例化模型。

新增模型时只需在 ``MODEL_REGISTRY`` 中登记，
训练/推理脚本无需改动即可通过 YAML 的 ``model.type`` 切换。
"""

from typing import Any, Callable, Dict

import torch.nn as nn

from .conditional_unet import ConditionalUNet
from .gpt_unet import GPTUNet
from .resnet_unet import ResNetUNet
from .trans_unet import TransUNet
from .unet import UNet

# 模型名 -> 构造函数，构造函数签名为 (**model_config) -> nn.Module
MODEL_REGISTRY: Dict[str, Callable[..., nn.Module]] = {
    "unet": UNet,
    "resnet_unet": ResNetUNet,
    "trans_unet": TransUNet,
    "conditional_unet": ConditionalUNet,
    "gpt_unet": GPTUNet,
}


def build_model(config: Dict[str, Any]) -> nn.Module:
    """根据配置创建模型实例。

    Args:
        config: 全局配置字典，读取 ``model`` 一节，
            其中 ``type`` 指定模型名，其余字段透传给构造函数。

    Returns:
        nn.Module: 实例化后的模型（尚未移动到计算设备）。

    Raises:
        ValueError: 未知的模型类型时抛出。
    """
    model_cfg = dict(config.get("model", {}) or {})
    model_type = model_cfg.pop("type", "unet")

    if model_type not in MODEL_REGISTRY:
        raise ValueError(
            f"未知模型类型 '{model_type}'，可选: {list(MODEL_REGISTRY)}"
        )

    return MODEL_REGISTRY[model_type](**model_cfg)


def is_conditional_model(config: Dict[str, Any]) -> bool:
    """判断当前配置对应的模型是否需要 marker 条件输入。

    Args:
        config: 全局配置字典。

    Returns:
        bool: 多标记条件模型返回 ``True``，其余返回 ``False``。
    """
    return config.get("model", {}).get("type") == "conditional_unet"
