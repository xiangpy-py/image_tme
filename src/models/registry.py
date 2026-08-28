"""模型注册表：按配置名统一实例化模型。"""

from typing import Any, Callable, Dict

import torch.nn as nn

from .adapter_unet import AdapterUNet
from .conditional_unet import ConditionalUNet, ConditionalUNetV2
from .gpt_unet import GPTUNet
from .resnet_unet import ResNetUNet
from .trans_unet import TransUNet
from .unet import UNet

MODEL_REGISTRY: Dict[str, Callable[..., nn.Module]] = {
    "unet": UNet,
    "resnet_unet": ResNetUNet,
    "trans_unet": TransUNet,
    "conditional_unet": ConditionalUNet,      # 原版：仅 bottleneck 条件
    "conditional_unet_v2": ConditionalUNetV2,  # 新版：Multi-scale FiLM
    "adapter_unet": AdapterUNet,               # 共享编码器 + Marker Adapter
    "gpt_unet": GPTUNet,
}


def build_model(config: Dict[str, Any]) -> nn.Module:
    model_cfg = dict(config.get("model", {}) or {})
    model_type = model_cfg.pop("type", "unet")

    if model_type not in MODEL_REGISTRY:
        raise ValueError(
            f"未知模型类型 '{model_type}'，可选: {list(MODEL_REGISTRY)}"
        )

    return MODEL_REGISTRY[model_type](**model_cfg)


def is_conditional_model(config: Dict[str, Any]) -> bool:
    """判断是否需要 marker_idx 输入（含所有条件/适配器模型）。"""
    conditional_types = {
        "conditional_unet",
        "conditional_unet_v2",
        "adapter_unet",
    }
    return config.get("model", {}).get("type") in conditional_types
