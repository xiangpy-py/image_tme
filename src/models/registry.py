"""模型注册表：按配置名统一实例化模型。"""

import inspect
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
    """按配置实例化模型。

    配置中目标模型不认识的参数键会被自动过滤——实验矩阵中
    「base 配置 + 覆盖模型类型」时，base 里残留的参数（如 UNet 的
    ``base_channels`` 之于 ResNetUNet）不会导致实例化失败。

    Args:
        config: 全局配置，读取 ``model`` 一节。

    Returns:
        nn.Module: 实例化后的模型。

    Raises:
        ValueError: 模型类型未注册时抛出。
    """
    model_cfg = dict(config.get("model", {}) or {})
    model_type = model_cfg.pop("type", "unet")

    if model_type not in MODEL_REGISTRY:
        raise ValueError(
            f"未知模型类型 '{model_type}'，可选: {list(MODEL_REGISTRY)}"
        )

    model_cls = MODEL_REGISTRY[model_type]

    # 仅保留构造函数声明了的参数；若构造函数接受 **kwargs 则原样传递。
    signature = inspect.signature(model_cls.__init__)
    accepts_kwargs = any(
        parameter.kind == inspect.Parameter.VAR_KEYWORD
        for parameter in signature.parameters.values()
    )
    if not accepts_kwargs:
        valid_keys = set(signature.parameters) - {"self"}
        model_cfg = {k: v for k, v in model_cfg.items() if k in valid_keys}

    return model_cls(**model_cfg)


def is_conditional_model(config: Dict[str, Any]) -> bool:
    """判断是否需要 marker_idx 输入（含所有条件/适配器模型）。"""
    conditional_types = {
        "conditional_unet",
        "conditional_unet_v2",
        "adapter_unet",
    }
    return config.get("model", {}).get("type") in conditional_types
