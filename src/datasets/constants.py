"""数据层常量定义。

集中管理赛题规定的标记名称与图像后缀，避免在各模块中
出现散落的魔法字符串，保证命名一致、易于扩展新标记。
"""

from typing import List

# 源染色：DAPI 细胞核染色图像，作为模型输入。
SOURCE_MARKER: str = "DAPI"

# 目标免疫组化标记：比赛指定的四类目标，顺序即多标记模型的 token 编号。
MARKERS: List[str] = ["CD68", "CD45RO", "HLA-DR", "Vimentin"]

# 数据集支持的图像后缀（赛题说明为 JPG，这里兼容常见格式）。
IMAGE_EXTENSIONS: List[str] = [".jpg", ".jpeg", ".png", ".tif", ".tiff"]


def sanitize_marker_name(marker: str) -> str:
    """将标记名转为适合作为目录名的小写标识。

    训练与推理脚本共用此约定：标记 ``HLA-DR`` 的实验目录为
    ``<实验名>_hladr``，保证 checkpoint 可被自动查找。

    Args:
        marker: 标记名，例如 ``"HLA-DR"``。

    Returns:
        str: 目录安全的小写标识，例如 ``"hladr"``。
    """
    return marker.lower().replace("-", "")
