"""image_tme 核心模块包。

按照功能划分为五个子包：

- ``datasets``: 数据读取、配对管理、数据增强与 DataLoader 构建。
- ``models``:   模型定义（U-Net / ResNet-UNet / TransUNet / 多标记条件模型）。
- ``losses``:   损失函数（L1、SSIM、感知损失及加权组合）。
- ``metrics``:  评测指标（SSIM、PSNR、比赛综合得分）。
- ``utils``:    配置加载、日志、随机种子、checkpoint、可视化等通用工具。
"""

__version__ = "0.1.0"
