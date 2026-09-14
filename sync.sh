#!/usr/bin/env bash
# 一键初始化 / 更新运行环境（自动适配本机显卡驱动）。
#
# 为什么不把 torch 写进 pyproject.toml / uv.lock：
#   PyTorch 的 wheel 与 NVIDIA 驱动强绑定，不同服务器支持的 CUDA 版本不同。
#   一旦写进锁文件，就会把某一台机器的构建固化下来，换机即失效
#   （甚至出现 torch.cuda.is_available()=False 而静默回退 CPU）。
#   因此改用 uv 官方自动探测来安装：
#     --torch-backend=auto 会查询本机 CUDA 驱动 / AMD GPU / Intel GPU，
#     选择最兼容的 PyTorch 索引；探测不到 GPU 时回退 CPU 构建。
#
# 用法：
#   bash sync.sh        # 首次部署、或非 torch 依赖变更后执行一次
set -euo pipefail

# torch 只约束下限，具体构建（+cu126 / +cu128 / +cu130 / +cpu）交由 uv 按驱动选择。
TORCH_SPEC="torch>=2.11.0"
TORCHVISION_SPEC="torchvision>=0.26.0"

# 1) 装齐非 PyTorch 依赖。
#    --inexact 是关键：uv sync 默认会清理「不在锁文件里的包」，而 torch 正是这种包，
#    不加该参数会被当成多余包删除（裸 uv sync 会误删 torch，请勿直接使用）。
uv sync --inexact

# 2) 按本机驱动自动选择 PyTorch 构建并安装。
uv pip install --torch-backend=auto "$TORCH_SPEC" "$TORCHVISION_SPEC"

# 3) 回显实际命中的构建，便于确认是否吃到 GPU 加速。
uv run --no-sync python -c \
    "import torch; print('torch', torch.__version__, '| CUDA available:', torch.cuda.is_available())"
