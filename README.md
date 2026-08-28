# image_tme：基于虚拟染色的免疫组化图像生成

面向「全球校园人工智能算法精英大赛 · 基于虚拟染色的免疫组化图像生成」赛题：
以 DAPI 染色图像为输入，生成 HLA-DR / CD45RO / Vimentin / CD68 四类目标 IHC 标记图像。

- 评测指标：`Score = 70% × SSIM + 30% × Normalize(PSNR)`
- 数据：多器官 mIHC 配对数据（colon / liver / stomach），256×256 JPG patch
- 创新加分项：一对多联合建模（同一 DAPI 输入生成多种目标标记）

## 目录结构

```
image_tme/
├── main.py                      # 统一 CLI 入口（等价于 image-tme 命令）
├── configs/                     # 实验配置
│   ├── baseline.yaml            # U-Net 基线
│   ├── transformer.yaml         # TransUNet（瓶颈自注意力）
│   ├── gpt.yaml                 # GPTUNet（Global Pixel Transformers 复现）
│   ├── multi_marker.yaml        # 条件 U-Net（一对多，瓶颈注入）
│   ├── conditional_v2.yaml      # 条件 U-Net V2（多尺度 FiLM 注入）
│   ├── adapter_unet.yaml        # 共享编码器 + Marker Adapter + 跨标记一致性
│   └── experiment_matrix.yaml   # 实验矩阵（短实验筛选 + 长训练）
├── scripts/                     # 执行脚本（analyze / split / train / experiments / inference）
├── src/
│   ├── datasets/                # 配对数据集、同步增强、ROI 划分、多尺度输入
│   ├── models/                  # UNet / ResNetUNet / TransUNet / GPTUNet / 条件与 Adapter 模型
│   ├── losses/                  # L1 + SSIM + Edge + Perceptual + CrossMarker 组合损失
│   ├── metrics/                 # SSIM / PSNR / 比赛综合得分
│   ├── engine/                  # Trainer（AMP/EMA）与 Predictor
│   └── utils/                   # 配置、日志、EMA、checkpoint、可视化
└── assets/                      # 赛题 PDF 与参考文献
```

## 环境准备（uv）

项目使用 [uv](https://docs.astral.sh/uv/) 管理依赖（Python 3.12）：

```bash
git clone https://github.com/xiangpy-py/image_tme.git
cd image_tme
uv sync                  # 创建虚拟环境并安装全部依赖（含 image-tme 命令）
```

此后所有命令统一用 `uv run` 前缀执行，无需手动激活虚拟环境。

## 数据准备

按赛题目录约定放置数据（兼容单器官扁平结构）：

```
data/
└── colon/
    ├── train/
    │   ├── DAPI/        ROI000_00_00.jpg ...
    │   ├── CD68/
    │   ├── CD45RO/
    │   ├── HLA-DR/
    │   └── Vimentin/
    └── test/
        └── DAPI/        # 测试集仅有输入
```

按 ROI 划分本地训练/验证集（防止同一 ROI 的相邻 patch 泄漏到验证集）：

```bash
uv run image-tme analyze --root data          # 可选：数据统计
uv run image-tme split   --root data --val-ratio 0.15
# 生成 data/splits/split.json，训练时自动加载
```

## 训练流程

### 方式一：单实验训练

```bash
# 训练配置中指定的单个标记
uv run image-tme train --config configs/baseline.yaml

# 命令行指定标记（无需改 YAML）
uv run image-tme train --config configs/baseline.yaml --marker CD68

# 一条命令依次训练全部四种标记（一对一任务推荐）
uv run image-tme train --config configs/baseline.yaml --marker all

# 一对多条件模型：一次训练覆盖全部标记
uv run image-tme train --config configs/conditional_v2.yaml
```

训练产物：

- `checkpoints/<实验名>/best.pth`：按验证集比赛综合得分保存的最优权重（启用 EMA 时自动保存更优的 EMA 权重）
- `checkpoints/<实验名>/last.pth`：最新权重
- `logs/<实验名>/`：逐 epoch 指标（history.csv/json）、实际生效配置（config.yaml）、训练日志

常用训练开关（YAML 的 `training` 节）：`amp`（混合精度）、`ema`（权重滑动平均）、
`scheduler`（`cosine` / `onecycle`）、`epochs` / `batch_size` / `lr`。

### 方式二：实验矩阵（赛马制推荐）

短实验筛选 → 排行榜 → Top-K 长训练：

```bash
# 1) 全部候选实验短跑（默认 15 epoch，单标记模型统一用 CD68 做代理）
uv run image-tme experiments --matrix configs/experiment_matrix.yaml --stage screening

# 2) 查看排行榜
uv run image-tme experiments --matrix configs/experiment_matrix.yaml --stage report

# 3) Top-K（默认 3）进入完整训练；单标记模型此时依次训练全部四种标记
uv run image-tme experiments --matrix configs/experiment_matrix.yaml --stage full
```

排行榜持久化在 `logs/leaderboard.csv`；筛选阶段实验目录以 `_screen` 后缀与正式训练区分。
在 `configs/experiment_matrix.yaml` 中以「base 配置 + overrides 覆盖」的方式增删实验。

## 推理与提交

```bash
# 单标记模型：按实验名自动查找四个标记的权重
uv run image-tme infer --config configs/baseline.yaml --exp exp001_unet_baseline

# 一对多条件模型：一个权重生成全部标记
uv run image-tme infer --config configs/conditional_v2.yaml --exp exp006_conditional_v2
```

输出组织为比赛要求的提交格式：`results/test/<MARKER>/<原名>_fake.jpg`。

## 模型与损失一览

| 模型 | type | 说明 |
| --- | --- | --- |
| U-Net | `unet` | 基线编码器-解码器 |
| ResNet-UNet | `resnet_unet` | ImageNet 预训练 ResNet 编码器 |
| TransUNet | `trans_unet` | 瓶颈处 Transformer 自注意力 |
| GPTUNet | `gpt_unet` | Global Pixel Transformers 复现（Dense Block + 多尺度输入） |
| ConditionalUNet | `conditional_unet` | 一对多：瓶颈注入 marker 嵌入 |
| ConditionalUNetV2 | `conditional_unet_v2` | 一对多：多尺度 FiLM 逐层注入 |
| AdapterUNet | `adapter_unet` | 一对多：共享编解码器 + 标记适配器，可配跨标记一致性损失 |

组合损失（`loss` 节权重，0 为关闭）：

```
L = λ_l1·L1 + λ_ssim·SSIM + λ_edge·Edge + λ_perceptual·Perceptual + λ_cross·CrossMarker
```

其中 `lambda_cross`（跨标记一致性）仅对 `adapter_unet` 生效，需同时开启
`model.return_shared: true`。
