# image_tme：基于虚拟染色的免疫组化图像生成

面向「全球校园人工智能算法精英大赛 · 基于虚拟染色的免疫组化图像生成」赛题：
以 **DAPI** 染色图像为输入，生成 **CD68 / CD45RO / HLA-DR / Vimentin** 四类目标 IHC 标记图像。

| 项目 | 说明 |
| --- | --- |
| 输入 | DAPI 染色图像 patch（256×256 JPG） |
| 输出 | 与输入同尺寸的目标 IHC 标记图像（灰度强度图） |
| 评测 | `Score = 70% × SSIM + 30% × Normalize(PSNR)` |
| 数据 | 多器官 mIHC 配对数据（colon / liver / stomach），训练集含配对真值，测试集仅给 DAPI |
| 提交 | `results/test/<MARKER>/<原名>_fake.jpg`，推理全自动、禁止人工后处理 |
| 加分项 | 一对多联合建模（同一 DAPI 输入生成多种目标标记） |

---

## 1. 快速开始

```bash
# 1) 安装依赖（Python 3.12，由 uv 管理）
uv sync

# 2) 按 ROI 划分训练/验证集（防数据泄漏）
uv run main.py split --root data --val-ratio 0.15

# 3) 训练（单标记 / 全部四标记 / 一对多条件模型）
uv run main.py train --config configs/baseline.yaml
uv run main.py train --config configs/baseline.yaml --marker all
uv run main.py train --config configs/conditional_v2.yaml

# 4) 测试集推理，生成比赛提交结果
uv run main.py infer --config configs/baseline.yaml --exp exp001_unet_baseline
```

> 所有指令与超参均可查看帮助：`uv run main.py --help`、
> `uv run main.py train --help`（每条子命令都有参数说明与示例）。

---

## 2. 代码结构（包 → 类 → 方法）

代码按三级组织：**包**表达职责域，**类**表达一组内聚能力，**方法**表达具体动作。
无模块级业务函数。

```
image_tme/
├── main.py                  # 组织器：只解析指令与超参数，功能全部转交 src
├── configs/                 # 实验配置（单实验 / 实验矩阵）
└── src/
    ├── layout.py            # 布局器：JobSpec + Layout
    ├── allocate.py          # 分配器：ResourceProfile + Allocator
    ├── operate.py           # 操作器：Operator（执行作业 + 编排用例）
    ├── feedback.py          # 反馈器：JobResult + Feedback
    ├── ensemble.py          # 集成器：Ensembler
    ├── data/                # 通用数据层（不依赖 torch）
    │   ├── constants.py     #   标记名、图像后缀、命名规范
    │   └── dataset.py       #   DatasetSource / DatasetSplitter / DatasetAnalyzer
    ├── utils/               # 通用工具链（不依赖其它子包）
    │   ├── config.py        #   ConfigManager
    │   ├── runtime.py       #   Runtime
    │   ├── logger.py        #   LoggerFactory / ExperimentLogger
    │   ├── checkpoint.py    #   CheckpointManager
    │   └── ema.py           #   ModelEMA
    └── train/               # 单作业训练子系统（不感知并行调度）
        ├── data.py          #   PairedTransform / VirtualStainingDataset /
        │                    #   MultiMarkerDataset / DataLoaders
        ├── models.py        #   网络构件 + 5 个模型 + ModelRegistry
        ├── losses.py        #   SSIM / SSIMLoss / SobelEdgeLoss /
        │                    #   CrossMarkerConsistencyLoss / CombinedLoss
        ├── metrics.py       #   Metrics / MetricAccumulator / AverageMeter
        └── engine.py        #   Trainer / Predictor
```

依赖方向自底向上，**不允许反向依赖**：

```
data / utils  <-  train  <-  layout  <-  allocate  <-  operate  ->  feedback
                                                                    （ensemble 复用 train）
```

### 类职责一览

| 包 | 类 | 职责 |
| --- | --- | --- |
| `utils` | `ConfigManager` | YAML 读取/落盘、命令行超参覆盖、配置深合并 |
| | `Runtime` | 随机种子固定、设备解析 |
| | `LoggerFactory` / `ExperimentLogger` | 控制台+文件日志 / 逐 epoch 指标 CSV+JSON |
| | `CheckpointManager` | 权重与优化器状态的保存/加载（含 torch 版本兼容） |
| | `ModelEMA` | 权重滑动平均，含 warmup 与 shadow 应用/还原 |
| `data` | `DatasetSource` | 标记目录发现（多器官/扁平双兼容）、图像读取与归一化 |
| | `DatasetSplitter` | 按 ROI 前缀划分训练/验证集并落盘，防止 patch 级泄漏 |
| | `DatasetAnalyzer` | 数据规模、尺寸、通道与像素分布统计 |
| `train` | `PairedTransform` | 配对增强：几何变换同步施加，光度扰动仅作用于输入 |
| | `VirtualStainingDataset` | 单标记配对数据集（DAPI → 指定标记），支持白名单与内存缓存 |
| | `MultiMarkerDataset` | 一对多数据集：每次随机采样一个目标标记并返回 `marker_idx` |
| | `DataLoaders` | 训练/验证/测试 DataLoader 并行调优（workers/prefetch/persistent） |
| | `DoubleConv` / `ResidualBlock` / `Down` / `Up` / `MarkerEmbedding` | 网络基础构件 |
| | `UNet` / `ResNetUNet` | 单标记模型（基线 / ImageNet 预训练编码器） |
| | `ConditionalUNet` / `ConditionalUNetV2` / `AdapterUNet` | 一对多条件模型（瓶颈嵌入 / 多尺度 FiLM / 共享编码器+标记适配器） |
| | `ModelRegistry` | 按 `model.type` 实例化模型、判断是否条件模型 |
| | `SSIM` / `SSIMLoss` / `SobelEdgeLoss` / `CrossMarkerConsistencyLoss` / `CombinedLoss` | 损失项与加权组合（`CombinedLoss.from_config`） |
| | `Metrics` / `MetricAccumulator` / `AverageMeter` | SSIM / PSNR / 比赛综合得分与累计统计 |
| | `Trainer` / `Predictor` | 训练全流程（AMP/EMA/compile）/ 推理与提交结果生成 |
| 编排 | `Layout` | 超参数 → 训练作业列表（单实验、实验矩阵两阶段） |
| | `Allocator` | 按本机 GPU 数量把作业切分为并行波次，支持再分配优先级 |
| | `Operator` | 执行波次、产出作业结果；每波次完成即写排行榜，并提供训练/矩阵等上层用例 |
| | `Feedback` | 排行榜写入与聚合、后缀还原、Top-K 筛选、再分配提示 |
| | `Ensembler` | 多实验预测结果逐像素平均 |

---

## 3. 指令与超参数

| 指令 | 作用 | 归属实现 |
| --- | --- | --- |
| `analyze` | 数据统计分析 | `DatasetAnalyzer.analyze` |
| `split` | 按 ROI 划分训练/验证集 | `DatasetSplitter.create` |
| `train` | 单实验训练 | `Operator.train_experiment` |
| `experiments` | 实验矩阵：筛选 + Top-K 长训练 | `Operator.run_experiments` |
| `infer` | 测试集推理并生成提交结果 | `Predictor.run_experiment` |
| `ensemble` | 多实验预测结果集成 | `Ensembler.run` |

### 3.1 analyze

```bash
uv run main.py analyze [--root data] [--output data/statistics.json]
```

| 参数 | 默认 | 说明 |
| --- | --- | --- |
| `--root` | `data` | 数据根目录，兼容 `data/<organ>/train` 与 `data/train` |
| `--output` | `data/statistics.json` | 统计结果输出路径 |

### 3.2 split

```bash
uv run main.py split [--root data] [--val-ratio 0.15] [--seed 42] [--save-dir data/splits]
```

| 参数 | 默认 | 说明 |
| --- | --- | --- |
| `--root` | `data` | 数据根目录 |
| `--val-ratio` | `0.15` | 验证集 ROI 占比，取值 (0, 1) |
| `--seed` | `42` | 随机种子，保证划分可复现 |
| `--save-dir` | `data/splits` | 划分结果保存目录，训练时自动加载其中的 `split.json` |

### 3.3 train

```bash
uv run main.py train --config <FILE>
                     [--marker NAME|all] [--epochs N] [--batch-size N] [--lr LR] [--device DEV]
```

| 参数 | 默认 | 说明 |
| --- | --- | --- |
| `--config` | 必填 | 训练配置 YAML |
| `--marker` | 配置中的 `data.marker` | `CD68`/`CD45RO`/`HLA-DR`/`Vimentin`，或 `all` 训练全部四类；条件模型可省略 |
| `--epochs` / `--batch-size` / `--lr` | `None` | 覆盖配置中的对应超参（不传则以配置为准） |
| `--device` | `None` | 覆盖 `runtime.device`：`auto`/`cpu`/`cuda`/`cuda:0` |
| `--resume` | 配置 `training.resume` | 从 `checkpoints/<实验名>/last.pth` 断点续训；恢复模型/优化器/调度器与训练进度，逐 epoch 曲线自动续接 |

单标记模型按标记展开为多个作业，实验名自动追加 `_<标记小写>` 后缀
（如 `exp001_unet_baseline_cd68`，`HLA-DR` → `_hladr`）；条件模型只产生一个作业。

**产物**

- `checkpoints/<实验名>/best.pth`：验证集综合得分最优的**推理权重**（EMA 更优时保存 EMA 权重）。仅含权重不含优化器状态，体积约为完整状态的 1/3
- `checkpoints/<实验名>/last.pth`：最新一轮的**完整训练状态**（权重 + 优化器 + 调度器），供 `--resume` 断点续训
- `logs/<实验名>/`：`history.csv`/`history.json`、实际生效配置 `config.yaml`、训练日志 `train.log`（每个实验独立文件）
- `logs/leaderboard.csv`：全部作业的排行榜，**每个波次完成即增量落盘**，阶段中途崩溃也不会丢失已完成作业的成绩

### 3.4 experiments（实验矩阵）

```bash
uv run main.py experiments --matrix <FILE> --stage screening|full|report|all [--device DEV]
```

| 参数 | 说明 |
| --- | --- |
| `--matrix` | 实验矩阵 YAML（含 `screening` / `full` / `experiments` 三节） |
| `--stage` | `screening`=短实验筛选；`full`=Top-K 长训练；`report`=仅打印排行榜；`all`=两阶段连跑 |
| `--device` | 覆盖运行设备 |
| `--resume` | 断点续训：各作业从各自 `last.pth` 恢复进度，已完成的实验秒退不重跑，脚本可反复执行 |

流程：布局器把矩阵展开为作业 → 分配器按 GPU 数量切分并行波次 → 操作器执行 →
反馈器写入排行榜并选出 Top-K → Top-K 以再分配提示的形式被优先调度进入长训练。
筛选阶段的实验目录带 `_screen` 后缀，避免覆盖长训练的产物。
配合 `--resume` 时，筛选/长训练两阶段都可安全中断续跑。

### 3.5 infer

```bash
uv run main.py infer --config <FILE> [--exp NAME] [--ckpt-all FILE] [--ckpt-<标记> FILE ...]
```

| 参数 | 说明 |
| --- | --- |
| `--config` | 推理配置 YAML（模型结构需与训练时一致） |
| `--exp` | 实验名，按命名约定自动查找全部标记的权重 |
| `--ckpt-all` | 多标记条件模型的统一权重（优先于 `--exp`） |
| `--ckpt-CD68` / `--ckpt-CD45RO` / `--ckpt-HLA-DR` / `--ckpt-Vimentin` | 单标记模型的权重（优先于 `--exp`） |

权重查找约定：

- 单标记模型：`checkpoints/<实验名>_<标记小写>/best.pth`
- 条件模型：`checkpoints/<实验名>/best.pth`

输出：`results/test/<MARKER>/<原名>_fake.jpg`（统一三通道 JPG，兼容按灰度或按 RGB 读取的评测脚本）。

### 3.6 ensemble

```bash
uv run main.py ensemble --config <FILE> --exps NAME [NAME ...]
```

把多个实验对同一标记的预测逐像素平均后输出，常用于把筛选出的 Top-K 模型集成。

---

## 4. 配置说明

配置分节如下（各模型示例见 `configs/`）：

| 节 | 字段 | 说明 |
| --- | --- | --- |
| `experiment` | `name` | 实验名，决定 `logs/`、`checkpoints/` 下的目录名 |
| `data` | `root` | 数据根目录 |
| | `marker` | 单标记目标；条件模型可省略 |
| | `split_dir` | ROI 划分文件目录（存在 `split.json` 即自动加载） |
| | `num_workers` | DataLoader 进程数；留空按 CPU 核心数自动探测 |
| | `cache` | 是否内存缓存解码后的图像（大内存机器建议开启） |
| | `prefetch_factor` | 每个 worker 预取的 batch 数 |
| `model` | `type` | `unet` / `resnet_unet` / `conditional_unet` / `conditional_unet_v2` / `adapter_unet` |
| | 其余字段 | 各模型构造参数，如 `in_channels` / `out_channels` / `base_channels` / `depth` / `num_markers` / `embed_dim` / `return_shared` / `backbone` / `pretrained` |
| `loss` | `lambda_l1` / `lambda_ssim` | L1 与 SSIM 损失权重 |
| | `lambda_edge` / `edge_kernel_size` / `edge_smooth_sigma` | 边缘损失权重与 Sobel/高斯参数 |
| | `lambda_cross` | 跨标记一致性权重（仅 `adapter_unet` 且 `return_shared: true` 生效） |
| `augmentation` | `hflip_prob` / `vflip_prob` / `rotate90` | 几何增强（对输入与真值同步施加） |
| | `brightness` / `contrast` / `noise_std` | 光度扰动（仅作用于输入 DAPI） |
| `training` | `epochs` / `batch_size` / `lr` / `min_lr` / `weight_decay` | 基础优化超参 |
| | `amp` | 混合精度（仅 CUDA 生效） |
| | `ema` / `ema_eval_every` | 权重滑动平均及其评估间隔 |
| | `scheduler` | `cosine` 或 `onecycle` |
| | `compile` / `compile_mode` | `torch.compile` 加速（仅 CUDA 生效） |
| `inference` | `output_dir` / `suffix` | 结果根目录与文件名后缀 |
| `runtime` | `device` | `auto` / `cpu` / `cuda` / `cuda:0` |
| | `seed` | 全局随机种子 |
| | `deterministic` | 置 `false` 启用 cuDNN 自动调优（更快） |
| | `num_threads` | CPU 线程数（仅 CPU 训练时生效） |

损失组合：

```
L = λ_l1·L1 + λ_ssim·SSIM + λ_edge·Edge + λ_cross·CrossMarker
```

### 实验矩阵格式

```yaml
screening:            # 筛选阶段
  epochs: 15
  marker: CD68        # 单标记模型使用的代理标记（保证横向可比）
full:                 # 长训练阶段
  epochs: 150
  top_k: 3            # 筛选排行榜前 K 名进入长训练
experiments:          # 候选实验 = base 配置 + overrides 覆盖
  - name: s01_v2_baseline
    base: configs/conditional_v2.yaml
    overrides:
      loss: {lambda_l1: 1.0, lambda_ssim: 1.0, lambda_edge: 0.0, lambda_cross: 0.0}
```

---

## 5. 数据组织

```text
data/
└── colon/                       # 也可直接是 data/（无器官层级）
    ├── train/
    │   ├── DAPI/   ROI000_00_00.jpg ...
    │   ├── CD68/   ROI000_00_00.jpg ...
    │   ├── CD45RO/
    │   ├── HLA-DR/
    │   └── Vimentin/
    └── test/
        └── DAPI/                # 测试集仅有输入
```

目录发现同时兼容多器官（`data/<organ>/train/<MARKER>`）与扁平
（`data/train/<MARKER>`）两种结构；目标标记统一按单通道灰度读取，
与模型 `out_channels: 1` 对应。

**train 与 test 的职责区分（务必分清）**

| 目录 | 内容 | 用途 |
| --- | --- | --- |
| `data/train/` | DAPI + 四类标记真值（同名配对） | 训练数据；由 `split` 命令**按 ROI 再划分**为本地训练集/验证集 |
| `data/test/DAPI/` | 仅 DAPI 输入，**无标签** | 赛方测试集输入；仅用于 `infer` 生成提交结果 |

- 赛题官方只发布「训练数据 + 测试集输入」，测试集标签不公开，且要求参赛者**自行划分本地训练集与验证集**。
- 因此本项目的比例划分产物是**本地验证集（val）**，用于模型选择与早停判断；它**不是**官方测试集。
- 划分以 ROI 前缀（`ROIxxx`）为最小单位，保证同一 ROI 的相邻 patch 只出现在同一侧，避免信息泄漏。
- `data/test/` 不参与任何训练或调参，只做推理，结果输出到 `results/test/<MARKER>/<原名>_fake.jpg`，打包上传即可。

---

## 6. 模型与训练要点

| 模型 | `model.type` | 说明 |
| --- | --- | --- |
| U-Net | `unet` | 基线编码器-解码器 |
| ResNet-UNet | `resnet_unet` | ImageNet 预训练 ResNet 编码器 + U-Net 解码器 |
| ConditionalUNet | `conditional_unet` | 一对多：瓶颈注入 marker 嵌入 |
| ConditionalUNetV2 | `conditional_unet_v2` | 一对多：多尺度 FiLM 逐层注入 |
| AdapterUNet | `adapter_unet` | 一对多：共享编解码器 + 标记适配器，可配跨标记一致性 |

训练要点：

- **防泄漏**：以 ROI 为单位划分训练/验证集，避免相邻 patch 跨集。
- **同步增强**：几何变换对 DAPI 与真值同步；亮度/对比度/噪声只作用于输入。
- **可复现**：固定全部随机源；每次运行把实际生效配置落盘到 `logs/<实验名>/config.yaml`。
- **数值稳定**：损失在 autocast 之外以 fp32 计算，避免 fp16 下 SSIM 灾难性抵消。
- **checkpoint 瘦身**：`best.pth` 只存推理权重，`last.pth` 才存完整训练状态（权重 + 优化器 + 调度器）。
- **断点续训**：`--resume`（或 `training.resume: true`）从 `last.pth` 恢复进度继续训练，已完成的 epoch 不会重跑。
- **并行**：多 GPU 时每个作业绑定一张卡、以 `spawn` 子进程并行；纯 CPU 环境自动串行。

---

## 7. 复现与约束

- 仅使用赛事官方数据；不使用任何闭源在线 API 完成图像生成。
- 推理全自动，无人工逐张处理或测试集后处理。
- 随机种子、数据划分文件、生效配置与逐 epoch 指标均已持久化，便于复现。
- 排行榜逐波次增量落盘，训练中断也不会丢失已完成作业的成绩。
- `runtime.deterministic: false` 时启用 cuDNN 自动调优以获得更快训练速度。

> **评测口径说明**：赛题仅给出 `Score = 70% × SSIM + 30% × Normalize(PSNR)`，
> 未定义 `Normalize(PSNR)`。本项目取最常见做法：将 PSNR 裁剪到 `[0, 40] dB`
> 后除以 40 归一化（见 [src/train/metrics.py](src/train/metrics.py)），
> 若官方口径不同需据此调整。

> 依赖清单见 `pyproject.toml`；本项目为应用型工程（`[tool.uv] package = false`），
> 统一通过 `uv run main.py ...` 调用。
