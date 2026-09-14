可以。基于我目前能看到的你这个 `image_tme` 仓库结构，我不建议推倒重写，而是**沿着你现在的“数据分析 → ROI 划分 → context → 实验赛马 → 推理 → ensemble”框架，逐层升级模型**。

你现在其实已经把“比赛工程化”这部分做得不错了：`main.py` 已经有 `experiments` 做短训筛选 + Top-K 长训，也有 `marker-report` 和 `ensemble`。

而比赛最终是 **70% SSIM + 30% PSNR**，所以我们的目标不是“做一个看起来最先进的生成模型”，而是：

> **让输出尽可能像 GT，同时保留局部结构和颜色/强度分布。**

---

# 一、我建议你最终做成这个版本

我会把你的主力模型定为：

> **Multi-Scale Residual Attention U-Net + Context + Marker FiLM**

结构大致：

```text
                    DAPI 256×256
                         │
             ┌───────────┴───────────┐
             │                       │
       Local branch            Context branch
       256×256                 512×512/2×
             │                       │
             └───────────┬───────────┘
                         ↓
                 Multi-scale Encoder
                         │
              ConvNeXt / Residual Block
                         │
                    Attention
                         │
                 Marker Embedding
                         │
                     FiLM
                         │
              Multi-scale Decoder
                         │
                 Residual Output
                         ↓
                  IHC 256×256×3
```

四个 marker：

```text
DAPI
 ├── CD68
 ├── CD45RO
 ├── HLA-DR
 └── Vimentin
```

但**不要一开始就把现在的模型全部改掉**。

应该按下面的顺序一点一点打。

---

# 二、第一阶段：先把你的“基准线”钉死

这是最重要的一步。

你现在已经有：

```text
analyze
split
context
train
experiments
infer
ensemble
marker-report
```

这套框架不要动。

特别是你的 ROI 划分：

```text
ROIxxx
 ├── patch 1
 ├── patch 2
 ├── patch 3
 ...
```

你已经按照 ROI 整体划分 train/val，避免相邻 patch 泄漏。这个设计是正确的。

### 先做：

```bash
uv run main.py analyze --root data

uv run main.py split \
    --root data \
    --val-ratio 0.15 \
    --seed 42
```

然后固定：

```text
seed = 42
split = 固定
validation = 固定
```

**之后所有实验绝对不能重新随机划分。**

否则你的实验分数没有可比性。

---

# 三、第二阶段：先把四个 marker 找出“谁是短板”

你已经写了：

```bash
uv run main.py marker-report
```

这其实非常适合赛马制。

最终得到类似：

| Marker   | SSIM | PSNR | Score |
| -------- | ---: | ---: | ----: |
| CD68     | 0.72 | 24.1 |  0.73 |
| CD45RO   | 0.69 | 23.5 |  0.70 |
| HLA-DR   | 0.61 | 21.8 |  0.64 |
| Vimentin | 0.76 | 25.2 |  0.77 |

那么不要平均用力。

比如：

```text
Vimentin     不动
CD68         小改
CD45RO       中改
HLA-DR       重点优化
```

因为比赛是四个 marker 的平均。

**最差 marker 往往是最值得投入的地方。**

你现在代码已经支持这个分析，这是我建议保留的机制。`main.py` 对 `marker-report` 的定位本身就是为了找四个标记的短板。

---

# 四、第三阶段：先把 Context 真正用起来

你现在已经实现了：

```bash
uv run main.py context \
    --root data \
    --output data/context \
    --scale 2
```

代码说明里，目前 context 是：

> 根据 ROI 网格拼接邻近 DAPI patch，然后生成上下文图，把输入从 3 channel 扩展到 6 channel。

这个方向我认为是**值得继续深挖的**。

因为 IHC 染色并不是只由当前 256×256 patch 的 DAPI 决定。

例如：

```text
       surrounding tissue
 ┌─────────────────────────┐
 │                         │
 │       ┌─────────┐       │
 │       │  patch  │       │
 │       │         │       │
 │       └─────────┘       │
 │                         │
 └─────────────────────────┘
```

周围组织结构可以帮助模型判断：

```text
细胞类型
组织区域
细胞密度
组织边界
免疫细胞聚集
```

---

# 五、但是 Context 不要只做一个 scale

你现在：

```yaml
scale: 2
```

我建议改成实验：

```text
baseline
context ×2
context ×4
multi-context
```

例如：

### EXP-A

```text
DAPI 256
```

### EXP-B

```text
DAPI 256
+ context ×2
```

### EXP-C

```text
DAPI 256
+ context ×4
```

### EXP-D

```text
DAPI 256
+ context ×2
+ context ×4
```

最后只留下：

```text
A/B/C/D
```

里面验证集 Score 最高的。

**不要凭肉眼判断 context 有没有用。**

---

# 六、第四阶段：模型升级，这是核心

这里才是我认为你现在最值得改的地方。

我建议不要直接：

```text
U-Net → Diffusion
```

而是：

```text
现有 U-Net
    ↓
Residual U-Net
    ↓
Residual + Attention U-Net
    ↓
Multi-scale Residual Attention U-Net
```

一步一步来。

---

# 七、第一版模型：Residual U-Net

你的普通 U-Net 如果现在类似：

```text
Conv
 ↓
Conv
 ↓
Down
```

改成：

```text
x
│
├───────────────┐
│               │
Conv             │
↓                │
Norm              │
↓                │
Activation        │
↓                │
Conv              │
│                │
└────── + x ─────┘
        ↓
       out
```

也就是：

```python
out = x + F(x)
```

这种结构对于图像恢复非常合适。

因为你的任务不是：

> “凭空生成一张 IHC。”

而是：

> “根据 DAPI 恢复对应的 IHC。”

所以应该尽量保留原始空间信息。

---

# 八、第二版：加入 Attention

然后在 encoder 深层和 bottleneck 加：

```text
Residual Block
      ↓
Attention
      ↓
Residual Block
```

我推荐首先尝试：

```text
CBAM
```

而不是一上来搞非常复杂的 Transformer。

结构：

```text
Encoder
   ↓
Residual
   ↓
CBAM
   ↓
Residual
   ↓
Bottleneck
```

为什么？

因为 256×256 图像，如果全程 Transformer attention，计算和显存开销没必要。

你的任务真正需要关注的通常是：

```text
细胞区域
细胞核
组织边界
高密度区域
局部结构
```

CBAM 这种轻量 attention 就比较合适。

---

# 九、第三版：Multi-scale

最终 encoder 做：

```text
256×256
   ↓
128×128
   ↓
64×64
   ↓
32×32
   ↓
16×16
```

分别保存：

```text
F1
F2
F3
F4
F5
```

decoder：

```text
F5
 ↓
up
 ↓
F4 ─────┐
         ↓
        block
         ↓
F3 ─────┐
         ↓
        block
         ↓
F2
 ↓
F1
 ↓
output
```

这样可以同时处理：

```text
大尺度组织结构
+
中尺度细胞结构
+
小尺度细胞细节
```

这和比赛给出的“多尺度特征融合、局部细节恢复、结构一致性”等设计方向也是一致的。比赛说明明确把这些列为模型设计建议。

---

# 十、第四版：四个 marker 不要简单硬共享

这是我比较看重的一点。

你的四个目标：

```text
CD68
CD45RO
HLA-DR
Vimentin
```

当然存在共同的信息：

```text
DAPI
 ↓
细胞结构
```

但是它们的染色模式完全一样是不可能的。

所以：

```text
Encoder
   ↓
Shared representation
   ↓
Marker condition
   ├── CD68
   ├── CD45RO
   ├── HLA-DR
   └── Vimentin
```

---

# 十一、加入 Marker Embedding + FiLM

例如：

```python
marker_embedding = Embedding(4, 128)
```

得到：

```text
CD68     → e1
CD45RO   → e2
HLA-DR   → e3
Vimentin → e4
```

然后：

```text
feature
   ↓
FiLM
```

核心：

```python
y = gamma(marker) * x + beta(marker)
```

也就是：

```text
shared feature
       +
marker information
       ↓
marker-specific feature
```

最终：

```text
DAPI
 ↓
Shared Encoder
 ↓
FiLM(CD68)
 ↓
Decoder
 ↓
CD68

DAPI
 ↓
Shared Encoder
 ↓
FiLM(CD45RO)
 ↓
Decoder
 ↓
CD45RO
```

这样就可以得到一个真正的：

> **one-to-many conditional virtual staining model**

而不是简单地训练四个模型。

比赛本身也明确鼓励一对多联合建模，而且最终还有创新加分空间。 

---

# 十二、Loss 是第二个核心战场

这里我建议你**不要一开始加入 GAN**。

因为你的评分：

```text
70% SSIM
30% PSNR
```



所以 loss 要直接围绕这个设计。

我建议第一轮：

```text
L =
0.35 L1
+
0.35 (1 - SSIM)
+
0.20 MSE
+
0.10 Gradient
```

即：

```python
loss = (
    0.35 * l1
    + 0.35 * ssim_loss
    + 0.20 * mse
    + 0.10 * gradient_loss
)
```

---

# 十三、为什么同时用 L1 + MSE？

因为：

### L1

负责：

```text
整体颜色
整体结构
减少异常值
```

### MSE

直接对应：

```text
PSNR
```

PSNR 本质上就是从 MSE 推导出来的。

所以：

```text
MSE ↑↓
PSNR ↓↑
```

### SSIM

负责：

```text
局部结构
对比度
纹理
```

而比赛 SSIM 权重最高。

因此：

```text
L1
+
MSE
+
SSIM
```

是非常符合这个比赛评分机制的组合。

---

# 十四、Gradient Loss 只给 10%

例如：

```text
∇prediction
vs
∇ground truth
```

让：

```text
细胞边缘
核边界
组织边缘
```

更清楚。

但是我不建议把 gradient loss 放太重。

否则模型容易开始“追边缘”，导致：

```text
SSIM ↓
PSNR ↓
```

所以先：

```text
0.10
```

即可。

---

# 十五、不要马上上 GAN

这是我比较明确的建议。

你可能会想到：

```text
U-Net
+
Discriminator
```

然后：

```text
GAN loss
```

但是这个比赛：

```text
SSIM 70%
PSNR 30%
```

GAN 很容易生成：

```text
视觉上更锐利
但是像素上偏离 GT
```

也就是：

```text
人眼看起来更好
↓
SSIM / PSNR 不一定更高
```

因此：

> **GAN 可以做一个后期实验，但不要作为你的第一主力模型。**

---

# 十六、Diffusion 也不要现在上

我也不建议现在直接：

```text
Stable Diffusion
DDPM
ControlNet
```

原因一样。

Diffusion 擅长：

```text
distribution modeling
```

而你这里实际上是：

```text
paired image reconstruction
```

而且 GT 已经存在。

你现在最需要的是：

```text
准确恢复
```

而不是：

```text
生成多样性
```

---

# 十七、你的实验矩阵应该彻底利用起来

你已经有：

```text
experiments
```

而且代码明确设计成：

```text
短实验筛选
      ↓
排行榜
      ↓
Top-K
      ↓
长训练
```



这个机制非常适合你的赛马制比赛。

我建议实验矩阵直接这样设计：

---

## 第一轮：结构实验

```text
EXP01  原始 U-Net
EXP02  Residual U-Net
EXP03  Residual + Attention
EXP04  Residual + Context
EXP05  Residual + Attention + Context
```

只跑：

```text
10~20 epochs
```

---

## 第二轮：Loss 实验

假设 EXP05 最好。

然后：

```text
EXP06  L1
EXP07  L1 + SSIM
EXP08  L1 + SSIM + MSE
EXP09  L1 + SSIM + MSE + Gradient
```

继续短训。

---

## 第三轮：Context 实验

```text
EXP10  scale=1
EXP11  scale=2
EXP12  scale=4
EXP13  scale=2+4
```

---

## 第四轮：Marker 条件实验

```text
EXP14  四个独立模型
EXP15  Shared Encoder
EXP16  Shared Encoder + Marker Embedding
EXP17  Shared Encoder + FiLM
```

---

# 十八、然后才进入“冠军模型”

最终可能得到：

```text
EXP17
Multi-scale
+
Residual
+
Attention
+
Context
+
FiLM
+
L1
+
MSE
+
SSIM
+
Gradient
```

然后：

```text
50~150 epochs
```

进行正式训练。

---

# 十九、但这里有一个很关键的优化：不要四个模型都长训

比如第一轮：

```text
CD68      0.73
CD45RO    0.71
HLA-DR    0.64
Vimentin  0.76
```

你就应该：

```text
HLA-DR → 重点投入
CD45RO → 第二重点
CD68 → 普通
Vimentin → 不折腾
```

这就是比赛和论文最大的区别之一。

论文：

> 每个组件都要公平比较。

比赛：

> **哪里有分，往哪里砸。**

---

# 二十、最后一定要 Ensemble

你现在已经有：

```bash
uv run main.py ensemble ...
```

而且实现逻辑是：

```text
多个实验
 ↓
逐像素平均
 ↓
最终预测
```



这个不要小看。

例如：

```text
Model A
Model B
Model C
```

预测：

```text
P = 0.5 A + 0.3 B + 0.2 C
```

通常比简单：

```text
P = A
```

稳定。

特别适合 SSIM/PSNR。

---

# 二十一、我甚至建议你最后做“加权 Ensemble”

不是：

```text
A 1/3
B 1/3
C 1/3
```

而是根据验证集成绩：

```text
A = 0.60
B = 0.25
C = 0.15
```

例如：

```text
A Score = 0.735
B Score = 0.728
C Score = 0.721
```

可以测试：

```text
0.6 A + 0.25 B + 0.15 C
```

和：

```text
0.5 A + 0.3 B + 0.2 C
```

谁最高就用谁。

---

# 二十二、训练速度方面，你的下一步也应该改

你现在 Python 依赖比较精简：

```text
torch
torchvision
numpy
opencv
Pillow
scikit-image
```

我建议增加：

```text
timm
```

然后可以直接尝试：

```text
ConvNeXt-Tiny
```

作为 encoder。

最终：

```text
ConvNeXt-Tiny Encoder
        ↓
Multi-scale features
        ↓
Residual Attention Decoder
        ↓
FiLM
        ↓
IHC
```

这会比你自己从零堆一个很深的 CNN 更值得尝试。

但是：

> **ConvNeXt 是第二阶段，不要第一天就换。**

先确定你现在的 U-Net baseline 到底多少分。

---

# 二十三、训练优化顺序

如果你的 GPU 够，我建议：

```text
AMP
↓
torch.compile
↓
channels_last
↓
num_workers
↓
pin_memory
↓
persistent_workers
```

尤其：

```python
torch.autocast(...)
```

以及：

```python
GradScaler
```

可以明显减少训练时间。

你的比赛是赛马制，所以：

> **单位时间能跑多少实验，本身就是竞争力。**

不是单纯追求单次训练最快。

---

# 二十四、真正的比赛路线，我建议这样

整个项目最后变成：

```text
                    数据
                     │
             ROI 防泄漏划分
                     │
              Context 生成
                     │
                     ▼
             ┌──────────────┐
             │ Experiment   │
             │ Screening    │
             └──────┬───────┘
                    │
          ┌─────────┼─────────┐
          ▼         ▼         ▼
       Model A   Model B   Model C
          │         │         │
          └─────────┼─────────┘
                    ▼
               Validation
                    │
              Score 排名
                    │
                 Top-K
                    │
                    ▼
             Full Training
                    │
          ┌─────────┼─────────┐
          ▼         ▼         ▼
        CD68     CD45RO    HLA-DR
          │         │         │
          └─────────┼─────────┘
                    ▼
                Ensemble
                    │
                    ▼
              JPG Round-trip
                    │
                    ▼
                 Submit
```

---

# 二十五、我给你的优先级

如果你现在只有有限时间，我会严格按照这个顺序：

| 优先级  | 工作                       |  预期价值 |
| ---- | ------------------------ | ----: |
| 🔥 S | 确定真实 baseline            | ★★★★★ |
| 🔥 S | marker-report 找短板        | ★★★★★ |
| 🔥 S | Residual U-Net           | ★★★★★ |
| 🔥 S | SSIM + MSE + L1 loss     | ★★★★★ |
| 🔥 S | Context                  | ★★★★☆ |
| 🔥 S | Attention                | ★★★★☆ |
| 🔥 S | Ensemble                 | ★★★★☆ |
| A    | Multi-scale              | ★★★★☆ |
| A    | FiLM marker conditioning | ★★★★☆ |
| A    | ConvNeXt encoder         | ★★★★☆ |
| B    | Gradient loss            | ★★★☆☆ |
| B    | GAN                      | ★★☆☆☆ |
| C    | Diffusion                | ★★☆☆☆ |

---

# 二十六、所以如果让我直接给你“执行清单”

**第一天不要改模型。**

### Step 1

```bash
analyze
split
context
```

确认数据没问题。

### Step 2

跑现在 baseline：

```text
CD68
CD45RO
HLA-DR
Vimentin
```

拿到四个真实分数。

### Step 3

运行：

```text
marker-report
```

找最差 marker。

### Step 4

建立：

```text
EXP01 baseline
EXP02 residual
EXP03 residual+attention
EXP04 residual+context
EXP05 residual+attention+context
```

短训。

### Step 5

取 Top 2。

### Step 6

对 Top 2 做：

```text
L1
L1 + SSIM
L1 + SSIM + MSE
L1 + SSIM + MSE + Gradient
```

### Step 7

取冠军。

### Step 8

在冠军上加入：

```text
FiLM + Marker Embedding
```

### Step 9

再测试：

```text
ConvNeXt encoder
```

### Step 10

最后：

```text
Top 3 models
    ↓
weighted ensemble
    ↓
正式提交
```

---

## 最关键的一点

**我现在不建议你直接让 Claude Code “重构整个项目”。**

你的 `main.py` 已经把比赛工程骨架搭起来了，而且 `experiments`、`marker-report`、`ensemble` 这些东西恰好是赛马制最需要的能力。

下一步应该是**精准改 `src` 里的 Model / Dataset / Loss / Trainer，而不是动 CLI 和实验管理层**。

如果你愿意，我下一步可以直接按照你这个仓库的结构，给你列出一份非常具体的：

```text
src/
├── models/
│   ├── ...
│   └── ...
├── losses/
│   └── ...
├── data/
│   └── ...
└── train/
    └── ...
```

**“哪些文件改、每个文件改什么、增加哪些类、配置 YAML 怎么写、实验矩阵怎么写”**的改造方案，直接做到可以交给 Claude Code 执行的程度。
