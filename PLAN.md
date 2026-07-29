# 少样本工业异常检测：第一阶段基准复现计划

更新日期：2026-07-24  
项目主题：基于不确定性路由的少样本工业异常视觉—语言证据融合  
本阶段边界：只建立可审计的基线、数据协议、评价流程和特征接口；不开发动态路由模块。

## 1. 阶段目标与验收标准

第一阶段必须在 **MVTec AD** 和 **VisA** 上建立可重复的 **1/2/4-shot** 实验协议，并形成两套结果：

1. **原论文协议复现**：严格使用各官方代码的默认配置，验证代码、权重、数据和指标链路。
2. **统一协议复现**：统一数据划分、shot 抽样、随机种子、输出格式和评价代码，用于论文公平比较。

阶段完成时必须具备：

- AnomalyCLIP、WinCLIP/WinCLIP+、PatchCore 的可运行基线；
- PromptAD、AnomalyDINO、ReMP-AD、AdaptCLIP 的官方代码、环境说明和复现结果；
- MVTec AD 与 VisA 的 1/2/4-shot 结果，每个 shot 至少 3 个固定种子；
- 图像级 AUROC、AP、F1-max；像素级 AUROC、AP、AUPRO；
- 每次实验的配置、代码提交号、随机样本清单、日志、逐图预测和可视化；
- AnomalyCLIP 全图与多层 patch 特征的导出接口，供下一阶段视觉参考分支使用；
- 一份“官方结果—本地原协议—统一协议”差异报告。

不以“所有论文数字逐位相同”为标准；必须保证任务设定、指标定义、数据划分和趋势一致，并解释可复现差异。

## 2. 当前机器与约束

- 操作系统：Windows / PowerShell。
- Python：3.10.11，当前无 Conda。
- GPU：NVIDIA GeForce RTX 3060 Laptop，6 GB 显存。
- NVIDIA 驱动：591.86；驱动报告最高 CUDA 13.1。
- 当前目录初始为空，尚不是 Git 仓库。

影响与处理：

- AnomalyCLIP 官方实验使用 RTX 3090 24 GB；本机先做**预训练权重推理复现**，batch size 固定为 1，必要时降低输入尺寸或使用 CPU 后处理。
- 不把 Windows 原生环境强行统一为一个依赖集合。每个官方方法使用独立 `.venv`，避免 PyTorch、CLIP、FAISS 和 CUDA 版本冲突。
- PatchCore/AnomalyDINO 的 FAISS 首轮使用 CPU；Windows 上 GPU FAISS 安装成本高，且 6 GB 显存不适合同时放模型和索引。
- 完整训练若超过显存或耗时预算，记录为“需要实验室服务器”的任务，但本地推理、协议和评价仍必须完成。

## 3. 项目目录

```text
sci_project/
├── PLAN.md                         # 本文件：阶段计划与命令
├── PROJECT_STATUS.md               # 当前状态、阻塞项、下一动作
├── README.md                       # 项目入口
├── configs/
│   └── protocol.yaml               # 统一协议唯一事实来源
├── data/
│   ├── README.md                   # 下载、许可、校验和、布局
│   ├── mvtec_ad/                   # 数据本体，不提交 Git
│   ├── visa_raw/                   # VisA 官方原始包
│   ├── visa_1cls/                  # 统一后的单类布局
│   └── splits/                     # 1/2/4-shot 抽样清单，提交 Git
├── docs/
│   ├── sources.md                  # 官方论文、代码、权重与数据链接
│   ├── environment_matrix.md       # 每个方法的依赖和已验证版本
│   └── reproduction_notes.md       # 差异、故障和解决方法
├── methods/                        # 第三方官方代码；尽量不直接修改
│   ├── anomalyclip/
│   ├── winclip/
│   ├── patchcore/
│   ├── promptad/
│   ├── anomalydino/
│   ├── remp_ad/
│   └── adaptclip/
├── src/industrial_ad/
│   ├── adapters/                   # 各方法到统一输出格式的薄封装
│   ├── datasets/                   # 数据索引、shot 采样、校验
│   ├── evaluation/                 # 统一指标
│   └── schemas/                    # 预测与实验记录格式
├── scripts/
│   ├── verify_system.ps1
│   ├── prepare_splits.py
│   ├── validate_dataset.py
│   └── run_smoke.py
├── experiments/
│   └── registry.csv                # 一行一个实验
└── outputs/                        # 不提交 Git
    ├── logs/
    ├── predictions/
    ├── anomaly_maps/
    ├── feature_cache/
    └── tables/
```

第三方仓库只记录来源 URL 和 commit SHA。若必须修补，补丁放在 `patches/<method>/`，不把研究代码直接写入第三方仓库。

## 4. 官方资源

### 数据集

- MVTec AD：<https://www.mvtec.com/research-teaching/datasets/mvtec-ad>
  - 15 类、5000 余张高分辨率图像，包含无缺陷训练集、正常/异常测试集和像素级标注。
  - 许可：CC BY-NC-SA 4.0，仅限非商业用途。
  - 官方页面要求填写表单后下载；不得用非官方镜像绕过许可流程。
- VisA：<https://registry.opendata.aws/visa/>
  - 12 类，10,821 张图像，含图像级和像素级标注。
  - 许可：CC BY 4.0。
  - 官方 AWS 对象：`s3://amazon-visual-anomaly/VisA_20220922.tar`

### 方法与权重

| 方法 | 年份 | 官方资源 | 第一阶段角色 |
|---|---:|---|---|
| PatchCore | 2021 | <https://github.com/amazon-science/patchcore-inspection> | 传统视觉与特征库基线 |
| WinCLIP | 2023 | 论文未公开原始代码；采用 AnomalyCLIP 内置复现并交叉核对 <https://github.com/caoyunkang/WinClip> | CLIP 早期零/少样本基线 |
| AnomalyCLIP | 2024 | <https://github.com/zqhang/AnomalyCLIP> | 项目基础框架与文本分支 |
| PromptAD | 2024 | <https://github.com/FuNz-0/PromptAD> | 仅正常样本 Prompt 学习 |
| AnomalyDINO | 2025 | <https://github.com/dammsi/AnomalyDINO> | 强纯视觉少样本基线 |
| ReMP-AD | 2025 | <https://github.com/cshcma/ReMP-AD> | 最接近的多模态检索基线 |
| AdaptCLIP | 2026 | <https://github.com/gaobb/AdaptCLIP>；权重 <https://huggingface.co/csgaobb/AdaptCLIP> | 最新通用零/少样本强基线 |

优先级按依赖关系而非年份排列：

```text
AnomalyCLIP 冒烟
→ AnomalyCLIP 原协议
→ WinCLIP/WinCLIP+ 与 PatchCore
→ 统一协议和统一评价
→ PromptAD 与 AnomalyDINO
→ ReMP-AD 与 AdaptCLIP
```

## 5. 统一实验协议

### 5.1 任务定义

- 目标类别只允许使用 K 张正常训练图作为参考，K ∈ {1, 2, 4}。
- 目标类别异常图和掩码只用于测试与评价，不参与训练、调参或参考库构建。
- 若方法需在源域训练：
  - 测 MVTec AD 时，以 VisA/论文规定的非目标源域训练；
  - 测 VisA 时，以 MVTec AD/论文规定的非目标源域训练；
  - 结果中必须明确写出源域，禁止产生目标域泄漏。
- 目标域不微调是主协议；需要目标正常图训练 Prompt 的方法单独标注为 `target_normal_tuning=true`。

### 5.2 Shot 抽样

- 固定种子：`0, 1, 2`；最终论文资源允许时扩展为 5 个种子。
- 每个“数据集 × 类别 × shot × seed”生成确定的相对路径清单，保存到 `data/splits/`。
- K-shot 集合使用嵌套策略：同一 seed 下，1-shot 是 2-shot 的子集，2-shot 是 4-shot 的子集，减少抽样噪声。
- 各方法必须读取同一份清单，禁止各自内部重新随机抽样。

### 5.3 预处理

- 原协议：保留官方 resize/crop/归一化。
- 统一协议：第一版固定输入短边/方形尺寸为 224，双线性图像插值、最近邻掩码插值；如某方法结构要求不同尺寸，记录例外并同时保留其原协议结果。
- 不将测试集统计量用于归一化。
- VisA 统一转换为 MVTec 风格的 `train/good`, `test/*`, `ground_truth/*` 布局，但保留原始数据只读副本。

### 5.4 指标

- 图像级：AUROC、AP、F1-max。
- 像素级：AUROC、AP、AUPRO（最大 FPR=0.30）。
- 效率：单图推理时间、峰值显存、参考库大小、可训练参数量。
- 数据聚合：
  - 先逐类别计算，再做 macro mean；
  - 不用把所有像素直接拼接后的 global AUROC 冒充类别均值；
  - 每个 shot 报告多 seed 的 mean ± std。

### 5.5 统一输出

每个方法的适配器必须输出：

```python
{
    "sample_id": str,
    "image_score": float,       # 越大越异常
    "pixel_map": "path/to.npy", # H×W，已对齐原图/GT
    "image_label": int,         # 0 正常，1 异常
    "mask_path": str | None,
    "runtime_ms": float,
    "metadata": {
        "method": str,
        "dataset": str,
        "category": str,
        "shot": int,
        "seed": int,
        "source_domain": str | None,
        "checkpoint": str,
        "commit_sha": str
    }
}
```

## 6. 分步执行清单

### P0：仓库与系统基线

1. 建立本目录结构、`.gitignore`、实验登记表和状态文档。
2. 运行 `scripts/verify_system.ps1`，记录 Python、GPU、驱动和磁盘状态。
3. 初始化主 Git 仓库；第三方源码目录不与本项目研究代码混合提交。
4. 克隆 7 个方法仓库并记录：
   - URL；
   - 默认分支；
   - commit SHA；
   - license；
   - README 中的 Python/PyTorch/CUDA 要求；
   - 权重下载地址。
5. 每个方法建立独立环境，环境名固定为 `.venv-<method>`。

验收：`PROJECT_STATUS.md` 中源码和环境矩阵均有明确状态，无“来源未知”的代码或权重。

### P1：数据获取与验证

#### MVTec AD

1. 打开官方页面并阅读 CC BY-NC-SA 4.0 条款。
2. 填写页面表单并下载官方压缩包到 `data/downloads/`。
3. 解压到 `data/mvtec_ad/`。
4. 运行：

```powershell
python scripts/validate_dataset.py --dataset mvtec --root data/mvtec_ad
```

验证 15 个类别、train/test/ground_truth 目录、掩码与异常图的对应关系，并保存文件数与校验摘要。

#### VisA

安装 AWS CLI 后可匿名下载：

```powershell
aws s3 cp --no-sign-request s3://amazon-visual-anomaly/VisA_20220922.tar data/downloads/VisA_20220922.tar
```

若无 AWS CLI，使用官方 Registry 页面提供的对象入口下载。随后：

```powershell
tar -xf data/downloads/VisA_20220922.tar -C data/visa_raw
python methods/anomalydino/<官方VisA整理脚本或项目适配脚本>
python scripts/validate_dataset.py --dataset visa --root data/visa_1cls
```

验收：数据校验无缺图、无孤立 mask、类别数正确；许可文件和来源写入 `data/README.md`。

### P2：生成统一 1/2/4-shot 清单

```powershell
python scripts/prepare_splits.py --dataset mvtec --root data/mvtec_ad --shots 1 2 4 --seeds 0 1 2
python scripts/prepare_splits.py --dataset visa --root data/visa_1cls --shots 1 2 4 --seeds 0 1 2
```

检查：

- 每份清单只包含正常训练图；
- 相同 seed 的 1⊂2⊂4；
- 运行两次所得文件哈希一致；
- 清单使用相对路径，可跨机器复用。

### P3：AnomalyCLIP 首个冒烟测试

1. 进入 `methods/anomalyclip`，完整阅读 README、`requirements.txt`、`test.sh` 和数据 JSON 生成脚本。
2. 建立 `.venv-anomalyclip`，安装与 6 GB GPU 兼容的 PyTorch CUDA wheel，再装官方依赖。
3. 从官方仓库链接下载 checkpoint，保存到官方要求目录，并记录 SHA256。
4. 为 MVTec 生成 `meta.json`。
5. 只跑 `bottle`，batch=1：
   - 先用 1 张正常图、1 张异常图确认推理；
   - 再跑 bottle 全测试集；
   - 输出图像分数、热力图和官方指标。
6. 检查热力图：
   - shape 与 GT 对齐；
   - 正常图无明显全屏高响应；
   - 异常图热点与缺陷区域大致一致；
   - image score 方向正确。
7. 保存完整命令、stdout/stderr、checkpoint、commit SHA、运行时间和显存。

验收：单类别进程退出码为 0，所有预测可追溯，至少人工检查 5 张正常和 5 张异常图。

### P4：AnomalyCLIP 原协议完整复现

1. MVTec AD：使用官方“在 VisA 上训练的权重”测试，避免目标数据训练泄漏。
2. VisA：使用官方“在 MVTec AD 上训练的权重”测试。
3. 输出全部类别和 macro mean。
4. 与官方表格比较，记录绝对差值与可能原因。
5. 对官方仓库 2025-04 修复前后的中间层局部输出行为做版本说明，固定当前 commit。

验收：两个数据集均可批量运行；官方与本地的指标名称、聚合方式和后处理已经核对。

### P5：特征导出接口

在不改变预测逻辑的前提下，以 hook/adapter 导出：

- 全局图像特征；
- 官方使用的每个中间层 patch token；
- normal/abnormal 文本锚点；
- 每层文本异常图；
- 最终异常图；
- 图像 resize/crop 与 token grid 元数据。

以 `.npz` 或分片 `.pt` 保存，文件名包含 dataset/category/sample/commit/checkpoint。先对 bottle 的 10 张图验证“导出前后预测逐元素一致或在浮点容差内一致”。

验收：缓存可独立加载；每个 patch 坐标可映射回输入图；不开启导出时性能无变化。

### P6：WinCLIP/WinCLIP+

1. 先运行 AnomalyCLIP 仓库内置 WinCLIP 复现，共用同一 CLIP 与数据环境。
2. 用独立第三方复现 `caoyunkang/WinClip` 交叉核对关键设置。
3. 零样本 WinCLIP 报单独表；少样本 WinCLIP+ 才进入 1/2/4-shot 主表。
4. WinCLIP+ 必须读取统一 shot 清单，输出所选正常参考图。
5. 记录窗口尺度、prompt ensemble、聚合方式和运行时。

验收：零样本与少样本结果不混表；能够明确解释 WinCLIP、WinCLIP+、AnomalyCLIP 的信息来源差异。

### P7：PatchCore

1. 克隆 Amazon 官方实现。
2. 建立 `.venv-patchcore`；首轮使用 CPU FAISS。
3. 原协议以 WideResNet50、layer2/layer3、官方 resize/imagesize 和 coreset 比例运行。
4. 先 bottle，再全部 MVTec。
5. 统一少样本协议中，PatchCore 的 memory bank 只使用统一 K 张正常图；若 K 太小时关闭或记录 coreset 的实际保留数。
6. 再跑 VisA 1/2/4-shot。

验收：可导出 memory bank、近邻距离和异常图；这部分接口可供下一阶段视觉参考分支设计参考。

### P8：统一评价层

1. 编写方法无关的预测 schema 和读取器。
2. 用合成小样本为 AUROC、AP、F1-max、pixel AUROC、pixel AP 和 AUPRO 写单元测试。
3. 对同一组 AnomalyCLIP 预测，同时用官方指标和统一指标计算，定位任何差异。
4. 生成：
   - `per_image.csv`；
   - `per_category.csv`；
   - `summary.csv`；
   - 失败样本可视化。

验收：同一预测重复评价结果完全一致；正常/异常分数方向错误会被自动检测。

### P9：PromptAD（2024）

1. 使用官方仓库的 Python 3.10 环境和安装脚本。
2. 分别核对分类与分割脚本，因为官方将两者拆开。
3. 原协议跑 MVTec 与 VisA。
4. 统一协议强制读取相同 K-shot 清单；记录它是否对目标正常样本进行 Prompt 训练、训练步数和每类耗时。
5. 输出文本引导图、视觉记忆图及固定融合结果（如官方实现可导出）。

验收：明确它与本项目的差异不是“是否使用正常参考”，而是目标 Prompt 学习与固定融合对区域级可靠性路由的差异。

### P10：AnomalyDINO（2025）

1. 官方仓库默认支持 MVTec、VisA 和 `--shots 1 2 4`。
2. 首轮命令：

```powershell
python run_anomalydino.py --dataset MVTec --shots 1 2 4 --num_seeds 3 --preprocess agnostic --data_root <mvtec-root> --faiss_on_cpu --eval_segm
python run_anomalydino.py --dataset VisA --shots 1 2 4 --num_seeds 3 --preprocess agnostic --data_root <visa-root> --faiss_on_cpu --eval_segm
```

3. 原协议结果保留官方内部抽样；统一协议改为读取 `data/splits/`。
4. 保存 DINOv2 模型版本、PCA masking/rotation augmentation 设置和 memory bank 耗时。

验收：纯视觉强基线在两个数据集上都有 mean±std，可与 CLIP 方法公平比较。

### P11：ReMP-AD（2025）

1. 固定官方 ICCV 2025 代码 commit 和预训练权重。
2. 先复现论文默认 shot/seed，再接入统一清单。
3. 重点导出：
   - 正常 token memory；
   - 检索/类别平衡权重；
   - 视觉—语言 prompt fusion 输出；
   - 最终热力图。
4. 仔细检查源域训练和目标域适配规则，避免与本项目主协议混淆。

验收：在复现报告中逐项比较 ReMP-AD 与拟议路由模块的信息来源、融合粒度、可靠性估计和训练要求。

### P12：AdaptCLIP（2026）

1. 官方代码：`gaobb/AdaptCLIP`；官方 checkpoint 来自 Hugging Face。
2. 生成与 AnomalyCLIP 兼容的 `meta.json`。
3. 先用 checkpoint 执行 `scripts/test_adaptclip.sh`。
4. 分开报告 zero-shot 和 1/2/4-shot；确认 visual/textual/prompt-query adapter 的源域训练规则。
5. 6 GB GPU 若 OOM，依次尝试 batch=1、关闭无关缓存、降低并发；不得静默改变会影响结果的分辨率。

验收：作为 2026 强基线进入最终表，任何硬件导致的协议变动必须单列。

## 7. 实验矩阵与资源控制

核心矩阵：

```text
7 methods × 2 datasets × 3 shots × 3 seeds
```

完整逐方法逐类运行数量很大，按三级闸门推进：

- Gate A：`bottle`, 1-shot, seed 0 冒烟。
- Gate B：MVTec 全类，1-shot, seed 0。
- Gate C：MVTec + VisA，1/2/4-shot，3 seeds。

只有上一级满足退出码、输出 schema、指标和可视化检查，才进入下一级。失败运行保留日志但不覆盖成功结果。

## 8. 结果表设计

主表按 shot 分组，报告 mean±std：

| Method | Target tuning | MVTec I-AUROC | MVTec P-AUROC | MVTec AUPRO | VisA I-AUROC | VisA P-AUROC | VisA AUPRO |
|---|---|---:|---:|---:|---:|---:|---:|

补充表：

- 原论文数值 vs 本地原协议；
- 统一协议 1/2/4-shot；
- 每类别结果；
- 运行时、显存、参数量、参考库大小；
- 失败案例和不同 seed 方差。

零样本方法单列；不能与使用 K 张正常参考的结果用同一列含混比较。

## 9. 进入第二阶段的门槛

只有同时满足以下条件才开始正常视觉参考分支和动态路由：

1. AnomalyCLIP 两个数据集的推理和特征导出稳定；
2. 至少 WinCLIP+、PatchCore、PromptAD、AnomalyDINO 完成统一 1/2/4-shot；
3. ReMP-AD 和 AdaptCLIP 至少完成官方协议或明确记录不可复现原因；
4. 统一评价通过单元测试并与官方实现交叉核验；
5. 所有 shot 清单、配置、日志和结果均可追溯；
6. 已形成文本分支与视觉分支误差互补性的初步统计，而不是凭直觉增加路由模块。

## 10. 立即执行顺序

当前从以下动作开始：

1. 完成项目骨架与状态记录；
2. 下载并固定 AnomalyCLIP 官方代码；
3. 检查官方依赖、测试脚本、权重链接和数据 JSON 生成方式；
4. 建立 AnomalyCLIP 独立环境；
5. 获取 MVTec AD（需在官方页面填写许可表单）和 VisA；
6. 执行 bottle 冒烟测试；
7. 冒烟通过后再下载其余方法并扩展实验。

## 11. 2026-07-28 执行检查点与后续顺序

已完成：

1. VisA 数据、统一 1/2/4-shot 清单、清单哈希和完整性检查；
2. 方法无关的 NPZ 评价层、六项统一指标和单元测试；
3. PatchCore 的 VisA 12 类 × 3 shots × 3 seeds 完整矩阵；
4. WinCLIP+ 的 VisA 12 类 × 3 shots × 3 seeds 完整矩阵；
5. 两种方法的逐运行结果、mean±std 汇总、实验登记和第三方补丁留档。

下一步严格按以下顺序执行：

1. AnomalyDINO：固定官方代码与环境，先做 VisA/candle 1-shot seed 0
   冒烟，再接统一清单和 NPZ 导出，最后跑完整 VisA 矩阵；
2. PromptAD：分别检查分类与分割入口，完成同样的 Gate A/B/C；
3. ReMP-AD：完成官方 checkpoint 推理，确认源域训练与目标域适配边界，
   至少形成可核验的官方协议结果；
4. AdaptCLIP：下载官方 checkpoint，以 batch size 1 做 6 GB 显存门控，
   再判断完整矩阵是否需要额外算力；
5. MVTec AD：等待用户通过官方许可页面取得数据后，立即执行数据校验、
   统一清单生成，并把已完成方法扩展到 MVTec；
6. 当至少 WinCLIP+、PatchCore、PromptAD、AnomalyDINO 的两个数据集矩阵
   齐全后，统计文本分支与视觉分支的样本级、区域级误差互补性，再进入
   动态路由模块。

当前外部阻塞：MVTec AD 不能绕过官方许可表单自动下载。在该数据到位前，
优先完成不受许可阻塞的 VisA 方法复现，不空等。

## 12. 2026-07-29 项目进度复盘与后续完整计划

### 12.1 当前已完成

1. 项目骨架、Git 记录、实验登记表、统一协议和数据清单已经建立。
2. VisA 数据已下载、校验并生成统一的 1/2/4-shot、3-seed 清单；清单哈希已固定。
3. 统一 NPZ 预测格式和评价脚本已完成，AUROC、AP、F1-max、pixel AUROC、pixel AP、AUPRO 五项测试通过。
4. PatchCore 已完成 VisA 全部 12 类 × 3 shots × 3 seeds。
5. WinCLIP+ 已完成 VisA 全部 12 类 × 3 shots × 3 seeds，并启用特征缓存。
6. AnomalyDINO 已完成官方代码固定、DINOv2 权重校验、Windows/子集/缓存适配，以及全部 9 组统一矩阵运行。
7. PromptAD 官方源码已固定，VisA/candle 1-shot 分类 Gate A 已成功，image AUROC 为 92.92%。
8. MVTec AD 仍受官方许可下载流程限制，尚未进入本地实验。

### 12.2 当前正在执行

1. 完成 PromptAD 的 VisA/candle 1-shot 分割 Gate A。
2. 将 PromptAD 数据抽样替换为冻结的统一 manifest，并导出共同 NPZ 格式。
3. 完成 PromptAD VisA 全矩阵，再进入 ReMP-AD 和 AdaptCLIP。

### 12.3 下一阶段的执行顺序

#### 阶段 A：收尾 AnomalyDINO

1. 检查 9 个运行目录是否都有 `evaluation_report.json`、12 类结果和 2162 张测试样本。
2. 重新运行统一评价单元测试和注册表校验。
3. 生成 `experiments/summaries/anomalydino_visa_unified/`。
4. 更新 `PROJECT_STATUS.md`、`docs/reproduction_notes.md` 和本计划。
5. 将 AnomalyDINO 与 WinCLIP+/PatchCore 的结果加入同一张对比表。

#### 阶段 B：PromptAD Gate A/B/C

1. 固定官方仓库 commit，记录依赖版本和 checkpoint 来源。
2. 先只跑 VisA/candle、1-shot、seed 0，确认分类和分割入口都能运行。
3. 将官方数据读取改为统一 manifest，导出共同 NPZ 格式。
4. 通过 Gate A 后跑 VisA 全 12 类 seed 0，再跑 3 seeds × 3 shots。
5. 如果官方方法需要目标域 Prompt 学习，单独记录训练样本是否来自统一正常样本，不能与零样本结果混列。

#### 阶段 C：ReMP-AD 与 AdaptCLIP

1. 对每个方法先做来源、commit、权重和许可证审计。
2. 分别完成单类 Gate A；若官方 checkpoint 或数据入口不可用，保留可复核的阻塞记录。
3. 只对能稳定输出共同 NPZ 的方法运行完整 VisA 矩阵。
4. 记录显存、运行时间、图像分辨率、是否目标域调参和异常图后处理。

#### 阶段 D：MVTec AD

1. 用户通过官方许可流程取得数据后，计算压缩包哈希并解压。
2. 生成并校验 MVTec metadata 和统一 1/2/4-shot manifest。
3. 按 Gate A → 全类别 seed 0 → 3 seeds × 3 shots 扩展已有方法。
4. 先完成 WinCLIP+、PatchCore、AnomalyDINO、PromptAD，再补 ReMP-AD/AdaptCLIP。

#### 阶段 E：统计、比较与第二阶段入口

1. 统一生成按方法/数据集/shot 的 mean±std 表、逐类别表和失败案例表。
2. 分开报告零样本、少样本和目标域 Prompt/Adapter 学习结果。
3. 分析文本分支与视觉分支的互补性，检查不同类别和缺陷区域的误差。
4. 满足第二阶段门槛后，再设计动态融合模块；在此之前不修改基线评价协议。

### 12.4 每个阶段的验收条件

- 所有成功运行均有固定配置、日志、输入清单哈希和可重算的预测文件。
- 汇总表只能由脚本从逐 run 文件生成，不能手工填数。
- 任何失败运行保留日志，不覆盖之前的成功结果。
- 结果比较必须注明数据集、shot、seed、目标域调参和分辨率。
