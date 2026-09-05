# Track-2 立项：多层/中段互补性测量（doc23 遗留空白，首次系统回答）

日期：2026-09-05　上游：doc30 路线图（Track-1 已归档 → Track-2 启动）；doc23 §3（CASF Wave1/2 从未跑，"真实多层互补性是否存在"从未被实验回答）。
性质：**测量先行**——先回答"中间层特征相对 A1（final DINO+final CLIP concat）是否存在可用的缺陷判别互补性"；有 headroom 才建最小融合，否则归档转 Track-3。

## 1. 动机与问题
- 全部已实验都只在**末层融合**上做（A1/DNC/容量/可学习对角/关系描述子），末层空间已证近平坦。
- 多层互补性假说（doc23）：不同深度的 DINO/CLIP token 对"缺陷类型×类别"的响应不同；把互补中间层**加入 A1 拼接**（仍是零训练、support-only、KNN）可能补上末层缺陷信号盲区。
- 问题预注册：**在 A1 拼接中加入中间层 token（同网格）能否在留出图/留出族结构下提高缺陷 Pixel-AP，且不伤 normal 路径？**

## 2. 数据、层位与红线
- 数据：6 类 MPDD，k2（先判；通过再补 k4）。support 图像 1024 渲染合成变体（三族×3 种子）与 15 光度 nuisance，经冻结提取器重编码；LOO memory（不含查询图）；不读 /test/good；不用真实缺陷；不训练任何参数。
- 层位（预注册）：
  - DINOv2 vitb14（448，32 网格）：final（复用 v14 缓存，即 L11 输出）与 **block5 中间输出**（新导出）；
  - CLIP ViT-L/14@336px（518，37→32 网格）：final（复用 v14 缓存 = pf[-1]）与 **pf[0]/pf[1]（一次前向内 DPAM 层 6/12）**（新导出）。
- 度量口径：16 网格 mean-pool；描述子经 A1 式行 L2/0.5 拼接后 `s=1−max_cos`；Pixel-AP（mask）按族汇总；normal 用尺度无关 nuisance-AUC。

## 3. Probe-M1 候选（只评价，不拟合）
| 候选 | 组成（16 网格 concat） |
|---|---|
| M0 | A1：dino_final + clip_final（基线） |
| M1 | A1 + dino_mid(block5) |
| M2 | A1 + clip_mid(层6) |
| M3 | A1 + dino_mid + clip_mid |

## 4. 门（预注册，按结果不调）
- **G-M1**：cutpaste 宏 AP（最弱族，A1≈0.76–0.83）：最优候选 − M0 ≥ **+0.05**（k2）；
- **G-M2**：erasure 宏 AP 相对 M0 ≥ −0.01（不得损害已近饱和的结构缺陷）；
- **G-M3**：normal 路径：最优候选 nuisance-AUC − M0 ≤ +0.05 且 ≤ 0.60。
- 通过 → 立项最小多层融合主线（冻结通道/拼接细节后进真实门路线）；失败 → 归档 Track-2 → Track-3。

## 5. 产物与成本
- `experiments/dynamic_fusion/innovation_t2_multilayer_20260905/`；脚本 `scripts/innovation_t2_multilayer_20260905/`。
- 新导出：dino block5 + clip 层6/12 于 support 图像（clean/9syn/15nui×6 类×k2），CPU ≈ 20–40 分钟；测量 ≈ 5–10 分钟。
- 报告：逐类×族 AP（M0–M3）、宏 Δ、normal AUC、层位实际提取说明。

## 6. 执行提示
> 执行 Track-2 Probe-M1：导出 dino block5 与 clip DPAM 层 6/12 的 support 合成特征（复用 v14 渲染/种子，只触碰 manifest support）；在 16 网格按 M0–M3 候选做 LOO memory KNN Pixel-AP（cutpaste/erasure 按族）与 nuisance-AUC；按 G-M1/G-M2/G-M3 判定。只评价不拟合，失败即归档并转 Track-3（doc32）。
