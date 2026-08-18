# METHOD_CARD — A1 Feature-Level Fusion (concat + KNN memory bank)

RunId: `freeze/a1_mpdd_w05` · 冻结时间 2026-08-17 · 开发数据集 MPDD（development）。

> 本文档描述**冻结后唯一允许**的推理协议。任何改动（权重、PCA、whiten、融合顺序、
> 阈值、参考选择规则）都视为新的非冻结方法，须重新走审计与冻结流程。

## 1. 输入输出

### 输入（每类别、每 seed/shot）
- **正常参考**：manifest 声明的 K 张正常图（`data/splits/mpdd/manifest.json`，K ∈ {1,2,4}）。
- **测试图像**：待评估图像（推理时无标签/掩码）。
- **分支特征**（冻结导出，见 REPRODUCE.md）：
  - `anomalydino_visual`：DINO `dinov2_vitb14` patch tokens（32×32 grid，768 维）。
  - `anomalyclip_text`：AnomalyCLIP `ViT-L/14@336px` patch tokens（37×37 grid，768 维），
    checkpoint `9_12_4_multiscale_visa/epoch_15.pth`。

### 输出
- 每测试图一张 anomaly map（448×448，值越大越异常）；像素级排名用于 AUROC/AP/AUPRO。

## 2. 推理流程（冻结，逐步骤）

1. 提取 DINO patch 特征（grid 32×32）。
2. 提取 CLIP patch 特征（grid 37×37），**bilinear resize 到 DINO grid**（32×32）。
3. 两分支 patch 各自 **L2-normalize**（按最后一维）。
4. **加权 concat**：`f = [0.5 * dino_norm ; 0.5 * clip_norm]`（1152 维，不 PCA、不 whiten、不去中心）。
5. 对 concat 特征整体 **L2-normalize**。
6. 参考 patch 特征建 **faiss IndexFlatL2**（k=1）memory bank。
7. 测试 patch `distance/2.0` → 每 patch 异常分数 → 上采样到 448×448 anomaly map。

## 3. 允许 / 禁止信息

### 允许
- K 张正常参考图的 patch 特征（构造 memory bank 与归一化统计）。
- 测试图自身的冻结分支特征。
- 位置信息（patch 坐标，用于 dists2map）。

### 禁止（冻结协议下绝不使用）
- 测试标签 `gt_sp`、测试掩码 `imgs_masks`、测试集整体统计（分位数/均值/阈值）。
- 任何由测试真值挑选的类别规则、权重或阈值。
- PCA / whitening / centering（实证破坏 KNN 原点语义，见项目记忆"严禁去中心化"）。
- 测试集的正常样本筛选（`test_normal_selection_used=false`）。

## 4. 失效条件（回退逻辑）

A1 是**固定权重静态融合**，无运行时路由，故失效处理为**离线标记**而非推理时切换：

| 条件 | 处置 |
|---|---|
| 任一分支参考特征缺失/NaN/Inf | 该 (seed, shot, category) 评估无效，按失败审计处理 |
| 参考图少于 K 张（manifest 不一致） | 无效，须修复 manifest 后重审 |
| 分支特征维度/grid 与冻结不符 | 无效（freeze_manifest 哈希不匹配） |
| 推理时遇到不可靠分支（未来若扩展路由） | 回退视觉分支 `anomalydino_visual`（安全默认），见 V3.3-clean/局部救援协议 |

## 5. 冻结证据

- 9 配置矩阵 ΔAP 全正（mean +0.0486），审计 9/9 通过。
- 权重扫描：w=0.4 vs w=0.5 差 +0.0009（噪声内）→ 保持等权 w=0.5（无超参）。
- 动态（参考自 KNN 紧凑度路由）vs 固定：+0.0009、胜率 44% → 动态未超过固定，冻结固定。
- 全部哈希：`experiments/dynamic_fusion/freeze/a1_mpdd_w05/freeze_manifest.json`。
