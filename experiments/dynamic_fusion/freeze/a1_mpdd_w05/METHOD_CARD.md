# METHOD_CARD — Reference-Conditioned Multimodal Feature Fusion with a Normal Memory Bank (A1)

RunId: `freeze/a1_mpdd_w05` · 冻结时间 2026-08-17 · 开发数据集 MPDD（development）。
修订：2026-08-18（S4：伪代码、schema、资源统计、失败案例与限制、四数据集验证证据）。

> **名称约定**：本方法为**固定权重特征融合**（concat + KNN memory bank），**不是动态路由**。
> 文档/论文不得将其包装为"动态路由成功"。权重 w=0.5 冻结，不随测试图变化。

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

### 伪代码

```
输入: R = K 张正常参考图; T = 测试图集; 冻结分支模型 DINO(vitb14), CLIP(ViT-L/14@336)
参数: dino_weight=0.5, pca_dim=0, whiten=False, stride=8, map_size=448, knn_k=1

def embed(x):
    d = DINO_patch(x)                         # (32*32, 768)
    c = CLIP_patch(x)[1:, :]                  # (37*37, 768)，去 CLS
    c = bilinear_resize_to_grid(c, 32x32)     # (32*32, 768)
    f = concat(0.5 * l2norm(d), 0.5 * l2norm(c))   # (32*32, 1152)
    return l2norm(f)                          # 整体再归一化

# 训练侧（memory bank 只用正常参考，绝不触碰测试真值）
M = concat([embed(im) for im in R])           # (K*32*32, 1152)
index = faiss.IndexFlatL2(1152); index.add(M)

# 推理侧（无标签/掩码）
for im in T:
    f = embed(im)
    dists, _ = index.search(f, k=1)           # 每个 patch 的最近正常参考距离
    score_map = (dists / 2.0).reshape(32, 32)
    anomaly_map = upsample(score_map, (448, 448))   # stride=8 语义

# 评价（仅 evaluator 读取测试真值）
AUROC, AP, AUPRO = compute_metrics(anomaly_map, gt_mask)   # stride=8 下采样
```

## 3. 输入输出 schema

### 特征缓存 npz（每分支、每 (seed,shot,class)）
| 字段 | 类型 | 说明 |
|---|---|---|
| `sample_ids` | (N,) str | 每图唯一 ID，跨分支必须逐图对齐 |
| `features` | (N, 32*32, D) float32 | patch 特征；D=768（dino/clip 原始）、1152（concat 由评估器现场合成） |
| `grid` | (H, W) int | 空间网格（dino 32×32；clip 37×37 经 resize 后 32×32） |
| `branch` | str | `anomalydino_visual` / `anomalyclip_text` |
| `normal_reference_ids` | (K,) str | 构建 memory bank 用的正常参考 sample_id（manifest 来源） |

### 评估报告 JSON（每配置）
| 字段 | 说明 |
|---|---|
| `mode` | `concat` / `dino` / `clip` |
| `seed`, `dataset`, `dataset_role` | 配置元数据（角色见第 8 节） |
| `mean_fused.{pixel_auroc,pixel_ap,pixel_aupro}` | 冻结 evaluator 指标 |
| `mean_dino_baseline_ap` / `mean_delta_ap_vs_dino` | baseline 与增益（baseline source 显式） |
| `per_category[]` | 逐类 fused/baselines/delta |
| `leakage_flags` | 五项全 `false` |

## 4. 允许 / 禁止信息

### 允许
- K 张正常参考图的 patch 特征（构造 memory bank 与归一化统计）。
- 测试图自身的冻结分支特征。
- 位置信息（patch 坐标，用于 dists2map）。

### 禁止（冻结协议下绝不使用）
- 测试标签 `gt_sp`、测试掩码 `imgs_masks`、测试集整体统计（分位数/均值/阈值）。
- 任何由测试真值挑选的类别规则、权重或阈值。
- PCA / whitening / centering（实证破坏 KNN 原点语义，见项目记忆"严禁去中心化"）。
- 测试集的正常样本筛选（`test_normal_selection_used=false`）。

## 5. 失效条件（回退逻辑）

A1 是**固定权重静态融合**，无运行时路由，故失效处理为**离线标记**而非推理时切换：

| 条件 | 处置 |
|---|---|
| 任一分支参考特征缺失/NaN/Inf | 该 (seed, shot, category) 评估无效，按失败审计处理 |
| 参考图少于 K 张（manifest 不一致） | 无效，须修复 manifest 后重审 |
| 分支特征维度/grid 与冻结不符 | 无效（freeze_manifest 哈希不匹配） |
| 推理时遇到不可靠分支（未来若扩展路由） | 回退视觉分支 `anomalydino_visual`（安全默认），见 V3.3-clean/局部救援协议 |

## 6. 环境与资源统计（实测，RTX 3060 Laptop 6GB）

| 项 | 数值 |
|---|---|
| GPU | NVIDIA GeForce RTX 3060 Laptop GPU，6 GiB；单进程串行 |
| DINO vitb14 全量导出（448px，约 1700-2200 图） | 约 10 分钟 |
| CLIP ViT-L/14@336 全量导出（518px 推理） | 约 25 分钟 |
| ref-only 导出（复用 k1 测试特征） | 每个约 4-8 分钟（含模型加载） |
| 评估 | CPU/faiss（IndexFlatL2 k=1），9 配置 × 3 mode 共 27 评估 |
| 冻结哈希条目 | 229 项（9 代码 + 2 checkpoint + manifest + evaluator + 特征/基线 npz） |
| 内存注意事项 | BTAD 03 类（441 图，32×42 grid）单进程峰值约 3.4 GiB，需预留内存 |

## 7. 冻结证据与冻结后验证

- **开发（MPDD）**：9/9 配置全正，mean ΔAP **+0.0486**（vs legacy v2 dino score）；matched feature-DINO-only 相对 legacy 为 +0.0227，**纯 concat 贡献 +0.0258**；审计 9/9。
- **权重决策**：w=0.4 vs w=0.5 差 +0.0009（噪声内）→ 保持等权 w=0.5（无超参、对称）。
- **动态对照**：参考自 KNN 紧凑度路由 vs 固定：+0.0009、胜率 44% → 动态未超过固定，冻结固定。
- **冻结后验证**（全部用冻结配置原样，未调参）：
  - BTAD（K1，external）：3/3 正，mean ΔAP +0.0726。
  - VisA（in-domain，checkpoint 在 VisA 训练过）：9/9 正，mean ΔAP +0.0524。
  - MVTec（external）：9/9 正，mean ΔAP +0.0320。
- **只读验证（S1）**：`scripts/freeze_a1_mpdd.py --verify` 严格只读，229 项全量 hash 验证通过，verify 前后 manifest SHA256 不变；篡改测试 8/8 通过（`tests/test_freeze_a1_mpdd.py`）。

## 8. 数据集角色（权威口径）

| 数据集 | 角色 | 含义 |
|---|---|---|
| MPDD | `development` | 冻结配置的合法开发集 |
| BTAD | `external_frozen_validation_k1_only` | 仅 K1 三 seed 全覆盖 |
| VisA | `in_domain_frozen_validation` | 非独立 holdout（CLIP checkpoint 训练数据） |
| MVTec | `external_frozen_validation` | 冻结后新验证 |

## 9. 失败案例与限制（failure cases & limitations）

- **逐类退化**：MPDD bracket 硬类小幅退化（约 -0.006~-0.020）；MVTec leather（-0.043）、hazelnut（-0.030）、capsule（-0.016）、grid（-0.006）；VisA candle（-0.020）、chewinggum（-0.039）。无灾难类，但低纹理/光泽类（leather）增益最弱。
- **CLIP-only 单分支显著弱**（MPDD -0.022 / VisA -0.090 / MVTec -0.057，全负）：方法依赖 DINO 视觉主干，CLIP 只作互补信号。
- **不是动态路由**：逐图/逐像素不改变权重，无法针对个别失败图自适应。
- **9/9 的含义**：同一测试集上的 9 组参考采样配置（3 seeds × 1/2/4-shot）均正，**不是 9 个独立数据集**；不做伪独立显著性。
- **BTAD 覆盖不对称**：仅 K1 全量验证，K2/4 缺冻结版特征缓存。
- **VisA 域内性**：验证结论限定为"冻结配置在训练域数据上仍有效"，非独立泛化证据。
