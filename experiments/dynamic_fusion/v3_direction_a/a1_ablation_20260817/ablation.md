# A1 特征级消融（dino-only / clip-only vs concat）

- RunId: `a1_mpdd_feature_level_ablation_20260818`（2026-08-18T03:55:12.886763+00:00 UTC）
- 数据集: MPDD（development），CPU/faiss，无 GPU。
- 目的: 矩阵 baseline 是 v2 **分数级**缓存；本消融跑 A1 **特征级**单分支 KNN，确认 concat 增益来自 CLIP 互补而非偶然。
- ΔAP 一律相对 v2 分数级 DINO baseline（`anomalydino_visual`），与矩阵口径一致。

## 配置

- dino-only: `--mode dino --pca-dim 0 --whiten 0`（vitb14 patch 特征，768 维）
- clip-only: `--mode clip --pca-dim 0 --whiten 0`（AnomalyCLIP ViT-L/14@336 patch 特征，768 维）
- concat（冻结）: 双分支 L2-normalize → concat(1152) → L2-normalize → KNN(k=1)，w=0.5（来自 `a1_matrix_20260817`）
- 单分支时 `--dino-weight 0.5` 仅决定模式名，实际只使用对应分支。

## 9 配置逐项（mean Pixel AP，Δ vs DINO baseline）

| seed | shot | dino-only Δ | clip-only Δ | concat Δ | concat−dino-only |
|---|---|---|---|---|---|
| 0 | 1 | -0.0008 | -0.0113 | +0.0290 | +0.0298 |
| 0 | 2 | +0.0129 | -0.0202 | +0.0410 | +0.0281 |
| 0 | 4 | +0.0333 | -0.0305 | +0.0596 | +0.0263 |
| 1 | 1 | +0.0325 | -0.0361 | +0.0396 | +0.0070 |
| 1 | 2 | +0.0317 | -0.0186 | +0.0569 | +0.0252 |
| 1 | 4 | +0.0242 | -0.0463 | +0.0555 | +0.0314 |
| 2 | 1 | +0.0172 | -0.0012 | +0.0434 | +0.0262 |
| 2 | 2 | +0.0207 | -0.0017 | +0.0506 | +0.0299 |
| 2 | 4 | +0.0328 | -0.0342 | +0.0614 | +0.0286 |

## 汇总

| 模式 | mean fused Pixel AP | mean ΔAP vs DINO | 正向配置数 | mean AUROC | mean AUPRO |
|---|---|---|---|---|---|
| DINO 单分支（特征级 KNN） | 0.3304 | +0.0227 | 8/9 | 0.9547 | 0.8805 |
| CLIP 单分支（特征级 KNN） | 0.2854 | -0.0222 | 0/9 | 0.9669 | 0.8959 |
| **concat + KNN（冻结 w=0.5）** | 0.3562 | **+0.0486** | 9/9 | 0.9646 | 0.9001 |

## 结论

1. **DINO 特征级 KNN 本身有正增益**：dino-only mean ΔAP +0.0227（相对分数级 baseline）。
2. **CLIP 单分支弱于 DINO baseline**：clip-only mean ΔAP -0.0222（CLIP 单独在 MPDD 上弱，符合历史）。
3. **concat 强于任一单分支**：concat mean ΔAP +0.0486，超过 dino-only（+0.0258）。
4. 因此 concat 增益不是简单平均或偶然，而是 **CLIP 在 concat 空间提供 DINO 缺失的互补信息**（KNN 邻居联合判定）带来的。
5. 泄漏审计：与矩阵一致，KNN memory bank 只用正常参考特征，测试标签/掩码仅用于最终评价（见 `a1_matrix_20260817/matrix_audit.json`）。

## 结论口径

- 本消融不改变冻结配置（concat w=0.5 仍为最优固定方案）。
- dino-only 特征级相对分数级有 +0.02~0.03 的提升值得记录，但它属于单分支特征级 KNN 的实现差异，不是融合增益。