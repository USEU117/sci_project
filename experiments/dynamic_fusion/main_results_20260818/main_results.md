# 统一性能表（main results）

RunId: `main_results_20260818` · 冻结配置: concat w=0.5, pca_dim=0, whiten=0, KNN k=1, stride=8, map=448

> 口径说明: 9/9 表示同一测试集上的 9 组参考采样配置（3 seeds × 1/2/4-shot），不是 9 个独立数据集。

| 数据集 | 角色 | 方法 | mean fused Pixel AP | mean ΔAP vs baseline | 正向配置 | baseline source |
|---|---|---|---|---|---|---|
| mpdd | development | A1 concat (frozen w=0.5) | 0.356238 | 0.048559 | 9/9 | legacy v2 dino score cache (v2_mpdd_predictions) + matched feature-level dino-only KNN |
| mpdd | development | feature-DINO-only KNN | 0.330408 | 0.022729 | 8/9 | legacy v2 dino score cache (v2_mpdd_predictions) + matched feature-level dino-only KNN |
| mpdd | development | CLIP-only KNN | 0.285444 | -0.022235 | 0/9 | legacy v2 dino score cache (v2_mpdd_predictions) + matched feature-level dino-only KNN |
| mpdd | development | A1 concat minus feature-DINO-only |  | 0.02583 | 9/9 | matched feature-level dino-only KNN (concat contribution isolated) |
| btad | external_frozen_validation | A1 concat (frozen w=0.5) | 0.645512 | 0.076575 | 9/9 | legacy v2 dino score cache (v2_btad_predictions, matched per (seed, shot)) |
| btad | external_frozen_validation | feature-DINO-only KNN | 0.620617 | 0.05168 | 9/9 | legacy v2 dino score cache (v2_btad_predictions, matched per (seed, shot)) |
| btad | external_frozen_validation | A1 concat minus feature-DINO-only |  | 0.024895 | 9/9 | matched feature-level dino-only KNN (concat contribution isolated; CLIP-only not evaluated standalone on BTAD) |
| visa | in_domain_frozen_validation | A1 concat (frozen w=0.5) | 0.372479 | 0.052353 | 9/9 | feature-level dino-only KNN (no v2 score cache on this dataset) |
| visa | in_domain_frozen_validation | feature-DINO-only KNN | 0.320125 | 0.0 | 0/9 | feature-level dino-only KNN (no v2 score cache on this dataset) |
| visa | in_domain_frozen_validation | CLIP-only KNN | 0.230102 | -0.090023 | 0/9 | feature-level dino-only KNN (no v2 score cache on this dataset) |
| mvtec | external_frozen_validation | A1 concat (frozen w=0.5) | 0.554611 | 0.031962 | 9/9 | feature-level dino-only KNN (no v2 score cache on this dataset) |
| mvtec | external_frozen_validation | feature-DINO-only KNN | 0.522649 | 0.0 | 0/9 | feature-level dino-only KNN (no v2 score cache on this dataset) |
| mvtec | external_frozen_validation | CLIP-only KNN | 0.465395 | -0.057254 | 0/9 | feature-level dino-only KNN (no v2 score cache on this dataset) |

## 重算校验（从 per-config report 重算 mean ΔAP）

- mpdd/concat: recomputed=0.048559 vs reported=0.048559 (diff 0.0) → PASS
- btad/concat: recomputed=0.076575 vs reported=0.076575 (diff 0.0) → PASS
- visa/concat: recomputed=0.052353 vs reported=0.052353 (diff 0.0) → PASS
- mvtec/concat: recomputed=0.031962 vs reported=0.031962 (diff 0.0) → PASS

- 全部通过: **True**