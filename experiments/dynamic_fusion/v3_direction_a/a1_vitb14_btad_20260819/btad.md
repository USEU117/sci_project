# A1 BTAD 冻结后验证（post-freeze, 阶段七）

- RunId: `a1_btad_matrix_20260819`（2026-08-18T23:46:45.566906+00:00 UTC）
- 数据集: BTAD（外部冻结后验证，3 类，741 测试图）。
- 冻结配置: concat + KNN(k=1) + distance/2，pca_dim=0，whiten=0，w=0.5，stride=8，map=448（与 freeze_manifest 一致，**未调参**）。
- baseline 口径: `anomalydino_visual` = legacy v2 dino **score** cache（`v2_btad_predictions`，按 (seed, shot) 匹配）；`dino` 模式的 `fused` 才是特征级 dino-only KNN。
- 特征导出: K1 全量（`raw_patch_features`）+ K2/K4 ref-only（测试特征复用 K1，黄金结论：测试特征与 (seed,shot) 无关）。
- 评估: CPU/faiss，18 个评估（9 配置 × concat/dino），无 standalone CLIP-only（与 MPDD 口径一致）。

## 汇总（mean Pixel AP）

| 模式 | mean fused AP | mean ΔAP vs legacy DINO | 正向配置 |
|---|---|---|---|
| feature-DINO-only KNN | 0.6206 | +0.0517 | 9/9 |
| **concat + KNN（冻结）** | 0.6455 | **+0.0766** | **9/9** |

## 三口径分解（对照 MPDD）

| 口径 | mean ΔAP | 正向配置 |
|---|---|---|
| ① concat vs legacy DINO score | +0.0766 | 9/9 |
| ② feature-DINO-only vs legacy DINO score | +0.0517 | 9/9 |
| ③ concat vs matched feature-DINO-only（纯融合贡献） | +0.0249 | 9/9 |

## 9 配置（concat）

| seed | shot | concat AP | legacy DINO AP | ΔAP | 正向类别 | 最大退化 |
|---|---|---|---|---|---|---|
| 0 | 1 | 0.6203 | 0.5646 | +0.0557 | 2/3 | -0.0281 |
| 0 | 2 | 0.6422 | 0.5506 | +0.0916 | 2/3 | -0.0145 |
| 0 | 4 | 0.6504 | 0.5589 | +0.0915 | 2/3 | -0.0084 |
| 1 | 1 | 0.6333 | 0.5594 | +0.0738 | 2/3 | -0.0469 |
| 1 | 2 | 0.6460 | 0.5725 | +0.0734 | 2/3 | -0.0392 |
| 1 | 4 | 0.6480 | 0.5833 | +0.0647 | 2/3 | -0.0492 |
| 2 | 1 | 0.6585 | 0.5701 | +0.0883 | 3/3 | +0.0102 |
| 2 | 2 | 0.6515 | 0.5792 | +0.0723 | 3/3 | +0.0083 |
| 2 | 4 | 0.6596 | 0.5819 | +0.0777 | 3/3 | +0.0052 |

## 逐类（concat，跨 9 配置平均 ΔAP）

| category | mean ΔAP vs legacy DINO | 正向 | mean ΔAP vs DINO-only | 正向 |
|---|---|---|---|---|
| 01 | +0.2063 | 9/9 | +0.0245 | 9/9 |
| 02 | -0.0181 | 3/9 | +0.0346 | 9/9 |
| 03 | +0.0415 | 9/9 | +0.0156 | 7/9 |

## 结论

1. **BTAD 冻结后验证（9 配置）**：concat mean ΔAP vs legacy DINO +0.0766（9/9 正）；纯融合贡献（vs matched feature-DINO-only）+0.0249。
2. 与 MPDD（+0.0486 / +0.0258）、VisA（+0.0524）、MVTec（+0.0320）的结论一致：concat 增益来自 CLIP 互补。
3. BTAD baseline 为 legacy v2 dino score cache（与 MPDD 同源），区别于 VisA/MVTec 的 feature-level dino-only KNN。

## 产物

- 评估报告: `a1_vitb14_btad_20260819/seed{{s}}_k{{2,4}}/{{concat,dino}}_pca0_whiten0_w0.5_report.json` + K1 旧目录 `a1_vitb14_btad_{fusion,dino}/seed{{s}}/`
- 本汇总: `a1_vitb14_btad_20260819/btad_summary.json` + `btad.md`
- 特征缓存: `outputs/dynamic_fusion/v3_direction_a/features_vitb14_btad_s{{seed}}_k{{shot}}/anomalydino_visual/` 与 `features_btad_s{{seed}}_k{{shot}}/anomalyclip_text/`