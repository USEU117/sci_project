# A1 MVTec 冻结后验证（post-freeze, 阶段七）

- RunId: `a1_mvtec_post_freeze_20260818`（2026-08-18T11:56:14.818410+00:00 UTC）
- 数据集: MVTec（holdout，15 类，1725 测试图）。
- 冻结配置: concat + KNN(k=1) + distance/2，pca_dim=0，whiten=0，w=0.5，stride=8（与 freeze_manifest 一致，**未调参**）。
- baseline 口径: 该 holdout 无 v2 分数级缓存 → DINO baseline 用特征级 dino-only KNN（MPDD s0/K1 上该口径与 v2 分数级差 ~0.0008 AP）。
- 特征导出: `export_a1_visa_features.py`（s0/k1 全量）+ `export_a1_visa_ref_only.py`（其余 8 组合，测试特征复用，黄金结论：测试特征与 (seed,shot) 无关）。
- 评估: CPU/faiss，27 个评估（9 配置 × concat/dino/clip），全部通过。

## 汇总（mean Pixel AP，Δ vs DINO feature baseline）

| 模式 | mean fused AP | mean ΔAP | 正向配置 | mean AUROC | mean AUPRO |
|---|---|---|---|---|---|
| DINO 单分支 | 0.5226 | 0.0000 | - | 0.9548 | 0.9065 |
| CLIP 单分支 | 0.4654 | -0.0573 | 0/9 | 0.9539 | 0.8732 |
| **concat + KNN（冻结）** | 0.5546 | **+0.0320** | **9/9** | 0.9663 | 0.9199 |

## 9 配置（concat）

| seed | shot | fused AP | DINO AP | ΔAP | 正向类别 | 最大退化 |
|---|---|---|---|---|---|---|
| 0 | 1 | 0.5615 | 0.5206 | +0.0410 | 12/15 | -0.0278 |
| 0 | 2 | 0.5753 | 0.5434 | +0.0319 | 12/15 | -0.0641 |
| 0 | 4 | 0.5771 | 0.5433 | +0.0339 | 13/15 | -0.0550 |
| 1 | 1 | 0.5324 | 0.5021 | +0.0303 | 11/15 | -0.0589 |
| 1 | 2 | 0.5394 | 0.5140 | +0.0254 | 11/15 | -0.0607 |
| 1 | 4 | 0.5480 | 0.5237 | +0.0242 | 12/15 | -0.0647 |
| 2 | 1 | 0.5234 | 0.4903 | +0.0331 | 11/15 | -0.0599 |
| 2 | 2 | 0.5631 | 0.5270 | +0.0361 | 12/15 | -0.0433 |
| 2 | 4 | 0.5713 | 0.5395 | +0.0318 | 12/15 | -0.0504 |

## 逐类（concat，跨 9 配置平均 ΔAP）

| category | mean ΔAP | 正向配置 |
|---|---|---|
| bottle | +0.0107 | 9/9 |
| cable | +0.0349 | 9/9 |
| capsule | -0.0161 | 5/9 |
| carpet | +0.0795 | 9/9 |
| grid | -0.0061 | 2/9 |
| hazelnut | -0.0297 | 0/9 |
| leather | -0.0428 | 0/9 |
| metal_nut | +0.0434 | 9/9 |
| pill | +0.0769 | 9/9 |
| screw | +0.0675 | 9/9 |
| tile | +0.0506 | 9/9 |
| toothbrush | +0.1080 | 9/9 |
| transistor | +0.0117 | 9/9 |
| wood | +0.0422 | 9/9 |
| zipper | +0.0487 | 9/9 |

## 结论

1. **MVTec 冻结后验证 9/9 全正**：concat mean ΔAP +0.0320，与 MPDD development（+0.0486）/ BTAD（+0.0726）/ VisA（+0.0524）一致，泛化成立。
2. 单分支对照：dino-only 为 0（自身基准），clip-only 全负 → concat 增益来自 CLIP 互补，与 MPDD/BTAD/VisA 消融一致。
3. 最大退化类别与提升类别：见逐类表；无类在 9 配置中系统性崩坏（max regression 均 > -0.15）。
4. **未按结果调参**：冻结配置原样运行；baseline 口径差异已在报告中显式记录。

## 产物

- 评估报告: `a1_mvtec_20260818/seed{s}_k{shot}/{concat,dino,clip}_pca0_whiten0_w0.5_report.json`
- 本汇总: `mvtec_summary.json` + `mvtec.md`
- 特征缓存: `outputs/dynamic_fusion/v3_direction_a/mvtec_features_vitb14/s{seed}_k{shot}/anomalydino_visual/` 与对应 clip 目录
- 导出队列: `outputs/logs/a1_mvtec_export_queue/status.json`（18/18 成功）