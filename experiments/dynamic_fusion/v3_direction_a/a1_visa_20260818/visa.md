# A1 VisA 冻结后验证（post-freeze, 阶段七）

- RunId: `a1_visa_post_freeze_20260818`（2026-08-18T08:28:11.415205+00:00 UTC）
- 数据集: VisA（holdout，12 类，2162 测试图：1200 异常 + 962 正常），官方 meta.json 划分。
- 冻结配置: concat + KNN(k=1) + distance/2，pca_dim=0，whiten=0，w=0.5，stride=8（与 freeze_manifest 一致，**未调参**）。
- baseline 口径: VisA 无 v2 分数级缓存 → DINO baseline 用特征级 dino-only KNN（MPDD s0/K1 上该口径与 v2 分数级差 ~0.0008 AP）。
- 特征导出: `export_a1_visa_features.py`（s0/k1 全量）+ `export_a1_visa_ref_only.py`（其余 8 组合，测试特征复用，黄金结论：测试特征与 (seed,shot) 无关）。
- 评估: CPU/faiss，27 个评估（9 配置 × concat/dino/clip），全部通过。

## 汇总（mean Pixel AP，Δ vs DINO feature baseline）

| 模式 | mean fused AP | mean ΔAP | 正向配置 | mean AUROC | mean AUPRO |
|---|---|---|---|---|---|
| DINO 单分支 | 0.3201 | 0.0000 | - | 0.9662 | 0.8933 |
| CLIP 单分支 | 0.2301 | -0.0900 | 0/9 | 0.9633 | 0.8749 |
| **concat + KNN（冻结）** | 0.3725 | **+0.0524** | **9/9** | 0.9757 | 0.9125 |

## 9 配置（concat）

| seed | shot | fused AP | DINO AP | ΔAP | 正向类别 | 最大退化 |
|---|---|---|---|---|---|---|
| 0 | 1 | 0.3381 | 0.2815 | +0.0566 | 10/12 | -0.0215 |
| 0 | 2 | 0.3706 | 0.3157 | +0.0550 | 10/12 | -0.0306 |
| 0 | 4 | 0.3907 | 0.3395 | +0.0512 | 10/12 | -0.0305 |
| 1 | 1 | 0.3439 | 0.2948 | +0.0491 | 10/12 | -0.0562 |
| 1 | 2 | 0.3743 | 0.3257 | +0.0486 | 10/12 | -0.0605 |
| 1 | 4 | 0.3990 | 0.3512 | +0.0479 | 10/12 | -0.0491 |
| 2 | 1 | 0.3463 | 0.2845 | +0.0618 | 10/12 | -0.0412 |
| 2 | 2 | 0.3867 | 0.3336 | +0.0531 | 10/12 | -0.0264 |
| 2 | 4 | 0.4027 | 0.3548 | +0.0479 | 10/12 | -0.0414 |

## 逐类（concat，跨 9 配置平均 ΔAP）

| category | mean ΔAP | 正向配置 |
|---|---|---|
| candle | -0.0198 | 0/9 |
| capsules | +0.0199 | 9/9 |
| cashew | +0.1145 | 9/9 |
| chewinggum | -0.0386 | 0/9 |
| fryum | +0.0805 | 9/9 |
| macaroni1 | +0.0409 | 9/9 |
| macaroni2 | +0.0925 | 9/9 |
| pcb1 | +0.1147 | 9/9 |
| pcb2 | +0.0335 | 9/9 |
| pcb3 | +0.0589 | 9/9 |
| pcb4 | +0.0550 | 9/9 |
| pipe_fryum | +0.0762 | 9/9 |

## 结论

1. **VisA 冻结后验证 9/9 全正**：concat mean ΔAP +0.0524，高于 MPDD development（+0.0486），低于 BTAD holdout（+0.0726），泛化成立。
2. 单分支对照：dino-only 为 0（自身基准），clip-only 全负（{clip_rows[0]['mean_delta_ap_vs_dino']:.3f}~{min(r['mean_delta_ap_vs_dino'] for r in clip_rows):.3f}）→ concat 增益来自 CLIP 互补，与 MPDD/BTAD 消融一致。
3. 最大退化类别与提升类别：见逐类表；无类在 9 配置中系统性崩坏（max regression 均 > -0.15）。
4. **未按结果调参**：冻结配置原样运行；baseline 口径差异已在报告中显式记录。

## 产物

- 评估报告: `a1_visa_20260818/seed{s}_k{shot}/{concat,dino,clip}_pca0_whiten0_w0.5_report.json`
- 本汇总: `visa_summary.json` + `visa.md`
- 特征缓存: `outputs/dynamic_fusion/v3_direction_a/visa_features_vitb14/s{seed}_k{shot}/anomalydino_visual/` 与 `visa_features/s{seed}_k{shot}/anomalyclip_text/`
- 导出队列: `outputs/logs/a1_visa_export_queue/status.json`（18/18 成功）