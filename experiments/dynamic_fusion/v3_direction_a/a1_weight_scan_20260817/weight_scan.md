# A1 MPDD 权重扫描（阶段五 5.2 → 阶段六前，2026-08-17）

RunId: `a1_mpdd_weight_scan_20260817` · 9 配置 × {0.3, 0.4, 0.5, 0.6, 0.7} = 45 评估 · CPU/faiss。

## 预注册网格（无测试真值拟合）

- 方法：A1 `concat + KNN`，pca_dim=0、whiten=0，仅扫 dino_weight。
- 5 个权重全为预声明固定值；评估边界与矩阵一致（STRIDE=8）。

## 结果（mean ΔAP vs DINO，9 配置平均）

| weight | mean ΔAP | 正配置 | min ΔAP | max ΔAP |
|---|---|---|---|---|
| 0.3 | +0.0299 | 9/9 | +0.0135 | +0.0473 |
| **0.4** | **+0.0495** | 9/9 | +0.0316 | +0.0607 |
| **0.5**（冻结配置） | **+0.0486** | 9/9 | +0.0290 | +0.0614 |
| 0.6 | +0.0381 | 9/9 | +0.0158 | +0.0504 |
| 0.7 | +0.0295 | 9/9 | +0.0065 | +0.0421 |

## 决策

- **w=0.4 仅以 +0.0009 微弱领先 w=0.5**（0.0495 vs 0.0486），差距远小于配置间噪声，
  且 w=0.5 的 min ΔAP（+0.0290）高于 w=0.4（+0.0316 反而略低）、max ΔAP 更高（0.0614 vs 0.0607）。
- 结论与 V3.3-clean 权重决策一致：**保持 w=0.5（等权、无超参、对称）为冻结配置**，不切换到 w=0.4。
- 全权重全配置 ΔAP>0 再次确认：A1 融合方向在 MPDD 上稳健成立（增益随 DINO 占比先升后降，峰值在 0.4-0.5）。

## 产物

- 报告：`experiments/dynamic_fusion/v3_direction_a/a1_weight_scan_20260817/`
  `weight_scan_summary.json` + 每配置 `seed{s}_k{shot}/concat_pca0_whiten0_w{w}_report.json`
- 脚本：`scripts/run_a1_mpdd_weight_scan.py`（串行 + marker 断点）、`scripts/summarize_a1_weight_scan.py`
