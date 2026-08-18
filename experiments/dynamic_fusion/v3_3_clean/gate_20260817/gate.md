# V3.3-clean MPDD seed0/K1 CPU Gate（2026-08-17）

RunId: `v3_3_clean_gate_20260817_mpdd_s0_k1` · 阶段三 · CPU，无 GPU，仅复用冻结缓存。

参考：`docs/DYNAMIC_FUSION_NEXT_STEPS.md` 阶段三。

## 输入缓存（全部冻结、只读）

- 测试预测：`outputs/dynamic_fusion/v2_mpdd_predictions/v2_mpdd_s0_k1_full_v1/{anomalydino_visual,anomalyclip_text}/{类}.npz`
  - DINO 448×448，CLIP 518×518（对齐后 resize 到 448）。
- 正常参考：`outputs/dynamic_fusion/v2_branch_cache/v2_mpdd_s0_k1_branch_cache_v1/...`
  - K=1 正常参考 + 5 个确定性视图，`pixel_maps` 作为校准来源。
- 每类逐文件 sha256 记录于报告 `provenance`。

## 预注册网格（无搜索、无测试真值拟合）

| 方法 | 说明 |
|---|---|
| `visual_only` | AnomalyDINO 原始图（默认安全输出） |
| `text_only` | AnomalyCLIP 原始图 |
| `v33_clean_w{0.40,0.50,0.60,0.70}` | V3.3-clean 固定权重（0.50 = 50/50） |
| `visual_fallback` | clean 文本强制不可靠 → 纯视觉锚定 |
| `old_v33_w060_invalid` | 旧泄漏 V3.3（gt_masks 校准），仅作无效标记对照 |

图像级阈值规则：`image_score > max(正常参考 image scores)`（仅参考导出）。
像素级指标 STRIDE=8。五个泄漏字段全 false。

## 结果（6 类聚合，Pixel AP = mean）

| 方法 | Pixel AP | Δ vs visual | Pixel AUROC | AUPRO | Img AUROC |
|---|---|---|---|---|---|
| visual_only | 0.2802 | — | 0.9578 | 0.8622 | 0.6643 |
| text_only | 0.2444 | -0.0358 | 0.8776 | 0.7234 | 0.7852 |
| v33_clean_w0.40 | 0.2975 | **+0.0173** | 0.9609 | 0.8741 | 0.6794 |
| v33_clean_w0.50 | 0.2945 | +0.0142 | 0.9602 | 0.8716 | 0.6774 |
| v33_clean_w0.60 | 0.2911 | +0.0108 | 0.9596 | 0.8699 | 0.6741 |
| v33_clean_w0.70 | 0.2873 | +0.0071 | 0.9590 | 0.8675 | 0.6711 |
| visual_fallback | 0.2802 | 0.0000 | 0.9578 | 0.8622 | 0.6643 |
| old_v33_w060_invalid | 0.3556 | +0.0754 | 0.9682 | 0.9004 | 0.7653 |

逐类（w0.40/0.50/0.60 Δ vs visual）：bracket_white +0.071/+0.058/+0.044、
connector +0.011/+0.007/+0.004、metal_plate +0.005/+0.010/+0.010、
tubes +0.007/+0.004/+0.003、bracket_brown ≈ 0、bracket_black +0.010/+0.006/+0.004。
**6/6 类全部正收益，无单类退化（max_reg > 0）。**

## Gate 判定（计划建议规则）

- mean Pixel AP 高于视觉：✅（全部 4 个 clean 权重）
- ≥4/6 类正收益：✅（6/6）
- 无单类大幅退化（max regression > -0.02）：✅（min 为 +0.0002）
- AUPRO 不整体下降：✅
- 审计与重复性：✅（15/15 CPU 测试，`tests/test_v3_3_clean.py`）

**结论：V3.3-clean 通过建议 Gate。** 4 个预注册权重全部通过，w=0.40 最优（+0.0173）。

## 关键发现：泄漏对旧 V3.3 结果的虚增

- 旧泄漏 V3.3（w=0.60，seed0 K1）mean ΔAP = **+0.0754**，是 clean（+0.0108）的 **7 倍**。
- 泄漏链路：用 `gt_masks` 挑「完全正常测试图」估计 median/IQR 校准统计 → 校准直接利用了测试掩码/标签信息 → 指标被系统性抬高。
- 因此历史 V3.3 主结果（3-seed LoO mean ΔAP = +0.0704）**不可作为论文证据**，其量级已被泄漏污染；clean 版本的"真实"增益约 +0.01~+0.017（s0/K1）。
- 好消息：clean 增益虽小但**方向一致且全类正、AUPRO 不降**，说明融合方向本身成立，只是旧数字不可信。

## 下一步（按计划决策）

Gate 通过 → 按计划"再判断动态路由是否超过最佳固定融合"。
当前 Gate 中动态元素（分支可靠性回退）未触发（text 分支全部可靠，`text_coverage=1.0`），
即 clean 动态路由 ≈ 最佳固定融合 w=0.40。在扩大矩阵（阶段五 5.2，需 GPU 授权）之前，
应先做一次"动态 vs 固定"的显式对照，或直接进入阶段四（视觉锚定的文本局部救援，CPU）。
