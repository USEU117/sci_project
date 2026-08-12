# V3.5 实验 — 方向B与方向C探索

日期: 2026-08-12  
参照: `docs/few_shot_industrial_ad_project_overview_expanded_20260812_v2.docx` 第7节

## 方向C: 图像级分层融合

**思路**: 在图像级使用 DINO anomaly score 进行 per-image gate，像素级保持 V3.3 静态融合。

三种gate策略:
- `discrete_gate`: 3-bin 离散门控 (lo/hi thresholds)
- `continuous_gate`: 连续 sigmoid 门控 (min/max/steepness)
- `agreement_gate`: cross-modal agreement 门控 (boost/threshold)

**结论**: Oracle 上限仅 +0.010 ΔAP — 图像级门控无法利用文本分支的像素级优势。

详见:
- `s0_report.json`, `s1_report.json`, `s2_report.json` — 多seed完整结果
- `src/industrial_ad/fusion/v3_5_strategies.py` — 核心实现

## 方向B: 缺陷词 Prompt Ensemble

**思路**: 用手写缺陷词变体替代 learnable prompts，增强文本分支。

6个快速变体: "damaged {}", "broken {}", "scratched {}", "deformed {}", "cracked {}", "stained {}"

**结论**: Mean ΔAP = -0.036 (原始 learned prompts: +0.075)，手写缺陷词远不如 learned prompts。

详见:
- `s0_eval.json` — 完整对比结果
- `scripts/defect_ensemble_utils.py` — 文本特征构建
- `scripts/export_anomalyclip_defect_ensemble.py` — GPU导出
- `scripts/evaluate_v3_5_defect_ensemble.py` — CPU评估

## 总结

| 方向 | 方法 | ΔAP | 结论 |
|------|------|-----|------|
| C | 图像级gate (oracle) | +0.010 | 无法超越V3.3静态 |
| B | 缺陷词ensemble | -0.036 | 远不如learned prompts |

两个方向都证明了: **文本分支的pixel-level信息是核心瓶颈**，图像级gate和手写文本都无法解决。
