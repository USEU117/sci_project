# Negative results index — 失败的动态/文本融合路线（仅作消融或补充材料）

以下路线有完整实验证据但**未进入论文主方法**，不得在正文宣称其有效：

1. **V3.3 / V3.4 dynamic fusion**：按不确定性/路由动态融合视觉分支。
   - 结论：动态路由不优于冻结的等权 concat；`dynamic_minus_fixed ≈ +0.0009`（噪声内）。
   - 证据：`experiments/dynamic_fusion/freeze/a1_mpdd_w05/freeze_manifest.json` 的 `dynamic_vs_fixed` 字段。
2. **V4 vision-text / 显式文本特征融合**：在推理中显式使用文本特征（AnomalyCLIP 文本侧）。
   - 结论：`explicit_text_features_used_at_inference = false`；A1 最终推理仅用两个视觉编码器的 patch 特征。
   - 证据：`experiments/dynamic_fusion/v4_vision_text_20260819/00_g0_audit/modality_semantics_audit.json`、
     `04_gate_decision/gate_decision.md`、`06_v2_g2_audit/g2_audit_report.json`。
3. **同 backbone 子空间融合**：在同一 DINO 特征子空间上做融合/路由。
   - 结论：互补性主要来自异构预训练视觉表征；同子空间路线未获增益（见 v4 g2 审计）。
4. **单分支 CLIP-only**：CLIP-only 弱于 DINO-only 与 concat，但提供互补信息（见各数据集 per-category 结果）。

这些证据只用于消融、讨论或补充材料；正文方法必须按 README 的冻结 A1 描述书写。
