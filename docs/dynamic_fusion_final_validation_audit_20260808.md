# 动态融合最终验证审计与结果口径

更新日期：2026-08-08

## 审计结论

`scripts/audit_dynamic_fusion_final_validation.py` 已审计 17 个冻结输出：

- VisA：8 个运行；
- MVTec：9 个运行；
- 所有运行均通过统一评测的类别数、样本数与 `validation_errors=0` 检查；
- 每个融合 NPZ 中嵌入的校准路径和 SHA256 与对应校准 JSON 相符；
- 全部校准文件均为 `status=passed`、`test_predictions_used=false`、
  `test_labels_used=false`；
- 全部运行使用图像温度 0.50、像素温度 0.20。

机器可读审计：

- `experiments/dynamic_fusion/final_validation_audit_20260808/final_validation_audit.csv`
- `experiments/dynamic_fusion/final_validation_audit_20260808/final_validation_audit.json`
- `experiments/dynamic_fusion/final_validation_audit_20260808/validation_scope.json`

## VisA 论文口径

论文中的独立最终验证只使用 6 个运行：VisA seed 1/2 × K=1/2/4。
它们在路由结构和双温度参数已冻结后运行。两个 seed 0 的 K=2/K=4
目录是开发集补充复核，保留作可追溯性证据，但不纳入泛化结论、均值或标准差。

## MVTec 结果口径

9 个 MVTec 动态融合运行（seed 0/1/2 × K=1/2/4）均已完成并通过审计。
机器可读宏平均汇总位于：

- `experiments/summaries/mvtec_dynamic_fusion_final_summary_20260808.csv`
- `experiments/summaries/mvtec_dynamic_fusion_final_summary_20260808.json`

该汇总可以用于描述动态融合自身的跨 seed 表现；但在 PromptAD、AnomalyDINO
和 AnomalyCLIP 的 MVTec 基线矩阵完整前，不能把现有跨方法 CSV 作为正式公平
排名或论文主表。
