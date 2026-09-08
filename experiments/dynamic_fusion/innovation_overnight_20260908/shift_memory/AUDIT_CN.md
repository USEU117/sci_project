# 旧路线重复性审计

结论：本方向与旧 D-DEVA 是近邻路线，但不是完整重复，因此允许做一次有界 support-only 诊断。

旧证据：

- `src/industrial_ad/innovation_v2/equivariant_augmentation.py` 定义 D-DEVA 的 photometric augmentation，并要求双编码器等变性过滤。
- `scripts/innovation_v2/export_deva_references.py` 对正常 reference image 重新渲染、重新提取，再逆变换回坐标；`photometric` pack 固定 brightness `{0.90, 1.10}` × contrast `{0.90, 1.10}`。
- `scripts/innovation_v2/run_small_gates.py` 将通过过滤的 augmented descriptors 直接追加到 normal memory，再用 A1 KNN 评估真实 MPDD。
- `experiments/dynamic_fusion/innovation_v2/FINAL_DECISION.md` 记录 D-DEVA photometric 候选在 k2/k4 的 mean ΔPixel-AP 约 `+0.000011`，与 `unfiltered_augmentation` control 无可见差异；k2 photometric memory growth 为 `5.0`。
- `experiments/dynamic_fusion/innovation_t2_multilayer_20260905/TRACK2_DECISION.md` 只在 clean memory 上统计 15 个 photometric nuisance 的 AUC；没有把 nuisance feature 追加到 memory。`scripts/innovation_overnight_20260908/probe_support_selection.py` 同样只选择 clean image memory，nuisance 只作诊断。

本次差异：

1. 不重新渲染或重编码图像，直接复用 V14 的 15 个 support-only nuisance feature（5 族 × 3 强度），固定每族第 0 强度作为部署预算探针。
2. 采用 leave-one-support-image-out：query 是 heldout support image 的 synthetic/nuisance episode，memory 只来自其他 support image；不允许同图 cell 进入 memory。
3. 重点测量 heldout nuisance 分数分布（均值、p95、AUC、p99 超阈率）以及 synthetic AP，并设置相同 memory 行数的重复 clean control 来剥离“只是 memory 变大”的效应。
4. `augmented_loo_family` 仅作为预注册的 leave-family-out 交叉诊断，使用 heldout nuisance 族排除 memory 中同族变体；它不模拟部署时已知 query 族，实际部署候选是无路由的 `augmented_all5`。

因此，本轮不能写成新的 augmentation 算法或真实泛化确认；若指标没有超过 clean-only 和 equal-budget control，应归档为旧 D-DEVA 在新的 support-only shift 口径下的重复阴性结果。
