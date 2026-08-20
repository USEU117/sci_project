# V4 视觉—文本动态融合：G2/G3 结果汇总与第 12 节决策

日期：2026-08-19
RunId：`v4_vision_text_20260819`
数据集角色：MPDD = `development`
权威标准：`docs/DYNAMIC_FUSION_NEXT_STEPS.md`

## 0. 一句话结论

G2 强视觉锚点 Gate **未通过**（V1 代理版失败；官方 SubspaceAD V2 完整 54 配置审计同样失败），因此当前结果不满足第 12 节 A/B/C 的任何"视觉升级"前提，最终落点锁定为 **D（新视觉未超过 AnomalyDINO）**：停止 V4 算法扩展，诚实交付当前 A1（双视觉固定融合，`paper_eligible = false`）。

---

## 1. G0：状态 / 语义 / 来源审计 —— 通过

- A1 身份确认为 `dual_visual_fixed_fusion`，不是显式视觉—文本融合。
  - 导出脚本 `scripts/export_anomalyclip_mpdd_features.py` 只调用 `model.encode_image(...)`；
  - `prompt_learner` 被加载但从未参与导出；`encode_text` / image-text similarity 从未计算；
  - 全部 7 项语义证据 `all_passed = true`（见 `00_g0_audit/modality_semantics_audit.json`）。
- A1 冻结只读 verify：`all_ok = true`，229 项，无缺失 / 大小 / hash 不一致。
- 候选来源锁定表完成（`candidate_source_lock.json`）：
  - V0 AnomalyDINO、T0 AnomalyCLIP text、V1 SubspaceAD-style、T1 ReMP-AD 可用；
  - V2 官方 SubspaceAD、V3 FoundAD、T2 AdaptCLIP、V4-alt FastRef 均 `not_cloned` 或 `blocked_empty_repo`。

## 2. G1：V4 schema 与防泄漏测试 —— 通过

- `src/industrial_ad/fusion/v4_contracts.py` 与 `tests/test_v4_contracts.py` 已存在。
- 回归测试合计 **49 passed**：
  - `tests/test_v4_contracts.py`
  - `tests/test_v3_3_clean.py`
  - `tests/test_v3_3_rescue.py`
  - `tests/test_freeze_a1_mpdd.py`
- RouterInput / EvaluationTarget 物理隔离与五项防泄漏字段契约成立。

## 3. G2：强视觉锚点 Gate —— 未通过（H1 不支持）

方法：`subspace_style_same_backbone`（PCA 正常子空间重建残差，复用冻结 DINO raw patch cache）。
对照：matched feature-DINO-only KNN（`dists2map` + L2-normalize + FAISS k=1 + `distance/2`）。

| pca_ev | mean ΔP-AP | mean ΔAUPRO | 非负配置 | 4/6 类正配置 | 最差类别 ΔAP | Gate |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| 0.95 | -0.012419 | +0.006422 | 1/9 | 1/9 | -0.134502 | failed |
| 0.99 | +0.001234 | +0.005377 | 4/9 | 2/9 | -0.135386 | failed |

硬标准未满足（`g2_matrix_report.json` → `gate`）：

- `mean_delta_ap >= +0.010`：均不满足（0.95 为负，0.99 仅 +0.0012）。
- “P-AP 持平且 AUPRO >= +0.010”的备选条件：AUPRO 均 < +0.010，不满足。
- `non_negative_configs >= 7/9`：均不满足。
- `positive_4_of_6_cat_configs >= 7/9`：均不满足。
- `worst_category_delta_ap >= -0.020`：均严重突破（约 -0.135）。

结论：V1 无法稳定超过 matched DINO-KNN，更遑论官方 AnomalyDINO。按 G2 失败条款，V1 归档，不得继续扩 PCA/whiten/centering 网格。

## 3.5 官方 SubspaceAD（V2）完整 G2 审计 —— 未通过（H1 仍不支持）

2026-08-19 按"V1 失败后允许做一次官方 smoke"条款完成两阶段验证：

- **smoke（2/2 类别 PASS）**：metal_plate P-AP 0.8513（vs DINO-KNN 0.7633）、bracket_black 0.0655（vs 0.0207），推翻"代理失败 ⇒ 官方必失败"的推断（`05_v2_smoke/smoke_report.md`）。
- **完整 54 配置审计（6 类别 × 3 seeds × 1/2/4-shot）**：官方协议（`dinov2-with-registers-giant` + 672px + aug30 rotate + `pca_ev 0.99` + mean layers -12..-18 + reconstruction + batch 1，commit `ef56d5c`），对照 matched feature-DINO-only KNN（冻结 G2 矩阵 pca 0.99 行）。

| 指标 | 值 | Gate 要求 |
| --- | ---: | --- |
| mean ΔP-AP | **+0.0472** | ≥ +0.010 ✓ |
| mean ΔAUPRO | +0.0634 | — |
| 非负配置 | **9/9** | ≥ 7/9 ✓ |
| positive_4of6 配置 | **9/9** | ≥ 7/9 ✓ |
| 最差类别 ΔAP | **connector -0.1167** | ≥ -0.020 ✗ |
| gate_passed | **false** | — |

逐类 mean ΔAP：bracket_black **+0.147**、metal_plate **+0.130**、tubes **+0.090**、bracket_brown **+0.027**、bracket_white **+0.006**、connector **-0.117**。5/6 类别正且幅度可观，唯 connector 系统性负（9/9 配置全负，-0.02 ~ -0.16）。

**结论**：官方实现确证强视觉（PCA 子空间重建）在 MPDD 多数类别上显著优于 matched DINO-KNN，但 connector 类别退化突破预注册底线，Gate G2 硬标准未满足。按预注册纪律**不接受"看多数类别"的豁免**；H1（强视觉锚点）不成立。证据：`06_v2_g2_audit/g2_audit_report.json`（54 行 `per_config.jsonl` 全部附于同目录）。

技术说明（写入 provenance）：官方 `PCAModel` 硬编码 `torch.float64`（cov 累积），在 6GB 显存 + giant 模型（仅剩 ~479MiB）下 fp64 峰值导致 CUDA OOM 静默终止；审计采用 fp32 PCA + 周期性 `empty_cache()` 的工程适配，指标计算路径（reconstruction scoring / post-process / AU-PRO）与官方 `main.py` 一致。P-AUROC/P-AP 为 stride-8 抽样、AU-PRO 全分辨率。

## 4. G3：显式文本分支 —— 方向成立，但互补性仅对弱视觉成立

- T0 / AnomalyCLIP 显式 text-conditioned anomaly map 导出成功（MPDD 6 类，`outputs/dynamic_fusion/v4_text_maps_mpdd`）。
- 方向翻转验证通过：交换 normal/abnormal prompt 后 map 方向如预期变化（`g3-verify-flip` 已完成）。
- 文本单分支 P-AP 较低：宏平均 `0.139236`，且主要集中在 `metal_plate`(0.484) / `tubes`(0.264)，其余四类 < 0.031。
- Oracle headroom（逐像素 best-of-two，用 GT mask）：
  - vs matched DINO-KNN：mean headroom `+0.359423`，6/6 类 >= 0.005，top 类占比 0.278；
  - vs V1 subspace：mean headroom `+0.323811`，6/6 类 >= 0.005，top 类占比 0.299。

**关键限制**：该 Text Gate 的“视觉参照”是弱视觉（V1 与 matched DINO-KNN），因为强视觉锚点尚未成立。因此这里的 Oracle headroom 只能证明“文本在弱视觉之上有互补信息”，**不能**等价于第 3 节 H2 所要求的“文本在强视觉锚点犯错处仍有独立有效信息”。H2 严格意义上尚未被验证。

## 5. 第 12 节 A/B/C/D 映射

- H1（强视觉锚点）＝ **失败**：V1 未通过 G2；官方 SubspaceAD（V2）完整 54 配置审计亦未通过（worst 类 connector -0.117 突破 -0.020 底线）。
- H2（显式文本对强视觉互补）＝ **未达验证条件**：强视觉不存在，文本 headroom 只对弱视觉成立。
- H3 / H4：尚未进入（依赖 H1/H2）。

按第 12 节定义：

- A（完整成功）：否。
- B（H1/H2 过，H3/H4 败）：否，因 H1 未过。
- C（H1 过，H2 败）：否，因 H1 未过。
- **D（新视觉未超过 AnomalyDINO）＝ 最终落点（已锁定，不再有挂起分支）**。

对应第 12 节 D 的诚实交付：停止算法扩展，交付当前 A1（双视觉固定特征融合，相对 matched internal baseline 稳定有效，但不宣称 SOTA 或动态路由）。`paper_eligible = false`，`dataset_role = development`。

## 6. 关键限制（写入报告，不得回避）

1. G2 对照是 **matched feature-DINO-only KNN**，不是官方 AnomalyDINO；未独立运行官方 AnomalyDINO 的 MPDD 对照。
2. 官方 SubspaceAD（V2）审计使用本地 giant 权重 + 官方协议，但 **PCA 精度从官方 fp64 降为 fp32、并加周期性 `empty_cache()`**（6GB 显存适配），理论上有微小数值差异；reconstruction 打分与后处理路径与官方一致。此限制已记录于 `06_v2_g2_audit/g2_audit_report.json`。
3. G3 文本 Oracle headroom 的参照是弱视觉，不能作为 H2 成立的证据。
4. 文本单分支绝对 P-AP 很低，且集中在 2/6 类，进一步说明文本不宜作为独立主支路。
5. G3 文本 map 口径：`--features-list [6,12,18,24]` + `--feature-map-layer [0]`，按官方 `methods/AnomalyCLIP-main/test.py` 的 `if idx >= args.feature_map_layer[0]` 语义会叠加全部 patch 特征层；**不得声称"仅使用 layer 0"**。该口径与官方实现一致，但作为 provenance 必须记录。
6. Text Gate 的其余预注册子项未完成：prompt/view 稳定性阈值、保存"文本有益/有害位置"清单。因主线已按 D 停止，这两项标记为**未完成**而非通过。
7. 官方完整审计的 connector 退化（-0.117）发生在 PCA 子空间重建口径下，是**类别级系统性问题**（9/9 配置全负），非随机波动；在 D 结论下不进一步归因，但作为负结果证据保留。

## 7. 结论与归档状态（已关闭，无需再拍板）

路线 2（官方 SubspaceAD smoke → 完整审计）已按 G2 失败条款执行完毕：smoke 2/2 PASS 但完整 54 配置审计 FAIL（connector 单类 -0.117）。预注册判据无豁免，按用户既定决策"失败就毫不犹豫回到路线 1"：

- **最终决策 = D**：停止 V4 算法扩展，A1 为最终诚实交付（相对 matched internal baseline 稳定有效，不宣称 SOTA / 动态路由）。
- `paper_eligible = false`；G4–G11 永久阻断；G3 维持 partial（未构成 H2 证据）。
- 归档证据：`06_v2_g2_audit/`（`g2_audit_report.json` + 54 行 `per_config.jsonl`）、`05_v2_smoke/`、`02_visual_gate/v1_archived.json`；权威状态已更新于 `docs/CURRENT_DYNAMIC_FUSION_STATUS.md` / `docs/current_dynamic_fusion_status.json`。
- 唯一剩余主线：S6 论文交付（A1 工程论文 / 负结果研究，由用户另行决策）。
