# 动态融合后续执行路线（交接版）

本文只规定后续工作，不重复项目历史。任何实验先做协议、输入和缓存审计，再做评价；旧输出只读保存，使用新的 RunId 和新目录，绝不覆盖旧结果。

## 0. 总原则

- 所有少样本实验使用 K=1/2/4 张正常参考图，seed、shot、数据集和方法均应写入独立 RunId。
- 路由和校准禁止使用测试标签、测试掩码、测试集整体统计，或由这些信息导出的阈值、正常样本筛选和类别规则。
- 视觉分支始终是默认安全输出；文本只能做有限、局部、可解释的救援，任何不可靠情况都必须回退视觉结果。
- 图像级和像素级分别校准、选择参数、建模和汇报；不得共享另一层级的测试真值。
- 每次运行保存命令、配置、日志、输入清单、随机种子、RunId、normal-reference manifest、checkpoint/code/cache/evaluator hash 和审计报告。

## 阶段一：V3.3 协议审计（CPU，无 GPU）

目标：先确定 V3.3 结果是否可用，再谈效果。

1. 审查 `src/industrial_ad/fusion/v3_3_strategies.py`、`scripts/evaluate_v3_3_gate_b.py`、`scripts/evaluate_v3_3_pipeline.py`，以及全部 V3.3 报告、配置、预测缓存、校准文件。
2. 对每个运行审计 sample_id 的唯一性与跨分支对齐、图像/像素图 shape、类别数、样本数、缓存来源、RunId、代码与输入 hash。
3. 检查 normal-only 筛选、校准、阈值和路由特征是否直接或间接读取 `gt_masks`、`labels` 或测试集统计。
4. 任何使用 `gt_masks` 选择正常测试样本来校准的旧结果，统一写入：`development_only_leaky_calibration=true`、`paper_eligible=false`。
5. 每个受审运行输出 `audit.json` 与 `audit.md`，写明证据路径、hash、泄漏字段、错误和结论。

通过条件：所有旧结果都有明确、可追溯的可用性标记。本阶段不得修改旧数值。

## 阶段二：V3.3-clean 协议修复（CPU，无 GPU）

目标：实现严格不读取测试真值的 V3.3-clean；CPU 全部通过后才能继续。

1. 拆分数据边界：
   - `RouterInput` 仅含预测、K 张正常参考、允许的无标签特征、sample ID 和元数据；
   - `EvaluationTarget` 仅含测试标签/掩码，且仅评价器可访问。
2. 校准仅能来自当前 seed/shot 的 K 张正常参考图，保存 median/IQR 或 MAD 与 q95/q99，并保存 manifest/checkpoint/cache/code hash。
3. 所有报告显式写出以下五项且必须全为 `false`：
   - `test_predictions_used=false`
   - `test_labels_used=false`
   - `test_masks_used=false`
   - `test_dataset_statistics_used=false`
   - `test_normal_selection_used=false`
4. 新增测试：禁止访问标签/掩码；改变 GT 不改变预测；sample ID 错位/缺失/重复必须失败；NaN/Inf 安全处理；文本不可靠或特征缺失时视觉回退；确定性；旧 V3.3 回归测试并明确协议差异。

## 阶段三：MPDD seed0 / K1 CPU Gate（无 GPU）

目标：复用冻结缓存，在小规模预注册开发 Gate 中判断 V3.3-clean 是否值得扩大。

1. 先固定输入缓存与唯一配置，预注册一个小网格，只允许少量预先声明的固定权重、阈值或残差上限。
2. 对照必须包括：AnomalyDINO、AnomalyCLIP、AdaptCLIP（只有缓存/协议通过时）、50/50、少量固定权重、V3.2、旧 V3.3（仅作无效标记对照）、V3.3-clean 和视觉安全回退。
3. 报告 Image AUROC/AP/F1、Pixel AUROC/AP/AUPRO、逐类 delta、rescue、harm、coverage、risk-coverage。
4. 建议 Gate：平均 Pixel AP 高于视觉；至少 4/6 类正收益；无单类大幅退化；AUPRO 不整体下降；审计与重复性均通过。

决策：V3.3-clean 通过时，再判断动态路由是否超过最佳固定融合；未通过则不跑大矩阵，转向视觉锚定的文本局部救援。

## 阶段四：视觉锚定的文本局部救援（先 CPU）

固定流程：`候选区域 -> 正常参考超界 -> prompt/增强稳定性 -> 背景拒绝 -> 有界单向文本残差 -> 视觉回退`。

- 候选区域只由视觉异常图产生；正常参考超界仅用 K 张正常参考局部统计。
- 文本判断需在预声明 prompt/增强下稳定；低纹理、边界伪影和明显背景必须拒绝。
- 文本只可在视觉候选区加入有界、单向残差，不得全图改写视觉分数。
- 允许：正常参考距离、视觉候选掩码、无标签文本稳定性、位置和模型自身置信度。
- 禁止：测试标签、测试掩码、测试集总体分位数、按测试指标挑选类别规则。
- 保存区域原因代码：`no_visual_candidate`、`reference_in_support`、`prompt_unstable`、`background_rejected`、`bounded_text_residual`、`visual_fallback`；完成模块、残差上限和失败案例消融。

## 阶段五：7.1-A1 审计与 MPDD 完整开发矩阵

### 5.1 7.1-A1 先行（CPU，无 GPU）

7.1-A1 的 **concat + KNN memory bank** 是当前最值得严格审计的候选。先固定唯一配置，核验 concat 特征与 memory bank 只使用正常参考特征，并核验类别、seed、shot、sample ID、缓存 hash 和评价边界。A2、CCA/A2b、A3、手写 prompt、图像级 gate 的负结果保留在报告中，但不继续扩展。

仅在 A1 通过协议与小 Gate 后，才决定是否补齐 MPDD 3 seeds × 1/2/4-shot。

### 5.2 必要时的 GPU 矩阵

- 目标：MPDD 的 K=1/2/4 × seed 0/1/2，共 9 个配置。
- AnomalyDINO 的 K/seed 相关缓存可能需要 GPU；先审计已有缓存能否复用。
- 先审计 AdaptCLIP 预测是否真正依赖 K/seed。若不依赖，不得重复推理，应复用同一份审计通过缓存并说明。
- 使用单 GPU、串行、可恢复队列；每项保存 RunId、marker、日志、缓存 hash；每完成一项立刻审计。
- 9/9 必须全部通过：无 schema/对齐错误、无 NaN/Inf。动态方案必须超过最佳固定融合，才能进入冻结。

## 阶段六：正式冻结（CPU，无 GPU）

冻结候选确定后，不得再随验证结果修改。生成并保存：

1. code/config/checkpoint/manifest/calibration/evaluator/预测缓存 hashes；
2. `freeze_manifest.json`；
3. `METHOD_CARD.md`（输入输出、允许/禁止信息、失效条件、回退逻辑）；
4. `REPRODUCE.md`（环境、命令、目录、验收标准、预期产物）。

## 阶段七：冻结后验证

- MPDD 仅按冻结版本复核，不再调参。
- VisA、MVTec、BTAD 按历史或补充验证角色报告，明确既有缓存和冻结后新验证的差异。
- 最好增加一个冻结前未查看的新外部数据集。
- 冻结后不得按结果重选规则、类别或候选。

### 阶段七状态（2026-08-18 更新）

- MPDD 复核 ✅（9/9 全正，mean ΔAP +0.0486）。
- BTAD 冻结后验证 ✅（2026-08-19 完成，9/9 正，mean ΔAP **+0.0766** vs legacy v2 dino score；三口径 0.0766/0.0517/0.0249，角色升级为 `external_frozen_validation`）。
- **VisA 冻结后验证 ✅（2026-08-18 完成）**：
  - 结果：9/9 配置全正，mean ΔAP **+0.0524**（vs DINO feature-level baseline）。
  - 逐类：10/12 类 9/9 全正（candle -0.020 / chewinggum -0.039 小幅退化，无灾难类）。
  - 审计：285 项检查全过；导出队列 18/18 成功（s0/k1 全量 + 其余 8 组合 ref-only，测试特征复用黄金结论）。
  - baseline 口径：VisA 无 v2 分数级缓存 → 用特征级 dino-only KNN 作 DINO baseline（MPDD s0/K1 上该口径与 v2 分数级差 ~0.0008 AP，可比）。
  - 产物：`experiments/dynamic_fusion/v3_direction_a/a1_visa_20260818/{visa.md,visa_summary.json,visa_audit.json}`。
  - 新脚本（不触碰冻结清单哈希）：`scripts/{export_a1_visa_features.py, export_a1_visa_ref_only.py, evaluate_a1_visa_frozen.py, run_a1_visa_export_queue.py, summarize_a1_visa.py, audit_a1_visa.py}`；`scripts/v2_mpdd_prediction_common.py` 新增 `index_visa`（不在冻结清单，MPDD/BTAD 行为不变）。
- MVTec：**冻结后验证 ✅（2026-08-18 完成）**：
  - 结果：9/9 配置全正，mean ΔAP **+0.0320**（vs DINO feature-level baseline；by_seed 0.036/0.027/0.034；by_shot 0.035/0.031/0.030）。
  - 逐类：15 类中 11 类 9/9 全正（toothbrush +0.108 / carpet +0.080 / pill +0.077 / screw +0.068 / tile +0.051 等）；capsule -0.016 / grid -0.006 / hazelnut -0.030 / leather -0.043 小幅退化，无灾难类。
  - 消融对照：dino-only 0；clip-only -0.0573 全负（0/9）→ concat 增益来自 CLIP 互补（与 MPDD/VisA 一致）。
  - 审计：327 项检查全过；导出队列 18/18 成功（s0/k1 全量 + 其余 8 组合 ref-only）。
  - baseline 口径：MVTec 无 v2 分数级缓存 → 与 VisA 同口径（特征级 dino-only KNN）。
  - 产物：`experiments/dynamic_fusion/v3_direction_a/a1_mvtec_20260818/{mvtec.md,mvtec_summary.json,mvtec_audit.json}`。
  - 脚本参数化（不触碰冻结清单哈希）：六个 VisA 脚本新增 `--dataset`（visa/mvtec），`scripts/v2_mpdd_prediction_common.py` 新增 MVTec→`index_mpdd` 路由（不在冻结清单，MPDD/BTAD/VisA 行为不变）。
- 新外部数据集：未执行（需新数据集授权）。

### 额外补强（GPT 验收对账，2026-08-18）

- A1 特征级消融（dino-only / clip-only × 9 配置）已完成：dino-only +0.0227（8/9 正）、clip-only -0.0222（0/9 正）、concat +0.0486（9/9 正）→ 确认 concat 增益来自 CLIP 互补（产物 `a1_ablation_20260817/`）。
- 验收对账总表：`experiments/dynamic_fusion/freeze/a1_mpdd_w05/ACCEPTANCE_MAPPING.md`（逐条映射 GPT 验收点 → 产物，含真实缺口清单）。

### 收敛状态（S0–S5 + D0/D1，2026-08-18）

依据 `docs/DYNAMIC_FUSION_DESIGN_REVIEW_AND_NEXT_PLAN.md`：

- **S0 状态对账** ✅：`experiments/dynamic_fusion/reconciliation/dynamic_fusion_state_reconcile_20260818_v1/{state.json,state.md,hashes.sha256,link_check.json}`。
- **S1 只读 verifier** ✅：`freeze_a1_mpdd.py --create/--verify` 互斥，`--verify` 严格只读全量 229 项验证通过、manifest hash 不变；篡改测试 8/8（`tests/test_freeze_a1_mpdd.py`）；报告 `freeze_verification.{json,md}`。
- **S2 角色与路径修正** ✅：VisA→`in_domain_frozen_validation`（60 处 JSON 修正）、MVTec→`external_frozen_validation`、BTAD→`external_frozen_validation_k1_only`（2026-08-19 9 配置收口后升级为 `external_frozen_validation`）、MPDD→`development`；VisA 路径 20260817→20260818；权威状态 `docs/CURRENT_DYNAMIC_FUSION_STATUS.{md,json}`；链接检查通过。
- **S3 统一性能表** ✅：`experiments/dynamic_fusion/main_results_20260818/`（13 行主表 + 36 行逐类；MPDD 三口径 0.0486/0.0227/0.0258；BTAD 三口径 0.0766/0.0517/0.0249；4 数据集 concat 均值重算 diff=0.0，`recompute_all_pass=true`）。
- **S4 正式方法包** ✅：`METHOD_CARD.md`（名称、伪代码、schema、资源统计、failure cases）、`REPRODUCE.md`（validate-only 检查、CPU 重算命令、角色与验收）。
- **S5 Git 归档** ✅：分 3 批提交（S1/S2/S3+S4）并 push；数据/cache 排除。
- **路线 D**：D0（headroom）**passed**（MPDD 9 配置 per-pixel best-of-3 Oracle，mean +0.5807）；D1（可预测性）**failed**（LOCO AUROC 0.592 < 0.60，特征置乱不下降 0.616）→ **动态路线永久归档**，仅保留证据 `route_d_d0_20260818/`。
- **项目完成判定**：第 12 节 1–10 条已满足（S1–S5 完成、D 路线失败已归档）；剩余唯一主线为 **S6 论文交付**。

## GPU 需求与预计时间

| 阶段 | GPU 需求 | 预计时间 |
| --- | --- | --- |
| V3.3 审计 | 不需要 | 数小时至 1 天 |
| V3.3-clean 与测试 | 不需要 | 1–3 天 |
| MPDD seed0/K1 Gate | 不需要，复用缓存 | 数小时至 1 天 |
| 局部救援设计/Gate | 首轮不需要 | 1–3 天 |
| A1 严格审计 | 不需要 | 数小时至 1 天 |
| MPDD 9 配置矩阵 | 仅 Gate 通过后，单 GPU 串行 | 约 1–3 天 |
| 冻结与结果整合 | 不需要 | 1–2 天 |

这些时间只用于排程；GPU 空闲不是启动授权。任何 GPU 任务都须先通过审计并获得明确授权。

## 每个实验必须保存的信息

- RunId、开始/结束时间、机器/GPU、命令、完整配置；
- 数据集版本、类别清单、K 张正常参考路径与 hash；
- checkpoint、代码、环境、缓存、校准、评价器 hash；
- 类别/样本数、sample ID/shape 对齐、NaN/Inf 检查；
- 五个泄漏字段、逐类指标、失败项、stdout/stderr、恢复点、最终审计结论。

## 后续 AI 每次开始的检查清单

1. 读取本文件、最新权威状态、freeze manifest 和最近 audit，区分历史文档与当前状态。
2. 检查同 RunId 是否已有完整产物或活跃进程；绝不重复或覆盖。
3. 检查 GPU 是否空闲，但不能因空闲自动开跑。
4. 先核验输入 hash、sample ID、shape、类别/样本数、五个泄漏字段和测试边界。
5. 先运行 CPU 测试/`validate-only`；发现协议问题即停止扩大实验。
6. 每一步写入新报告、日志和状态，失败结果也必须保留。

## 接手后第一轮立即行动（不要启动 GPU）

1. 新建 V3.3 审计 RunId/目录，不改旧 V3.3 输出。
2. 完成阶段一，重点核对是否以 `gt_masks` 做校准；生成 `audit.json` 与 `audit.md`。
3. 若发现泄漏，实施阶段二的 RouterInput/EvaluationTarget 分离和 CPU 测试；不要运行 GPU 推理。
4. 对 7.1-A1 唯一 concat + KNN 配置做正常参考来源、特征来源、缓存对齐审计；其他路线仅归档负结果。
5. 仅在阶段二和 A1 审计都通过后，才讨论 MPDD seed0/K1 CPU Gate；第一轮不得启动 GPU。
