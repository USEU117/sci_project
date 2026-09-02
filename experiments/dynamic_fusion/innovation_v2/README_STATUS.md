# A2 Innovation Program — 状态与协议记录（2026-09-02）

任务书：`docs/paper_writing_preparation_20260830/12_MULTI_ROUTE_ALGORITHM_INNOVATION_EXECUTION_AND_ACCEPTANCE_CN_20260902.md`

## 1. 保护声明

- A1 冻结证据（`submission_repro_20260827/`、`experiments/dynamic_fusion/freeze/`）不修改；
- 既有未提交工作（`docs/` 状态文档、任务书 12、RCEC 复核记录）不触碰、不清空；
- 无 `git reset` / `checkout --` 操作。

## 2. 路线 E 权限 Gate（任务书 9.1）

用户于 2026-09-02 明确授权路线 E（NCPRA）正式标签评估。

**协议变化已记录**：论文定位由 "zero-trainable-parameter" 调整为
"lightweight normal-only adaptation"（仅在路线 E 被选为 winner 时生效；
否则 A1 维持零训练口径）。适配器只由正常参考 patch 训练，backbone 冻结，
无任何测试标签/mask/测试统计参与训练或选 epoch。

## 3. 执行波次进度

| 波次 | 内容 | 状态 |
|---|---|---|
| Wave 0 | 共享审计与框架 | ✅ 通过（缓存完整、参考顺序一致、A1 回归 <1e-6） |
| Wave 1 | 六路线 MPDD seed0 Small Gate | ✅ 完成：27 候选全部失败，无 winner |
| Wave 2 | D DEVA 正常参考增强 | ✅ 完成（GPU 导出 462 行，6 类 × 3 shots × 3 packs） |
| Wave 3 | E NCPRA 轻量训练 | ✅ 已授权并完成（4 候选全负） |
| Wave 4 | Full MPDD + winner selection | ⏸ 跳过：无候选通过 Small Gate（任务书停止规则 §17） |
| Wave 5 | 冻结 + 一次性验证 | ⏸ 跳过：保留 A1（无新 winner，验证集未被用作开发集） |
| — | 最终决策 | ✅ `FINAL_DECISION.md`：全部 ARCHIVE，A1 继续为主方法 |

## 4. 小门结果速览（ΔPixel-AP = 相对 A1）

| 路线 | 候选数 | 最优 mean Δ | 结论 |
|---|---|---|---|
| A LNDC | 3 | −0.085 | 全负，早停 |
| B DSAM | 6 | −0.012 | 全负（对齐 beats control 但局部窗口劣于全局 KNN） |
| C CEQA | 4 | +0.002799 | 全正但 <+0.003 且多数不 beats control，早停 |
| D DEVA | 6 | +2.4e-05 | ≈0，tau 滤波无效应，早停 |
| E NCPRA | 4 | −0.0052 | 全负，早停 |
| F FAGR | 4 | −0.0052 | 全负，早停 |

每路线决策文件见 `01_small_gates/<ROUTE>/SMALL_GATE_DECISION.json`；
逐候选报告（含 control 与完整指标）见 `01_small_gates/<ROUTE>/<CANDIDATE>/`。

## 5. 执行期修复记录（诚实上报）

1. **DSAM non-finite 修复**：RANSAC affine 把部分 query patch 中心映射到参考图外，
   L∞ 窗口为空得 inf → 修复为"窗口空时回退全局最近邻"（`_constrained_min_dist`），
   修复前后 translation 结果一致（1e-6 级），affine 不再崩溃。
2. **DEVA CLIP 等变性网格修复**：`aligned.c_ref` 已被 resize 到 DINO 32×32 网格，
   与 CLIP 原始 37×37 网格不匹配 → `build_augmented_memory` 新增
   `clip_ref_grid`（读取 `{cat}_k{shot}_identity.npz` 的 identity_c），
   e_C 在 CLIP 原生分辨率计算。
3. 两处修复均补跑对应路线并重写决策文件。

## 6. 环境

- `.venv-patchcore`：torch 2.0.0+cu118、faiss 1.7.4、cv2 4.8.1、CUDA 可用。
- 新增代码全部位于 `src/industrial_ad/innovation_v2/`、`scripts/innovation_v2/`、
  `tests/innovation_v2/`、`configs/innovation_v2/`。
- DEVA 增强缓存写 `outputs/dynamic_fusion/innovation_v2_deva/`（gitignored）。
