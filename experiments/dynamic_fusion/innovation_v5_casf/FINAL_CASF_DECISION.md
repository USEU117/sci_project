# FINAL CASF DECISION — 提前归档（2026-09-02）

程序：`innovation_v5_casf`（A4 入选路线，类条件 CASF）
任务书：`docs/paper_writing_preparation_20260830/15_CASF_CATEGORY_CONDITIONAL_ALGORITHM_AND_EXPERIMENT_PLAN_CN_20260902.md`
决策类型：Wave 0 后**提前归档**（用户 2026-09-02 确认，未运行 Wave 2 小门）。

## 1. 决策

**CASF 类条件路线不升级，A1 继续保持唯一主方法与论文口径（零训练、normal-only）。**
不实现 P0/P1 候选，不运行 MPDD 小门，不触碰任何外部验证集。

## 2. 决策依据（均为 Wave 0 放大探针证据，MPDD development s0/k2）

1. **Gset 仅 1/6 类**：按预注册规则（mean headroom ≥ +0.02 且 ≥2/3 family seeds ≥ +0.02），
   放大探针（24ep/家族 × 3 seeds）只有 bracket_white 达标（mean +0.0593，votes 2/3）；
   bracket_brown −0.0168、connector −0.0425、tubes −0.0414、bracket_black +0.0026、metal_plate +0.0157。
   早期 12-episode 探针的 3/6 类支持（bracket_brown +0.587）被证明为小样本 sym 训练崩溃，不稳健。
2. **绝对判别力低**：bracket_white 的 asym-trained Dice 仅 0.019–0.096；合成监督的头几乎不触发，
   机制控制（CTRL-SYM/CTRL-NODIS）大概率与它持平或更优——没有值得投入小门的机制余量。
3. **pooled-6 门槛算术不可达**：小门要求全部 6 类平均 ΔPixel-AP ≥ +0.005 且 3/3 shot 为正；
   5 个 inactive 类为纯 A1（Δ=0），等价要求 bracket_white 单类三-shot 平均 ΔPixel-AP ≥ +0.030
   且每 shot 为正，叠加 train-seed std ≤0.005 等门槛。与证据量级不匹配。

## 3. 诚实边界（归档不意味着什么）

- CASF 只在 MPDD development seed0/k2 的合成探针尺度上被检验；不构成对
  "跨分支非对称伪异常监督"在任何数据集/任何规模下的彻底证伪；
- **类条件设计的教训仍然有效并可引用**：非对称监督价值呈类条件（1/6 类可learnable，5/6 类被
  symmetric 控制主导），单一全局伪异常配方在多类混合数据上会被拖累；
- 不做任何论文新主张；A1 口径、冻结证据、12/13/14/15 号文档审计链不变。

## 4. 证据文件

- `Wave0_gate_probe/PROBE_SUMMARY.json`（6 类 × 3 seeds 全表）
- `Wave0_gate_probe/GSET.json`（冻结 Gset = ["bracket_white"]）
- `Wave0_gate_probe/<cat>_s0_k2_seed<ix>.json`（逐 run 明细）
- `README_STATUS.md`（波次记录与决策）

## 5. 最终状态

| 波次 | 状态 |
|---|---|
| Wave 0（五目录 + 放大探针 + Gset） | ✅ 完成 |
| Wave 1（多层导出） | ⏸ 未启动（不再需要） |
| Wave 2（P0/P1 小门） | ⏸ 跳过（本决策） |
| 冻结/外部验证 | ⏸ 跳过（无 winner） |
| 最终决策 | ✅ ARCHIVE — A1 保持主方法 |
