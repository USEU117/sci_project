# CASF 类条件执行状态（innovation_v5_casf，2026-09-02）

任务书：[15_CASF_CATEGORY_CONDITIONAL_ALGORITHM_AND_EXPERIMENT_PLAN_CN_20260902.md](../../../docs/paper_writing_preparation_20260830/15_CASF_CATEGORY_CONDITIONAL_ALGORITHM_AND_EXPERIMENT_PLAN_CN_20260902.md)
保护声明：只允许 MPDD development；BTAD/MVTec/VisA 不可见；A1 冻结证据与 12/13/14/15 号文档不改写；无 reset/checkout。

## 执行波次进度

| 波次 | 内容 | 状态 |
|---|---|---|
| Wave 0 | 五目录 + 放大合成探针（24ep×3 seeds×6 类）→ 冻结 Gset | ✅ 完成，Gset = {bracket_white} |
| Wave 1 | 多层特征导出与身份回归（仅 P2 需要） | ⏸ 未启动 |
| Wave 2 | P0/P1(+控制) MPDD 小门 | ⏸ 跳过（提前归档） |
| 冻结/外部验证 | — | ⏸ 跳过（无 winner） |
| 最终决策 | ✅ ARCHIVE（用户确认）→ A1 保持主方法 | 见 FINAL_CASF_DECISION.md |

## Wave 0 结果（放大探针，MPDD dev，s0/k2，24ep/家族 × 3 family seeds）

| 类别 | per-seed headroom (s0/s1/s2) | mean hr | votes | active |
|---|---|---|---|---|
| bracket_black | −0.0066 / +0.0118 / +0.0025 | +0.0026 | 0/3 | ✗ |
| bracket_brown | −0.1064 / +0.0410 / +0.0149 | −0.0168 | 1/3 | ✗ |
| bracket_white | +0.0968 / +0.0187 / +0.0624 | **+0.0593** | 2/3 | ✓ |
| connector | +0.0269 / −0.1413 / −0.0130 | −0.0425 | 1/3 | ✗ |
| metal_plate | +0.0931 / −0.0734 / +0.0273 | +0.0157 | 2/3 | ✗ |
| tubes | −0.0706 / −0.1288 / +0.0753 | −0.0414 | 1/3 | ✗ |

**冻结 Gset = ["bracket_white"]**（`Wave0_gate_probe/GSET.json`；PROBE_SUMMARY.json 含逐 seed 全表与规则元数据）。

### 与早期探针的差异（诚实上报）
- 任务书 15 §0 引用的 D3 seed0/k2 12-episode 探针给出 3/6 类支持（bracket_brown +0.587 强信号）——
  与放大探针**明显不一致**：bracket_brown 在 24ep×3seeds 下降到 −0.017（其早期 +0.59 由小样本下 sym 训练崩溃驱动，
  非稳健信号）；metal_plate 0.020 → 0.016（<0.02）；bracket_white 保持 +0.059。
- 因此按任务书 §2.4「正式 Gset 以放大探针为准」，CASF-active 类别仅剩 bracket_white 一类。

### 对后续小门的算术约束（§6.1 pooled-6）
小门 mean ΔPixel-AP 门槛为 pooled-6（全部 6 类平均）≥ +0.005，且 3/3 shot 为正。
5 个 inactive 类为纯 A1（Δ=0），因此要求 **bracket_white 单类三-shot 平均 ΔPixel-AP ≥ +0.030（+3.0 点）且每 shot 为正**，
并叠加 train-seed 稳定性（std ≤0.005）、CTRL-SYM ≥ +0.002、CTRL-NODIS ≥ +0.001、gated ≥ CTRL-NOGATE 等机制门槛。
在 A1 已对该类有较好基线、且探针绝对 Dice 很低（asym Dice ≤0.06）的背景下，该门槛的达成概率很低——
这是运行 Wave 2 前必须向用户言明的预期。

## 代码与测试
- `src/industrial_ad/innovation_v5_casf/gate_probe.py`（探针 + 确定性 Gset 规则）、`scripts/innovation_v5_casf/run_gate_probe.py`
- `tests/innovation_v5_casf/test_gate_probe.py`：7 passed（规则边界、config 防漂移、RNG 确定性、统计量语义、dev-only）
- 探针零 mask/GT 接触：只读正常 reference feature，不调用 evaluator GT。

## 下一步（待用户决策）
1. **运行 Wave 2 小门**：Gset={bracket_white}，实现 P0/P1 + 6 控制，MPDD s0×shot{1,2,4} 全类 pooled 门槛——预期大概率失败（见上），失败即按任务书归档、A1 保持；
2. **提前归档**：以放大探针证据（1/6 类、绝对 Dice 低、pooled 算术不可达）为基础直接归档 CASF，不再投入小门实现；
3. 或用户修订门槛/范围（须显式批准，防 cherry-pick）。

## 最终决策（2026-09-02，用户确认「提前归档」）

**CASF 类条件路线归档，不升级；A1 保持唯一主方法与零训练论文口径。** Wave 0 后未运行小门，
决策与依据见 `FINAL_CASF_DECISION.md`（Gset 仅 bracket_white 一类、绝对 Dice ≤0.096、pooled-6
门槛需该单类 ≥+0.03 ΔPixel-AP）。类条件设计教训（非对称监督价值类条件化、单类可learnable）保留
为 Discussion 层可引用观察，不构成论文新主张。
