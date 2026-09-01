# FINAL_RCEC_DECISION

日期：2026-09-02
任务书：`docs/paper_writing_preparation_20260830/11_RCEC_INNOVATION_IMPLEMENTATION_AND_ACCEPTANCE_HANDOFF_CN_20260901.md`

- **状态：`ARCHIVE`**
- **决策：RCEC v1 在 MPDD 小门（Phase 2）失败，归档为开发集负结果；论文主方法继续使用 A1。**

---

## 1. RCEC 是否超过 A1（而不只是超过 DINO）？

**否。** 12 个预注册候选在 MPDD seed0 × shot {1,2,4} 上全部低于 A1：

| 候选 (direction, k, λ) | s0/k1 ΔPixel-AP | s0/k2 ΔPixel-AP | s0/k4 ΔPixel-AP | 平均 |
|---|---:|---:|---:|---:|
| dino_to_clip, k1, 0.25 | −0.0092 | −0.0110 | −0.0212 | −0.0138 |
| dino_to_clip, k1, 0.50 | −0.0271 | −0.0539 | −0.0749 | −0.0520 |
| dino_to_clip, k3, 0.25 | −0.0034 | −0.0107 | −0.0181 | −0.0107 |
| dino_to_clip, k3, 0.50 | −0.0168 | −0.0400 | −0.0605 | −0.0391 |
| dino_to_clip, k5, 0.25 | +0.0003 | −0.0060 | −0.0158 | −0.0071 |
| dino_to_clip, k5, 0.50 | −0.0088 | −0.0319 | −0.0520 | −0.0309 |
| symmetric, k1, 0.25 | −0.0123 | −0.0038 | −0.0113 | −0.0091 |
| symmetric, k1, 0.50 | −0.0175 | −0.0159 | −0.0277 | −0.0204 |
| symmetric, k3, 0.25 | −0.0126 | −0.0072 | −0.0129 | −0.0109 |
| symmetric, k3, 0.50 | −0.0190 | −0.0211 | −0.0316 | −0.0239 |
| symmetric, k5, 0.25 | −0.0129 | −0.0075 | −0.0132 | −0.0112 |
| symmetric, k5, 0.50 | −0.0195 | −0.0220 | −0.0319 | −0.0245 |

所有 delta 均为相对 A1 的 Pixel-AP 差（绝对 AP 点）。最接近的候选（`dino_to_clip_k5_lam0.25`）平均仍为 −0.0071，且仅 1/3 shot 为正，不满足小门任何一条（mean≥0、2/3 shot 正、worst≥−0.010）。

## 2. 通过了哪些 Gate，失败了哪些 Gate

**通过：**
- 工程：`tests/test_rcec.py` 18 项全过（含 A1 回归 map<1e-5 / Pixel-AP<1e-6、DINO-duplicate 距离保持、LOO 排除规则、泄漏防护、分块一致性）。
- 输入审计（Phase 0）：四数据集双分支缓存与 manifest 一致；DINO/CLIP 参考块顺序均按 manifest 列表（导出器源码审计 + 数量校验）；A1 concat map 与冻结实现回归一致。
- 防泄漏：所有报告五项 leakage flags 全 `false`；算法路径不接收测试标签/mask。

**失败：**
- **MPDD 小门（Phase 2）**：12/12 候选未通过 → 按任务书 Phase 2 停止规则触发 early stop。

**未执行（因早停，按规则不允许）：**
- Phase 3 完整开发矩阵、Phase 4 配对消融、Phase 5 冻结、Phase 6 冻结验证。

## 3. 是否存在验证集调参或其他泄漏？

**否。** 全部候选只在 MPDD seed0 × shot {1,2,4} 上运行；BTAD、MVTec AD、VisA 未被读取。12 个候选均为任务书 3.7 预注册的 `direction × k × λ` 网格（2×3×2），未增加任何额外超参数。校准只用正常参考 LOO 统计。

## 4. pairing shuffle 是否支持“一致性”解释？

**不适用。** 小门失败即停止，未运行配对破坏消融；因此本实验**不宣称**任何“跨编码器一致性有效”的机理结论（任务书 12 节禁止）。

## 5. 哪些数据集、类别、shot 得益，哪些退化？

小门范围内（MPDD seed0）所有候选在所有 shot 上相对 A1 退化。趋势：
- λ 越大（条件分数权重越高）退化越明显 → 说明 `r_C|D` 相对 `s_A1` 是更弱的异常证据；
- shot 越大退化越明显（1→2→4）；
- `dino_to_clip` 平均略优于 `symmetric`；
- k=5 略优于 k=1（更宽的 DINO 邻居集合使 CLIP 距离更平滑，但依然无增量）。

## 6. 增益是否值得增加计算和方法复杂度？

**否。** 没有正增益；无需评估复杂度代价。

## 7. 论文主方法应使用 RCEC 还是继续使用 A1？

**继续使用 A1。** RCEC 按任务书 12 节以负结果进入 Discussion/Future Work：
> 正常参考条件下的跨编码器邻域分歧没有在开发集上稳定超过固定拼接；简单互补收益并不必然转化为可利用的局部一致性信号。

## 8. 证据指向

| 证据 | 位置 |
|---|---|
| 小门全结果（机器可读） | `experiments/dynamic_fusion/rcec_v1/development_mpdd/SMALL_GATE_REPORT.json` |
| 早停报告 | `experiments/dynamic_fusion/rcec_v1/development_mpdd/RCEC_V1_EARLY_STOP_REPORT.json` |
| 逐配置报告（36 份） | `experiments/dynamic_fusion/rcec_v1/development_mpdd/small_gate/*/s0_k{1,2,4}/report.json` |
| 单元/技术测试 | `tests/test_rcec.py`（18 passed） |
| 核心源码 | `src/industrial_ad/fusion/rcec.py` |
| 共享 runner 逻辑 | `scripts/rcec_common.py` |
| 配置 | `configs/rcec_v1.yaml` |
| 任务书 | `docs/paper_writing_preparation_20260830/11_RCEC_INNOVATION_IMPLEMENTATION_AND_ACCEPTANCE_HANDOFF_CN_20260901.md` |
