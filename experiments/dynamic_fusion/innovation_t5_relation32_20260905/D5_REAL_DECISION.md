# 方向 5 主线结案（真实 MPDD 关系描述子门）— 归档（合成增益未向真实泛化）

日期：2026-09-05　立项：`docs/.../36_DIRECTION5_MAIN_REAL_GATE_PLAN_CN_20260905.md`（doc36）；探针上游：doc35 Probe-H1 PASS（32-grid 关系描述子合成 cutpaste Δ=+0.152/+0.148）。
脚本：`scripts/innovation_t5_relation32_20260905/run_d5_real_gate.py`
数据：A1 冻结特征缓存 seed0 k2/k4；真实 MPDD test 只读评价；无拟合、无 /test/good 进 memory、候选运行前冻结。
口径：完全复用 A1 冻结 concat；候选 R5-C1（concat z,3×3 邻域均值）、R5-C2（concat z,4 邻）；memory 与查询同规则；总体六类宏 + parts_mismatch 子组。

## 结果（seed0，六类宏 Δ 相对 A1）
| shot | 候选 | mean ΔAP | worst ΔAP | parts_mismatch ΔAP | ΔAUROC |
|---|---|---|---|---|---|
| k2 | C1（3×3） | **−0.0142** | −0.0577（bracket_white） | −0.0122 | −0.0043 |
| k2 | C2（4 邻） | −0.0390* | −0.1552（connector） | — | — |
| k4 | C1（3×3） | **−0.0335** | −0.0771（bracket_black） | −0.0254 | −0.0032 |
| k4 | C2（4 邻） | −0.0768 | −0.1989（connector） | −0.1134 | −0.0163 |

（*k2 C2 宏 Δ 由 JSON 完整计算；C1 为最优候选仍全负。）

## 门判定（预注册 doc36，按结果未调）
- **G-R1（真实增益 ≥ +0.01）失败**：C1 k2 −0.014、k4 −0.034（两 shot 均负）。
- **G-R2（无灾难类）失败**：k2 worst −0.058、k4 worst −0.077。
- **G-R3（parts_mismatch 子组 ≥ 0）失败**：k2 −0.012、k4 −0.025。
- G-R4（AUROC ≥ −0.005）通过（−0.0043/−0.0032），不足以翻转结论。

**决策：`D5_REAL_GATE_FAIL`**（doc36：关系增益未在真实 MPDD 出现 → 归档方向 5 主线 → 转方向 6）。

## 诚实解读
1. **合成代理再次不预测真实（反向失败）**：同一批 32-grid 关系描述子在 support-合成 LOO 上 cutpaste +0.15（远超门），在真实 test 上全为负（−0.01~−0.08，两 shot、六类一致）。与 Track-3 coreset（合成"无损"但真实单类崩）合成方向相反，但同一条纪律教训：**冻结特征上的合成增益不能作为真实像素排序增益的证据**。
2. 可能的真实机制：真实 MPDD 的上下文类缺陷（parts_mismatch）与合成 cutpaste 的扰动分布不同；邻域均值项把 test 图与 support 图之间的正常光度/结构差异也放进了描述子（test 图非 support 域），污染距离排序。C2（高维 5×1536）最严重（connector −0.199），高维拼接在跨域时放大域间隙。
3. 顺带得到方向 6 直接输入：A1 在 parts_mismatch 子组的绝对水平 k4=0.278（含 bracket_brown/connector 等），低于总体 0.388——真实上下文类缺陷仍有明确剩余缺口，但**它不能用"加邻域描述子"来补**；方向 6 只做量化诊断，不再加关系机制。
4. 不按结果调参（不做"仅某些类启用 C1"等后调）；探针结论（信息存在）成立，主线结论（真实可用）不成立，两者分开归档。

## 产物
`REAL_D5_s0_k2.json`、`REAL_D5_s0_k4.json`（逐类明细）、doc36、本结案记录。方向 5 关闭 → 转方向 6（真实 parts_mismatch 子组诊断，doc37）。
