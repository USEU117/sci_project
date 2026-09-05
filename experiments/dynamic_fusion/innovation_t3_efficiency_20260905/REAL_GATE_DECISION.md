# Track-3 主线结案（真实 MPDD 门：A1 + 50% 棋盘 coreset）— 归档

日期：2026-09-05　立项：`docs/.../33_TRACK3_MAIN_REAL_GATE_PLAN_CN_20260905.md`（doc33）；探针上游：doc32 Probe-E1（合成代理 PASS，T1=50% coreset）。
脚本：`scripts/innovation_t3_efficiency_20260905/run_real_gate_coreset.py`
数据：A1 冻结特征缓存（seed0，k2/k4，`outputs/dynamic_fusion/v3_direction_a/features_vitb14_s0_k{2,4}/` 与 `features_s0_k{2,4}/`）；真实 MPDD test 只读评价；ref 棋盘掩码 (i+j)%2==0 为确定性几何操作；无 /test/good 进 memory、无拟合、不调参。
口径：完全复用 A1 冻结 concat（pca0/whiten0/w0.5，32 网格，faiss k=1 → 448 map，Pixel-AP stride=8）。对照 = full memory（K 图×1024 patch/图）vs 棋盘 50% coreset（K 图×512 patch/图）。

## 结果（seed0，六类宏）
| shot | mean full AP | mean cs50 AP | **mean ΔAP** | worst ΔAP（类别） | AUROC Δ | memory 比 |
|---|---|---|---|---|---|---|
| k2 | 0.3437 | 0.3367 | **−0.0070** | **−0.0328（connector）** | +0.0006 | 0.5× |
| k4 | 0.3737* | — | **−0.0074** | **−0.0527（connector）** | — | 0.5× |

（*k4 full 基线按本脚本 full 分支实测；两 shot 逐类明细见 `REAL_GATE_s0_k{2,4}.json`。）

## 门判定（预注册，按结果未调）
- **G-R1（宏无损 ≥ −0.01）通过**：k2 −0.0070、k4 −0.0074。
- **G-R2（无灾难类别 ≥ −0.03）失败**：connector 在 k2 掉 **−0.033**、k4 掉 **−0.053**，均低于 −0.03；其余类别 |Δ| ≤0.014。
- G-R3（memory 0.5×）结构性成立，但不满足无损前提。

**决策：`REAL_GATE_FAIL`**（doc33：G-R1 ∧ G-R2 全过才 PASS；单类崩溃两次 shot 复现 → 合成代理结论未向真实 MPDD 泛化 → 归档 Track-3 主线）。

## 诚实解读
1. **宏层面确实接近无损**（Δ≈−0.007，AUROC/AUPRO 均 ≈0），大部分类别 50% coreset 完全无伤甚至微升（bracket_black/white 为正）。效率主张在"均值"口径上可支撑。
2. **但 connector 单类系统性崩溃**（k2 −0.033、k4 −0.053，AUPRO 几乎不变而 Pixel-AP 大降）：该类别 AP 中等（0.30/0.38）、缺陷 patch 占比小，棋盘半采样恰好移除对少量真实缺陷 patch 的关键近邻支撑 → Pixel-AP（强调 top-正样本排序）对此极敏感。AUROC/AUPRO 无感说明不是全局分数坍缩，而是少量高优先级缺陷 patch 的最近邻丢失。
3. 结论：**均匀/几何 coreset 对"像素级精确排序"不是无损压缩**——它把 memory 单元视为 i.i.d.，而真实缺陷打分依赖少量高杠杆正常 patch。合成代理（LOO 族 AP）未捕获这一真实风险（合成 memory 与真实支持图分布的差异）。
4. 不按结果调参（不做"connector 豁免"或逐类 coreset 率搜索）：那会引入按真实结果的选择，违背 doc28/doc33 纪律。若未来重启效率主线，正确方向是**保排序的 memory 修剪**（如基于正常路径重要性/距离贡献保留，而非几何均匀采样）或**量化/维度压缩**（不动 memory 单元集合），需另立协议重新预注册。

## 产物
`REAL_GATE_s0_k2.json`、`REAL_GATE_s0_k4.json`、`doc33`、本结案记录。Track-3 主线关闭；三方向（Track-1/2 机制增益、Track-3 效率压缩）探针+真实门全部尝试完毕并如实归档。
