# 广度探针 R10 结案（2026-09-09 夜，续）— RCW/BRC 未过门

预注册见 `ROUND10_PLAN_CN.md`；门同全 breadth；parity k2 0.343706 / k4 0.388328 **精确复现**。

## 逐族结果（Δ 相对 C0=冻结 A1 concat top-1；宏为六类）

| 族 | lead | k2 宏 ΔAP / k2 worst | k4 宏 ΔAP / k4 worst | k2 ΔAUROC / k4 ΔAUROC | 判定 |
|---|---|---|---|---|---|
| RCW 支持侧互惠覆盖权重 | RCW_10 | +0.0022 / −0.0070 | −0.0024 / −0.0127 | +0.0005 / −0.0002 | FAIL（k2 未达门且 k4 转负 → **不跨 shot**） |
| BRC 双向互惠一致性 | BRC_10 | +0.0007 / −0.0014 | +0.0011 / −0.0030 | −0.0005 / −0.0005 | FAIL（宏 ≪ 门；但两 shot 方向一致、worst 近零 = 又一共良性亚门） |

成员明细：RCW_30 k2 −0.0001/worst −0.019、k4 −0.0141/worst −0.045（重加权过头伤 tubes/bracket_white）；BRC_30 k2 −0.0025、k4 ±0.0000。

## 逐类方向与诊断量

- RCW_10 k2：bracket_white **+0.0084**、connector **+0.0101**、bracket_brown +0.0017、metal +0.0008；受损 tubes **−0.0070**、black −0.0006。
  k4 反转：black +0.0061/metal +0.0025，而 white **−0.0127**、tubes −0.0065、connector −0.0030 → **支撑越多、互惠稳定度的收益被"每类最优权重不同"吃掉**，与 CB 的 k2→k4 归零同源。
- BRC_10：k2 white +0.0054 为主；k4 black +0.0062/white +0.0027；全类幅度 ≤0.007。
- 诊断量（为论文/后续可复用的事实）：memory 行的跨支持图互惠稳定度均值随 shot 升高而下降（bracket_brown 0.51→0.49；bracket_black 0.31 最低），
  query↔memory 互惠率 0.52–0.70（connector 最高 0.70）——即"正常 patch 与其最近样例互为最近邻"只在一半到七成的 patch 上成立，本可成为可靠性信号，但纠正幅度过小（≤+0.001）不入主张。

## 方法学注记（本轮确认，写入口径）

**本 harness 的指标是"逐类 Pixel-AP 再宏平均"**，因此：
1. 任何**类内单调**变换（全局单调、逐类单调、conformal p 值/阈值校准、CDF 校准）**ΔAP 恒等于 0**——这类机制可证明为 null，不必再测；
2. 只有**逐图单调**（WZ/LCN，已闭）或**非单调**机制才有非零效应；
3. 逐类 Pixel-AP 的口径也解释了历史中大量"近中性（±0.005）"族：它们只在少数类上小幅进出，宏平均后趋零。
这条为后续探索提供剪枝依据，避免重复 null 轴。

## 结论与账目
- R10 两族真实门全负；**累计 32 机制族在真实 MPDD s0 k2/k4 闭合为负（R1–R10 29 族 + Web1 2 族 + FastRef 1 族）**，A1 仍唯一冻结方法。
- 良性亚门族名录更新：PA（救中类、worst≈0，唯一真正无损）、DST_L1（帮强类、k4 worst 破门）、BRC（两 shot 极小正、worst 近零）、WZ/COMB/SUB/CB（各有 worst 或跨 shot 缺陷）。

## 文件
- 预注册：`ROUND10_PLAN_CN.md`；脚本：`probe_breadth9.py`（脚本号承接 R9 自增）
- 结果：`round10/RESULTS_s0_k2.json` / `RESULTS_s0_k4.json`
