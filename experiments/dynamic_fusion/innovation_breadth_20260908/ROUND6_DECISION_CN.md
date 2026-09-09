# 广度探针 R6 预注册与结案（2026-09-09）— 数据/记忆库轴 CB、PA

轴：**数据 / 记忆库侧**（区别于 R1–R5 的打分、特征、后处理、几何轴），由 web 文献立项：
VisionAD（arXiv:2504.11895，"one-for-all class-aware visual memory bank + support/query augmentation"）
与自训练式伪标注扩库思想。忠实实现需多尺度多层特征 + 真实几何增强（GPU 重编码，本环境无 GPU）→ 本轮回放为**离线近似库捷径**的证伪。
预注册门与全 breadth 相同（lead 宏 ΔAP ≥ +0.01 且 worst ≥ −0.03 且 ΔAUROC ≥ −0.005，k2/k4 均须）；
control parity：k2 0.343706 / k4 0.388328，**精确复现**。

## 族与逐族 lead（六类宏；Δ 相对 C0=冻结 A1 concat top-1）

| 族 | lead | k2 宏 ΔAP / k2 worst | k4 宏 ΔAP / k4 worst | k2 ΔAUROC / k4 ΔAUROC | 判定 |
|---|---|---|---|---|---|
| CB 跨类 one-for-all 联合库 | CB_full(k2)/CB_selfplus(k4) | +0.0040 / −0.0048 | +0.0000 / −0.0175 | +0.0007 / +0.0008 | FAIL（宏 < 门；k4 近零） |
| PA 伪标注库自精化（逐图 leave-own-image-out） | PA_q40 | +0.0044 / −0.0001 | +0.0037 / +0.0001 | +0.0044 / +0.0033 | FAIL（宏 < 门；worst 两 shot 近零） |

成员明细：
- CB_full：k2 +0.0040 / worst −0.0048（受损类 bracket_black）；k4 −0.0008 / worst **−0.0322**（略破 worst 门）；
  CB_selfplus（自身库+质心最近邻类）：k2 +0.0025；k4 +0.0000 / worst −0.0175。
- PA_q20：k2 +0.0009 / k4 +0.0005；PA_q40：k2 +0.0044 / worst −0.0001；k4 +0.0037 / worst +0.0001。PA 两成员均"无最差类损伤"。

## 逐类方向（k2，lead 口径，ΔAP）

| 类 | 控制宏基线 | CB_full | PA_q40 |
|---|---|---|---|
| bracket_black | 0.0146 | −0.0048 | −0.0001 |
| bracket_brown | 0.0334 | +0.0008 | +0.0026 |
| bracket_white | 0.1060 | **+0.0249** | +0.0031 |
| connector | 0.2991 | +0.0029 | +0.0051 |
| metal_plate | 0.8784 | +0.0001 | +0.0030 |
| tubes | 0.7307 | +0.0001 | **+0.0125** |

## 解读（区别于既往"救硬类伤强类"律的新质态）

1. **CB**：k2 上把全类参考并库，收益集中在**中难类 bracket_white**（+0.025，相对增益 ~24%），强类几乎不受扰，
   唯一受损的是**最难类 bracket_black**（−0.0048，仍远在 worst 门内）；但 k4 支撑增多后跨类噪声使收益归零，
   CB_full 的 bracket_black 损伤扩大到 −0.032 越过 worst 门 → 跨类传递**只在极端少样本下微正且不稳定**，不构成方法。
2. **PA**：把分支一致、双分支低距离的"置信正常"查询 patch 以**逐图 leave-own-image-out** 方式扩入参考库（杜绝自匹配退化）。
   两 shot 宏均为正（+0.0044/+0.0037）且 worst 类 ΔAP ≈ ±0.0001、ΔAUROC 均正 —— 是累计 25 个机制族中
   **唯一既无最差类损伤、又两 shot 宏方向一致为正**的族。它止步于**预注册宏门 +0.01**（缺口 ~0.006），
   而非鲁棒性权衡失败——这是与 FastRef/R1-R5 所有"宏胜-worst 深负"族根本不同的新质态。
3. **纪律提醒**：PA 未过门即按门结案；**不**基于已见结果后发调参（改阈值/扩库量/选类门控去凑 +0.01），
   否则即门拟合。PA 的机制与"双分支一致性的置信正常选取"可作为论文方法学注记（read-only 观察），
   或作为未来**预注册新基座/新族**（需 GPU 重编码 + 更多支撑下重验）的候选动机。

## 结论

- R6 双族在真实 MPDD s0 k2/k4 全门闭合为负；A1 仍为唯一冻结方法。
- 累计广度轮次 R1–R6 + Web1 + FastRef = **25 个机制族在真实门闭合为负**；PA 为唯一"良性亚门"族（宏差 0.006、worst 无损）。
- 本轴（数据/库侧）的忠实版本——VisionAD 式多层特征 + 真实旋转/flip support/query 增强、CIF 式 hypergraph 结构先验记忆、
  旋转等变 memory——全部需要真实 GPU 重编码/结构先验，列为**需授权深研候选**（已入 WEB_LITERATURE_MAP 与 FastRef 调研文档谱系）。

## 文件
- 脚本：`scripts/innovation_breadth_20260908/probe_breadth6.py`
- 结果：`experiments/dynamic_fusion/innovation_breadth_20260908/round6/RESULTS_s0_k2.json` / `RESULTS_s0_k4.json`
