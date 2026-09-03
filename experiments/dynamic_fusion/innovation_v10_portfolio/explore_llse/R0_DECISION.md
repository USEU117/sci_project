# X LLSE 局部线性重构残差 — R0 决策（2026-09-03）

协议：`R0_PROTOCOL.json`（含 2026-09-03 amendment：行打乱对照 → per-query random-8 局部性破坏对照）
脚本：`scripts/innovation_v10_portfolio/run_r0_explore_llse.py`（seed 参数化，s0 写 R0_RESULT.json、s1/s2 写 R0_CONFIRM_s{seed}.json）

## 机制
A1 对每个 query patch 记 **到单个最近邻记忆 patch 的距离**（1-NN）。LLSE 改为记
**query 到其 top-8 近邻张成的局部线性流形的最小二乘重构残差**：
r² = ‖q‖² − wᵀ(Nᵀq)，w = solve(NᵀN + 1e-3·I, Nᵀq)。机制与 1-NN 不同，且 AP 对分数做
任意单调变换不变，因此 ΔAP≠0 意味着发生了非单调重排，不是 A1 分数场的平滑变换。

## 结果（MPDD, shot=1, fused A1 空间, 6 类, CPU）

| seed | A1 mean AP | LLSE mean ΔAP | n 正类 | 最差类 | random-8 对照 Δ | true−random |
|---|---:|---:|---:|---:|---:|---:|
| s0 | 0.309212 | **+0.0089** | 4/6 | bracket_black −0.0028 | −0.179 | +0.188 |
| s1 | 0.342511 | **−0.0028** | 3/6 | bracket_black **−0.0565** | −0.204 | +0.201 |
| s2 | 0.315919 | **+0.0014** | 2/6 | connector **−0.0205** | −0.188 | +0.189 |

逐类 ΔAP（s0 / s1 / s2）：metal_plate **+0.0128 / +0.0122 / +0.0195**；
tubes **+0.0362 / +0.0399 / +0.0354**；connector +0.0063 / +0.0152 / **−0.0205**；
bracket_white +0.0031 / −0.0248 / −0.0198；bracket_black −0.0028 / −0.0565 / −0.0002；
bracket_brown −0.0022 / −0.0029 / −0.0059。

## 门判定
- g0 identity：PASS（s0 A1 mean AP 0.309212 ≈ 冻结 0.3092，Δ=1.2e-5 ≤ 1e-4）。
- g1（mean≥+0.003 且 ≥4/6 正）：s0 PASS；s1/s2 **FAIL**（multi-seed 不稳）。
- g2（worst≥−0.015）：s0 PASS（−0.0028）；s1 **−0.0565 FAIL**；s2 **−0.0205 FAIL**。
- g3 对照（true top-8 增益 > random-8 增益 ≥ +0.002）：**3/3 seed PASS**（+0.188/+0.201/+0.189）。

## 结论：ARCHIVED（exploration；seed 稳健性确认失败）
修正后的 random-8 对照证明 LLSE 的增益确实来自**邻域局部性**（打散邻域后残差分数
场彻底失去分离能力，3/3 seed 一致），机制本身是真实且新颖的——但真实增益的**幅度
与方向不稳定**：s0 看似通过 g1，s1/s2 平均转为负，bracket_black / bracket_white /
connector 出现 seed 依赖的大幅回退。按任务书 R0→确认 纪律，**不升级、不进入候选主线**。

## 留存观察（per-category）
- **tubes 与 metal_plate：三 seed 一致增益（+0.013 ~ +0.040）**。两类的共同点：
  缺陷区域连贯、A1 AP 本已较高（0.68+/0.85+），LLSE 残差在"缺陷位于正常流形之外"
  的假设下能进一步分离。
- bracket_brown 三 seed 一致小幅负（≈−0.003~−0.006）；低 AP 类对参考图像选择敏感。

## 对 Scenario E 的影响
无。主线仍为 A1（冻结），LLSE 仅作为 per-category 观察与负结果档案的一部分，
与 A–F 的归档路线并列。若未来有"缺陷区域连贯"的数据集/任务（如大尺度缺陷
语义分割）或在融合打分层做 weighted ensemble，LLSE 残差可作为 A1 的补充证据源
复用，但单独不足以支撑论文主张。
