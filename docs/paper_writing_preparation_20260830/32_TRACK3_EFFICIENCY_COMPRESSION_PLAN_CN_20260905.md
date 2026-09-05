# Track-3 立项：推理效率压缩探针（doc30 §2 第三方向，端到端成本目标）

日期：2026-09-05　上游：doc30 路线图（Track-1 已归档 → Track-2 已归档 → Track-3 启动）；doc31 Probe-M1（M0=A1 基线已在 k2 代理上实测）；论文 Fig07（效率/内存成本定位）。
性质：**效率—保持权衡测量**——先回答「A1（DINO+CLIP 双分支全 memory 1536-D）是否存在几乎无损的效率压缩操作点」；有操作点才立项蒸馏/量化主线，否则归档（三个方向全部尝试完毕）。

## 1. 动机与问题
- A1 端到端成本两大来源：(a) 双 backbone 编码（DINOv2-b14 448 + CLIP-L/14 518）；(b) 查询 cell 与全 memory（每 memory 图 256 cells × 1536-D）的 `1−max_cos` 检索。
- 预注册问题：**在 support-only LOO 结构上，对 A1 描述子/memory 做确定性轻量压缩（coreset 子采样 / 线性降维），缺陷 Pixel-AP 是否几乎无损（≤0.03），同时检索成本至少减半？**
- 不重复：单分支（clip-only/dino-only）问题已由 v14 冻结线 G2 系统回答，不在本探针重复；本探针只测「保留双分支信息但压缩 memory/维度」这一 v14 未答的空白。

## 2. 数据、压缩候选与红线
- 数据：同 Track-2 全部 support-only 合成缓存（v14_p1_support dino/clip k2 + t2 无关，只用 v14 final），16 网格 cell，LOO memory，不读 /test/good，不用真实缺陷，不训练。
- 基线 **T0 = M0（A1）**：`z=rowL2([0.5·dino_final, 0.5·clip_final])`（1536-D），`s=1−max_cos(cell, memory)`；k2 六类宏已测：cutpaste AP=0.7768、erasure AP=0.9658、nuisance-AUC=0.5374（PROBE_M1_RESULTS.json）。
- 压缩候选（全部确定性、零拟合/仅 support 拟合）：
  | 候选 | 操作 | 检索成本比（vs T0） |
  |---|---|---|
  | T1 | memory 棋盘 50% coreset（256→128 cells/图） | 0.5× |
  | T2 | memory 棋盘 25% coreset（256→64 cells/图） | 0.25× |
  | T3 | 双分支拼接后 PCA→384 维（support clean cells 拟合） | 0.25×（维度） |
  | T4 | 双分支拼接后 PCA→192 维（support clean cells 拟合） | 0.125×（维度） |
- PCA 拟合只用该 cat 全部 K 张 support clean cells（data-role 允许：均属 manifest support），对 memory 与查询统一投影；不做通道级量化/蒸馏（超出纯测量范围）。
- 度量：cutpaste/erasure Pixel-AP 宏、normal nuisance-AUC（口径同 Track-1/2）。

## 3. 门（预注册，按结果不调）
- **G-T1（几乎无损保持）**：存在候选 T∈{T1..T4}：cutpaste 宏 AP ≥ T0−0.03 且 erasure 宏 AP ≥ T0−0.01（结构缺陷已近饱和，损失容忍更严）；
- **G-T2（效率成立）**：同一候选检索成本 ≤ 0.5× T0（T1 即达标，T2–T4 更强）；
- **G-T3（normal 稳定）**：同一候选 nuisance-AUC ≤ 0.60 且相对 T0 ≤ +0.05。
- 通过 → 立项 Track-3 主线（真实 MPDD 门 + 蒸馏/量化落地）；失败 → 归档 Track-3，三方向全部尝试完毕并汇总。

## 6. 主线执行结果（doc33，2026-09-05）
- **`REAL_GATE_FAIL`**：真实 MPDD seed0 k2/k4 六类宏 ΔPixel-AP = −0.0070/−0.0074（G-R1 宏无损过），但 **connector 单类崩溃**（−0.033/−0.053，G-R2 失败，两 shot 复现）→ 几何均匀 coreset 对像素级精确排序非无损，合成代理未捕获真实高杠杆近邻风险。Track-3 主线归档，详见 `experiments/dynamic_fusion/innovation_t3_efficiency_20260905/REAL_GATE_DECISION.md`。

## 4. 产物与成本
- `experiments/dynamic_fusion/innovation_t3_efficiency_20260905/`；脚本 `scripts/innovation_t3_efficiency_20260905/probe_e1_compress.py`。
- 成本：纯 CPU 评价（复用 v14 缓存），6 类 k2 ≈ 5–10 分钟。
- 报告：逐类×族 AP（T0–T4）、宏 Δ、AUC、实际成本比。

## 5. 执行提示
> 执行 Track-3 Probe-E1：读 v14 support 缓存（dino/clip k2）；构造 T0 基线与 T1–T4（coreset 棋盘采样 / support PCA 降维）；按 (cat, h, 族) LOO memory KNN 算 Pixel-AP 与 nuisance-AUC；输出宏表与成本比；按 G-T1/G-T2/G-T3 判定。零拟合仅评价，失败即归档并给出三方向汇总。
