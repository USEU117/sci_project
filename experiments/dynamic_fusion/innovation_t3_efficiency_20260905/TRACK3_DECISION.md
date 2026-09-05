# Track-3 结案（Probe-E1：A1 效率压缩是否几乎无损）— 探针通过

日期：2026-09-05　立项：`docs/.../32_TRACK3_EFFICIENCY_COMPRESSION_PLAN_CN_20260905.md`（doc32 §2/§3）
脚本：`scripts/innovation_t3_efficiency_20260905/probe_e1_compress.py`
数据：复用 v14 support-only 合成缓存（dino/clip k2，16 网格 A1 cell 1536-D）；无 /test/、无真实缺陷、零训练。
设计：每 cat k2 × 留出图 h：memory=(K−1) 图 clean cells；Pixel-AP（cutpaste/erasure）与 nuisance-AUC 按候选；coreset 只压缩 memory 端，查询全 256 cells 打分。

## 结果（6 类宏，k2）
| 候选 | cutpaste AP | erasure AP | nuisance-AUC | 检索成本 |
|---|---|---|---|---|
| T0 = A1（1536-D 全 memory） | 0.7768 | 0.9658 | 0.5374 | 1.0× |
| T1 = 50% 棋盘 coreset（128 cells/图） | **0.7744** | **0.9707** | 0.5275 | **0.5×** |
| T2 = 25% 棋盘 coreset（64 cells/图） | 0.6924 | 0.9683 | 0.5202 | 0.25× |
| T3 = PCA→384-D | 0.7432 | 0.8547 | 0.5247 | 0.25× |
| T4 = PCA→192-D | 0.7261 | 0.8298 | 0.5196 | 0.125× |

## 门判定（预注册，按结果未调）
- **G-T1（几乎无损保持）通过**：T1 cutpaste Δ=−0.0024（≥−0.03）、erasure Δ=+0.0049（≥−0.01）。
- **G-T2（效率 ≥0.5×）通过**：T1 成本 0.5×。
- **G-T3（normal 稳定）通过**：T1 AUC 0.5275（≤0.60，且 −T0 = −0.010）。
- T2/T3/T4 未过 G-T1（T2 掉 memory 过多、PCA 线性压缩破坏结构缺陷近饱和表示）。

**决策：`TRACK3_PROBE_PASS`**（doc32：通过 → 立项 Track-3 主线，真实 MPDD 门 + 蒸馏/量化落地）。

## 诚实解读
1. **memory 端确定性 50% 棋盘 coreset 在 k2 代理上几乎无损**（cutpaste −0.002、erasure 略升、AUC 略降）——A1 的 memory 单元存在明显信息冗余，半采样不损失判别力，还微降 nuisance 敏感度。
2. 线性降维（PCA→384/192）显著损害 erasure（−0.11~−0.14）：A1 1536-D 中 clip/dino 各自的 768-D 正交信息对结构缺陷判别是必要的，纯维度截断不可行；25% coreset 开始掉 cutpaste（−0.084）。
3. 效率增益来源明确为 **memory 单元数减半（检索/存储减半），不牺牲双分支信息**——与"单分支蒸馏"路线互补但更轻、零训练、可立即落地。
4. 探针在 support 合成代理上通过；**真实 MPDD 六类像素门**（test 侧、同协议 A1 基线对比）是主线立项后的验收门槛，尚未触碰（正确：机制门通过才进真实门）。

## 产物
`PROBE_E1_RESULTS.json`（聚合）、`experiments/dynamic_fusion/innovation_t3_efficiency_20260905/`、本结案记录。

## 下一步（待用户确认是否立项主线）
doc32 预注册：探针通过 → 立项 Track-3 主线 = 在真实 MPDD 六类同协议下验证「A1 + 50% coreset memory」相对 A1 全 memory 的像素 AP 无损性与端到端成本下降，作为论文 Fig07 效率主张的可选支撑；不涉及新 backbone、不训练。是否启动由用户决策（三个方向探针阶段已全部完成：T1 失败、T2 失败、T3 通过）。
