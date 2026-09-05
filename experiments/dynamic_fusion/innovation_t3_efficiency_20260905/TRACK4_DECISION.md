# Track-4 结案（保排序 memory 压缩，真实 MPDD 门）— 边界：主假设证实，严格门 G-F3 差 0.001

日期：2026-09-05　立项：`docs/.../34_TRACK4_PRESERVE_RANKING_CORESET_PLAN_CN_20260905.md`（doc34）
脚本：`scripts/innovation_t3_efficiency_20260905/run_t4_preserve_coreset.py`
数据：A1 冻结特征缓存 seed0 k2/k4；真实 MPDD test 只读评价；候选规则运行前冻结；确定性；无拟合、无 /test/ 进 memory。
口径：完全复用 A1 冻结 concat（pca0/whiten0/w0.5，32-grid，faiss k=1 → 448 map，Pixel-AP stride=8）；只替换 ref 行子集（50%）。

## 结果（seed0，六类宏 Δ 相对 full memory）
| shot | 候选 | mean ΔAP | worst ΔAP（类别） | ΔAUROC | ΔAUPRO |
|---|---|---|---|---|---|
| k2 | P1 棋盘（对照） | −0.0070 | −0.0328（connector） | +0.0006 | +0.0004 |
| k2 | C1 高杠杆 top50% | −0.0157 | −0.0769（connector） | −0.0036 | −0.0035 |
| k2 | **C2 farthest-point 覆盖** | **−0.0031** | **−0.0102（connector）** | −0.0021 | **−0.0060** |
| k4 | P1 棋盘（对照） | −0.0074 | −0.0527（connector） | −0.0007 | −0.0042 |
| k4 | C1 高杠杆 top50% | −0.0155 | −0.0848（connector） | −0.0026 | −0.0063 |
| k4 | **C2 farthest-point 覆盖** | **−0.0018** | **−0.0046（connector）** | −0.0013 | **−0.0031** |

P1 口径校准成功：两 shot 均重现 doc33 的 connector 崩溃（−0.033/−0.053），确认脚本与 Track-3 一致。

## 门判定（预注册 doc34，按结果未调）
C2（farthest-point 覆盖）：
- **G-F1（宏无损，两 shot）通过**：k2 −0.0031 ≥ −0.01；k4 −0.0018 ≥ −0.01。
- **G-F2（无灾难类，两 shot）通过**：k2 worst −0.0102 ≥ −0.03；k4 worst −0.0046 ≥ −0.03。
- **G-F3（整体不损）边界未过**：ΔAUROC 两 shot 均 ≥ −0.005（−0.0021/−0.0013）；**ΔAUPRO k2 = −0.0060 低于 −0.005 门 0.001**（k4 −0.0031 通过）。
C1（高杠杆）：G-F1/G-F2 均失败（connector 反而更差，−0.077/−0.085）——保留高引用 hub 单元却删除低引用"尾"单元，反而丢掉了少量真实缺陷所需的外缘近邻；高杠杆假设不成立。
C2 per-shot：k4 三门全过（win）；k2 仅 AUPRO 一项差 0.001。

**决策（如实）：主假设证实但严格门未全过。** C2 决定性修复了 Track-3 归档的唯一根因（connector 单类崩溃，−0.033/−0.053 → −0.010/−0.005），宏 AP 两 shot 无损、无灾难类；唯一缺口是 k2 AUPRO 门差 0.001。按纪律不按结果放宽门限；是否将 C2 视为"通过/边界接受"或归档转下一方向，由用户裁决（见 doc34 §6 队列：高分辨率关系描述子、parts_mismatch 诊断）。

## 诚实解读
1. **几何均匀 coreset 的失败根因诊断得到直接验证**：同 50% 压缩下，从"棋盘均匀删"换成"farthest-point 覆盖删"，connector 崩溃消失（两 shot）。原因：棋盘按空间位置删，破坏对少量真实缺陷 patch 的高杠杆近邻；farthest-point 在特征流形上均匀保覆盖，任何查询 patch 都留有距离相近的保留近邻。
2. C1（按 2 近邻引用频次保 hub）反证：**"被正常 patch 高频引用"不等于"对缺陷判别重要"**——hub 高度冗余且自相似，删除低引用的流形边缘单元反而损失判别细节。
3. 唯一未达标项（k2 AUPRO −0.006 vs −0.005）量级 ≈ 宏 0.001，且 k4 达标、Pixel-AP 两 shot 均达标；AUPRO 对阈值面积比 Pixel-AP 更平滑，此缺口是否构成"非无损"存在合理分歧。
4. 若用户接受 C2 为通过：A1+50% farthest-point coreset 可作为论文 Fig07 效率主张（无损宏 AP、无灾难类、修复棋盘缺陷），保留 A1 主配置不变。若归档：本结果作为"保排序 coreset 优于几何 coreset"的机制证据留档。

## 产物
`TRACK4_REAL_s0_k2.json`、`TRACK4_REAL_s0_k4.json`（逐类明细）、doc34、本结案记录。
