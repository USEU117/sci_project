# Track-4 立项：保排序 memory 压缩（诊断性修正 Track-3 真实门失败）

日期：2026-09-05　上游：doc33 REAL_GATE_FAIL（Track-3 主线归档）→ 用户选定新方向 **保排序 memory 压缩**，并保留队列：高分辨率关系描述子、真实 parts_mismatch 诊断 待后续依序尝试。
性质：**真实门测量（直接真实验证）**——Track-3 已证明合成代理无法预测真实单类崩溃（棋盘 coreset 在合成 LOO 上"无损"，真实 connector 却掉 −0.033/−0.053），故 Track-4 不做合成探针，直接在真实 MPDD 上预注册验证"保排序修剪"是否修复该崩溃。

## 1. 失败诊断与假设
- doc33 失败根因：**几何均匀（棋盘）coreset 把 memory 单元视为 i.i.d.**，50% 采样恰好移除少量真实缺陷 patch 的关键近邻 → Pixel-AP（强调 top 正样本）大降，而 AUROC/AUPRO 无感。
- 假设：**memory 单元的可删除性不均匀**——被大量正常 patch 引用为最近邻的"高杠杆"单元（正常流形代表点）与均匀覆盖流形的"中心"单元决定排序稳定性；**保留高杠杆/覆盖性单元、删除孤立低引用单元**，可在 50% 压缩下保持真实 Pixel-AP。
- 与 Track-3 的区别：从"几何位置均匀删除"改为"**按排序/流形结构保留**"；仍然零训练、零拟合、support-only、确定性。

## 2. 候选（预注册；全部确定性，压缩率 50%，作用于 ref memory 端）
| 候选 | 保留规则（50%） | 直觉 |
|---|---|---|
| P1（已知失败对照） | 棋盘 (i+j)%2==0 | 复现 doc33 失败，验证口径 |
| C1 | **高杠杆 top-50%**：在 L2 归一化的 A1 空间中，以全部 ref clean patch 为 query、对 memory 做 k=2 最近邻；统计**第二近邻**（排除自身）引用频次 → 保留频次最高的 50% 单元 | 正常流形的"代表/hub"单元是被引用最多的排序决定者 |
| C2 | **farthest-point 覆盖 top-50%**：确定性 seed 从首个 cell 起，贪心选离已选集最远的 cell，直至 50% | 均匀覆盖正常流形 → 任一真实 patch（含缺陷）都有不远的保留近邻 |

memory 构建与打分完全复用 A1 冻结 concat（pca0/whiten0/w0.5，32-grid，faiss k=1 → 448 map，Pixel-AP stride=8）；只替换 ref 行子集；查询端不变。

## 3. 门（预注册，按结果不调；两 shot 均需满足，避免单 shot 噪声）
- **G-F1（宏无损）**：存在 C∈{C1,C2}，seed0 **k2 与 k4 均**：六类宏 ΔPixel-AP(C − full) ≥ **−0.01**；
- **G-F2（无灾难类）**：同一 C，k2 与 k4 均：每类 ΔPixel-AP ≥ **−0.03**（P1 预期两 shot 均违规，作口径验证）；
- **G-F3（整体不损 + 效率）**：同一 C：宏 ΔAUROC ≥ −0.005 且 ΔAUPRO ≥ −0.005；memory patch 数 = 0.5×。
- 通过 → Track-4 结案：A1+保排序 coreset 无损成立（论文 Fig07 效率主张可用此策略）；失败 → 归档并转队列下一方向。

## 4. 数据与纪律
- A1 冻结特征缓存 seed0 k2/k4（`outputs/dynamic_fusion/v3_direction_a/`）；真实 MPDD test 只读评价。
- 纪律：候选规则在运行前冻结；不按真实结果增删候选/调压缩率/豁免类别；不读 /test/good 进 memory；无拟合（C1/C2 仅在 ref clean patch 内做结构选择，不触碰 test 标签）。

## 5. 产物
- `scripts/innovation_t3_efficiency_20260905/run_t4_preserve_coreset.py`
- `experiments/dynamic_fusion/innovation_t3_efficiency_20260905/TRACK4_REAL_s{seed}_k{shot}.json`、`TRACK4_DECISION.md`

## 6. 执行结果（2026-09-05，seed0 k2/k4 六类真实门）
- **C2（farthest-point 覆盖 50%）决定性修复 Track-3 根因**：connector 单类崩溃从棋盘 −0.033/−0.053 收敛到 −0.010/−0.005；宏 ΔPixel-AP = −0.0031/−0.0018（G-F1 两 shot 过）、无灾难类（G-F2 两 shot 过）、ΔAUROC 两 shot 过；**仅 k2 ΔAUPRO = −0.0060 低于 −0.005 门 0.001**（k4 −0.0031 达标）→ 严格门 G-F3 未全过，主假设证实。
- C1（高杠杆 hub 保留）失败（connector −0.077/−0.085）：hub 高频引用 ≠ 缺陷判别重要。
- 状态：边界结果已如实归档于 `experiments/dynamic_fusion/innovation_t3_efficiency_20260905/TRACK4_DECISION.md`；是否按"通过（Fig07 效率支撑）"接受由用户后续裁决。

## 7. 后续方向队列（用户指示：尽量都尝试，按序）
| # | 方向 | 触发 |
|---|---|---|
| 5 | 高分辨率关系描述子（32-grid 邻域/跨尺度，重启 Track-1 主线的明确建议） | Track-4 结案后 |
| 6 | 真实 parts_mismatch 子组诊断（A1 冻结下量化上下文类缺陷剩余缺口，纯测量） | 方向 5 后 |
