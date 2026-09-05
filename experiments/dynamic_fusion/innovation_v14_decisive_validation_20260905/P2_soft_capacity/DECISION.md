# P2 DECISION.md — 图像域容量转移门结果与判定（P2-A，16 网格）

日期：2026-09-05　计划：doc28 §6.1　冻结：ε=0.05、τ=4.0、ρ uniform、grid16（32→16 mean-pool）、Q=256。

## 1. 结构
对 (cat, shot=2) 每个留出 support 图 h：anchors = 其余 (K−1) 张 support clean 图的 16 网格 cells（k2：A=256）；queries = h 的 clean / 9 合成 episode（cutpaste/erasure/scratch ×3 seeds，1024 渲染后重编码）/ 15 光度 nuisance / 9 错配控制（dino 特征与另一 episode 的 clip 特征配对）。方法：dino、clip、concat 的 semi-relaxed OT 容量 premium（每行 semi-OT 期望成本 − 同分支自由熵 soft 成本）。

## 2. k2 结果（宏平均 over cats×h）
| 门 | 判定式 | concat | dino | clip | 判定 |
|---|---|---|---|---|---|
| A1 容量转移 | inside(syn) − clean 图 prem p95 ≥ +0.02 | −0.0173 | −0.0280 | −0.0037 | **FAIL** |
| A2 spillover | far / ring − clean p95 ≤ +0.02 | far −0.0154, ring −0.0174 | 负 | 负 | PASS（无远背景抬升，因信号本就缺失） |
| A3 nuisance | nuis p95/max − clean ≤ +0.02 | −0.0001 / −0.0003 | ≈0 | ≈0 | PASS |
| A4 面积单调 | Spearman(area, inside) > 0 | 0.190 | 0.265 | 0.491 | PASS |
| A5 配对 | matched − mismatch ≥ 0 | −0.0012 | ≈0 | +0.0022 | **FAIL（concat/dino）** |

单分支 CAP-D（dino 或 clip 单独过 A1&A2）：否。

**决策：`P2A_FAIL_IMAGE_DOMAIN`。** 三个分支的缺陷区容量 premium 均 ≤ 正常背景 p95（合成区 premium ≈ 0.003–0.012，正常 p95 ≈ 0.02–0.04）。即：把缺陷复制/擦除/划痕在**真实图像重编码**（含边界、重采样、Transformer 上下文）后，软容量约束没有在缺陷区产生可测的额外匹配成本。按 doc28 §6.1："若真实图像重编码后容量信号消失，停止，不进入 MPDD GT" → **P2-B（真实 MPDD 容量门）不执行**。

## 3. k4 稳健性检查（未完成）
k4（anchors=768）在 solo CPU 下 27 分钟仍未完成任一类别（预计 6 类 >2 小时），且 k2 已决定性失败（三分支全负），按 §6.1 停止规则与成本控制终止，记录为"因时间/资源未完成"，不作证据。产物中保留 `IMAGE_PROBE_k2.json` / `P2A_GATES_k2.json` 作为 k2 存档（canonical `IMAGE_PROBE.json`/`P2A_GATES.json` 即 k2 内容）。

## 4. 机制解读（诚实归因）
- cutpaste 复制的是**正常纹理块**：其 patch 级特征与 normal memory 仍高度相似，只在语义上下文层面异常 → 自由匹配距离已能把该类缺陷分开（P1-B 中 cutpaste AP 0.65+），但列拥挤带来的容量 premium 在缺陷区不高于正常背景 → 容量耦合相对自由匹配无增量。
- 因此容量机制只能记为"未在图像域建立证据"，不支持"跨分支容量融合"或"单分支容量辅助(CAP-D)"的提法。

产物：`IMAGE_PROBE.json`（含 `_k2` 存档）、`P2A_GATES.json`（含 `_k2` 存档）。
