# v10 Portfolio Ledger (task book 19, 2026-09-03)

Status vocabulary: `NOT_STARTED / RUNNING / PASS / FAIL / ARCHIVED`.

Baseline: git HEAD `7b164bf` (v7 Scenario C). v8/v9 are on-disk uncommitted
user work (preserved untouched). MPDD = only development set; BTAD/MVTec were
consumed by v8 (post-hoc diagnosis only); VisA = AnomalyCLIP source domain.

| 路线 | 状态 | 主增益 (Pixel-AP) | 最差类 | 对照差 | 成本 | 下一步 |
|---|---|---|---|---:|---:|---|
| A CRAM 跨参考一致性记忆 | **ARCHIVED** (R0 FAIL) | A1 −0.0028 / A2 −0.0076 (mean k2+k4 vs A0) | connector −0.029 | real −0.0028 vs shuffled −0.0037 (g6 FAIL) | 低(CPU) | 归档；per-category 观察留存(bracket_white) |
| B MESP 多视图等变稳定 | **ARCHIVED** (几何审计 PASS；promise 对照 FAIL) | dino B1 +0.0034 (5/6正) | — | real vs misaligned −0.0002 (g4 FAIL) | 中(GPU) | 归档；全融合 R0 未执行（成本不值得） |
| C CAPM 规范化对齐位置记忆 | **ARCHIVED** (R0 FAIL; 可行性 PASS) | mean ΔAP −0.0275 (0/6 正) | tubes −0.0909 | real vs random −0.0033 (FAIL) | 中 | 归档；对齐管线可复用 |
| D NORC 正常参考区域 conformal | **ARCHIVED** (无伙伴路线可门控；机制就绪) | — (安全模块, 非 AP 路线) | — | g0/g1 PASS by construction; g2/g3 无伙伴 → 归档 | 低-中 | 设计+机制入负结果档案 |
| E STR 频谱纹理残差 | **ARCHIVED** (R0 FAIL) | Δ(STR−A1) −0.027 (region info-value) | bracket_black −0.126 | true-vs-misaligned +0.030 (moot) | 低(CPU) | 归档；bracket_brown 类别专项观察 +0.041 |
| F SPRG 自监督部件关系图 | **ARCHIVED** (可行性门1 FAIL) | 跨样本节点匹配率 36% (<90%) | — | 匹配余弦高(0.87-0.97)但链稳定性崩 | 高 | 立即停止；MPDD 无稳定部件对应 |
| X LLSE 记忆局部线性重构残差 | **ARCHIVED** (R0 PASS → seed 确认 FAIL) | s0 mean ΔAP +0.0089 (4/6正)；三 seed 均值 +0.0025 | s1 bracket_black −0.0565 | random-8 破坏对照 PASS 3/3 (+0.19) → 机制确为 locality 驱动 | 低(CPU) | 归档；tubes/metal_plate 三 seed 一致增益(+0.013~+0.040)观察留存 |
| X2 CSS 图内上下文自一致性 | **ARCHIVED** (g2 融合门 FAIL) | CSS-alone mean AP 0.040 (reference-free 真信号)；fused-sum −0.042 / fused-max −0.025 (3/6正) | metal_plate −0.365 | corr(defect)≤0 on A1 强类 → 边界环线索与大缺陷内部自一致冲突 | 低(CPU) | 归档；bracket_* 弱类被拯救(+0.08~+0.16) = A1 弱类参考图不具代表性的类别证据 |

Per-route deliverable dirs: `experiments/dynamic_fusion/innovation_v10_portfolio/<route>/`
with `R0_PROTOCOL.json` (before results) → `R0_RESULT.json` → `R0_DECISION.md` → `FAILURE_ANALYSIS.md`.
