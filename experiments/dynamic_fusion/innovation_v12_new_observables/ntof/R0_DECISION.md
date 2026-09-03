# V12 NTOF — Nuisance-Tangent Orthogonal Fusion 决策（2026-09-03）

协议：`R0_PROTOCOL.json`（doc 22 §3.4 预注册 normal-only 机制门）
导出：`run_r2_ntof_export.py`（dino vitb14/448→32×32；AnomalyCLIP 518→37×37；仅 normal 图）
门跑：`run_r2_ntof_gates.py`（CPU）
范围：MPDD development seed0 × shot1 × 6 类；5 族干预×3 强度 + 每族 1 held-out；support（1 ref 图）有限差 top-r SVD 切空间；全部 normal-only，无 GT、无 bad、无 MVTec AD 2。

## 结果（r=3 融合，中位数跨 6 类；held-out illumination FP 门）

| 度量 | 值 | 门 |
|---|---:|---|
| g1 FP 比中位（FP_NTOF/FP_A1，需 ≤0.75） | **0.934**（分类别 0.876~1.083） | FAIL（仅降 ~7%，远低于 25%） |
| g2 合成缺陷保留（需 ≥0.90，median/类） | cutpaste 2.17 / erase 1.93 / scratch 2.23 | PASS |
| g3 对照 effect（true 需领先 ≥0.05） | true 0.060 vs rand 0.032 / wrong 0.045 / shuf 0.034 | FAIL（差 0.01–0.03） |
| g4 rank {2,3,4} fp 比 | 0.941 / 0.934 / 0.928 | 稳定（方向一致） |
| g5 单编码器对照 effect | dino 0.094 / clip −0.462 / concat 0.128 vs ntof 0.060 | FAIL（融合不如 dino-only） |
| effect（bracket_black 类） | true 0.259 vs wrong 0.258 / shuf 0.203 | 对照不显著 |

## 门判定与结论：NTOF R0 **FAIL → ARCHIVED**，转 PRS（doc 22 §10/§13）

- **g1 失败（机制假说被拒）**：单个 support 图上、由 15 个配对干预估计的全局低秩（r≤4）线性切空间，不能覆盖 unseen 光照强度/族的深层特征漂移。median FP 仅降 ~7%（需 ≥25%）。
- **g3 失败（方向无关）**：随机/错误类别/打乱配对切空间与真实切空间的 FP-reduction 效应差仅 0.01–0.03（需 ≥0.05）→ 残余投影的大部分"降 FP"来自对 ||d|| 的普遍收缩，而非对 nuisance 方向的正确剔除。正交性机制未得到证据支持。
- **g5 失败**：双编码器 0.5/0.5 融合的残余分在多数类上 ≤ dino-only；跨编码器融合在本资产上再次无宏观增益（与 CECW/MTCOA 一致）。
- g2 PASS 仅说明该投影不会破坏强合成缺陷响应（ratio≈1.9–2.5 甚至放大），不作为路线通过依据。
- 按协议 stop-rule 与 doc 22 §10：**NTOF 归档；不得用真实 bad 调干预范围或 rank；下一路线 = PRS（扰动响应谱，doc 22 §5）**；若 PRS 也失败，依序 CL-RPF / PSMF。
