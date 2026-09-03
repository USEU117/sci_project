# X2 CSS 图内上下文自一致性 — R0 决策（2026-09-03）

协议：`R0_PROTOCOL.json`（含 2026-09-03 amendment：fused-max 次级诊断）
脚本：`scripts/innovation_v10_portfolio/run_r0_explore_css.py`
机制：score(q) = q 与其**同图内** Chebyshev 半径 2 的 24 个空间邻域的 mean(1−cos)。
完全不用参考图——与全部已归档路线（A–F/LLSE/子空间/text）的"参考记忆/语义证据"
机制不相交。仓库审计确认此前无任何 self-similarity/context-consistency 实现。

## 结果（MPDD s0/k1，fused A1 空间，6 类）

| 类 | A1 AP | CSS AP | fused-sum Δ | fused-max Δ | corr(defect) | corr(normal) |
|---|---:|---:|---:|---:|---:|---:|
| bracket_black | 0.0105 | 0.0144 | **+0.160** | **+0.130** | −0.03 | 0.55 |
| bracket_brown | 0.0344 | 0.0128 | +0.012 | +0.008 | −0.11 | 0.47 |
| bracket_white | 0.0848 | 0.0104 | +0.078 | +0.064 | +0.33 | 0.55 |
| connector | 0.1261 | 0.0104 | −0.054 | −0.029 | −0.12 | 0.56 |
| metal_plate | 0.8713 | 0.1700 | **−0.365** | **−0.292** | **−0.38** | 0.45 |
| tubes | 0.7282 | 0.0242 | −0.082 | −0.032 | −0.23 | 0.46 |
| **mean** | 0.309212 | 0.0404 | **−0.0418** | **−0.0250** | — | — |

（g0 identity：A1 0.309212 ✓；融合规则均为固定逐图 z-score，无任何类别/阈值调参。）

## 门判定
- g0_identity：PASS。
- g1_promise（reference-free 流在 A1 最弱类有正信号）：**PASS**——bracket_black
  CSS AP 0.0144 > A1 0.0105（A1 AP<0.05 的最难类之一）；CSS-alone mean AP 0.040。
- g2_fusion（sum 规则）：**FAIL**（mean −0.0418、3/6 正、worst −0.365）。
- fused-max 次级诊断：同样 FAIL（mean −0.0250、3/6 正、worst −0.292）。

## 结论：ARCHIVED（exploration；融合门失败）
CSS 是**真实但弱**的 reference-free 证据源，且在 A1 最弱类（bracket_*）融合后可大幅
拯救（+0.078~+0.160）；但它在 A1 强类（metal_plate/tubes）上**系统性破坏**：corr(defect)
为负（−0.38/−0.23）表明大块连贯缺陷的内部像素是自一致的（CSS 只在其边界环响），
把 CSS 抬高的正常纹理边界像素混入后 AP 崩塌。诊断显示 CSS 主要在**正常像素**上与 A1
正相关（0.45–0.56），在缺陷像素上弱/负相关——即它对"哪些正常像素难"与 A1 看法一致，
却对"缺陷在哪"提供不了与 A1 互补的稀疏、区域选择性证据。

## 留存观察与含义
1. **reference-free 线索的价值只在 A1 失效处**：CSS 补上了 bracket_black 这类"参考图
   代表性差"的类——这首次给出了 A1 错误模式的类条件证据（A1 弱类 = 参考图不具代表性）。
2. 想同时保住强类需 **class-conditional 权重**（按类别决定是否信任 CSS）——这需要类别
   先验或元学习，超出"固定无调参规则"且引入泄漏风险 → 不纳入。
3. 根因与 A–F/LLSE 同一叙事再获一个独立支持：MPDD 六类缺陷形态差异大（bracket 小缺陷 vs
   metal_plate/tubes 大区域缺陷），**任何单一固定证据源都无法全局稳定超越 A1**。
