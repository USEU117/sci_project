# Track-2 结案（Probe-M1：多层/中段特征是否给 A1 final 增加可用互补性）— 归档

日期：2026-09-05　立项：`docs/.../31_TRACK2_MULTILAYER_COMPLEMENTARITY_PLAN_CN_20260905.md`（doc31 §3/§4）
脚本：`scripts/innovation_t2_multilayer_20260905/export_t2_midlayers.py`（中段特征导出）、`probe_m1_complementarity.py`（探针）
数据：复用 v14 support-only 合成缓存 + 本项目 t2 中段导出缓存（`outputs/dynamic_fusion/t2_multilayer_support/k2/`）；无 /test/、无真实缺陷；冻结编码（DINO hook 复用 v14 同路径，CLIP DPAM pf[0]/pf[1]）。
设计：每 cat k2 × 留出图 h：memory=(K−1) 图 clean cells；在留出图 h 的留族 episode 上按描述子 `s=1−max_cos` 算 Pixel-AP（cutpaste/erasure；scratch 不计），另算尺度无关 nuisance-AUC（clean cells vs 15 photometric）。候选（16 网格 cell，逐层 L2，A1 式 0.5 加权 concat 再 L2）：
- M0 = A1：dino_final(L11) + clip_final(L24)　（v14 缓存）
- M1 = M0 + dino_mid(block5)　（t2 dm_*，32 网格→16）
- M2 = M0 + clip_mid(resblock6)　（t2 c6_*，37 网格→32→16）
- M3 = M0 + dino_mid + clip_mid(resblock12)　（t2 dm_* + c12_*）

## 结果（6 类宏，k2）
| 指标 | M0 | M1(+dino b5) | M2(+clip r6) | M3(+dino b5+clip r12) |
|---|---|---|---|---|
| cutpaste AP | 0.7768 | **0.7956** | 0.7297 | 0.7933 |
| local_erasure AP | 0.9658 | 0.9576 | **0.9873** | 0.9683 |
| nuisance-AUC(clean vs 15 photo) | 0.5374 | 0.5475 | 0.5583 | 0.5542 |

best（cutpaste）= M1：Δcutpaste=**+0.0188**；erasure Δ=−0.0082；AUC(M1)−AUC(M0)=+0.0101。

## 门判定（预注册，按结果未调）
- **G-M1（互补性存在：best cutpaste Δ ≥ +0.05）失败**：M1 仅 +0.019。
- G-M2（erasure 不损失 ≥ −0.01）通过：−0.0082 在线上。
- G-M3（normal 路径稳定：AUC best − M0 ≤ +0.05 且 best ≤ 0.60）通过：M1 0.5475。

**决策：`TRACK2_PROBE_FAIL_ARCHIVE`**（doc30 §2 / doc31：门未过即归档 Track-2 → 依序启动 Track-3 推理效率压缩）。

## 诚实解读
1. DINO block5 中段（M1）给出**一致但微小**的 cutpaste 增益（+0.019，6 类宏），与 Track-1 C1 邻域描述子的 +0.014 同量级；CLIP resblock6 中段（M2）在 cutpaste 上反而损失（−0.047），只在 erasure 上略升（+0.021）。
2. 即：final 层之后的额外早期/中段表示，对 A1 已覆盖的结构缺陷几乎无剩余互补性；对上下文型缺陷只回收极小缺口，不足以支撑"多层互补测量"主线（+0.05 门是预注册阈值，±0.02 量级不构成可报告的机制增益）。
3. 中段特征带来的 nuisance-AUC 抬升（+0.010~+0.021）与增益不成比例——早期层对光度扰动更敏感，进一步支持"不加中段"。
4. 未触碰真实 MPDD 门（正确：未过机制门不进真实门）。真实 parts_mismatch 等观察留给 Track-3 之后的任何机制主线，不再在此重复。

## 产物
`PROBE_M1_RESULTS.json`（聚合）、`outputs/dynamic_fusion/t2_multilayer_support/k2/*.npz`（6 类中段缓存）、本结案记录。Track-2 关闭 → 依 doc30 §2 启动 Track-3（推理效率压缩立项，doc32）。
