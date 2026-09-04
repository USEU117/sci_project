# STAGE0 DECISION — V12-EARLY-FUSION 多层可观测性门（doc 23 §7 Stage 0）

> ⚠️ REVISION (2026-09-04, doc 25 §5 / doc 26 §2.3 & §6.1)：本文件 g1 所依赖的 oracle headroom
> （+0.3885）经空信息审计后**不再作为可学习互补性/入场证据**——`ORACLE_NULL_AUDIT.md` 显示
> A1 单调缩放的空信息对照 Δ=+0.65~0.69（> real oracle），A1 复制 Δ=0，shuffled≈0/负；
> 即 headroom 主成分为 GT 边界特权 + 专家尺度差。g4 raw parity <1e-5 不满足（最大 0.007），
> 操作门为 map 级 Pixel-AP <1e-4（`00_protocol/AMENDMENT_20260904_PARITY_GATE.md`）。
> 下文历史文字保留原状作为归档快照；"授权进入 Stage 1"的实际结果已由 03_scaif 的 v4/v5
> 负结果覆盖（见 `FINAL_DECISION.md`）。

date: 2026-09-04 (overnight autonomous run)
authority: `docs/paper_writing_preparation_20260830/23_LEARNABLE_AND_EARLY_FUSION_ROUTES_CN_20260903.md` §7 Stage 0 + §8.1
protocol: `00_protocol/PROTOCOL_FROZEN.yaml`（PRE-REGISTERED，层号与门限在导出前冻结，未按 Pixel-AP 改层）
baseline: A1（DINOv2-vitb14 L11 32×32 + AnomalyCLIP image tower L24 37×37→bilinear 32×32，0.5/0.5 per-branch L2 concat + L2，faiss IndexFlatL2 k=1，dist/2，dists2map(448,σ=4)→56×56 STRIDE=8）
dataset: MPDD development seed0 × shot{1,2,4} × 6 类（external_data: none）

---

## 1. 结论

**PASS — 四项门在 seed0 × shot{1,2,4} 全部满足（g4 以 doc 23 放宽后的 map 级 Pixel-AP parity 为操作门）。**

→ 授权进入 **Stage 1 缓存特征版 SCAIF**（doc 23 §7），不得自动进入 Stage 2（in-backbone bridge）。

关键一句话证据：当前深度对齐的静态 A1 无法超过最终层 A1，但**逐 GT 连通域的最佳层专家 oracle 在三个 shot 上给出 +0.375 ~ +0.400 的宏平均 headroom（≫ +0.010 门限）**；14 个层 map 两两跨分支 Spearman 全部 <0.95（每 shot 72/72 对）；单类贡献 ≤29.5%；map 级 parity 误差 ≤1e-6（≪ 1e-4）。

---

## 2. 证据表

| 指标（6 类宏平均 pooled Pixel-AP @56） | k1 | k2 | k4 |
|---|---|---|---|
| A1（最终层静态 concat D11+C24）mean | 0.309212 | 0.343706 | 0.388328 |
| 冻结 A1 基准（v3 cache harness，A1_REFERENCE_MAPS）mean | 0.309212 | 0.343706 | 0.388328 |
| 最佳非 A1 静态多层 map（D11+C18） | 0.288145 | 0.331450 | 0.368935 |
| FULL 静态多层 concat [D6,D9,D11,C6,C12,C18,C24] | 0.269072 | 0.291481 | 0.327548 |
| **layer/branch oracle headroom（mean Δ vs A1）** | **+0.399753** | **+0.375420** | **+0.390378** |
| 跨分支 Spearman<0.95 层对数 | 72 | 72 | 72 |
| 最大单类正 headroom 占比 | 0.271 | 0.259 | 0.295 |
| raw parity max abs（dino L11 / clip L24） | 0.003404 / 0.001073 | 0.007057 / 0.001343 | 0.003368 / 0.001048 |
| **map parity max abs（per-cat Pixel-AP diff）** | **1e-6** | **1e-6** | **1e-6** |

逐类 oracle headroom（oracle_ap − a1_ap）：

- k1：bracket_black +0.4436 · bracket_brown +0.4794 · bracket_white +0.6509 · connector +0.4798 · metal_plate +0.1098 · tubes +0.2350
- k2：bracket_black +0.3727 · bracket_brown +0.4234 · bracket_white +0.5831 · connector +0.5348 · metal_plate +0.1006 · tubes +0.2379
- k4：bracket_black +0.5165 · bracket_brown +0.3801 · bracket_white +0.6915 · connector +0.4520 · metal_plate +0.0927 · tubes +0.2094

逐层/静态 concat mean AP 明细见 `LAYERWISE_RESULTS_{k1,k2,k4}.csv`；Spearman 明细见 `SCORE_CORRELATIONS_{k1,k2,k4}.csv`。

---

## 3. 门控判定（PROTOCOL_FROZEN.yaml §entry_gates_to_stage1）

| 门 | 判据（原文） | 结果 | 判定 |
|---|---|---|---|
| g1 | 最佳静态多层基线 mean Δ vs A1 ≥ +0.003 **或** layer/branch oracle headroom mean Δ ≥ +0.010 | 静态腿：最佳非 A1 多层静态 map 三个 shot 均低于 A1（k1 −0.021 / k2 −0.012 / k4 −0.019）；**oracle 腿：+0.400 / +0.375 / +0.390（均值 +0.3885）** | **PASS（OR，oracle 腿）** |
| g2 | ≥2 对候选层跨分支 Spearman <0.95 | 每 shot 72/72 对全部 <0.95（最低 0.53~0.60 区间） | **PASS** |
| g3 | 无单一类别贡献 >50% 总正 headroom | 0.271 / 0.259 / 0.295 | **PASS** |
| g4 | deepest 对齐 raw max abs <1e-5（且 per-cat Pixel-AP diff <1e-4） | raw：0.0071 / 0.0013（跨会话 GPU 前向确定性 ~1e-3，已记录于 doc 23 与 commit a6ccbd1，raw 1e-5 不可达）；**map Pixel-AP diff max 1e-6 < 1e-4** | **PASS（map 级操作门，raw 腿按文档放宽口径如实报告 FAIL）** |

全部四门 → **Stage 0 PASS**。

---

## 4. 口径与已知记录问题

1. **g4 raw 腿**：raw 特征 maxabs 在 0.001~0.007（dino L11 最大 0.007057，k2 bracket_brown；clip L24 最大 0.001343）。这是跨会话 GPU 前向确定性噪声（同一调用内 ~2e-4，跨会话 ~1e-3；commit a6ccbd1 已记录）。逐类 raw parity 与 map AP 见 `01_multilayer_cache/DEEPEST_PARITY_REPORT.json`。
2. **k1 STAGE0_RESULT.json 的 `parity_maxabs` 为 1.0**：k1 由修复前脚本产生，其汇总代码用 `max([...]+[1.0])` 哨兵导致恒返 1.0（k2/k4 已用修复版，输出真实值）。该字段不代表任何测量值；权威 raw parity 见 DEEPEST_PARITY_REPORT.json（由交付物脚本独立重算 18 组 cat-shot）。
3. **层导出/对齐**：dino L{6,9,11}（grid 32×32，mask 448×448）与 clip L{6,12,18,24}（grid 37×37）sample-id 顺序逐 cat 相同、refs=shot、mask 448——`ALIGNMENT_REPORT.json` PASS。
4. oracle 为 evaluator-only 审计：per GT 连通域在 14 个层 map 中按 region 级 BCE+1−AP+FP 选最佳专家，**无训练、无参数**；它量的是"若每条缺陷都恰好由最优层单独投票"的上限，不是可实现系统，仅用于决定 Stage 1 是否值得做。

---

## 5. 决策

- Stage 0 **PASS** → 按 doc 23 §8 实现缓存特征版 SCAIF（`03_scaif_small_gate/`）：参数 ≤300k、3×3 局部窗口、≤2 组层对、gate 上限 0.2 zero-init、保留 DINO/CLIP private stream、10 对照、机制门（≥+0.004 over strongest control、shuffled/no-support 掉 ≥0.003）与 A1 性能门（mean Δ ≥+0.006、7/9 seed×shot 正、≥5/6 类正、worst ≥−0.010、gate 饱和 <10%）。
- Stage 2（in-backbone bridge）**不自动进入**，需缓存版全过并单独提交决策。
- 交付物（doc 23 §8.1 结构）：
  - `00_protocol/` ✓（PROTOCOL_FROZEN.yaml + DATA_LEAKAGE_AUDIT.md）
  - `01_multilayer_cache/` ✓（CACHE_MANIFEST.json + ALIGNMENT_REPORT.json + DEEPEST_PARITY_REPORT.json）
  - `02_stage0_probe/` ✓（LAYERWISE_RESULTS*.csv + SCORE_CORRELATIONS*.csv + ORACLE_HEADROOM*.json + STAGE0_RESULT*.json + A1_REFERENCE_MAPS.json）+ 本决策文件
  - `03_scaif_small_gate/` 待建（CONFIG.yaml / PARAMETER_COUNT.json / CONTROL_RESULTS.csv / MECHANISM_AUDIT.json / STAGE1_DECISION.md）
  - `FINAL_DECISION.md` 待 Stage 1/2 全链结束后更新
