# 创新路线夜间执行记录（2026-09-03，自动运行，截至 04:30）

> 背景：用户授权 09-02 夜至 09-03 08:00 自主占用本机，先按 16 号任务书推进 S0-DG-SAFE
> 分波，再做后续创新探索；关切点：**“只有两个视觉分支结合，创新程度可能不足”**。
> 本文件记录实际执行结果。所有结论均来自 MPDD development 上的诊断，BTAD/MVTec/VisA 未触碰。

## 0. 原则（沿用 16 号任务书 §5/§7）
- MPDD 是唯一 development；外部集冻结。
- 每个方向先做“独立信息价值”小门，失败即归档并记录 DECISION。

## 1. S0-DG-SAFE 主路径：Wave1 过、Wave2 失败 → 整路线归档（A1 保持主方法）

| 波次 | 结果 | 证据 |
|---|---|---|
| Step 4 导出（18 npz，s0×k{1,2,4}，~1.7h GPU） | ✅ | `sub_maps_s0/*.npz`；audit 重跑与冻结审计逐位一致 |
| Wave0 identity replay | ✅ | 逐类 mean\|Δ\|=1e-6..1.3e-4 ≤5e-4（GT 须用官方 PIL NEAREST 路径） |
| Wave1 互补上限 | ✅（cond A） | 固定 mean 组合 pooled Δ=**+0.0164** ≥+0.005；oracle headroom 仅 +0.0066 |
| Wave2 正常-only 可靠性 | ❌ **归档** | Spearman ρ(r_sub,Δ)=**+0.3387<0.40**；connector r_sub 未入后 25% |

Wave2 失败机制（详见 `Wave2_reliability/WAVE2_ARCHIVE.md`）：
- B_tail 实测为 k 的纯函数（k1=1.7047,k2=2.3026,k4=2.9444=log((2+9k)/2)）：逐像素池内最大 z 恒为
  量化上限且占池 ~1/n>1% → P99 必然命中 → 无类别信息（任务书公式在此小池校准形式下退化）。
- U_layer 在 k1 池(9 值)同样强量化（多类同值 0.1054、部分类=0）。
- 因此 r_sub 排名主要由 U_aug 与量化噪声决定 → 无法把 connector（SUB 最差）排入低可靠组。
- 结论：**正常-only 稳定性保护门在本 few-shot 小池形式下不可行**；禁止训练 router（16 §2.7）。

## 2. 创新方向探索结果（对“两个视觉分支结合”关切的证据化回答）

### P5-B：S2-GPMR 责任熵预检（CPU，02:10）→ **归档**
- 分桶对角 GMM(4 分量)于缓存 vitb14 ref 特征；test patch 责任熵 vs 像素 GT 误差
  avg |ρ|=**0.019**；−loglik avg 0.12（metal_plate 0.39 例外）。
- 16 §4.3 停止门触发。留档 `p5b_gpmr_precheck/P5B_GPMR_PRE.json`。

### P5-C：同主干内分支子集 concat vs DINO-only（CPU，02:30）→ **无增益**
- pooled：concat(A1) 0.3456 / dino-only 0.3175；mean combo Δ=**−0.0103**（负）；oracle +0.004。
- 解释：同主干内“分支子集”不构成可融合的几何异质。

### P5-D：同主干 vitb14 的 PCA 子空间重建专家 vs A1（CPU，03:55）→ **弱增益**
- pooled：subspace 0.3337 vs concat 0.3456；mean combo Δ=**+0.0026**（<+0.005 门）。
- 对照：异构 giant 子空间（Wave1）mean combo Δ=+0.0164。
- 结论：A1↔SUB 的可融合互补性 = **几何类型差异（记忆 KNN vs 子空间重建）× 骨干尺度**共同贡献；
  单靠同骨干换几何不足以过门，换骨干(giant)也不只是“更大模型”而是给了更强的正常子空间。

### P5-A-lite：S1-HGLC 图像级 DINO CLS（GPU，04:15）→ **归档（DINO CLS 轴）**
- vitb14 CLS→k-shot 参考最近余弦距离；pooled Image-AP 0.677 vs A1 图像分 0.745 → Δ=**−0.068**，
  16 §3.3 门槛（≥+0.010）未过。仅 connector 正（k2/k4 +0.04/+0.09）。
- CLIP global / AnomalyCLIP image-level logit 未测（需复刻官方 prompt/eval，工程量大）；
  按 16 §6 留待用户确认后再定（建议先做 image-level logit 低成本探针）。

### S1-HGLC 完整探针（AnomalyCLIP 跨模态，用户 07:05 确认后执行，08:20 收尾）→ **部分过门/融合模块归档**
- Export：冻结 AnomalyCLIP（ViT-L/14@336 @518 + 9_12_4_multiscale_visa/epoch_15，通用 "object" learned prompt）
  → CLIP-global 图像嵌入 + 异常文本概率 p_abn（zero-shot、类别无关、无需 refs）。swap 方向 sanity 恰为 1−p ✓。
- 图像级（doc §3.3 item 2，pooled Image-AP 口径与 P5A 一致）：
  - **TEXT 0.8234 vs A1-max 0.7985 → Δ=+0.0249（过门 ≥+0.010）**；DINO CLS 0.6769（Δ=−0.1216，与 P5A 完全一致）、
    CLIP-global 0.7035（Δ=−0.095）。
  - 文本信号在 connector（+0.151）与 bracket_black/brown（+0.06~0.07）强；metal_plate（A1=1.0 饱和）与 bracket_white 为负。
- 像素级校准（doc §3.3 item 3-5）：z_A1 + β·ReLU(p_abn−0.5)·h(z_A1)，β∈{0.1,0.25,0.5}，h∈{z/(1+z), top-q}：
  - z/(1+z) 最优 β=0.5：pooled ΔPixel-AP **+0.0040 < +0.005**（worst cat +0.0002，无类退化）；
    打乱门控控制 −0.0005 → 门控信息真实但极小。top-q 负（−0.005，metal_plate −0.037）。
- **结论**：文本信号不能作为像素级融合模块（差门槛 0.001），S1-HGLC 融合路线按冻结门槛归档；
  但 **zero-shot 跨模态图像级文本证据是真实独立发现**（超越 A1 局部记忆图像分），可作图像级筛查/全局证据叙述。
  证据：`s1_hglc/S1_HGLC_DECISION.md`、`S1_HGLC_DIAG.json`、`S1_CALIB.json`。

## 3. 对“创新不足”关切的综合结论（建议口径，供论文讨论）
1. 本项目已把“双视觉分支像素级融合（KNN 记忆 × 子空间重建）+ 正常-only 可靠性保护”在冻结协议下
   完整实证到 Wave2：**固定均值融合有 +0.016 的小增益，但 doc 的保护门不可构造（可靠性无信号）**。
   这不是“没做”，而是“严格证伪并留下机制解释”——本身就是可写进论文的负结果/边界刻画。
2. 互补性来自**几何异质 × 尺度异质**，不是“分支数量/更大骨干”的平庸效应（P5-C/P5-D 对照）。
3. 若继续提创新，最可行的剩余轴 **AnomalyCLIP 图像级文本证据（S1 完整版）已于 08:20 完成实证**：
   zero-shot 文本概率图像级 pooled Image-AP 0.8234 > A1 0.7985（Δ=+0.0249），但其像素级门控增益
   +0.0040 < +0.005，故不作为像素融合模块；可作**图像级筛查/全局-局部一致性的跨模态证据**写入论文
   （详见上方 S1-HGLC 完整探针小节）。Wave2 的“小池 z 量化退化”方法学改进已由 Wave2c 复核为无效。

## 4. 明确不做
- 不宣传“两个 DINO 系分支 = 多模态”；不在 Full MPDD 前碰外部集；不训练监督 router；
- 不按类别名硬编码切换；不为 Wave2 失败后的 S2/S1 变体投入 GPU 之前先等用户确认。
