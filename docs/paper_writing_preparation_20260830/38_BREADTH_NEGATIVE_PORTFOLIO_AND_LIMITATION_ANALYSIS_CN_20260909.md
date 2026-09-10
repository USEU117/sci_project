# 38 机制轴负结果组合与 limitation/analysis 撰写材料（2026-09-09 夜）

用途：把 breadth 系列 R1–R12（32 族）+ Web1（2 族）+ FastRef（1 族）共 **35 个机制族**在真实门下的负结果，
以及 R8 缺陷尺寸分解诊断，整理为可直接并入稿件 §4.2.6（Remaining Limitations）与 §5（Future Work）的材料。
所有数字来自本仓库归档 JSON（见 §6 复现入口），未做任何门后调参。

## 1. 证据口径（与稿件一致）

- 冻结方法 A1：DINOv2-B/14 与 AnomalyCLIP-DPAM 描述子各 resize 至 32 网格、逐行 unit、0.5/0.5 concat、faiss top-1（1−cos）→ 448 map → 逐类 Pixel-AP/AUROC（stride 采样掩码）。
- 真实门（预注册）：seed0、k2/k4 两 shot、六类宏 **Pixel-AP Δ≥+0.01 且 worst ≥−0.03 且 ΔAUROC ≥−0.005**；control parity k2 0.343706 / k4 0.388328（全部精确复现）。
- 候选机制只允许使用**冻结特征与支持集**；**不接触任何 test 标签**；不做按类启用、不做门后调参。

## 2. 35 族负结果组合（按轴归类）

| 轴 | 族（代表配置） | lead 宏 ΔAP（k2/k4） | 结论 |
|---|---|---|---|
| 特征标定/选择 | CS 通道标准化；SCA 仅缩放；CSP 低方差通道选择 | −0.11~−0.026 | 通道方向（含均值方向）承载信息，标定/剪裁均破坏 |
| 跨图一致性 | AGR2、RANK 秩 copula、RCW 支持侧互惠、BRC 双向互惠 | ~0 ~ +0.002 | 中性或亚门；不跨 shot |
| 匹配深度/度规 | K5 top-5 均值、GAP d2−d1、NDW d2/几何均值、DST L1/Cheb | −0.099 ~ +0.005 | 仅 L1 亚门（帮强类、k4 worst 破门） |
| 匹配尺度 | MRS 块池 32→16/8；MG 32+16 并存 min | −0.023~−0.127 | 尺度轴关闭（两次独立证伪） |
| 位置先验 | PLC 位置锁定（无配准） | −0.269/−0.292 | 灾难；MPDD 无逐像素配准先验 |
| 图内自归一 | WZ median/MAD；LCN 4×4 块对比 | WZ +0.019（worst −0.078）；LCN 灾难 | 宏正但 worst 深负 / 灾难 |
| 形态/几何/区域后处理 | MH top-hat；GV 旋转投票；RGN 连通域均值池化 | −0.17 ~ −0.29 | 448 map 上无正面空间（域内排序即 AP 价值） |
| 记忆（行数/密度/权重） | RW 典型性、BS 子库袋装、MDN mixup 致密化、CRAM/CAPM 等（V10） | 近恒等（|Δ|≤0.005） | memory 行数与密度**不是**瓶颈 |
| 记忆（数据轴） | CB 跨类联合库；PA 伪标注扩库（逐图留一） | CB +0.004/0.000；**PA +0.0044/+0.0037** | 跨类只在极端少样本微正；**PA 唯一 worst≈0** |
| 重构/流形/适配类打分 | CEN 质心、SUB 子空间残差、FastRef OT 精化、LSE/LLSE（V10） | −0.22 ~ +0.005 | 普遍"救硬类伤强类"，宏口径不超 A1 |
| 部件/结构一致性 | COP k=8 部件布局；T5 关系描述子 32 网格 | −0.28 ~ −0.014 | 无跨旋转对齐时不可用（真缺口证据） |
| 分数级集成 | MAP min/max/mean；COMB 掺 CLIP 距离 α 扫描；DIS 分支分歧 | +0.0057 峰值 | 弱正不入主张 |
| 评估/工程（非算法门） | 896 高分辨率、双尺度融合、FP16/INT8 存储、coreset 压缩 | 存储 **PASS**（FP16 0.5×、INT8 0.25×）；其余不成门 | 工程可落地项：低精度存储 |

## 3. 三条可写入论文的分析结论

**(a) 宏口径下的"救硬类伤强类"规律与唯一例外。**
凡以"重标定/重构/流形/测试期适配"为核心的机制，其宏增益均来自对上下文近随机类（bracket_brown 及部分 bracket_black/white）的 +0.01 量级修复，
同时伴随强类（metal_plate/tubes/connector）−0.1 以上的损失（WZ/SUB/CEN/FastRef 四连印证）。
**唯一例外是 PA（置信正常伪标注扩库，逐图 leave-own-image-out）**：k2 +0.0044/worst −0.0001、k4 +0.0037/worst +0.0001，两 shot 宏方向一致且最差类无损，
但仍低于预注册 +0.01 宏门（缺口 ~0.006）。→ 论文表述：等权融合下，"修复难类"与"保护强类"在本特征/本协议下不可兼得；PA 是唯一温和无损的候选方向，留待更高基座或更多支持时预注册复核。

**(b) 逐类宏口径下的单调性剪枝（方法学贡献）。**
本评测为"逐类 Pixel-AP 再宏平均"，故**任何类内单调变换的 ΔAP 恒等于 0**：全局/逐类单调校准、CDF/分位数校准、conformal p 值类机制可证明为 null。
只有**逐图单调**（WZ/LCN 类）或**非单调**机制才可能有非零效应。这界定了"可探索机制空间"的边界，也解释了历史中大量 ±0.005 的中性族。

**(c) 缺陷尺寸/类型诊断：两个不同性质的缺口。**
R8（尺寸分桶，A1 控制图）与 D6（缺陷类型）合并给出：
- 强类（metal_plate/tubes/connector）异常几乎全为大连通域且像素级饱和（AUROC 0.97–0.995）；小缺陷在各桶的像素 AUROC 仍高达 0.92–0.997，其低 AP 主要是**正负像素极端不平衡的指标稀释**，而非排序错误。
- **bracket_black 的 k2 困境是采样现象**：k2 图像级 AUROC 0.461（低于随机）vs k4 0.797，全局 Pixel-AP 提升 12.6×；两图参考无法覆盖该类正常纹理变异性。
- **bracket_brown 是唯一跨尺寸一致的表示缺口**：即便 ≥1000px 的大连通域，AUROC 仅 0.886–0.889，两 shot 图像级 AUROC ≈0.57，与 parts_mismatch/错位缺陷（异常区视觉正常）同源。
→ 结论：剩余误差由**支持集采样**与**上下文/错位类表示缺口**两项主导；两者都不能由单图全局距离的机制修复。

## 4. 可直接使用的英文段落草稿

**(4.1) 增强版 §4.2.6 Limitations（替换/补充现第二段）**

> We additionally bounded the mechanism search space rather than reporting only positive configurations. Thirty-five probe families spanning feature calibration and channel selection, cross-image consistency, matching depth, metric and scale, positional priors without registration, within-image normalization, morphological/geometric/region-level map post-processing, memory cardinality and density, cross-category and pseudo-labeled memory growth, reconstruction-style and test-time-adapted scoring, and component-consistency terms were pre-registered and evaluated under the same gate (per-shot macro Pixel-AP gain ≥ +0.01, worst-category change ≥ −0.03, and non-degrading AUROC, at both 2- and 4-shot). None passed. Two regularities emerged. First, mechanisms that improve the context-dominated categories (bracket_brown and part of bracket_black) consistently degrade the strongly-saturated categories (metal_plate, tubes, connector) by an order of magnitude more than the gain, so equal-weight fusion cannot trade one for the other at the dataset-macro level; the only exception, pseudo-labeled memory growth with leave-one-image-out exclusion, improves the macro slightly at both shot levels without any worst-category loss but remains below the +0.01 bar. Second, because the reported metric averages per-category Pixel-AP, any class-wise monotone recalibration is provably rank-preserving and therefore exactly null; only per-image monotone or non-monotone transformations can change the ranking, which bounds what remains to be explored.

**(4.2) 可选新增 §4.2.7 Bounded search and negative controls（若审稿要求更系统的负结果）**

> A size- and type-resolved decomposition of the control maps shows where the residual error lies. For metal_plate, tubes and connector, annotated defects occupy almost exclusively large connected components and detection is saturated (per-bucket AUROC 0.97–0.995); the low Pixel-AP of the small-defect buckets elsewhere reflects extreme positive-to-negative pixel imbalance rather than ranking failure (per-bucket AUROC 0.92–0.997). The two genuinely difficult populations differ in nature: bracket_black is shot-sensitive (image-level AUROC 0.461 at 2-shot versus 0.797 at 4-shot, with a 12.6× increase in Pixel-AP), i.e. a support-sampling effect, whereas bracket_brown remains rank-deficient at every defect size and both shot levels (large-component AUROC ≈0.89, image-level AUROC ≈0.57), i.e. a representation gap for context-level mismatch defects. Neither is addressable by the single-image global-distance mechanisms surveyed above.

**(4.3) Future Work 候选（明确标注需要授权/训练）**

> Closing the remaining gap plausibly requires one of: (i) cross-image structural priors or alignment for context/mismatch defects; (ii) re-encoding the support set with real geometric augmentation and multi-layer feature integration; (iii) training-based or test-time prototype adaptation under an explicitly audited, leakage-free protocol; or (iv) validation on additional external datasets. These routes require GPU re-encoding, training authorization, or new data provisioning and are therefore outside the present frozen-CPU evidence boundary.

## 5. 与现有稿件口径的两点核对提示

1. 稿件 §2/§4 将第二分支表述为 "DPAM-configured CLIP visual descriptors"；冻结缓存中该分支为 `anomalyclip_text`（AnomalyCLIP DPAM 文本提示条件的 patch 描述子）。**建议统一术语**后再引用本文件的"text branch"说法，避免审稿歧义。
2. §4.2.6 现有"stride sampling 可能漏小缺陷"与本文档 R8 的"小缺陷 AP 稀释"是**两件事**（前者是评测栅格效应，后者是不平衡稀释）；建议分别表述，勿合并。

## 6. 复现入口
- 总账：`experiments/dynamic_fusion/innovation_breadth_20260908/AXIS_LEDGER_AND_CLOSURE_CN.md`
- 各轮：`ROUND5..ROUND12_DECISION_CN.md`（含预注册 PLAN）、`DECISION_CN.md`（R1）、`ROUNDS2_3_DECISION_CN.md`、`R4_WEB1_DECISION_CN.md`、`fastref/FASTREF_RESEARCH_CN.md`
- 脚本：`scripts/innovation_breadth_20260908/{probe_breadth.py, probe_breadth2..11.py, probe_web1.py, probe_fastref.py, analyze_evaldecomp.py}`
- 结果 JSON：`experiments/dynamic_fusion/innovation_breadth_20260908/{round2..round12*, web1, fastref, round8_evaldecomp}/RESULTS_*`
- 尺寸诊断：`ROUND8_EVALDECOMP_CN.md` + `round8_evaldecomp/EVALDECOMP_s0_k{2,4}.json`
