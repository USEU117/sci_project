# A2 健康探针记录（可学习对角重标定，k2+k4）— 最终归档

日期：2026-09-05　脚本：`scripts/innovation_a2_learnable_matching_20260905/a2_health_probe.py`
数据：复用 v14 support-only 缓存（dino/clip k2/k4，32 网格；support 图像渲染+冻结提取器重编码；无 /test/）。
探针结构：每 (cat, shot∈{2,4}) × 留一族（3 族轮换）：在其余 2 族（K 图、leave-one-image-out memory）上用 AUC-logistic 对（L-B，对易样本不饱和）训练 80 步，评估留族。
打分 = A1 融合：`z=L2([0.5·L2(w_D⊙d), 0.5·L2(w_C⊙c)])`，`s=1−max_cos(z, memory)`；w∈[0,1.5]、恒等初始化、恒等收缩正则 λ_w=1e-3、lr=1e-3。

## 结果（每 shot 18 折；thin_scratch 族在 32 网格不可评 → 12 折可评）
| shot | 宏 ΔAP（trained−identity） | 达标折数（≥+0.005） | H2 normal 路径 |
|---|---|---|---|
| k2 | **+0.0040** | 3/12 | 通过（clean p95 rel +2.1%、nui +2.1%、abs +0.003） |
| k4 | **−0.0003** | 2/12 | 通过（clean p95 rel +2.5%、nui +2.7%、abs +0.003） |

- 权重确有移动（wD std≈0.04）且损失单调小幅下降 → 优化器工作，但**目标近乎平坦**。
- 决策：**HEALTH_PROBE_FAIL_ARCHIVE**（H1 未达 2/3 折阈值；两 shot 均失败）。

## 逐类 trained−identity 宏 AP（cutpaste / local_erasure）
| 类别 | k2 cut / k2 eras | k4 cut / k4 eras |
|---|---|---|
| bracket_black | +0.0019 / +0.0000 | +0.0140 / +0.0013 |
| bracket_brown | −0.0021 / −0.0026 | +0.0029 / −0.0013 |
| bracket_white | +0.0173 / **+0.0273** | +0.0077 / −0.0047 |
| connector | +0.0005 / −0.0008 | −0.0022 / −0.0014 |
| metal_plate | +0.0125 / −0.0060 | +0.0026 / −0.0103 |
| tubes | −0.0014 / +0.0019 | −0.0033 / −0.0094 |

## 解读（诚实归因）
1. 行归一化后的 fused cosine 空间里，逐通道对角重标定对排序的**可学空间极小**：非饱和排序目标 80 步在 k2/k4 均无系统改善（+0.004 / −0.0003 宏），达标折 3/12 与 2/12。类别级正值（bracket_white、个别 cutpaste 折）在 shot/族间不稳定，不构成可迁移机制。
2. 损失下降缓慢（~2%）说明该参数化下增益上界低，而非优化失败——与 v14 中闭式通道规则的零收益一致；也解释了为何 A1 fused KNN 对逐通道重标定几乎不敏感。
3. 按 doc28 §8.2 纪律，健康探针失败即归档该模块；阶段 II（低秩跨分支项）须阶段 I 通过后才可立项 → **本路线在本轮关闭**。

## 状态
- 产物：`HEALTH_PROBE{,_k2,_k4}.json`、`HEALTH_PROBE_GATES{,_k2,_k4}.json`、本记录。
- 下一步（需用户）：doc28 §8 选择 A（收口 A1 论文）；或提出全新非"同构匹配微调"的研究目标后再立项。可学习对角路线与 zero-training 手工规则两线均已由实验如实闭合。
