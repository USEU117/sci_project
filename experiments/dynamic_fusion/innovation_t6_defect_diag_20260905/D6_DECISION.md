# 方向 6 结案：A1 真实 defect-type 剩余缺口诊断 — 量化收口

日期：2026-09-05　立项：`docs/paper_writing_preparation_20260830/37_DIRECTION6_DEFECT_TYPE_DIAG_PLAN_CN_20260905.md`（doc37）
上游：doc36 D5_REAL_GATE FAIL → D5_REAL_DECISION 转方向 6（parts_mismatch 纯诊断，不再加机制）
脚本：`scripts/innovation_t6_defect_diag_20260905/run_d6_defect_type_diag.py`
数据：A1 冻结 concat 只读（compact 448 map + dino masks/sample_ids）；seed0 k2/k4；无拟合、无候选。
口径：与 doc36 C0 完全一致（Pixel-AP stride 8；defect-type 按 sample_id test 目录名切分；type 子组图集合内计算；type 宏=等类权）。

## 对账（必须复现 doc36）
| shot | mean A1 AP（实测/冻结） | parts_mismatch 宏 AP（实测/REAL_D5） | max|cat ΔAP| |
|---|---|---|---|---|
| k2 | 0.343697 / 0.343706 | 0.246402 / 0.246406 | 4.3e-05 |
| k4 | 0.388327 / 0.388328 | 0.278069 / 0.278069 | 1.8e-05 |

## defect-type 切分（Pixel-AP / Pixel-AUROC / image-AUROC；k2 → k4）
| defect type（出现类） | n | k2 | k4 |
|---|---|---|---|
| parts_mismatch（bracket_brown 34 + connector 14） | 48 | 0.246 / 0.950 / 0.671 | 0.278 / 0.954 / 0.722 |
| 　└ bracket_brown 内 | 34 | 0.0255 / 0.936 / 0.550 | 0.0251 / 0.939 / 0.544 |
| 　└ connector 内 | 14 | 0.4673 / 0.964 / 0.900* | 0.5310 / 0.970 / 0.900 |
| bend_and_parts_mismatch（bracket_brown） | 17 | 0.0921 / 0.882 / 0.613 | 0.0966 / 0.889 / 0.631 |
| scratches（bracket_black/white/metal_plate） | 86 | 0.383 / 0.979 / 0.776 | 0.463 / 0.989 / 0.899 |
| hole（bracket_black） | 12 | 0.0788 / 0.975 / 0.422 | 0.0605 / 0.936 / 0.659 |
| defective_painting（bracket_white） | 13 | 0.0813 / 0.996 / 0.792 | 0.0835 / 0.997 / 0.787 |
| major_rust（metal_plate） | 14 | 0.747 / 0.958 / 1.000 | 0.743 / 0.962 / 1.000 |
| total_rust（metal_plate） | 23 | 0.945 / 0.978 / 1.000 | 0.969 / 0.987 / 1.000 |
| anomalous（tubes） | 69 | 0.736 / 0.986 / 0.966 | 0.759 / 0.988 / 0.963 |

（*connector image-AUROC 为 k4 值；k2 0.900。）

## 诚实解读（量化结论）
1. **"parts_mismatch 子组弱"是 bracket_brown 拉低的，不是 parts_mismatch 本身弱**：connector 的 parts_mismatch Pixel-AP 达 0.467/0.531（不弱于多数表面缺陷），bracket_brown 的 parts_mismatch 仅 0.0255/0.0251、bend_and_parts_mismatch 0.092/0.097。子组宏 0.246/0.278 是"bracket_brown 34 图 + connector 14 图"等类权平均的产物；doc36 用"parts_mismatch 缺口 0.278 vs 0.388"表述需在此修正为**逐类拆分后的 bracket_brown 上下文缺陷缺口**。
2. **剩余缺口是"检测不出"而非仅"定位差"**：bracket_brown 上下文族在图像级（map max 聚合，对同 cat normal）只有 image-AUROC 0.55/0.61（k2/k4 各 type），接近难分；Pixel-AUROC 0.88–0.94 的"像素可分"是在全部为异常图的子集合内相对排序，掩盖了正常/异常不可分。这解释了方向 5 C1/C2（邻域描述子只放大距离/域差异）为何必然失败：**问题在正常支持流形上没有该结构的对应簇，任何"同图邻域增强"都不产生跨图的结构先验**。
3. **非上下文类的小缺口同源**：hole（bracket_black，12 图）image-AUROC 0.42/0.66、defective_painting（bracket_white，13 图）0.79/0.79——小面积/弱外观缺陷同样是"图像级可检性"瓶颈；而 rust/tubes/connector-pm 等大面积强外观缺陷已接近饱和（image-AUROC 0.90–1.00）。
4. **A1 边界在机制层面的统一表述**：冻结特征 + 全局 KNN 的正常域相似度范式，对"细结构装配/上下文偏移 + 稀疏小面积"缺陷的可检性有明确上限；该上限由特征与 support 分布决定，本方向（relation 描述子）与既有 z-score 融合家族均无法补。超过当前数据/范围才可能再启动机制（需结构先验或多视角 normal 流形），不在本方向内。

## 决策：`D6_DIAG_CLOSE`（纯诊断收口，无新机制）
- 产物：`D6_REPORT_s0_k2.json`、`D6_REPORT_s0_k4.json`（逐类×type 明细）、本结案记录。
- 对 doc36 表述修正一处：parts_mismatch 缺口定位为 **bracket_brown 上下文缺陷族**（n=51），而非 connector 的 parts_mismatch（后者 A1 表现良好）。
- 论文用途：defect-type 表作为 A1 像素级增益的诚实边界/机制局限证据；上下文/细结构装配缺陷（bracket_brown pm/bend）列为 documented limitation + future work（需授权新数据或结构先验机制）。
- 方向 6 关闭。创新路线在 MPDD s0 当前数据/目标约束内全部收口，A1 仍为唯一冻结方法。
