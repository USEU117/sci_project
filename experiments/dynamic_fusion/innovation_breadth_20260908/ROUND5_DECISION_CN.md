# 广度探针 R5 预注册与结案（2026-09-09）— MH/GV 双族灾难性失败

轴：**后处理 / 几何捷径**（区别于 R1–R4 的打分/特征变换轴、Web1 的 PCA/部件轴、FastRef 的 OT 轴）。
预注册门与全 breadth 相同（lead 宏 ΔAP ≥ +0.01 且 worst ≥ −0.03 且 ΔAUROC ≥ −0.005，k2/k4 两 shot 均须）；
control parity：k2 0.343706 / k4 0.388328，本轮回放**精确复现**。

## 族与逐族 lead（六类宏；Δ 相对 C0=冻结 A1 concat top-1）

| 族 | lead | k2 宏 ΔAP / k2 worst | k4 宏 ΔAP / k4 worst | k2 ΔAUROC / k4 ΔAUROC | 判定 |
|---|---|---|---|---|---|
| GV 特征网格旋转投票（伪等变探针，无内容重编码） | GV_mean | −0.1703 / −0.4644 | −0.2097 / −0.4795 | −0.0828 / −0.0729 | FAIL（灾难） |
| MH map 形态学 top-hat（盘半径 r=2/4，小亮凸起增强） | MH_r4 | −0.2937 / −0.7091 | −0.3298 / −0.7265 | −0.3440 / −0.3435 | FAIL（灾难） |

成员明细：GV_min k2 −0.2104 / k4 −0.2524（worst 至 −0.617）；GV_mean 同表；
MH_r2 k2 −0.3055 / k4 −0.3448（worst −0.732）；MH_r4 同表。
**两 shot 全成员、全族宏为深负**，无任何近似过门迹象。

## 解读

1. **GV（旋转投票）**：把 query 的 32×32 特征网格旋转 0/90/180/270 后对每个方向取 bank top-1 距离的 min/mean，是最廉价的"几何不变性"捷径。深度负收益说明
   **MPDD 缺陷定位对坐标完全敏感**：旋转网格即破坏 patch 与像素的对应关系（上游 dists2map 假设坐标对齐），不是可用的伪等变增强。真实旋转不变需对图像内容重编码（ViT 在旋转图上推理，需 GPU/授权，未越线）。
2. **MH（top-hat）**：只保留"比结构元小"的亮凸起，本质是**尺寸带通**。真实 MPDD 缺陷多为大而弥散/细长的金属划痕、油渍、缺损（超出 r=2/4 盘），top-hat 把异常主体抹掉 → 全局灾难。结合此前 CLIP/形状先验结论：**无监督单图后处理无法匹配真实缺陷的尺度多样性**。
3. 语义：map 空间的**非单调几何/形态学算子**即使保留所有正响应也无法提升宏指标；能改变排名的算子对真实缺陷全是破坏性的。

## 结论

- R5 双族在真实 MPDD s0 k2/k4 全门闭合为负；A1 仍为唯一冻结方法。
- 累计广度轮次 R1–R4 + Web1 + FastRef + R5 = **23 个机制族在真实门闭合为负**。
- "几何/形态学捷径"轴永久关闭；Rot-AD / 多尺度真实增强（VisionAD 式 support+query augmentation、CIF 式 hypergraph 消息传递）均需真实 GPU 重编码，列为需授权候选。

## 文件
- 脚本：`scripts/innovation_breadth_20260908/probe_breadth5.py`
- 结果：`experiments/dynamic_fusion/innovation_breadth_20260908/round5/RESULTS_s0_k2.json` / `RESULTS_s0_k4.json`
