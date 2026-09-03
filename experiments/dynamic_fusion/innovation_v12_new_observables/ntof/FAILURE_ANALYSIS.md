# V12 NTOF — 失败分析（2026-09-03）

> NTOF（doc 22 §3，P1 第一新主线）normal-only R0 机制门失败。本文分析失败层级与为何不追修。

## 1. 假说层级：光照漂移在深层特征中不是"单 support 图上的低秩线性切空间"

- 干预在图像域是逐像素光度变换（曝光/γ/白平衡/线性梯度/镜面斑），但 DINO/CLIP 深层特征对它们的响应是非线性、非均匀的：同一族不同强度沿不同方向移动特征（尤其 CLIP 的 L2 归一化球面），15 个局部有限差（1 张 support 图）张成的 r≤4 子空间只捕获了可解释漂移的一小部分。
- 结果：held-out（更强/未见强度）illumination 图经 NTOF 投影后 FP 仅降 ~7%（g1 需 ≥25%）；部分类（bracket_brown、tubes）even 升高（effect 为负），说明该类别上投影方向本身带偏。
- 依 doc 22 §3.6 的纪律，若机制门不过，不得用真实 bad 图调干预范围或 rank——直接归档。

## 2. 对照层级：投影的"降 FP"主要是范数收缩，而非方向正确性

- true/wrong-category/random/shuffled-pairing 四种切空间的 FP-reduction effect 都在 0.03–0.26 量级且相互差 <0.05（g3 需 ≥0.05）。bracket_black 上 true effect 0.259 与 wrong-category 0.258 几乎相同。
- 解释：把查询偏差 d 投影到任意 3 维子空间外都会缩短 ||d||，轻微降低超阈像素数；真实 nuisance 方向相比随机方向没有带来额外 ≥5% 的 FP 削减 → **正交残差机制未被验证**。任何"减去子空间"类方法的增益必须解释这一点（对照模板：random/wrong/shuf 均须显著更差）。

## 3. 融合层级：双编码器均分融合再次无增益（跨 CECW/MTCOA/NTOF 一致负）

- dino-only residual effect 0.094 高于 fused ntof 0.060（g5）；clip 分支在 brown/tubes/connector 大负（−0.3~−4.5）。
- 累计证据（MTCOA g2 唯一通过、CECW coupling 惰性、NTOF fusion 无益）表明：**在冻结 DINOv2-B14/AnomalyCLIP、K≤4、无训练协议下，双视觉编码器固定融合不产生稳定的宏观定位增量**；跨编码器类互补仅在小缺陷分量上出现过一次（MTCOA small-stratum），不足以支撑任何像素级融合主张。

## 4. 有效/复用的资产
- g2 保留率 PASS（ratio≈1.9–2.5）：说明该子空间投影不会消除强合成缺陷响应——后续若换机制（如 per-patch/多图局部切空间、非线性校正或更强干预族）仍可复用本干预导出管线（5 族×3 + held-out）作为基线生成器。
- DINO-only/CLIP-only/concat 三对照与本 A1 距离开销相同；可并入 PRS 的必做对照池。

## 5. 归档与后续纪律
- 归档 NTOF（doc 22 §10：NTOF normal-only R0 不过 → PRS）。不追修（不可在 bad 上选 rank/强度；不叠加 TTA/多 ref 来"救"）。
- 下一路线：PRS（doc 22 §5，扰动响应谱）：以 5 族干预为强度阶梯，检验"特征对扰动强度的响应谱偏离正常包络"是否构成独立异常观测量；门 doc 22 §5.3。仍为 normal-only-first 机制门。
