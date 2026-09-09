# FastRef 原型精化方向：深度调研与验证记录（2026-09-09）

范围：方法机制精读、审稿意见、基准表现、在我方冻结 A1 上的忠实验证。

## 1. 方法事实（arXiv:2506.21398 / CVPR 2026 版；ICLR 2025 前身被拒）
- 定位：**测试期（per-query）原型精化**，叠加在 prototype/memory 类 FS-IAD（PatchCore / FastRecon / WinCLIP / AnomalyDINO）上，label-free、无训练。
- 目标（每张 query 图 t）：min_{W,T} dis(f_q^t, W·M_s) + λ·OT_ε(p, q)
  - M_s∈R^{n×c}：正常原型（support 特征，可用 coreset）；W∈R^{m×n} 组成式"composition refinement"；f_q∈R^{m×c} 为 query patch（m=h×w）。
  - p=对细精化原型 W·M_s 的均匀分布；q=对原正常原型 M_s 的均匀分布；OT_ε 为熵正则最优输运，代价 C_ij=dis((WM_s)_i, M_s(j,:))。λ 平衡"贴合 query 特征"与"不偏离正常分布（抑制 query 中的异常被学进原型）"。
- 求解（EM 交替）：
  - E 步（固定 W）：Sinkhorn 求 T（作者：实践中 <10 次内循环足够）。
  - M 步（固定 T）：闭式行更新。结合行边际 Σ_jT_ij=1/m，等价于对每个 query patch 更新
    v_i ← (f_i + λ·Σ_j T_ij·M_s(j)) / (1 + λ/m)，
    即"把 query patch 向它的 OT 配对的正常原型方向拉回"（异常抑制），迭代数次。
- 打分（§4.3 "through reconstruction"；ICLR 算法第 4 步）：s_j = min_{r∈细精化原型集} dis(f_j^q, r)；即"距离测试期适配后的正常流形"。
- 声称：1/2/4-shot 下对 MVTec/VisA/MPDD/RealIAD 普遍增益；表 4 给 MPDD 2-shot PatchCore(EfficientNet_b5)：I-AUROC 62.5→69.3(+6.8)、P-AUROC 87.6→92.9(+5.3)（MPDD 上以图像级增益为主）。

## 2. 审稿与社区意见（重要）
- ICLR 2025 前身被 **Reject**；AC 意见要点："OT 组件动机不足、缺乏支撑其抑制异常的分析；**各数据集收益不均匀、多处增益 marginal**"。原作者随后扩展至 CVPR 2026。
- 与 SubspaceAD 的定位对照：SubspaceAD 与 FastRef 皆属"非样例 KNN"打分，一个静态 PCA 子空间、一个测试期 OT 适配；两者在 **logical/misplaced（MPDD parts_mismatch 族）上都只字未提或承认不解决**——该类缺口仍是开放问题。

## 3. 我方验证口径与实现（probe_fastref.py）
- 冻结 A1 融合特征（0.5 DINO + 0.5 CLIP，32 网格 unit 行）；原型=支持图 patch 全量 bank；query 特征即测试图 patch。
- 忠实复刻 EM：L_out=3、Sinkhorn E=20、ε=0.1·mean(C₀)；λ∈{0.1,1,10}（min-to-set，族 FRF）+ row-residual λ=1（族 FRR）；门同各轮（宏 ΔAP≥+0.01 且 worst≥−0.03，双 shot）。
- 已跑 k2 全 6 类（parity 精确 0.343706）：FRF lead(λ=10) 宏 **−0.029**（worst −0.112）；FRR(λ=1) −0.029（worst −0.111）；小 λ 灾难（λ=0.1 → −0.32）。
- 类内诊断：**bracket_brown（D6 最难上下文类）单类 +0.011（0.033→0.044）**，而强类大幅受损 → "测试期适配救硬类、伤强类"与 SUB/CEN/LCN 同构的再现。λ↑ 单调改善（0.1→10：−0.32→−0.19→−0.029）提示更强抑制倾向正，但按纪律不在赛后扩 λ。

## 5. 结果与决策（补 k4）
- k2（全 6 类，parity 精确）：FRF lead(λ=10) 宏 **−0.0293** / worst −0.1116；FRR(λ=1) −0.0290 / −0.1107；λ=1 −0.19；λ=0.1 −0.32（小 λ 灾难）。
- k4（parity 精确 0.388328）：FRF lead(λ=10) 宏 **−0.0401** / worst −0.1961；FRR −0.0402 / −0.1957。
- 两 shot 均未过门（宏 ≥ +0.01 且 worst ≥ −0.03）→ **决策 `FASTREF_A1_FUSED_FAIL_ARCHIVE`**：FastRef 型测试期原型精化叠加在冻结 A1 融合特征上不可用；不立项、不扩 seed。
- 复现的现象（贯穿 WZ/SUB/CEN/FRF 的稳健规律）：**重构/流形/适配类打分均表现为"救硬类（bracket_brown 上下文族近随机类）+0.01 量级、毁强类 −0.1 以上"**；宏口径下任何此类机制都无法超 A1 样例 KNN。写入论文 limitation/方法学注记（"macro gains of residual/adaptation scorers are driven by near-random classes while destroying strong ones"）。
- 遗留判断：真缺口仍需跨图结构先验/对齐/训练；FastRef 无法仅凭单图测试期 OT 适配解决。若未来要复现其文献增益（MPDD 图像级 +6.8 类），需按其 AnomalyDINO/PatchCore 原生口径与公开实现核对，但与本项目 A1 像素级主张无直接增益关系。
- 产物：probe_fastref.py；fastref/RESULTS_s0_k{2,4}.json；本文档。
Sources: [FastRef arXiv:2506.21398](https://arxiv.org/abs/2506.21398) · [FastRef CVPR 2026 PDF](https://openaccess.thecvf.com/content/CVPR2026/papers/Li_FastRef_Fast_Prototype_Refinement_for_Few-shot_Industrial_Anomaly_Detection_CVPR_2026_paper.pdf) · [FastRef 机制综述(moonlight)](https://www.themoonlight.io/de/review/fastreffast-prototype-refinement-for-few-shot-industrial-anomaly-detection) · [ICLR'25 前身审稿页(含 Reject 意见)](https://openreview.net/forum?id=gTsLBDMZrL) · [FastRecon ICCV'23](https://openaccess.thecvf.com/content/ICCV2023/papers/Fang_FastRecon_Few-shot_Industrial_Anomaly_Detection_via_Fast_Feature_Reconstruction_ICCV_2023_paper.pdf)
