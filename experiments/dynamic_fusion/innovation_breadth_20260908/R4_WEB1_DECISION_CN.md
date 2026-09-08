# R4 + Web1 结案（2026-09-09 自主段）：文献启发的 7 机制族全负 + 文献映射

预注册门同前（lead 宏 ΔAP ≥ +0.01 且 worst ≥ −0.03，k2 与 k4 均须）；parity 均精确（0.343706/0.388328）。

## R4（probe_breadth4.py：SCA / RANK / IMGP / CEN / DIS + COMB 细扫）
| 族 | lead | k2 宏 ΔAP | k4 宏 ΔAP | k2 worst | k4 worst | 判定 |
|---|---:|---:|---:|---:|---:|---|
| SCA 逐通道仅缩放（不中心化） | SCA | −0.0225 | −0.0345 | −0.099 | −0.097 | FAIL |
| RANK 分支秩 copula | RANK | −0.0073 | −0.0231 | −0.114 | −0.128 | FAIL |
| IMGP 图级原型门控 | IMGP_G25 | −0.0012 | −0.0025 | −0.007 | −0.008 | FAIL |
| CEN 参考质心距离 | CEN | −0.2195 | −0.2665 | −0.699 | −0.724 | FAIL（原型≪样例） |
| DIS 跨分支分歧 | DIS_MIX | −0.0244 | −0.0235 | −0.090 | −0.059 | FAIL |
| COMB α∈{0.05..0.30} 细扫 | a≈0.25–0.30 | +0.0057~+0.0062 | +0.0042~+0.0050 | −0.003~−0.004 | −0.008~−0.012 | FAIL（峰值仍 < 门） |

## Web1（probe_web1.py：SUB 子空间残差 / COP 部件一致性，文献驱动）
| 族 | lead | k2 宏 ΔAP | k4 宏 ΔAP | k2 worst | k4 worst | 判定 |
|---|---:|---:|---:|---:|---:|---|
| SUB concat 子空间重构残差（r=256） | SUB_c_r256 | +0.0053 | +0.0028 | −0.039 | −0.051 | FAIL |
| SUB DINO-only 残差（r=64，SubspaceAD 口径） | SUB_d_r64 | −0.0236 | −0.0379 | −0.079 | −0.119 | FAIL |
| COP k=8 部件聚类布局一致性 | COP_mix | −0.2814 | −0.3277 | −0.714 | −0.739 | FAIL（无跨旋转对齐，粗模板不稳定） |

## 文献映射（新一轮检索，web_citation 见文末）
| 来源方法（会议/年份） | 核心机制 | 我们的判定 |
|---|---|---|
| SubspaceAD（CVPR 2026, arXiv:2602.23013） | 冻结 DINOv2 + PCA 正常子空间**重构残差**打分，无记忆库 | 已验：concat 高秩残差仅 +0.005（k2）/k4 衰减；DINO-only 负。且作者在补充材料自认"不解决 logical/部件错位"→ 与 D6 一致；**不构成 A1 升级** |
| PSAD（AAAI 2024）／UniVAD（arXiv 2412.03342） | 部件/组件分割 + 组成-排列逻辑一致性 | 方向正确（对 MPDD parts_mismatch），但需**跨旋转/跨样本对齐**；我们的 COP 无对齐、k-means 粗粒度 → 失败。忠实实现需更多工作 → 授权后深研候选 |
| FastRef / FastRecon+（CVPR 2026） | 推理期用 query 统计 + OT 迭代**精化原型**（特征迁移 + 异常抑制），可加到 PatchCore/AnomalyDINO/WinCLIP | **本项目从未试过的真新机制族**（我们从未按 query 自适应 memory/原型）。实现需 OT 嵌套优化与防泄漏审计，不宜无人值守半实现 → 授权/对照其开源后深研候选 #1 |
| CAReg（TNNLS 2024）／RegAD | 类别无关注册学习（跨类训练） | 需跨类训练集/GPU → 需授权 |
| GraphCore VIIF（ICLR 2023，MPDD 大增益） | 旋转等距不变特征（图像旋转变化处理） | MPDD 视角变化确为因素；我们的特征无显式旋转不变处理 → 旋转增强 memory 探针列为候选（需真实重编码） |
| 其它（GATE-AD、PACKD、BayPrAnoMeta 等） | GAT 图建模 / 蒸馏 / 贝叶斯 meta | 需训练/GPU → 需授权，暂不列为无训练候选 |

## 决策
- 本段（R4 + Web1）7 机制族全负；累计广度轮次 R1–R4 + Web1 = **19 机制族在真实门闭合为负**，A1 仍唯一冻结方法。
- 检索结论支持的**三个真新方向**（需授权/精确实现，非本段能可靠完成）：
  1. **FastRef 型推理期原型精化**（label-free、无训练，需 OT 求解器与严谨防泄漏实现）；
  2. **部件-组成一致性（PSAD/UniVAD 路线）**，前提是跨旋转对齐（先做旋转鲁棒编码）；
  3. **MPDD 旋转/姿态不变 memory**（GraphCore 启发），需真实重编码 4 方向支持图。
- 论文口径提醒（再次确认）：任何"宏增益"（含 SUB/COMB 的 +0.005 类弱正）都须同报 worst 类与逐类方向；0.5% 量级提升不入 A1 主主张。

## 产物
`probe_breadth4.py / probe_web1.py`；`round4/、web1/` 各 `RESULTS_s0_k{2,4}.json`；本结案与 `WEB_LITERATURE_MAP_CN.md`。
Sources: [SubspaceAD (arXiv:2602.23013)](https://arxiv.org/html/2602.23013v3) · [PSAD (AAAI2024)](https://arxiv.org/pdf/2312.13783.pdf) · [UniVAD (arXiv:2412.03342)](https://arxiv.org/html/2412.03342) · [FastRef (CVPR2026)](https://openaccess.thecvf.com/content/CVPR2026/papers/Li_FastRef_Fast_Prototype_Refinement_for_Few-shot_Industrial_Anomaly_Detection_CVPR_2026_paper.pdf) · [CAReg (TNNLS2024)](https://arxiv.org/pdf/2406.08810v2) · [GraphCore (ICLR2023)](https://arxiv.org/pdf/2301.12082v1.pdf)
