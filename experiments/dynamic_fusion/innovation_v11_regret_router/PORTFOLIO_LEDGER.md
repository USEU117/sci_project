# v11 Regret-Router Ledger (doc 21, 2026-09-03)

Authority: `docs/paper_writing_preparation_20260830/21_INNOVATION_BREAKTHROUGH_PORTFOLIO_AND_ACCEPTANCE_CN_20260903.md`.
Status vocabulary: `NOT_STARTED / RUNNING / PASS / FAIL / ARCHIVED`.
Data roles frozen per doc 21 s9.1: MPDD = exhausted development (R0/candidates only);
BTAD/MVTec consumed (diagnostic only); VisA source; Real-IAD intermediate (NOT downloaded);
MVTec AD 2 final untouched (NOT downloaded). No external set may be used for any tuning.

| 阶段 | 状态 | 关键数字 | 门 | 下一步 |
|---|---|---|---|---|
| RSR Oracle audit (E0 A1 / E1 text / E2 LLSE / E3 CSS, MPDD s0 k1) | **ARCHIVED** (R0 FAIL) | Oracle mean Δ +0.400 (6/6 ≥+0.01) | g3 FAIL: 被选缺陷像素 97.2% 归 LLSE（text 0.6%/CSS 2.3%）；g4 FAIL: 去 CSS headroom 反升 −0.039；g1/g2/g5 PASS | 池不平衡；补 SubspaceAD 级新机制专家或转 BC-MCR |
| RSR pseudo-regret router (Phase 2) | NOT_STARTED（被 Oracle 阻断） | — | doc 21 s4.7 | 仅在重新平衡专家池并重过 Oracle 后启动 |
| BC-MCR blind-center repair (structural gate) | **ARCHIVED** (R0 FAIL) | perm/dup FULL≈CTRL_POS≈CTRL_CTX 0.72 (≥A1+0.22)；missing FULL 0.405 < A1 0.582 <… CTRL_POS 0.520 | 6 门全 FAIL：盲性与配对对照成立，但训练无旧检索增益、support 无贡献、missing 机制性反向 | 真实异常门不启动；需新 missing 代理/新分数才可复活 |
| Topo-Head (image metric only) | NOT_STARTED | — | doc 21 s7.8 | 仅 image-metric 分支，需单独协议 |

## 结论（2026-09-03）
doc 21 两条首选主线在本资产内均已被实验关闭：
1. RSR 因专家池失衡（g3/g4）被 Oracle 门阻断；
2. BC-MCR 因训练式盲中心无法检测"结构抹除为均值"的 missing 代理（g1–g5 全失败）被归档，
   真实异常验收永不触发。
剩余可选项全部需要新前置：NR-MoE / Meta-RSR（source 元训练与老师协议变更）、
AARC（GPU 高分辨率 + small-defect oracle）、Object-Set（LOCO 新数据/新论文协议）、
Topo-Head（仅 image metric，可低成本独立跑）。
总体仍指向 doc 21 s12 最后一条与 Scenario E：停止算法搜索，以 A1 + 系统负结果收尾。
