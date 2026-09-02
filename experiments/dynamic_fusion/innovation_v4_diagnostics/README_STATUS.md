# A4 Diagnostic Program — 信息价值诊断矩阵与路线决策（2026-09-02）

任务书：`docs/paper_writing_preparation_20260830/14_BROAD_ALGORITHM_INNOVATION_PORTFOLIO_CN_20260902.md`（§10 诊断 / §12 下一位指令）
执行范围（用户确认）：**严格按文档「诊断先行」**——只做 D1/D2/D3 信息价值诊断，全部在 MPDD development（seed 0）上；
不跑四数据集、不直接实现 RG-MCR/SF-NM/RG-OT 正式模型、不使用冻结验证集（BTAD/MVTec/VisA 保持不可见）。

## 1. 保护声明

- A1 冻结证据（`submission_repro_20260827/`、`experiments/dynamic_fusion/freeze/`）不修改；
- 既有未提交工作（`docs/` 任务书 12/13/14、RCEC 复核记录、A2 D/E 纠错约定）不触碰、不清空；
- 只允许 MPDD development（`assert_development_only` guard）；labels/masks 只由 evaluator 在分数写完后加载；
- 无 `git reset` / `checkout --` 操作。

## 2. 执行波次进度

| 波次 | 内容 | 状态 |
|---|---|---|
| Wave 0 | 共享审计 + 五目录 + 协议 | ✅ 完成 |
| Wave 1 | common 模块 + 11 项单测 | ✅ `pytest tests/innovation_v4_diagnostics` 11 passed |
| Wave 2 | D1 缺陷尺度与频率诊断（6 类 × k{1,2,4}，18 格） | ✅ 完成，全 headroom=0 |
| Wave 3 | D2 结构上下文诊断（6 类 × 3 扰动型） | ✅ 完成，perm/dup 部分正、missing 负、node-OT ≈ 随机 |
| Wave 4 | D3 双分支监督价值诊断（6 类） | ✅ 完成，mean headroom +0.0668，3/6 类 ≥ +0.02 |
| Wave 5 | 诊断矩阵 + ≤2 路线建议 | ✅ 本文档（不直接立项） |

代码：`src/industrial_ad/innovation_v4_diagnostics/`（common/spectral/diagnostics）、`scripts/innovation_v4_diagnostics/`（run_d1/run_d2/run_d3）。
逐格报告与汇总：`D1_scale_frequency/D1_SUMMARY.json`、`D2_structure_context/D2_SUMMARY.json`、`D3_supervision_value/D3_SUMMARY.json`。

## 3. D1 — 缺陷尺度与频率诊断（SF-NM / DC-SZoom 门槛）

方法：GT 面积三层 small(<0.5%)/mid(0.5–5%)/large；frozen A1 分层 Pixel-AP；
无参数两尺度 stationary-wavelet 谱描述符（24 维）memory 1-NN；
oracle 实例级最优互补 headroom + Spearman rank corr。
门槛（§10 D1）：**small 层 headroom ≥ +0.03 Pixel-AP 且 corr < 0.90 → 优先 SF-NM/DC-SZoom**。

small 层逐格（非空格）：

| 类别 | k1 headroom/corr | k2 headroom/corr | k4 headroom/corr | 备注 |
|---|---|---|---|---|
| bracket_black | 0.0 / −0.529 | 0.0 / −0.500 | 0.0 / −0.495 | a1_ap 0.009→0.157；freq_ap≈0.0005 |
| bracket_brown | 0.0 / −0.329 | 0.0 / −0.288 | 0.0 / −0.289 | a1_ap 0.015→0.021；freq_ap≈0.0006 |
| bracket_white | 0.0 / −0.207 | 0.0 / −0.210 | 0.0 / −0.254 | a1_ap 0.12–0.15 |
| tubes | 0.0 / −0.078 | 0.0 / −0.059 | 0.0 / −0.111 | 仅 4 张 small |
| connector / metal_plate | — | — | — | small 层无图像（缺陷不 <0.5% 面积） |

mid/large 层 headroom 亦全为 0（例：metal_plate large freq_ap 0.07 vs a1 0.87；tubes mid a1 0.74–0.76）。

**判定：未通过。** headroom 处处 = 0，oracle 从未选择谱图；rank corr 全负（缺陷区纹理“平坦化”→
谱特征更接近正常参考，AUROC 约 0.2 低于随机）。预注册无参 wavelet 评分对 MPDD 不构成对 A1 的互补。
→ SF-NM 与 DC-SZoom 均**不获优先**。

## 4. D2 — 结构上下文诊断（RG-MCR / RG-OT 门槛）

方法：只用正常 reference/正常 test feature 合成 patch permutation / missing-block / duplicate-block；
比较 A1 全局 KNN、context repair residual（5×5 去 3×3 ring 在 normal ring-bank 上 top-1 预测中心）、node-only OT（anchor 64）。
门槛（§10 D2）：**context 对三类扰动 AUROC 均 ≥ 0.80 且比 A1 高 ≥ 0.10 → 优先 RG-MCR**；RG-OT 仅在 context 失败但 node-OT 明显有效时 smoke。

6 类均值 patch-AUROC：

| 扰动型 | A1 | context-residual | Δ(context−A1) | node-OT(图像) |
|---|---|---|---|---|
| permutation | 0.492 | **0.708** | +0.216 | 0.502（随机） |
| missing | 0.569 | 0.508 | **−0.061** | 0.557 |
| duplicate | 0.517 | **0.723** | +0.206 | 0.509（随机） |

**判定：未通过。** context 在 permutation/duplicate 上比 A1 强 +0.21（结构“搬移/复制”可被 ring 上下文定位），
但三类均未达 0.80，且 missing（删除）类 context 反低于 A1：记忆检索式 in-painting 会用相似背景“填平”缺块、
残差≈0，删除型结构缺失需要生成式修复而非记忆预测。node-OT 图像级 AUROC ≈ 0.50–0.56（随机）→ RG-OT 不满足 smoke 条件。
→ RG-MCR 与 RG-OT 均**不获优先**（RG-MCR 保留机制洞察：只对移植型结构有信号，删除型是硬伤）。

## 5. D3 — 双分支监督价值诊断（CASF 门槛）

方法：按 A3 CASF 规定在正常 reference 上生成 branch-asymmetric 伪异常（dino/clip/both 1:1:1，transplant/tangent-noise）；
asymmetric 家族 vs symmetric（both-only）控制分别训练同一 tiny logistic head（CASF 4 通道输入统计），
在 held-out asymmetric 上评估。门槛（§10 D3）：**asym−sym 机制控制差值 < 0.02 pseudo Dice → CASF 降级**。

逐类（Dice：asym 训练 − sym 训练，同 held-out 评估）：

| 类别 | asym dice | sym dice | headroom | 判定 |
|---|---|---|---|---|
| bracket_black | 0.056 | 0.095 | −0.040 | 不支持 |
| bracket_brown | **0.675** | 0.088 | **+0.587** | 强支持 |
| bracket_white | 0.069 | 0.010 | +0.059 | 支持 |
| connector | 0.070 | 0.116 | −0.046 | 不支持 |
| metal_plate | 0.041 | 0.021 | +0.020 | 支持（边缘） |
| tubes | 0.129 | 0.309 | −0.180 | 不支持 |

汇总：mean headroom **+0.0668 ≥ 0.02**；3/6 类 ≥ +0.02（bracket_brown 贡献主导，符号不稳定）。

**判定：通过（类条件）。** 按文档规则 mean ≥ 0.02 → CASF **保留**，但信号强依赖 bracket_brown 族，
进入正式任务书时须按类条件/分层评估设计，不能假设 6 类同质。

## 6. 诊断矩阵与 ≤2 路线建议

| 路线（优先级） | 门槛诊断 | 结果 | 决策 |
|---|---|---|---|
| RG-MCR（1） | D2 | 未通过（missing 型失败，三类未达 0.80） | 归档（暂不立项） |
| SF-NM（2） | D1 | 未通过（headroom 全 0） | 归档（暂不立项） |
| RG-OT（3） | D2 | 未通过（node-OT ≈ 随机，无 smoke） | 归档（暂不立项） |
| CASF（4） | D3 | **通过（mean +0.0668；类条件）** | **入选** |
| DC-SZoom（5） | D1（同 SF-NM） | 未通过 | 归档（暂不立项） |

**推荐：1 条路线进入下一份正式执行任务书 = CASF（类条件设计）。**
第二名额空缺：本版图其余诊断门均未通过；若希望第二方向，只能以「RG-MCR 生成式修复变体」这类
超出预注册范围的机制改动进入，需用户显式决策（§1 禁止换名复活，改机制不属于换名，但属于扩权）。
诊断后另写只含最终路线的严格任务书；本程序不直接实现 CASF 正式模型，未跑任何验证集。

## 7. 执行期修复记录（诚实上报）

1. D1 `spearmanr().statistic` → `.correlation`（scipy 版本差异）。
2. D1 空 tier 汇总 KeyError → `tier_pooled_map_scores` 空返回统一补 `n_pos_px` 等字段。
3. D2 Sinkhorn IBP 广播维度错误（p/q 形状 64 vs 1024）→ 修正为源/目标各自约束 `K@v` / `K.T@u`。
4. D2 missing 语义从“局部移植”改为“块内均值抹平”（真实结构缺失），否则不可见。
5. 单测 2 项断言修正（1 像素 mask 保证 small 层、guard 测试语义），最终 11 passed。

## 8. 环境与运行

- `.venv-patchcore`（CPU：pywt/faiss/scipy/sklearn/numpy）；未占用 GPU。
- 复现：`python scripts/innovation_v4_diagnostics/run_d1.py`（约 27 min，6 类 × 3 shots），
  `run_d2.py` / `run_d3.py`（各约 1–2 min，6 类）。
- 下一步（待用户确认）：写 `15_...` 正式任务书（CASF 类条件严格候选/门槛），再执行。
