# S0 DG-SAFE 执行状态（innovation_v6_dgsafe，2026-09-02）

任务书：[16_POST_NEGATIVE_RESULTS_NEW_FUSION_ROUTES_AND_ACCEPTANCE_CN_20260902.md](../../../docs/paper_writing_preparation_20260830/16_POST_NEGATIVE_RESULTS_NEW_FUSION_ROUTES_AND_ACCEPTANCE_CN_20260902.md)（S0 为主路线）
保护：只允许 MPDD development；BTAD/MVTec/VisA 冻结不可见；A1 冻结证据与既有决策不改写；GT 只由 evaluator 加载。

## 进度

| 步骤（任务书 §6） | 内容 | 状态 |
|---|---|---|
| Step 1 | v6 五目录骨架 + wave0_protocol.json | ✅ |
| Step 2 | audit runner 加 `--export-maps`（默认关，旧行为不变） | ✅ 已实现、py_compile 通过 |
| Step 3 | identity/sample-id/几何/泄漏单测（`tests/innovation_v6_dgsafe`） | ✅ 7 passed |
| Step 3.5 | CPU 前提复现（无 GPU） | ✅ PREMISE_SUMMARY.json |
| Step 4 | MPDD s0×k{1,2,4} 六类 map 导出 | ⏸ 待 GPU（需先下载 giant 权重） |
| Step 5 | S0-Wave 1 互补上限诊断 → WAVE1_DECISION.md | ⏸ 依赖 Step 4 |

## CPU 前提复现结果（`premise/PREMISE_SUMMARY.json`，54 单元）

复算 A1（compact concat 448/stride-8）均值 Pixel-AP **0.3556**（冻结 0.356 ✓）；SUB 取冻结审计 per_config。

| 类别 | SUB−A1 mean ΔPixel-AP | 正单元 | 最差单元 | 对照 16 号表 |
|---|---:|---|---:|---|
| bracket_black | +0.1529 | 9/9 | +0.0169 | +0.15 ✓ |
| bracket_brown | +0.0314 | 9/9 | +0.0214 | +0.03 ✓ |
| bracket_white | −0.0013 | 3/9 | −0.0556 | ~0.00 ✓ |
| connector | **−0.1536** | 0/9 | −0.2539 | −0.16 ✓ |
| metal_plate | +0.0382 | 8/9 | −0.0126 | +0.04 ✓ |
| tubes | +0.0644 | 9/9 | +0.0389 | +0.06 ✓ |

overall mean ΔPixel-AP **+0.022**（16 号写 +0.0213），正单元 38/54 ✓。→ 前提成立：确有一个「强专家与 A1 逐类互补、
但可靠性不可事先识别」的真问题，值得进入像素级融合测试。

## 关键事实（用于 GPU 步骤）
- 代码/venv(`.venv-anomalyclip`)/manifest/数据齐全；唯一缺口是 **giant 权重 model.safetensors**（原 %TEMP% 目录已被清理）。
- 目标放置：`methods/SubspaceAD/checkpoints/dinov2-with-registers-giant/`（已有 config.json），
  runner 以 `--model-dir` 指向；运行须从项目根启动（MPDD manifest 相对路径）。
- 导出内容：每 (cat,seed,shot) 一个 npz：sample_ids + amap_raw (N,48,48) float16 + ref_ids + 参数/commit。

## 进度更新（2026-09-03 凌晨，GPU 权重已下载 4,546,030,112 B 校验通过）

| 步骤 | 结果 |
|---|---|
| Step 4 导出 s0×k{1,2,4} 六类（18 npz） | ✅ 01:46 完成；export_out_s0/per_config 18 行 |
| Wave 0 identity replay | ✅ passed；逐类 mean\|Δ\| 1e-6..1.3e-4 ≪ 5e-4（首次因 GT 用 cv2 而非官方 PIL NEAREST 重采样误判失败，修正后通过——map 复现本身逐位一致） |
| Wave 1 互补诊断 | ✅ **passed**（cond A：固定 mean 组合 pooled Δ **+0.0164** ≥ +0.005；cond B oracle headroom +0.0066 仅 2/6 类正，未达） |
| Wave 2 正常-only 可靠性 | ❌ **归档**（03:40）：ρ(r_sub,Δ)=0.3387<0.40；connector 未入后 25% → 见 WAVE2_ARCHIVE.md |
| 创新探针 P5-B/C/D/A-lite | 归档/无增益/弱增益/归档（见 INNOVATION_ROADMAP_20260903.md） |
| Wave 2c 分布级尾估计探针（CPU，05:50–06:35） | ❌ **复核归档**：重校准不能救回 Wave2 门。冻结 B_tail 的 k 纯度被逐位证实（每 k 六类同值 1.7047/2.3026/2.9444）；V1 pooled-CDF ρ=−0.07；V2 逐像素高斯表观 ρ=0.62/connector 入组均为 clip+平局伪影（两特征 ρ=−0.93、r_sub2' 仅 7 个离散值、q25 平局吞 9/18、误标 bracket_brown×2+tubes|2）→ 见 Wave2_reliability/dist_tail_probe/ |
| S1-HGLC 完整探针（AnomalyCLIP 跨模态，07:05–08:20） | 图像级 **TEXT 过门**（pooled Image-AP +0.0249 vs A1，zero-shot）；像素级校准 +0.0040 < +0.005 → **融合模块按冻结门槛归档**；CLIP-global −0.095、DINO CLS −0.1216（与 P5A 一致）→ 见 s1_hglc/S1_HGLC_DECISION.md |

## 结论（03:40 后，Wave2c 于 06:35 复核）
- S0-DG-SAFE 在 Wave2 被冻结公式实证拒绝：正常-only 稳定性无法识别何时信任 SUB。
  **A1 仍是主方法**；固定 mean 组合 +0.0164 属静态事实，但无保护、connector 仍退化，且按协议
  禁止在无保护情况下选用（doc Wave3 候选均需 r_sub 保护）。
- Wave2c：把"小池 z 量化"换成分布级尾估计（pooled-CDF / 逐像素高斯）均**不能**救回可靠性门，
  负结果是校准公式鲁棒的；唯一有真实风险方向的特征是 U_layer（|ρ|≈0.32），其 B/C 网格未落盘。
- 夜间创新探索：S2-GPMR 归档（熵无信号）；同主干 concat/DINO-only 无增益；同主干 vitb14 子空间
  弱增益(+0.0026)；DINO CLS 图像级轴归档(−0.068)。→ 融合价值=几何异质×尺度异质。

## 证据链位置
`Wave0_replay/ WAVE0_REPLAY.json` · `Wave1_complementarity/` · `Wave2_reliability/`
`reliability/ reliability_raw.json`（+18 pool npz，npz 被 gitignore）·
`p5b_gpmr_precheck/` · `p5c_intrasystem/` · `p5d_samebackbone_subspace/` · `p5a_global_diag/`
