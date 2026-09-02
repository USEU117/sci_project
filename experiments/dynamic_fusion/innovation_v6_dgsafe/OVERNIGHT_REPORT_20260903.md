# 夜间自主运行报告（2026-09-02 23:42 – 2026-09-03 04:30 主体完成）

## 一、做了什么（时间线）

| 时间 | 事件 | 结果 |
|---|---|---|
| 00:34 | 下载 DINOv2-with-registers-giant 权重（4,546,030,112 B） | ✅ 校验通过 |
| 00:35–01:46 | P1：MPDD s0×k{1,2,4} 六类 SubspaceAD 导出（--export-maps，18 npz） | ✅ |
| 01:48 | Wave0 identity replay | ✅ 逐类 mean\|Δ\| ≤1.3e-4（≤5e-4 容差）|
| 01:53 | Wave1 互补上限诊断 | ✅ cond A：固定 mean 组合 pooled Δ=**+0.0164** |
| 02:05–03:32 | Wave2a 可靠性构建（官方 A 子空间重拟合 + B/C 层组探针 + 9 版本 light-aug 池） | ✅ 18 配置 |
| 03:35 | Wave2b GT 诊断 | ❌ **ρ=0.3387<0.40；connector 未入后 25%** |
| 02:10 / 02:30 / 03:55 / 04:15 | P5-B(S2熵) / P5-C(同主干分支) / P5-D(同主干子空间) / P5-A-lite(DINO CLS) | 归档/无增益/弱增益/归档 |

## 二、关键科学结论
1. **S0-DG-SAFE 归档（Wave2 失败）**：正常-only 稳定性可靠性无法解释真实风险。
   机制：任务书 B_tail 式逐像素小池校准（k≤4）严重量化退化（B_tail 实测仅随 k 变化）；U_layer
   在 k1 池同样退化。→ doc 16 的“可靠性保护融合”在冻结公式下不可构造，禁止训练 router。
   **A1 仍是主方法**；固定 mean 组合 +0.0164 是静态事实，但无保护且 connector 仍退化（-0.064@448）。
2. **对“两视觉分支创新不足”的证据化回答**（三个对照探针，均 448/冻结 GT）：
   - concat(A1)+DINO-only 子集：mean combo Δ=−0.0103 → 分支数量不是关键；
   - concat(A1)+vitb14 同骨干子空间：Δ=+0.0026 → 同骨干换几何不够；
   - concat(A1)+giant 异构子空间：Δ=+0.0164 → 需要**几何异质 × 尺度异质**。
   → 融合增益不是“两个视觉分支”的平庸效果；但 doc 的可靠性保护门不成立，故 S0 不能作为主创新。
3. **S2-GPMR 责任熵无信号（|ρ|≈0.02）归档；DINO CLS 图像级轴低于 A1（ΔImage-AP=−0.068）归档。**
   尚未测的剩余候选：AnomalyCLIP 图像级文本 logit（S1 完整版，跨模态，需工程投入与你的确认）。

## 三、证据与代码位置
- 波次证据：`experiments/dynamic_fusion/innovation_v6_dgsafe/`（Wave0_replay、Wave1_complementarity、
  Wave2_reliability/、reliability/reliability_raw.json、p5b/p5c/p5d/p5a 子目录、README_STATUS.md、
  INNOVATION_ROADMAP_20260903.md）
- 脚本：`scripts/innovation_v6_dgsafe/`（run_wave0_replay / run_wave1_diagnostic /
  run_wave2a_build_reliability / run_wave2b_diagnostic / run_p5b_gpmr_precheck /
  run_p5c_intrasystem / run_p5d_samebackbone_subspace / run_p5a_global_diag）
- 协议：`configs/innovation_v6_dgsafe/`（wave0_protocol.json、reliability_probe.json 冻结附录）
- git：主体提交 `6a22543`（前序 9899adb/67e0d27/8422f6f/2e2cf9b），每波一提交。
- npz（sub_maps_s0、reliability/pools、audit export_out_s0）被 .gitignore 排除（大文件），
  但 JSON/MD 证据链全部入库。

## 四、复现/续作命令
- Wave2a 确定性重跑（同 seed，~1.5h）：`python scripts/innovation_v6_dgsafe/run_wave2a_build_reliability.py
  --model-dir methods/SubspaceAD/checkpoints/dinov2-with-registers-giant`（会覆盖 reliability_raw.json）
- 若未来要做 S1 完整版（AnomalyCLIP image-level logit 探针）或外部集一次性验证，
  需你确认后由主流程启动；两者均未触碰冻结数据。

## 五、建议的下一步（等你决定）
1. 读 `Wave2_reliability/WAVE2_ARCHIVE.md` 与 `INNOVATION_ROADMAP_20260903.md`；
2. 决定是否投入 S1 完整版（AnomalyCLIP 文本/图像级证据）或把 Wave2 的“小池 z 量化退化”
   作为一个方法学改进小实验（分布级尾估计，可能仍无法过门——预期不高）；
3. 或直接以“A1 主方法 + 上述严格负结果与边界刻画”推进论文写作（负结果本身构成严谨性证据）。
