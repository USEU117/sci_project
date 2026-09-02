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
- 上次 s0×k{1,2,4}（18 配置）约 30–40 min GPU（RTX 3060 6GB，fp16 memmap 加载）。
- 导出内容：每 (cat,seed,shot) 一个 npz：sample_ids + amap_raw (N,48,48) float16 + ref_ids + 参数/commit。

## 下一步
1. 下载 dinov2-with-registers-giant safetensors（~5.5GB）→ 2. `--export-maps` 跑 MPDD s0×k{1,2,4} →
3. Wave 0 identity replay（每类 |Δpixel_ap| ≤5e-4 vs 冻结审计）→ 4. Wave 1 互补上限诊断 + oracle → 5. WAVE1_DECISION.md。
