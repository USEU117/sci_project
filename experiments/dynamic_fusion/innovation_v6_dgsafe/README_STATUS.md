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
| Wave 2 正常-only 可靠性 | ⏳ 下一步（Wave2a GPU ~60–90 min 后 Wave2b 判 GT） |

## 下一步
1. Wave2a：构造 normal augmentation/layer 探针，冻结 reliability_raw.json（协议附录
   `configs/innovation_v6_dgsafe/reliability_probe.json`）→ 2. Wave2b：读 GT 检查
   Spearman(r_sub, ΔSUB−A1) ≥ 0.40 与 connector 后 25% → 3. 通过才做 Wave3 小门 C0/C1/C2+控制。
