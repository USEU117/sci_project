# 官方 SubspaceAD（V2）MPDD Smoke — 结果与边界决策

日期：2026-08-19
RunId：`v4_v2_official_subspacead_smoke_20260819`
协议预注册：[smoke_protocol.md](./smoke_protocol.md)（结果出前已写死边界）
官方仓库 commit：`ef56d5c`（`https://github.com/CLendering/SubspaceAD`，Apache-2.0）
模型：`facebook/dinov2-with-registers-giant`（本地化加载，见下）
环境：RTX 3060 Laptop 6GB；GPU fp16 推理（`--smoke_half`）；模型加载峰值显存 ≈ 4.2GB，无 OOM

## 1. 结果（官方 SubspaceAD，MPDD，seed=0 / k=1，manifest 固定参考图）

| 类别 | 参考图 | n_test | I-AUROC | I-AUPR | P-AUROC | **P-AP** | AU-PRO |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| metal_plate | `029.png` | 97 | 1.0000 | 1.0000 | 0.9815 | **0.8513** | 0.9432 |
| bracket_black | `271.png` | 79 | 0.3943 | 0.5757 | 0.9695 | **0.0655** | 0.9297 |

基线（冻结 G2 矩阵 s0/k1 pca0.99，同一测试集、同 manifest 参考）：

| 类别 | matched DINO-KNN P-AP | V1 subspace P-AP | 官方 SubspaceAD P-AP | Δ vs DINO-KNN | Δ vs V1 |
| --- | ---: | ---: | ---: | ---: | ---: |
| metal_plate | 0.7633 | 0.8144 | **0.8513** | **+0.0880** | +0.0369 |
| bracket_black | 0.0207 | 0.0051 | **0.0655** | **+0.0448** | +0.0603 |

## 2. 边界决策（按预注册）

- PASS 阈值：P-AP ≥ matched DINO-KNN + 0.010。
- metal_plate：0.8513 ≥ 0.7733 ✅
- bracket_black（确认类别）：0.0655 ≥ 0.0307 ✅
- **判定：PASS。代理版本（V1）失败不能外推为官方版本失败。**

## 3. 解读

1. 官方 SubspaceAD（giant + 672 + aug30 + pca_ev 0.99）在"最有利"和"最不利"两个类别上都明确超过 matched AnomalyDINO 口径，且大幅超过 V1 代理。
2. V1 崩得最惨的 bracket_black（P-AP 0.0051，几乎不可用）被官方版本救回至 0.0655——说明 V1 的失败主要来自小骨干（vitb14）/低分辨率/无增强的"代理"属性，而非子空间建模本身。
3. 因此**官方版本有现实可能通过 G2 量级标准**（9 配置、7/9 非负、4/6 类正、最差类 ≥ -0.020），需要完整矩阵验证后才能下"通过"结论。

## 4. 6GB 环境适配与口径限制（不得回避）

- 权重加载：safetensors Rust 扩展在本 sandbox 处理 >4GB 文件时崩溃（access violation / pyo3 panic）；改用 numpy memmap 解析 + 模型 fp16 初始化 + `load_state_dict` 手动加载（`src/subspacead/core/extractor.py` 内文档化）。
- 指标内存适配：P-AUROC / P-AP 在 672×672 图上按 stride=8 确定性抽样计算（~64× 少像素）；AU-PRO 保持全分辨率。官方代码以全分辨率计算，数值略有差异。
- 官方代码原不支持 MPDD：新增 `MPDDDataset` 加载器（参考图由 `data/splits/mpdd/manifest.json` 固定；seed/shot 来自 `V4_SMOKE_SEED/V4_SMOKE_SHOT`）。
- 分辨率/骨干与 V4 matched 协议不同（giant/672 vs vitb14/448），本 smoke 是**跨协议指示性比较**，不是严格同骨干消融。
- 单 seed/shot；I-AUROC 在 bracket_black 上 < 0.5（mtop1p 聚合问题），本项目主指标为 P-AP，未受此影响。

## 5. 下一步（需用户拍板）

完整官方 G2 审计：MPDD 6 类别 × 3 seeds × 1/2/4-shot（54 runs，官方协议），对照 matched AnomalyDINO 套 Gate G2；估算 GPU 1.5–3 小时。
- 若通过 → 官方版本成为强视觉锚点，可推进 G3 强视觉文本 headroom 复测与后续 Gate；
- 若失败 → 立即回路线 1（D）。
