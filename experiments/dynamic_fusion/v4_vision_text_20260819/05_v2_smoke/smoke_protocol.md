# 官方 SubspaceAD（V2）MPDD Smoke — 预注册协议与决策边界

日期：2026-08-19
RunId：`v4_v2_official_subspacead_smoke_20260819`
官方仓库 commit：`ef56d5c`（`https://github.com/CLendering/SubspaceAD`，Apache-2.0）

## 1. 目的

消除"代理版本（V1 `subspace_style_same_backbone`）失败是否代表官方版本（V2）失败"的疑问。
只做**一次边界明确**的 smoke；失败即回路线 1（第 12 节 D），不再扩展。

## 2. 协议（尽量保持官方默认）

- 脚本：`methods/SubspaceAD/main.py`（官方未改推理/评分路径）
- 数据集：MPDD（development），类别 `metal_plate`（V1 代理最有利的类别——若官方在最有利类别都不能过关，则强证据支持"官方也失败"）
- 参考采样：manifest 固定 `metal_plate/train/good/029.png`（s0/k1），与 V4 G2 矩阵完全一致
- 测试集：全部 97 张（26 good + 71 缺陷）
- 模型：`facebook/dinov2-with-registers-giant`（DINOv2 registers-giant）
- 分辨率 672；`agg_method=mean`，layers `-12..-18`；`k_shot=1`，`aug_count=30`（rotate）；`pca_ev=0.99`；`score_method=reconstruction`；`batch_size=1`；`seed=0`；`bg_mask_method=None`

## 3. 6GB 显存适配（记录在案，不得装作"无改动官方运行"）

1. `--smoke_half`：特征模型 fp16 推理（权重/激活减半）；
2. `need_saliency=False`（`bg_mask_method=None` 时官方本就不使用 saliency）：跳过 `output_attentions`，避免 40 层注意力张量 OOM；
3. `_aggregate_layers` 返回前 `.float()` 还原 float32 特征（与官方 fp32 行为一致）；
4. 新增 `MPDDDataset` 加载器（官方不支持 MPDD；参考图由 manifest 固定，seed/shot 来自 `V4_SMOKE_SEED/V4_SMOKE_SHOT`）；
5. 新增 Pixel AP 输出列（官方只有 P-AUROC；本项目主指标为 P-AP）。

评分公式（重建残差）、PCA（ev 0.99）、后处理（post_process_map 448/672、gaussian blur sigma=4）均未改动。

## 4. 基线（冻结 G2 矩阵 s0/k1 pca0.99，同一测试协议）

| 方法 | metal_plate pixel_ap |
| --- | ---: |
| matched feature-DINO-only KNN（AnomalyDINO 口径） | 0.763261667 |
| V1 subspace（代理版本） | 0.814379 |

## 5. 决策边界（出结果前预注册）

- **PASS**：官方 SubspaceAD `metal_plate` P-AP **≥ 0.7733**（matched DINO-KNN + 0.010）。
  含义：官方版本在最有利类别上明确超过 matched AnomalyDINO，代理失败不能直接外推为官方失败 → 允许再做**一个**确认类别（有界），然后重新向用户汇报。
- **FAIL**：P-AP **< 0.7733**，或发生 OOM/基础设施失败。
  含义：即使官方 giant+672+aug 在最有利类别也无法超过 matched AnomalyDINO → 与 V1 结论一致 → **立即回路线 1**，V2 归档，不继续扩展。
- 单类别 smoke 是"方向性"判据，**不等于** G2 通过；G2 完整标准（9 配置、7/9 非负、4/6 类正、worst ≥ -0.020）在通过本 smoke 后才可讨论。

## 6. 报告字段

run_id、官方 commit、config hash、reference、n_test、P-AUROC / P-AP / AUPRO、VRAM 峰值、运行时长、有限值检查、五项泄漏字段（全 false）、失败原因（若有）。
