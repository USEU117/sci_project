# RCEC Phase 0 — 输入与复现审计报告

日期：2026-09-02
任务书：`11_RCEC_INNOVATION_IMPLEMENTATION_AND_ACCEPTANCE_HANDOFF_CN_20260901.md` Phase 0

## 1. 工作区保护

执行前 `git status --short` 记录的既有未提交修改（用户工作）均未触碰：
- `docs/CURRENT_DYNAMIC_FUSION_STATUS.md`、`docs/PAPER_DETAILED_CHINESE_DRAFT_20260827.md`、
  `docs/submission_reproducibility_20260826/VERSIONED_EVIDENCE.sha256`、
  `submission_repro_20260827/evidence/p1/p1_d_fairness_table.md`
- 未跟踪：`docs/paper_writing_preparation_20260830/`、`docs/PROJECT_PROGRESS_AND_MANUSCRIPT_PLAN_FOR_SUPERVISOR_EN_20260827.md`、
  `experiments/dynamic_fusion/v3_direction_a/clip_only_controls_20260830/`、
  `scripts/build_manuscript_figure_package.py`、`scripts/evaluate_clip_only_complete_metrics.py`

未使用 `git reset --hard` / `git checkout --`。

## 2. 缓存布局与可用性

特征根：`outputs/dynamic_fusion/v3_direction_a/`

| 数据集 | DINO 目录 | CLIP 目录 | 类别数 | DINO grid | CLIP grid | 维度 |
|---|---|---|---|---|---|---|
| MPDD | `features_vitb14_s{S}_k{K}/anomalydino_visual` | `features_s{S}_k{K}/anomalyclip_text` | 6 | 32×32 | 37×37 | 768/768 |
| BTAD | `features_vitb14_btad_s{S}_k{K}/anomalydino_visual` | `features_btad_s{S}_k{K}/anomalyclip_text` | 3 | 32×32 / 32×42 | — | 768 |
| VisA | `visa_features_vitb14/s{S}_k{K}/anomalydino_visual` | `visa_features/s{S}_k{K}/anomalyclip_text` | 12 | 32×32 | — | 768 |
| MVTec AD | `mvtec_features_vitb14/s{S}_k{K}/anomalydino_visual` | `mvtec_features/s{S}_k{K}/anomalyclip_text` | 15 | 32×32 | — | 768 |

每个 `.npz` 包含：`patch_features (N,H,W,768)`、`ref_patch_features (shot,H,W,768)`、
`sample_ids (N,)`、`gt_sp (N,)`、`imgs_masks (N,map,map)`、`grid_size`、分支/seed/shot 元数据。

`run_rcec_mpdd_development.py --validate-only` 对 MPDD 全 9 配置校验：DINO 类别集 == CLIP 类别集 == manifest 类别集，且每类别 `ref_patch_features.shape[0] == manifest 参考数` —— **passed**。

## 3. 参考顺序审计

`export_anomalydino_mpdd_features.py` 与 `export_anomalyclip_mpdd_features.py` 源码均按
`manifest["categories"][cat][seed][shot]` 的列表顺序逐参考图 append `ref_blocks`，因此：
- 两分支参考块第 0 维顺序一致（同一 manifest 列表）；
- 参考身份 = manifest 列表顺序 + patch (row, col)，DINO 与 CLIP 在 resize 对齐后同位置配对成立。

验证：`tests/test_rcec.py::test_memory_metadata_pairing`（合成）与 MPDD 实测（ref 数量校验）。

## 4. A1 回归核对

任选 MPDD s0/k1 `metal_plate`：RCEC 的 `s_A1` 路径（对齐+分支 L2+0.5/0.5 concat+整体 L2+KNN k=1+distance/2+dists2map）
与冻结 `evaluate_a1_feature_fusion.fuse_category(..., "concat", pca_dim=0, whiten=False, map=448, w=0.5)`：
- 448×448 map 最大绝对误差 `< 1e-5`；
- Pixel-AUROC / Pixel-AP / Pixel-AUPRO 误差 `< 1e-6`。

DINO-duplicate 1536-D 控制（`compute_dino_duplicate_dists`）：map 误差 `< 1e-5`，Pixel-AP 误差 `< 1e-6`（单元级随机向量距离保持 `< 1e-6`）。

## 5. 环境

- 运行环境：`.venv-patchcore`（torch 2.0.0+cu118、faiss 1.7.4、sklearn 1.2.2、scipy 1.9.1、cv2 4.8.1、yaml 6.0.3）。
- 该环境已补装 `pytest`（测试执行器，不改变推理依赖）。

## 6. 结论

输入清单完整；A1 回归一致；不存在未解释的样本 ID、grid 或特征维度冲突。Phase 0 验收通过。
