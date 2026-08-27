# A1 投稿复现包（compact）— submission_repro_20260827

方法（冻结，不再改动）：DINOv2 ViT-B/14（448px）与 AnomalyCLIP ViT-L/14@336 图像塔（518px）
双视觉编码器 patch 特征 → 各分支 L2 归一化 → 等权 concat（w=0.5/0.5）→ 整体 L2 →
KNN(k=1) 正常记忆库 → distance/2 = 像素异常图（map=448, stride=8）。**无文本特征参与最终推理，无动态路由。**
方法口径以 `METHOD_SPEC_V2.md` 为准（concat 维度 **1536**，旧文档 1152 为错误）。

## 包含什么

- `config/`：冻结配置（`frozen_a1.json`）与 split manifest/权重 SHA256（`split_manifest_hashes.json`）。
- `evidence/`：版本化论文表（`paper_tables/`）、P0 机器审计快照、CPU 回归记录、逐配置重建证据（`per_config/`）、负结果索引。
- `predictions_compact/maps/`：**可重放的逐图 patch anomaly maps**（每 `dataset × seed × shot × category` 一个
  compressed float16 npz）：`sample_ids`、concat 与 matched DINO-only 低分辨率 patch map、grid/map/stride 元数据、
  reference IDs、特征缓存 SHA256。**包内不包含 GT mask**。
- `recompute_tables.py`：包内独立 CPU 脚本，从 compact maps + 用户数据 mask 重算逐类/逐配置/四数据集论文表
  （`--verify-only` 做结构校验，无需数据）。
- `METHOD_SPEC_V2.md` / `LICENSES_AND_DATA.md`：方法口径与数据/权重许可索引；自研代码已选 **MIT（2026, LiYuening）**（仓库根 `LICENSE`），MPDD/BTAD 再分发条款发布前仍须向作者确认。
- `evidence/p1/`：**P1 论文实验收尾证据**——`p1_a_bootstrap_ci.*`（36 配置 bootstrap CI + dataset×shot 三 seed mean±std，dataset 均值与主表差 ≤5e-4）、`p1_b_*`（worst/negative categories 与逐图失败样例 ID）、`p1_c_efficiency.*`（训练参数/推理时间/峰值 VRAM/记忆库规模/包大小）、`p1_d_fairness_table.*`（11 方法协议对照）、`p1_acceptance.json`。**P1-A/B/C/D 全部通过**（`p1_complete=true`）。
- `environment/`、`logs/`、`manifest.json`、`rebuild_manifest_v2.json`、`SOURCE_COMMIT.txt`、`SHA256SUMS`。

本包**不包含**数据集原图、第三方权重——不可再分发，须按 `LICENSES_AND_DATA.md` 与 `config/split_manifest_hashes.json`
获取并校验。

## 论文表数字（P0-3 重建，历史对照，容差 5e-4）

A1 concat vs matched feature-DINO-only 的 ΔPixel-AP（9 配置 = 3 seeds × 1/2/4-shot 参考采样）：

| 数据集 | 角色 | 重建 ΔAP | 历史 ΔAP | 误差 | 9/9 全正 |
|---|---|---|---|---|---|
| MPDD | development | +0.025829 | +0.025830 | 1e-6 | 是 |
| BTAD | external frozen validation | +0.024895 | +0.024895 | 0 | 是 |
| VisA | in-domain frozen validation | +0.052353 | +0.052353 | 0 | 是 |
| MVTec AD | external frozen validation | +0.031962 | +0.031962 | 0 | 是 |

注意：3 seeds × 3 shots 是同一测试集上的参考采样配置，不是 9 个独立数据集。
四数据集一致正结论仅针对 Pixel-AP；完整指标表显示 BTAD 的 Image-AP 与 Image-F1-max 相对 matched DINO-only 下降，论文不得扩写为所有图像级与像素级指标全面提升。

## 如何复现

### CPU 一键重算论文表（从 compact maps）

```powershell
# 结构校验（无需数据）
python recompute_tables.py --verify-only

# 完整重算（需要四个数据集的 GT mask，按官方许可获取后指定路径）
python recompute_tables.py --data-root mpdd=<abs> --data-root btad=<abs> `
    --data-root visa=<abs> --data-root mvtec=<abs>
```

脚本从 `predictions_compact/maps/` 读取逐图 maps，从数据根读取 mask 并按 sample_id 对齐，
重放 `dists2map(448) + stride=8` 指标，与 `evidence/per_config/` 逐配置报告比对（容差 5e-4），
输出到 `tables_recomputed/`。

### 从原图重建（GPU smoke）

仓库侧：`scripts/smoke_a1_one_class_one_image.py`（一类一图，实测 concat 维度 1536）
与 `scripts/p0_3_evaluate_a1_rebuild.py`（从特征缓存现场重算）。

### 机器审计

`scripts/audit_submission_repro_package.py` 全门禁通过 → `submission_repro_package_complete=true`
（见 `evidence/P0_LIVE_AUDIT_REBUILD_20260827.json` 与更新后的 P0_ACCEPTANCE_AUDIT）。

## 禁止事项

- 不从 test label/mask 选权重、阈值、类别规则或路由条件。
- 不把 VisA 写成独立外部验证（AnomalyCLIP checkpoint 与 VisA 有训练域关系）。
- 不把 9 配置当作 9 个独立数据集做显著性检验。
- 不重新开启动态路由/文本融合实验；不把 A1 写成 vision-language fusion。
