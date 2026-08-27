# P0 投稿复现工作最终验收（2026-08-27）

## 1. 最终结论

结论必须分成“技术复现”和“公开发布”两层：

- **P0 研究数值重建：PASS。** 四数据集、36 个配置的 A1 concat vs matched feature-DINO-only 均已重建，数据集平均值与历史结果的绝对误差不超过 `5e-4`。
- **P0 技术复现包：PASS。** `P0_ACCEPTANCE_AUDIT_20260827.json` 的 P0A–P0I 全部为 `true`，`research_rebuild_complete=true`、`submission_repro_package_complete=true`。
- **公开发布准备：仍有两项人工事项。** 作者须选择本仓库代码许可证，并把 `LICENSES_AND_DATA.md` 中数据集官方来源名称补成可点击的精确 URL 后逐项复核许可条款。完成前可以继续统计与写稿，但不要公开发布代码包。

本文件取代此前同名文档中的 `CONDITIONAL` 结论；旧问题及其关闭证据保留在第 3 节，供审计追溯。

## 2. 独立复验结果

2026-08-27 在当前 `main` 工作树完成了以下只读/临时输出复验：

1. `python scripts/audit_submission_repro_package.py`：P0A–P0I、`research_rebuild_complete`、`submission_repro_package_complete` 全部为 `true`，与保存的 JSON 一致。
2. `python submission_repro_20260827/recompute_tables.py --verify-only`：324/324 个类别 payload 通过结构检查。
3. 从 compact maps + 用户本地 mask 完整重算：
   - MPDD seed0/K1：6/6 类通过，配置聚合相对参考表最大误差 `1e-5`；
   - MVTec AD seed1/K4：15/15 类通过，配置聚合最大误差 `3e-6`，覆盖已知最敏感的 wood float16 项。
4. `.venv-anomalyclip` 下执行 `python -m pytest -q tests`：`122 passed in 5.80s`。历史 `81 passed` 是当时较早测试集的有效快照，不是当前测试总数。
5. 独立读取并重新计算 `SHA256SUMS`：447 条受校验记录、0 缺失、0 mismatch；包内当前共 448 个文件（含 `SHA256SUMS` 自身）。
6. `SOURCE_COMMIT.txt` 指向 `12e1fcf180f7eacd41854acbbc865a4190df7c98`，该提交存在且生成时 `dirty=false`；其后本地提交仅补来源指针、哈希和状态文档，未改变 compact maps 或重算代码。

完整重算产生的临时报告已删除，没有混入复现包或 Git 工作树。

## 3. 原条件问题的关闭状态

| 原问题 | 当前状态 | 关闭证据 |
|---|---|---|
| `predictions_compact` 只有汇总数字 | 已关闭 | `predictions_compact/maps/` 含 324 个逐 `dataset×seed×shot×category` NPZ，含 sample/ref IDs、两种 patch maps、grid/stride 和来源哈希 |
| 包内无 CPU 重算脚本 | 已关闭 | `submission_repro_20260827/recompute_tables.py` 支持 `--verify-only` 及从合法数据根 mask 完整重算 |
| 无源码提交标识 | 已关闭 | `SOURCE_COMMIT.txt` 固定 `12e1fcf...` 与生成命令 |
| 重建结果混同历史 byte identity | 已关闭 | `rebuild_manifest_v2.json` 明确 numerical equivalence=true、byte identity=false，旧 freeze manifest 未改 |
| 1152 维及 multimodal 命名歧义 | 已关闭 | `METHOD_SPEC_V2.md` 固定 DINO 768 + CLIP image tower 768 = 1536，并声明 A1 无文本推理 |
| 包内哈希覆盖不足 | 已关闭 | 447 条清单记录独立复验全部通过 |
| 自研代码 LICENSE 未选择 | **未关闭，发布前人工决定** | 根目录当前没有 `LICENSE`；不得由 AI 擅自替作者选择 |
| 数据集官方 URL 不够精确 | **未关闭，发布前核验** | 当前 `LICENSES_AND_DATA.md` 只有来源名称，需补实际 URL 并复核当日条款 |

## 4. P0 方法与数值证据

- 双分支缓存：DINO 324 + AnomalyCLIP image tower 324，共 648 个 NPZ。
- 实测维度：DINO `32×32×768`；AnomalyCLIP `37×37×768`；对齐后 concat 为 1536。
- 36 个配置全部重建；四数据集平均 ΔPixel-AP：MPDD `+0.025829`、BTAD `+0.024895`、VisA `+0.052353`、MVTec AD `+0.031962`。
- 五项泄漏 flag 全 false；A1 是固定等权双视觉 patch 融合，不是动态路由，也没有文本特征进入最终推理。
- compact maps 使用 float16；逐类允许 `5e-3` 的回放容差，配置/表级聚合仍须在 `5e-4` 内。不得宣称与历史压缩文件逐字节相同。

## 5. 接下来应做什么

P0 不需要再跑四数据集特征导出，也不需要重新设计动态融合。下一主线是 P1：

1. 由 324 个 compact maps 生成 paired image bootstrap 或 category bootstrap 的 ΔPixel-AP 置信区间，并明确统计单位。
2. 生成按 dataset×shot 汇总的三 seed mean±std、逐类增益、worst/negative categories 和可追溯失败样例清单。
3. 完成效率表：训练参数、推理时间、峰值 VRAM/RAM、memory-bank 大小、复现包大小；若缺运行时数据，只补最小代表性 benchmark，不重跑全矩阵。
4. 完成公平性表：backbone、输入分辨率、shot/seed、训练域、目标正常图调优、测试时适应、评价实现和 baseline source。
5. 作者完成 LICENSE/官方 URL 决策后再公开发布复现包；Git 当前相对 `origin/main` ahead 3，推送前应先审阅提交。
6. P1 通过后再按双编码器固定视觉融合主线重写论文，并最后实时核验 SCI 四区期刊要求。

当前不建议新增动态路由、视觉—文本路由、SubspaceAD 或新 backbone 实验。MVTec AD 2 仍是可选增强项，不是 P1 前置条件。
