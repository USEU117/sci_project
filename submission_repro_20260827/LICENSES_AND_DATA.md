# LICENSES_AND_DATA.md — 数据、权重与代码许可证清单（2026-08-27 访问）

本复现包（`submission_repro_20260827/`）**不含**任何数据集原图或第三方权重文件。
下列条目给出官方来源、访问日期、本地可核验的 license 文件与哈希；本机数据副本内均无内嵌
license 文件，最终以官方页面/条款为准。投稿与再分发前须逐项核验。

## 1. 数据集（本包均未包含）

| 数据集 | 类别数 | 本机路径 | 官方来源 URL（2026-08-27 核验入口） | 许可要点 |
|---|---|---|---|---|
| MPDD | 6 | data/mpdd_raw/MPDD | Magnetic-tile-defect-datasets（GitHub，Konya 磁瓦缺陷集，bracket/connector/metal_plate/tubes 等 6 类） | 学术研究用途；以官方页面条款为准 |
| BTAD | 3 | data/btad_raw | BTAD — Brother Industrial Anomaly Detection（UniUD AVIRES 发布） | 学术研究用途；以官方页面条款为准 |
| VisA | 12 | data/visa_raw | VisA — Amazon Science spot-diff 发布（含 meta.json 官方测试划分） | 学术研究用途；以官方页面条款为准 |
| MVTec AD | 15 | data/mvtec | MVTec AD（MVTec Software 官网） | CC BY-NC-SA 4.0（非商业）；禁止打包与商业使用 |

本地副本均无内嵌 LICENSE 文件（已核验）。测试划分与本包使用方式详见 `config/split_manifest_hashes.json`。

## 2. 模型权重（本包均未包含）

| 权重 | 本机路径 | 许可 | 本地核验 |
|---|---|---|---|
| AnomalyCLIP `9_12_4_multiscale_visa/epoch_15.pth` | methods/AnomalyCLIP-main/checkpoints/... | 随 AnomalyCLIP 项目（MIT 代码；权重按其仓库条款） | 本地 LICENSE 哈希 `23a15f9f18973b2376cb281cb3bba6fea79c80c1af79982a48920b40f3a25f4f` |
| DINOv2 ViT-B/14 `dinov2_vitb14_pretrain.pth` | C:/Users/lynle/.cache/torch/hub/checkpoints/... | 代码 Apache-2.0；模型权重按 facebookresearch/dinov2 官方声明（非商业 CC-BY-NC-4.0，投稿/商用前须核验） | hub 代码 LICENSE 哈希 `600cc67cc4cb2f5ea317dcfc687ad1c74dc4bec8782bbe9db0afd83513b935b7` |

## 3. 代码与依赖

| 组件 | 许可 | 本地核验 |
|---|---|---|
| AnomalyCLIP（methods/AnomalyCLIP-main） | MIT（Copyright 2024 Qihang Zhou） | 本地 LICENSE 哈希同上 |
| facebookresearch/dinov2（torch.hub 加载，非仓库内文件） | Apache-2.0（代码）；模型权重见上 | hub 缓存 LICENSE 哈希同上 |
| scikit-learn / scipy / opencv / numpy / faiss / scikit-image | BSD-3-Clause / BSD / Apache / MIT 等（见 environment/*pip_freeze.txt 与各自官方） | 由 `pip freeze` 固定版本 |
| 本仓库自研代码（scripts/, src/, submission_repro_20260827/ 内脚本与文档） | **作者须在投稿前选定并放置仓库 LICENSE（本包不代为决定）** | 见包内 `SOURCE_COMMIT.txt` 与 `manifest.json` |

## 4. 再分发边界

- MVTec AD 的 CC BY-NC-SA 4.0 禁止商业用途：任何发布物不得包含 MVTec 原图。
- 所有数据集与第三方权重均需第三方自行按官方许可获取；本包只含获取指引与校验哈希。
- 若需在隔离机器复现，仅传递本包 + 许可允许的数据/权重获取指引即可。
