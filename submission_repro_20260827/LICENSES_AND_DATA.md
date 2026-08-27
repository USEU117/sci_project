# LICENSES_AND_DATA.md — 数据、权重与代码许可证清单（2026-08-27 访问）

本复现包（`submission_repro_20260827/`）**不含**任何数据集原图或第三方权重文件。
下列条目给出官方来源、访问日期、本地可核验的 license 文件与哈希；本机数据副本内均无内嵌
license 文件，最终以官方页面/条款为准。投稿与再分发前须逐项核验。

## 1. 数据集（本包均未包含）

| 数据集 | 类别数 | 本机路径 | 官方来源 URL（2026-08-27 核验入口） | 许可要点 |
|---|---|---|---|---|
| MPDD | 6 | data/mpdd_raw/MPDD | Metal Parts Defect Detection Dataset：<https://github.com/stepanje/MPDD> | 官方仓库未展示标准 LICENSE 文件；发布复现包前须向作者/数据下载页确认使用与再分发条款。**不是磁瓦数据集。** |
| BTAD | 3 | data/btad_raw | beanTech Anomaly Detection，作者托管下载：<https://avires.dimi.uniud.it/papers/btad/btad.zip>；论文：<https://arxiv.org/abs/2104.10036> | 直接下载端未提供可核验的标准许可证文本；只提供获取链接，不再分发数据，投稿前保留访问记录并核验作者条款。 |
| VisA | 12 | data/visa_raw | AWS Open Data：<https://registry.opendata.aws/visa/>；官方仓库：<https://github.com/amazon-science/spot-diff> | CC BY 4.0；本包不含原图，论文按官方要求引用。 |
| MVTec AD | 15 | data/mvtec | MVTec Software 官方页：<https://www.mvtec.com/research-teaching/datasets/mvtec-ad> | CC BY-NC-SA 4.0（非商业）；禁止把原图打入本包或用于商业用途。 |

本地副本均无内嵌 LICENSE 文件（已核验）。测试划分与本包使用方式详见 `config/split_manifest_hashes.json`。

## 2. 模型权重（本包均未包含）

| 权重 | 本机路径 | 许可 | 本地核验 |
|---|---|---|---|
| AnomalyCLIP `9_12_4_multiscale_visa/epoch_15.pth` | methods/AnomalyCLIP-main/checkpoints/... | 项目代码为 MIT；官方仓库未给 checkpoint 单独许可证，故本包不再分发权重 | 本地 LICENSE 哈希 `23a15f9f18973b2376cb281cb3bba6fea79c80c1af79982a48920b40f3a25f4f`；官方仓库 <https://github.com/zqhang/AnomalyCLIP> |
| DINOv2 ViT-B/14 `dinov2_vitb14_pretrain.pth` | C:/Users/lynle/.cache/torch/hub/checkpoints/... | 标准 DINOv2 code/model card 当前标为 Apache-2.0；须与实际下载模型及固定 commit 对应，不能套用 Cell-DINO/医学扩展模型的非商业许可证 | hub LICENSE 哈希 `600cc67cc4cb2f5ea317dcfc687ad1c74dc4bec8782bbe9db0afd83513b935b7`；官方 model card <https://github.com/facebookresearch/dinov2/blob/main/MODEL_CARD.md> |

## 3. 代码与依赖

| 组件 | 许可 | 本地核验 |
|---|---|---|
| AnomalyCLIP（methods/AnomalyCLIP-main） | MIT（Copyright 2024 Qihang Zhou） | 本地 LICENSE 哈希同上 |
| facebookresearch/dinov2（torch.hub 加载，非仓库内文件） | Apache-2.0（代码）；模型权重见上 | hub 缓存 LICENSE 哈希同上 |
| scikit-learn / scipy / opencv / numpy / faiss / scikit-image | BSD-3-Clause / BSD / Apache / MIT 等（见 environment/*pip_freeze.txt 与各自官方） | 由 `pip freeze` 固定版本 |
| 本仓库自研代码（scripts/, src/, submission_repro_20260827/ 内脚本与文档） | **MIT（2026, LiYuening）— 已于 2026-08-27 选定**，见仓库根 `LICENSE` | 根 `LICENSE` 已放置；包内 `SOURCE_COMMIT.txt` 与 `manifest.json` 记录来源 |

## 4. 再分发边界

- MVTec AD 的 CC BY-NC-SA 4.0 禁止商业用途：任何发布物不得包含 MVTec 原图。
- MPDD 与 BTAD 的标准许可证文本当前未在其发布入口中明确核验到；公开复现包只保留获取链接和引用，不包含其原图，并在投稿/发布前向作者或托管方确认条款。
- 所有数据集与第三方权重均需第三方自行按官方许可获取；本包只含获取指引与校验哈希。
- 若需在隔离机器复现，仅传递本包 + 许可允许的数据/权重获取指引即可。
