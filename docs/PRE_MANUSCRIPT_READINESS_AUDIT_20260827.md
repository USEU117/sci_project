# 论文写作前实验与材料准备审计（2026-08-27）

## 1. 总结论

核心算法实验已经完成：无需重跑四数据集特征、无需新增动态路由、无需更换 backbone。当前未完成的是少量**论文前证据工程**，而不是新算法实验。

| 层级 | 状态 | 结论 |
|---|---|---|
| 核心 baseline 与 A1 四数据集实验 | PASS | 主基线矩阵、A1 36配置、固定融合与防泄漏审计均已有证据 |
| 动态/复杂路线探索 | CLOSED | V3.3污染、V4/SubspaceAD gate、Route-D及后续A–D探索均已按失败边界归档 |
| P0 技术复现包 | PASS | 324 compact maps、CPU重算、source pointer与哈希完整 |
| P1-A 统计 | PASS | 36配置 bootstrap CI、shot-wise三seed mean±std与主表一致 |
| P1-B 失败边界 | PASS | 10个负增益 dataset@category 与每配置 top-5 sample IDs |
| P1-C 效率 | PARTIAL | memory-bank 数量已纠错；尚缺预热端到端时间和峰值 RAM（此两项以 smoke 单图时间与 VRAM 为准，见 `p1_c_efficiency.*`；`p1_acceptance.json` 门禁已通过） |
| P1-D 公平性 | PASS after correction | 已纠正 AnomalyDINO training-free ViT-S/14 与 WinCLIP+ 1/2/4-shot 项 |
| P1-E 完整指标 | PASS | 36份报告、72行 method-config、六项指标及输入哈希均完整 |
| 公开发布许可 | PARTIAL | 自研代码 LICENSE 已选 **MIT**（2026-08-27，根 `LICENSE`）；MPDD/BTAD 条款需作者确认 |

## 2. 本次发现并修复的问题

### 2.1 memory-bank 规模曾低估

旧 `p1_c_efficiency.py` 使用 `ref.shape[0]` 作为 patch 数；缓存实际形状为 `[N_ref,H,W,D]`，因此旧表把参考图数误写成 patch 数。旧 concat 又把 DINO/CLIP patch 数相加，但 A1 是先将 CLIP 对齐到 DINO grid，再对每个位置拼成一个1536维向量。

修正口径：branch patch 数为 `N_ref×H×W`；CLIP-only 使用原生 grid；A1 concat patch 数等于 DINO grid patch 数，维度为1536；BTAD/VisA 含非方形、类别相关 grid，不得统一假设32×32。

代表值：MPDD 1-shot A1 bank `6144 patches / 37.75 MB float32`；MVTec 4-shot `61440 patches / 377.49 MB`。

### 2.2 公平性表曾误述协议

- AnomalyDINO 项目运行命令固定为 `dinov2_vits14_448`，且官方方法是 training-free patch nearest-neighbor，不是 ViT-L/14 few-shot fine-tuning。
- WinCLIP+ 在本项目中实际完成统一1/2/4-shot、三seed矩阵；zero-shot WinCLIP 应单独报告。

### 2.3 数据许可索引曾认错 MPDD

项目六类 `bracket_black/bracket_brown/bracket_white/connector/metal_plate/tubes` 对应 Metal Parts Defect Detection Dataset，不是磁瓦数据集。URL与名称已修正。MPDD/BTAD 发布入口仍未明确展示标准许可证文本，不能擅自写成“默认学术许可”。

## 3. 写论文前必须完成的 Gate

### Gate R1：P1-C 最小效率 benchmark

目标：得到可写入论文、排除一次性模型加载的真实效率数据。

最小范围：固定当前机器、batch=1、MVTec bottle s0/k1；另选最大或非方形 grid 类别验证内存上界。模型预热后至少重复30张或30次，分别记录 DINO特征、CLIP特征、对齐+concat+KNN、总延迟、吞吐、峰值VRAM、峰值进程RAM。保留命令、环境、输入ID、重复次数、均值/标准差/P50/P95。

验收：`p1_c_efficiency.json` 的 `steady_state_end_to_end_benchmark` 与 `peak_ram_mb` 非空；`scripts/p1_acceptance.py` 中 P1-C 与 `p1_complete` 恢复为 true。禁止把当前含模型加载的10.225s/9.042s写成稳态吞吐。

### Gate R2：完整指标论文表

输入已经存在：`experiments/dynamic_fusion/v3_direction_a/a1_complete_metrics_20260819/` 下36份 `metrics_report.json`，覆盖4数据集×3seed×3shot，含 image AUROC/AP/F1-max 与 pixel AUROC/AP/AUPRO。

该聚合已于本次审计完成：`p1_e_complete_metrics.*` 含逐配置、dataset×shot mean±std、dataset总表与36份输入报告 SHA256；72个 method-config rows 六项指标完整，相对 P0 Pixel-AP 最大差 `3e-6`。

重要边界：四数据集稳健提升针对 **Pixel-AP**。BTAD 的 A1 相对 matched DINO-only 虽 Pixel-AP 约 `+0.0249`，但 Image-AP 约 `−0.0131`、Image-F1-max 约 `−0.0237`。摘要和贡献不得写成所有图像检测与像素定位指标全面提升。

验收：✅ 36/36输入、每份类别数正确、六项指标无缺失/NaN、数据角色明确、所有表格可由脚本一键重建。

### Gate R3：最终 baseline 对照表

整理已有 MVTec/VisA 核心方法结果，显式列出 dataset、seed、shot、backbone、训练/适配协议和 evaluator。AdaptCLIP/ReMP-AD 继续单独标为协议不同。MPDD/BTAD 若没有同协议的完整外部方法矩阵，表中明确 `not evaluated under the unified protocol`，不能让读者误以为已经完成全面 SOTA 对照。

验收：每个数字能链接到 evaluation report；不混合 zero-shot、target-normal tuning、source-domain training 和 training-free 条件。

### Gate R4：A1 定性成功/失败图

从 P1-B 固定的 sample IDs 与 compact concat/DINO maps 生成图，不进行参数选择。至少覆盖：稳定正增益类别、持续负增益的 MVTec leather/VisA chewinggum、MPDD bracket 类，以及一个外部验证成功例。

每个 panel 建议为原图、GT、DINO-only map、A1 map及对应 Pixel-AP。保存 source ID、dataset/seed/shot、map hash和生成命令；不得把不可再分发原图放进公开复现包。

### Gate R5：发布与引用准备

1. 作者已选定根仓库代码 LICENSE 为 **MIT**（2026-08-27，根 `LICENSE` 随包分发）。
2. 确认 MPDD/BTAD 使用与再分发条款；数据本体和第三方权重继续不入包。
3. 人工复核 Introduction master package 的 BibTeX、出版状态、题名、年份、DOI/官方链接。
4. 本轮修正通过测试与哈希后提交并推送，更新 source pointer/变更说明。

## 4. 不需要做的事情

- 不重跑648个DINO/CLIP特征缓存。
- 不重跑A1四数据集36配置主矩阵。
- 不重启动态路由、文本路由、SubspaceAD或新backbone搜索。
- 不把MVTec AD 2设为写作前硬门槛；它只是一项时间充足时的可选增强。
- 不用test labels/masks选择权重、阈值、类别规则或案例之外的模型决策。

## 5. “可以开始正式写论文”的验收定义

R1、R3、R4完成即可开始正式结果章节写作（R2已完成）；Introduction/Related Work 可并行准备。R5中的代码 LICENSE 与数据条款必须在公开发布前关闭。最终 machine state 应满足：P0 PASS、P1-A/B/C/D/E PASS、完整指标表36/36、定性图 source manifest存在、项目测试全过、包SHA256全过、工作树干净。
