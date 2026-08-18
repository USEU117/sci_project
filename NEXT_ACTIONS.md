# 下一步执行清单

更新日期：2026-08-10

本文件只记录当前真实待办。历史队列和早期计划保留在日志、`PLAN.md`、`PROJECT_STATUS.md` 和带日期的实验报告中，不再作为当前状态。

## 当前权威状态

- PromptAD MVTec 剩余 5 组训练已启动（GPU 串行，从 s1_k2 capsule seg 断点续跑）；基线 ReMP-AD/AdaptCLIP 待 GPU 空出后接力。
- 基线前置已就绪：AdaptCLIP 环境依赖冲突已修复（transformers 4.38.2）、ReMP-AD 环境可用、`data/visa -> data/visa_raw` junction 已建立；两个基线脚本已写好并通过 `-ValidateOnly` 预检（`scripts/start_adaptclip_mvtec_gate_a.ps1`、`scripts/start_remp_ad_mvtec.ps1`）。
- 用户已确认「等 PromptAD 全部完成再接力基线」；看门狗脚本 `scripts/run_baselines_after_promptad.ps1` 已后台启动（poll 120s / max 96h / stall 45min）：① `status.json=completed` 后自动跑 AdaptCLIP Gate A（bottle, batch 1）→ ReMP-AD MVTec（train VisA → test 4/2/1，batch 1）；② 检测到队列进程消失但未 completed 时自动断点重启队列；③ `state=blocked` 则停止并记录；④ marker 停滞 >45min 仅告警不杀进程。看门狗日志 `outputs/logs/baselines_after_promptad.log`。
- VisA 四种基线共 36/36 运行通过统一审计。
- MVTec：PatchCore 9/9、WinCLIP+ 9/9、AnomalyDINO 9/9、PromptAD 4/9、DynamicFusion 9/9。
- 动态融合 V1 已冻结并完成最终验证、科学分析、完整消融和可视化材料；不得再使用已查看的最终结果调参。
- V3.3 论文主线已完整：metal_plate 天花板（safe 退火）✅、BTAD holdout ✅、图像级指标 ✅。
- AnomalyCLIP 只作为 zero-shot 结果和动态融合文本分支，不冒充少样本矩阵。
- 机器可读状态快照：`experiments/summaries/project_state_snapshot_20260809.json`。

## P0：状态同步（已完成）

- [x] 重新审计 VisA 36 组结果。
- [x] 重建 MVTec 方法/seed/shot 完整性矩阵。
- [x] 确认 AnomalyDINO 为 9/9，废止旧 8/9 记录。
- [x] 确认 PromptAD 为 4/9、剩余 5 组。
- [x] 将旧 `overnight_status.json` 标记为 `superseded`。
- [x] 确认正式 PromptAD 队列为 `paused_by_schedule`。
- [x] 同步动态融合实际分支：AnomalyDINO + AnomalyCLIP。
- [x] 同步动态融合冻结参数和最终验证状态。

## P1：动态融合科学分析、消融和可视化（已完成）

- [x] 审计 17 个冻结运行的分支来源、校准文件和禁止信息标记。
- [x] 比较 DynamicFusion、原始 AnomalyDINO、AnomalyCLIP 和固定权重方案。
- [x] 完成按 dataset、shot、seed、category 的差值分析。
- [x] 分析图像权重、像素权重、正常/异常权重和路由比例。
- [x] 定位主要原因：正常参考校准饱和、熵置信度误判和样本相关权重改变排序。
- [x] 整理 K=1/2/4 完整消融表，并明确缓存限制下未做的层间特征消融。
- [x] 生成成功案例、失败案例、分支热图和像素权重图。
- [x] 生成校准诊断、shot 对比、逐类别热图、路由统计和双温度消融图。
- [x] 生成 9-sheet Excel 数据包并完成视觉检查；输入来源表覆盖 693 行、285 个唯一文件 SHA256。
- [x] 形成客观结论：允许报告审计结果和局部收益，禁止宣称全面优于最强单分支。

主要材料：

- `docs/dynamic_fusion_scientific_analysis_20260809.md`
- `docs/dynamic_fusion_ablation_and_visualization_20260809.md`
- `experiments/summaries/dynamic_fusion_scientific_analysis_20260809/`
- `outputs/dynamic_fusion/figures/20260809_scientific_analysis/`
- `outputs/dynamic_fusion/analysis_20260809/dynamic_fusion_scientific_analysis_20260809.xlsx`

## P2：论文初稿材料（已完成中文 V0.1 与英文 SCI 风格 V0.2）

- [x] 根据已完成材料搭建论文方法、实验、结果、消融、失败案例和局限性章节。
- [x] 将 DynamicFusion 的局部定位收益与总体图像级失败同时写清楚。
- [x] PromptAD 始终标记 `target_normal_tuning=true`。
- [x] AnomalyCLIP zero-shot 与少样本方法分开报告。
- [x] 效率表先留出正式 GPU 显存和推理耗时栏，不使用估计值冒充测量值。
- [x] 生成 Word 初稿：`outputs/paper_draft_20260810/基于不确定性路由的少样本工业异常视觉语言证据融合方法_论文初稿_V0.1.docx`。
- [x] 使用 Microsoft Word 更新目录并导出 PDF，完成 33 页逐页视觉检查。
- [x] 依据 Elsevier、IEEE 和 Springer Nature 的通用作者规范，将现有证据重写为英文期刊论文，而不是逐句直译中文初稿。
- [x] 生成英文 Word 初稿：`outputs/paper_draft_20260810/Leakage-Safe_Uncertainty_Routing_English_SCI_Draft_V0.2.docx`。
- [x] 英文稿采用单栏期刊中性格式，包含 248 词单段摘要、6 个关键词、编号章节、声明、数据/代码可用性和生成式 AI 使用说明；不使用中文学位论文封面和目录。
- [x] 英文稿完成 16 页逐页视觉检查、DOCX 完整性检查、标题/图片/表格几何审计和 19 条正文引用—参考文献一致性检查。
- [ ] 选定具体目标期刊后，按该刊 Guide for Authors 和 Word/LaTeX 模板做最后一次格式迁移，并复核该刊当前分区、收稿范围、篇幅和图表要求。
- [ ] 补充作者姓名、单位、ORCID、通讯作者、基金、贡献分工、利益冲突和公开仓库/归档编号。
- [ ] 长 GPU 实验完成后，只更新明确标为待补的结果表、效率表和相应讨论，保留 V1 失效证据。

## P3：PromptAD MVTec 剩余 GPU 矩阵（已完成）

- [x] 复核 `s1_k2` 断点：已有 5/30 个阶段标记；从 capsule 分割恢复。
- [x] 重建缺失的训练入口 `_run_promptad_mvtec.ps1`（此前队列脚本引用但文件不存在），并修复根目录路径解析 bug。
- [x] 通过 `--validate-only` 确认 completed=4（s0_k1/s0_k2/s0_k4/s1_k1）、pending=5（s1_k2/s1_k4/s2_k1/s2_k2/s2_k4），并启动断点续跑队列。
- [x] 完成 `s1_k2`。
- [x] 完成 `s1_k4`。
- [x] 完成 `s2_k1`。
- [x] 完成 `s2_k2`。
- [x] 完成 `s2_k4`。
- [x] 每组验收 15 类、1,725 样本、零 schema 错误、无 Traceback（`status.json=completed`，9/9 组合全部完成）。
- [ ] 全部完成后生成 3-seed 均值±标准差和论文主表（PromptAD 部分）。

## P2.5：DynamicFusion V2 在 13 日前的落实

- [x] V1 关键证据 SHA256 冻结及防覆盖检查。
- [x] 独立 V2 配置、RunId、泄漏审计字段和输出保护。
- [x] 排序保持校准、尺度下限与饱和/并列/排序诊断。
- [x] 图像/像素超支持范围检测。
- [x] 视觉默认安全路由和图像/像素独立权重路径。
- [x] V2 最小缓存消融接口和 CPU 烟雾测试。
- [x] 全部 45 项测试通过；VisA seed-0 正常参考回顾性校准审计 48/48 通过，仅作软件验证。
- [x] MPDD、BTAD 归档下载、SHA256/ZIP 检查、安全解压、类别/掩码审计和嵌套 manifest 已完成。
- [x] MPDD 6 类、BTAD 3 类；两者均通过 1/2/4-shot × 3 seeds 嵌套验证和入选文件 SHA256 检查。
- [x] 数据协议冻结文件已生成并复核通过，仍保持 `parameters_frozen=false`、`holdout_metrics_allowed=false`。
- [x] 已为 MPDD/BTAD 的 1/2/4-shot × 3 seeds 生成 18 组正常参考视图，并建立 36 个两分支缓存任务；RunId、命令、输入 SHA256 和禁止执行状态均已写入 `experiments/dynamic_fusion/v2/branch_cache_queue/queue.json`。
- [x] AnomalyDINO 与 AnomalyCLIP 的 36/36 条缓存命令均通过 `validate_only` 干跑；未加载长推理、未读取测试标签、未产生 BTAD 指标、未使用 GPU。
- [x] MPDD/BTAD 正常参考两分支缓存 36/36 完成，缓存审计 36/36、校准审计 18/18 通过，失败 0；未读取 BTAD 测试指标。
- [x] 完整 MPDD 预测缓存 Gate A：seed 0、1-shot 的 AnomalyDINO/AnomalyCLIP 均完成，6 类、458 图的跨分支样本/标签/掩码/数值/泄漏审计通过。
- [x] MPDD 3-seed × 1/2/4-shot 完整预测缓存矩阵 9/9 完成并逐组审计通过；固定 zero-shot AnomalyCLIP 预测按来源 SHA256 复用，AnomalyDINO 按组合重新生成。
- [x] MPDD seed 0 候选筛选完成：六个候选完全回退为纯视觉；`pixel_wide_w25` 保持图像指标不变、宏平均 AUPRO +0.00305，但收益仅来自 4-shot `bracket_white`，暂不能冻结。
- [x] seed 1/2 重复验证未复现 seed 0 收益：`pixel_wide_w25` AUPRO 相对纯视觉约 -0.0000006，因此不冻结该候选。
- [x] 像素独立修复后的 seed 0 重跑完成：像素融合真实启用后 AUPRO 下降 0.079～0.112，纯视觉胜出；微小 Pixel AUROC/AP 变化不能抵消区域定位明显退化。
- [x] seed 1/2 的 `pixel_only_w15` 分别取得 AUPRO +0.0361/+0.0236，但与 seed 0 的 -0.0791 方向相反，存在明显抽样敏感性。
- [x] 统一 200 阈值最终确认完成：总体 AUPRO +0.00324，但逐 seed 为 -0.00315/+0.01770/-0.00482，仅 1/3 seed 为正，未通过重复性门槛。
- [x] V2 参数已正式冻结为 `visual_only_safe_fallback`，图像/像素文本权重均为 0；11 项证据 SHA256 冻结复核通过，49/49 测试通过。
- [ ] 参数冻结后允许进入 BTAD 最终验证；下一步先准备并干跑 BTAD 完整预测适配器，再运行冻结方案，不再根据 BTAD 结果调参。
- [x] 只在 MPDD 上确定 V2 阈值和权重上限，并生成正式参数冻结文件。
- [ ] 参数已冻结，下一步可读取 BTAD 指标并开展最终验证；结果只能报告，禁止回头调参。

当前执行状态：`experiments/dynamic_fusion/v2/branch_cache_queue/runtime/status.json`。代码/协议预冻结、数据协议冻结、参考视图和分支缓存命令干跑均已验证；GPU 队列只生成正常参考分支缓存并做逐任务审计/校准，不读取 BTAD 测试指标。

## P4：ReMP-AD 和 AdaptCLIP 门控（已完成）

- [x] AdaptCLIP：取得并校验官方 checkpoint（`12_4_128_train_on_visa_3adapters_batch8/epoch_15.pth`，SHA256 匹配）、修复环境依赖（venv 装 transformers 4.38.2 解决 huggingface_hub 冲突）。
- [x] 建立 `data/visa -> data/visa_raw` junction，满足 ReMP-AD `train_data_path` 与 AdaptCLIP 默认数据根。
- [x] 写好 `scripts/start_adaptclip_mvtec_gate_a.ps1`（bottle、batch size 1、VisA checkpoint、`--dataset mvtec`）并通过 `-ValidateOnly`。
- [x] 写好 `scripts/start_remp_ad_mvtec.ps1`（train VisA → test MVTec，k_shot 4/2/1，batch_size 参数化）并通过 `-ValidateOnly`。
- [x] AdaptCLIP bottle Gate A（batch size 1）运行完成（2026-08-14，bottle: I-AUROC 99.2 / P-AUROC 95.6 / P-AUPRO 90.8）。
- [x] ReMP-AD 训练（VisA，15 epoch，loss 0.2555）+ MVTec 测试 4/2/1 完成（2026-08-14，seed 10）。
- [x] AdaptCLIP MVTec 完整矩阵（全 15 类，1-shot，seed 0/1/2）完成并导出统一结果（2026-08-17，`outputs/unified/adaptclip_mvtec_seed_{0,1,2}_shot_1/`）。
- [x] ReMP-AD MVTec 统一导出完成（k4/k2/k1；`--prediction_cache_dir` 补丁 + evaluate_unified，`outputs/unified/remp_ad_mvtec_k{4,2,1}/`；图像分数采用官方口径 0.5*(文本概率+few-shot 归一化 max)，与官方 auroc_sp 一致）。

## GPU 执行规则

1. 单 GPU、单任务、串行运行。
2. 已通过配置自动跳过；从类别级 marker 断点恢复。
3. 每个任务退出后先验收，再切换下一组。
4. OOM 只有限重试；连续失败达到阈值后停止并保留证据。
5. PromptAD 队列权威状态文件为 `outputs/logs/promptad_mvtec_resumable_queue/status.json`。
6. 在用户重新确认 GPU 窗口前，不启动剩余长队列。

## 2026-08-11 BTAD 冻结验证（已完成）

- [x] BTAD 完整预测适配器和双分支干跑通过：3 类、741 张测试图、290 张异常图。
- [x] BTAD seed 0 / 1-shot 双分支 Gate A 通过；741 个样本的 ID、标签、图和掩码、数值及 holdout 元数据均通过审计，Gate A 未计算指标。
- [x] BTAD 3-seed × 1/2/4-shot 完整预测矩阵 9/9 完成，配对审计 9/9 通过，失败 0；固定 zero-shot 文本缓存按哈希和不变性声明复用。
- [x] 仅使用 MPDD 已冻结的 `visual_only_safe_fallback` 在 BTAD 运行一次最终评估；采用 200 阈值 AUPRO，共生成 27 行逐类别/seed/shot 结果。
- [x] 明确记录：BTAD 未用于候选比较或参数选择，禁止根据结果回头调参。
- [x] 项目 CPU 测试 49/49 通过。
- [ ] 后续只做结果分析、表格和论文融合；本轮不修改论文。
