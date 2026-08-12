# 项目状态纠偏与同步报告

更新日期：2026-08-09

## 1. 本次目的

本次工作以磁盘上的统一评估报告、冻结融合NPZ、校准JSON和实际进程为证据，重新整理
项目当前状态。历史失败日志和早期计划均保留，但不再把旧状态当作当前状态。

## 2. 当前运行状态

- 实验训练进程：无。
- GPU：RTX 3060 Laptop 6 GB；审计时约7%利用率、1,853 MiB显存，为Windows
  桌面/后台占用，不是项目训练。
- 正式PromptAD MVTec队列：`paused_by_schedule`。
- 正式队列断点：已完成`s0_k1`、`s0_k2`、`s0_k4`、`s1_k1`；待完成
  `s1_k2`、`s1_k4`、`s2_k1`、`s2_k2`、`s2_k4`。
- `s1_k2`内部断点：5/30个阶段标记；bottle和cable的分类/分割已完成，capsule
  分类已完成，日志停在capsule分割第一次尝试。
- 旧`outputs/logs/overnight_status.json`原来错误显示`running`，现已标记为
  `superseded`并指向正式状态文件。

## 3. 最新完整性矩阵

### VisA

| 方法 | 完成数 | 验收范围 | 状态 |
|---|---:|---|---|
| PatchCore | 9/9 | 12类、2,162样本、3 seeds × 1/2/4-shot | 完成 |
| WinCLIP+ | 9/9 | 同上 | 完成 |
| AnomalyDINO | 9/9 | 同上 | 完成 |
| PromptAD | 9/9 | 同上；`target_normal_tuning=true` | 完成 |
| DynamicFusion | 6/6 | seed 1/2独立最终验证 × 1/2/4-shot | 完成 |

VisA基线新鲜审计结果为36/36运行通过，零错误。

### MVTec

| 方法 | 完成数 | 当前缺口 | 状态 |
|---|---:|---|---|
| PatchCore | 9/9 | 无 | 完成 |
| WinCLIP+ | 9/9 | 无 | 完成 |
| AnomalyDINO | 9/9 | 无 | 完成 |
| PromptAD | 4/9 | s1/k2、s1/k4、s2/k1、s2/k2、s2/k4 | 暂停待续 |
| DynamicFusion | 9/9 | 无 | 完成 |
| AnomalyCLIP | 1个zero-shot结果 | 不属于少样本矩阵 | 单独报告 |
| ReMP-AD | 0 | manifest/NPZ适配、Gate A | 待门控 |
| AdaptCLIP | 0 | checkpoint、Gate A、6 GB显存检查 | 阻塞/待门控 |

每个完整MVTec矩阵运行必须有15类、1,725样本和零schema错误。最新矩阵确认
AnomalyDINO为9/9，旧的8/9记录已作废。

## 4. 动态融合状态纠偏

最终运行的真实分支来源已经从命令和校准JSON确认：

- 视觉分支：AnomalyDINO；
- 文本分支：AnomalyCLIP；
- 图像温度：0.50；
- 像素温度：0.20；
- decision margin：0.15；
- min weight：0.05。

旧配置中的`text_guided_branch: winclip_plus`只对应早期工程对齐/冒烟，不对应冻结最终
验证，现已修正。动态融合最终审计为17/17通过：VisA 8个审计输出（其中6个独立最终
验证、2个seed-0补充复核）和MVTec 9个最终输出。所有校准均为：

```text
status=passed
test_predictions_used=false
test_labels_used=false
```

配置现在允许报告已经审计的结果，但保持`superiority_claim_allowed=false`：当前结果尚不
支持动态融合全面优于最强单分支，必须先完成跨方法和逐类别科学分析。

## 5. 已修正的过期或矛盾记录

1. AnomalyDINO MVTec：8/9改为9/9。
2. PromptAD MVTec：旧“7组运行中”改为4/9完成、5组暂停待续。
3. 后台状态：旧`running`改为`superseded`；正式状态为`paused_by_schedule`。
4. 动态融合文本分支：WinCLIP+改为最终实际使用的AnomalyCLIP。
5. 第二阶段：双温度消融、冻结、VisA最终验证和MVTec最终验证均标记完成。
6. `NEXT_ACTIONS.md`已重建，只保留当前真实待办。
7. `PLAN.md`新增2026-08-09权威断点，早期章节仅作为历史计划。
8. `PROJECT_STATUS.md`增加状态读取规则和当前权威章节。

## 6. 当前真正的剩余任务

1. CPU：动态融合分支差距、逐类别失败、路由权重、消融和可视化分析。
2. GPU：PromptAD MVTec剩余5组。
3. CPU/GPU Gate A：ReMP-AD和AdaptCLIP。
4. CPU：完成方法接纳后重建公平主表、论文图表和结果章节。
5. 工程：整理临时脚本、依赖、测试环境和最终复现入口；不删除历史证据。

## 7. 权威证据优先级

发生冲突时按以下顺序判断：

1. 单次运行的`evaluation_report.json`和`summary.csv`；
2. 2026-08-09完整性矩阵与项目快照；
3. 冻结融合审计和校准JSON；
4. `PROJECT_STATUS.md`最新章节与`NEXT_ACTIONS.md`；
5. 旧日期计划、控制台文字和历史队列状态。

机器可读文件：

- `experiments/summaries/project_state_snapshot_20260809.json`
- `experiments/summaries/current_method_status_20260809.csv`
- `experiments/summaries/visa_result_audit_20260809.csv`
- `experiments/summaries/visa_result_audit_20260809.json`
- `experiments/summaries/mvtec_method_seed_shot_completeness_20260809.csv`
- `experiments/summaries/mvtec_method_seed_shot_completeness_20260809.json`
- `experiments/summaries/mvtec_paper_main_table_template_20260809.csv`

## 8. 验收结论

项目状态同步通过：VisA审计36/36，MVTec方法计数与PromptAD正式队列待办一致，动态
融合17/17审计通过，旧队列已停用，当前没有训练进程。后续应从`NEXT_ACTIONS.md`的
P1动态融合CPU科学分析继续，不应从早期未完成清单恢复任务。

附加工程验收：状态构建脚本语法检查通过，JSON/YAML均可解析，Git差异检查无空白
错误；使用`.venv-anomalyclip`运行动态融合测试，25/25项通过。
