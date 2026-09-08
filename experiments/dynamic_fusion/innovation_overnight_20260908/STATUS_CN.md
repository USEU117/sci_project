# 夜间探索运行状态

状态更新：夜间实验已结束并交接。简明交接与成果见 `D:/STUDY/My_github/sci_project/docs/OVERNIGHT_HANDOFF_20260908_CN.md`。

所有已汇报的方法探针完成；最后追加的FINAL_AUDIT_CN.md与resolution_gap分析因子代理额度耗尽未完成，不能视为已有产物。保持唤醒于07:00自动释放，heartbeat 7现已暂停。下文为04:58历史运行记录，其“最终阶段”是当时的计划，不代表已完成。

已完成：
- 初始计划、历史文件读取、第一轮文献新颖性排查。
- 分辨率审计：6类k2/k4，thin_scratch原图mask均非空，但16/32/64多数覆盖网格全部消失；见resolution_audit/REPORT_CN.md。这是已知不可评现象的量化扩展，不是模型改进。
- 原像素mask探针完成：native_resolution/REPORT_CN.md；16→32网格thin_scratch AP .0901→.2137，但cutpaste .7574→.7179。存在分辨率取舍，不可简单称高分辨率普遍更好。
- 同对话heartbeat ID=7建立，每30分钟继续，截止07:00；没有新建用户任务。
- 临时系统保持唤醒helper PID=19032，允许显示器关闭，07:00自动释放；脚本keep_awake.ps1。不修改全局电源设置。

首轮已全部完成，详见WAVE1_SUMMARY_CN.md。PRIOR_AUDIT_CN.md、precision/、spatial/、tiling/均已有实测结论。GPU当前无任务。

第二轮已完成：support_selection/无稳健胜出；shift_memory/原始分数偏移略降但6倍memory，root要求复核尺度混淆；tiling/AUDIT_CN.md更正FP阈值与独立mask计数。详见WAVE2_SUMMARY_CN.md。

第三轮已完成，见WAVE3_SUMMARY_CN.md。主要质量信号：full896在2类相对full448宏AP+ .053995；工程确认：precision_native/k4全3族FP16/INT8存储压缩过门；高频raw-only负结果。

第四轮已完成，见WAVE4_SUMMARY_CN.md。6类full896合成AP+ .072160，所有类scratch改善；双尺度2类concat只略优late mean，需诚实保留控制。

第五轮已完成，见WAVE5_SUMMARY_CN.md。真实458图全部配对：高res pixelAP略降、imageAUROC上升；存储组合字节预算近似lowres但运行成本仍高；concat≈late。停止新增GPU/方法探索，不按test结果调参。

最终阶段：
- audit_routes（Plato）：FINAL_AUDIT_CN.md，只读真实完整性/指标与字节核验。
- probe_tiling（Pascal）：resolution_gap/现有逐图记录的定位/跨图排名分解，CPU，无重提取。
- probe_precision（Mill）、probe_spatial（Parfit）：空闲，已完成。
- root：总报告、索引、停止状态和自动化收尾。07:00前停所有本次重计算。

接续规则：先检查现有agent/实验是否仍活跃和文件是否更新，不重复启动。历史t3文档k4数值据JSON证据更支持误植；详见PRIOR_AUDIT_CN.md。候选只能比较本脚本同配置baseline，不能跨表拼接。初步最有希望的是存储压缩（工程）和细节输入（有取舍）；不重新激活旧动态分数加权/通道重标定。

07:00停止计算后应更新FINAL_REPORT_CN.md，列每方向证据等级（实测/待测/归档）、复现命令、局限和深研优先级，并将heartbeat 7设PAUSED。勿将当前计划或未完成实验当成结论。
