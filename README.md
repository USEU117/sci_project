# Few-shot Industrial Anomaly Detection

本仓库用于复现少样本工业异常检测基线，并开发“基于不确定性路由的视觉—语言证据融合”方法。

当前阶段只做基准复现。详细任务、下载入口、命令、协议和验收标准见 [PLAN.md](PLAN.md)；实时进度见 [PROJECT_STATUS.md](PROJECT_STATUS.md)。

核心目标：

- 数据集：MVTec AD、VisA；
- shot：1 / 2 / 4 张正常参考图；
- 基础方法：AnomalyCLIP、WinCLIP+、PatchCore；
- 近期强基线：PromptAD、AnomalyDINO、ReMP-AD、AdaptCLIP；
- 输出：图像级检测、像素级定位、效率统计与可复现实验记录。

