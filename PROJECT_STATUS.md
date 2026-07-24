# 项目状态

更新日期：2026-07-24

## 已确认

- 项目根目录初始为空，未发现既有代码或 Git 历史。
- Python 3.10.11 可用；未安装 Conda。
- GPU 为 RTX 3060 Laptop 6 GB；驱动 591.86。
- 第一阶段范围与老师要求已写入 `PLAN.md`。
- 已核验 AnomalyCLIP、PatchCore、PromptAD、AnomalyDINO、ReMP-AD、AdaptCLIP 的代码来源。
- MVTec AD 官方下载需要接受/填写许可信息；VisA 可从 AWS Open Data 匿名获取。

## 进行中

- AnomalyCLIP 官方仓库下载。

## 尚未开始

- 数据集下载与校验；
- 各方法独立虚拟环境；
- checkpoint 下载；
- bottle 冒烟测试；
- 全类别和统一 1/2/4-shot 实验。

## 当前阻塞与风险

- MVTec AD 需要用户在官方网页提交许可表单后取得下载。
- 6 GB 显存不足以照搬部分官方训练配置；先做预训练 checkpoint 推理。
- Windows 原生 FAISS GPU 支持不稳定；PatchCore 与 AnomalyDINO 首轮使用 CPU FAISS。

## 下一动作

1. 完成 AnomalyCLIP 源码下载并记录 commit。
2. 审查其 README、依赖、测试脚本和权重入口。
3. 建立 `.venv-anomalyclip`。
4. 获取数据并运行 bottle 冒烟。

