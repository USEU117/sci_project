# MVTec 结果范围与论文主表模板

本文件由 `scripts/build_mvtec_completeness_and_paper_template.py` 的机器可读矩阵支撑。

## 正式主表规则

一个方法在某个 K 值下只有 seed 0/1/2 三个配置都通过 15 类、1,725 测试样本、零
schema 错误的统一评测后，才计算 `mean ± std`，并标为 `paper_ready=yes`。
缺失配置显示 `-- (n=x/3)`，不得以不完整均值比较排名。

## 当前范围

- PatchCore、WinCLIP+、AnomalyDINO、DynamicFusion：按完整性矩阵决定是否进入主表。
  AnomalyDINO 的原缓存目录混有 VisA 与 MVTec，因此必须使用只含 15 个 MVTec
  类别的 CPU 重评测结果，不能复用旧的 27 类汇总。
- PromptAD：其 MVTec 队列已暂停到 GPU 窗口，当前不完整的 K 值不进入主表。
- AnomalyCLIP 当前只具备 zero-shot 输出，不写入少样本横向主表。
- ReMP-AD 与 AdaptCLIP 尚未通过 Gate A，不写入主表。

机器可读文件：

- `experiments/summaries/mvtec_method_seed_shot_completeness_20260808.csv`
- `experiments/summaries/mvtec_paper_main_table_template_20260808.csv`
