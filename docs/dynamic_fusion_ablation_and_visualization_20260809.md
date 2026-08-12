# 动态融合消融与可视化材料说明

日期：2026-08-09  
状态：已完成

## 1. 已整理的消融内容

消融表覆盖 VisA seed 0 的 K=1/2/4，包括：

- 校准后视觉单分支；
- 校准后文本单分支；
- 固定视觉权重 0、0.25、0.50、0.75、1.00；
- 单温度动态路由 0.20；
- 单温度动态路由 0.50（K=2/K=4）；
- 图像温度 0.50、像素温度 0.20 的冻结双温度方案。

评价指标完整包含 Image AUROC、Image AP、Image F1、Pixel AUROC、Pixel AP 和 AUPRO。

表格文件：`experiments/summaries/dynamic_fusion_scientific_analysis_20260809/ablation_summary.csv`。

## 2. 图表清单和用途

1. `calibration_saturation_diagnostic.png`
   - 左图展示校准分数达到 0.999 以上的比例。
   - 右图展示校准前后 Image AUROC 的变化。
   - 用于论文解释最主要的失败原因。

2. `mvtec_visual_vs_dynamic_by_shot.png`
   - 比较 MVTec 三个 shot 下原始 AnomalyDINO 和动态融合的 Image AUROC。
   - 用于客观展示总体差距。

3. `mvtec_category_image_auroc_delta_heatmap.png`
   - 展示每个类别、每个 shot 的动态融合减去原始视觉分支的差值。
   - 用于寻找 carpet、cable、grid 等主要失败类别。

4. `route_weight_summary.png`
   - 展示图像视觉权重、像素视觉权重，以及正常/异常图像权重差异。
   - 用于证明图像任务和像素任务不能用同一权重解释。

5. `visa_ablation_split_temperature.png`
   - 展示固定权重、单温度和双温度的 Image AUROC 与 AUPRO。
   - 用于论文消融部分说明双温度的作用。

6. `mvtec_success_failure_cases.png`
   - 每行包含原图、真值、AnomalyDINO、AnomalyCLIP、动态融合和视觉像素权重。
   - 前三行为定位改善案例，后三行为定位退化案例。

7. `selected_cases.csv`
   - 记录案例编号、类别、对比度变化、图像分数和视觉权重。
   - 用于追溯图片选择规则，避免人工只挑好看的结果。

## 3. 使用时的注意事项

- 所有成功/失败案例按“融合后异常区域对比度减去视觉分支对比度”自动选择，不是手工挑图。
- 可视化来自 MVTec seed 0、K=4 冻结结果，只用于机制解释。
- 图中热点颜色只表示同一张热图内部的相对响应，不应把不同模型的颜色亮度直接当作绝对分数比较。
- 论文正文建议放校准诊断图、双温度消融图和成功/失败案例图；完整类别热图和路由统计可放补充材料。

## 4. 工作簿内容

Excel 工作簿包含 9 个工作表：Dashboard、README、Run Comparison、Category Diagnostics、Route Statistics、Ablation、Cases、Provenance 和 Input SHA256。

它既可以直接筛选，也保留了运行级、类别级、案例级数据和 285 个唯一输入文件的 SHA256，适合后续写论文表格、重新排序类别或核对某一张可视化的来源。

## 5. 当前完成边界

今天要求的两项工作已经完成：

- 动态融合科学分析：完成；
- 消融与可视化材料：完成。

尚未做的是论文初稿、效率表中的正式 GPU 显存/推理耗时测量，以及任何 V2 新算法实验。GPU 队列也没有启动或修改。
