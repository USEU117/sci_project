# 今夜创新探索总报告

日期：2026-09-08。当前版本：待最终独立审计与误差分解补充；全部方法实验已经完成。

## 主要判断

本夜没有找到足以替换冻结A1、或可以直接写成论文新算法贡献的稳定方法。最可靠的可落地优化是**保留全部memory单元的低精度存储**；最值得研究的质量问题是**细节输入、完整上下文与跨图异常排序之间的取舍**。简单提高分辨率在合成缺陷上有效，却没有改善真实总体像素AP，因此不能直接升级主方法。

## 多方向结果

|方向|实测范围与关键结果|建议|
|---|---|---|
|合成mask可评价性|6类k2/k4；thin_scratch在16/32/64多数覆盖网格全部消失，但原图非空|修复后续探针评价，不据不可评推断模型无信息|
|同特征原16/32网格|原像素mask下scratch AP .090→.214；cutpaste .757→.718|保留尺寸/缺陷族取舍证据|
|空间对应+局部NN|6类k2；cutpaste −.256、erasure −.121|此配置归档，不继续扫空间带宽|
|图级参考图覆盖选择|6类k4留图；选2张仅+.0063，12/24 folds胜出|收益不稳定，暂不深挖|
|光度normal memory扩充|6类k2；6倍memory，合成AP+.0013，nuisance-clean AUC .537→.522|弱鲁棒性入口，成本大；不能仅凭分数缩小宣称有效|
|原图低层高频/梯度|2类24mask，AP约.0044，接近面积基线|归档；既有高频方法也无独立证据|
|2×2图像裁块重编码|2类；scratch提升，cutpaste显著损失，总体.507→.478|丢失上下文的具体实现不保留|
|完整896高分辨率输入|6类72合成mask，AP .463→.535；完整真实458图结果见下|有研究信号，不是通用定位升级|
|固定双尺度concat与late对照|6类：concat .5488、late .5470，差.0018|主要收益已有简单均值解释，不作为独立融合创新|
|全memory FP16/INT8存储|6类k2及k4/native32确认：FP16减半，INT8约四分之一，AP几乎不变|优先工程优化；不是编码/检索加速|
|高分辨率×INT8存储组合|6类24mask，高res INT8字节为lowres FP32的1.00098倍，高res AP几乎保持|可研究存储预算下的细节保留；解码后运行memory仍是highres|

不同脚本在特征归一化、map平滑/插值、mask网格和聚合方式上存在差异；表内差值来自各自配对control，绝对AP不可跨行比较。

## 真实数据是决定性限制

真实MPDD开发诊断完整覆盖6类458张图：176正常、282异常。固定manifest seed0/k2 normal support memory，test仅评价，没有拟合或参数选择。两个臂仅DINO输入分辨率不同，使用同距离、同448 map后处理，全448像素计算AP/AUROC。

|类别宏指标|full448|full896|
|---|---:|---:|
|Pixel-AP|0.317636|0.310542|
|Pixel-AUROC|0.950996|0.953672|
|Image-AUROC|0.732330|0.754412|

高分辨率帮助图像级检出，但像素AP略降且有类别退化。它不与A1双分支融合构成同配置比较；这些test数据已有历史开发诊断，不能当作独立确认集。完整逐类、11个defect-type及逐图记录见[real_resolution/STATUS_CN.md](real_resolution/STATUS_CN.md)。

stride8使6/282个真实异常mask完全消失（2.13%），均来自bracket_white。后续小缺陷研究应保留全像素或明确尺寸条件指标。每64取1导致正像素数降至约1/64本身是预期抽样，不应误称98%缺陷消失。

## 推荐深研顺序

1. **先做可靠的存储工程优化。** FP16最稳；INT8需保存per-channel scale、明确解码路径。扩大真实bank的量化一致性检查后，才进入主pipeline。现有结果不支持CPU低精度直接检索加速，也不证明运行内存下降。
2. **将后续算法目标明确到细小缺陷检出或精确定位。** 完整高分辨率的图像AUROC与像素AP方向不同，不能混为一个目标。后续应检验固定预算的区域重编码、上下文保留和跨图校准；选区/阈值只用合法normal开发数据，避免按test缺陷类型路由。
3. **先完善小缺陷评价，再训练新模块。** 原像素mask、面积分层、正常图高尾与跨图pooled指标同时报告；合成缺陷不能仅靠网格二值mask筛选。重新设计更接近真实细结构的合成任务需要新协议，不能把旧测试结果当调参集。
4. **以简单控制作为创新门。** 新的尺度机制必须明显超过固定late均值，并说明相同计算预算下的优势；单纯加分辨率、裁块、INT8、等权concat本身不是新颖性。

## 文献与新颖性边界

[HiAD](https://arxiv.org/abs/2508.12931)已研究高分辨率多尺度检测与计算资源分配；[Learning to Be a Transformer to Pinpoint Anomalies](https://arxiv.org/abs/2407.04092)讨论高分辨率与尺寸稳健评价。因此本夜得到的是本项目的可行性/失败边界，而非这些操作本身的原创性。正常域偏移方向还需对照[SPARC](https://arxiv.org/abs/2608.18585)。本夜只做一手摘要初筛，未复现上述论文。

## 复现与证据入口

- 本夜脚本：项目根目录`scripts/innovation_overnight_20260908/`，使用现有`.venv-anomalyclip/Scripts/python.exe`。
- 各实验目录包含PROTOCOL、RESULTS、中文结论和命令；大feature/真实逐图JSON存于项目根目录`outputs/dynamic_fusion/overnight_20260908_*`。
- 阶段汇总：[第一轮](WAVE1_SUMMARY_CN.md)、[第二轮](WAVE2_SUMMARY_CN.md)、[第三轮](WAVE3_SUMMARY_CN.md)、[第四轮](WAVE4_SUMMARY_CN.md)、[第五轮](WAVE5_SUMMARY_CN.md)。
- 历史与指标审计：[PRIOR_AUDIT_CN.md](PRIOR_AUDIT_CN.md)、[WAVE2_AUDIT_CN.md](WAVE2_AUDIT_CN.md)、[WAVE3_AUDIT_CN.md](WAVE3_AUDIT_CN.md)、[WAVE4_AUDIT_CN.md](WAVE4_AUDIT_CN.md)。

所有新实验独立存档，未覆盖冻结A1、历史原始结果或论文主表。负结果、真实退化、未验证的新颖性和计算成本均保留。
