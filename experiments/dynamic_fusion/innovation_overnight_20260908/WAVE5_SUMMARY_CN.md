# 第五轮：真实诊断与存储组合（2026-09-08 04:58 北京）

## 真实诊断改变了优先级
MPDD完整458张test（176 normal、282 anomaly），6类，固定seed0/k2 normal memory，full448与full896共916条配对记录，694秒完成。仅分辨率不同，DINO、距离与448 map后处理相同。全448像素（非stride8）类别宏：

|指标|full448|full896|变化|
|---|---:|---:|---:|
|Pixel-AP|.317636|.310542|−.007094|
|Pixel-AUROC|.950996|.953672|+.002676|
|Image-AUROC|.732330|.754412|+.022082|

合成细划痕改善没有转化为真实总体像素AP改善。bracket_white、connector退化较明显；不能根据这些test结果挑类/调参再作确认。两臂是DINO-only，不是与冻结A1融合比较。Image-level改善可保留为另一个目标的探索入口，尚非独立确认。

stride8：282异常mask中6个完全消失（2.13%），均在bracket_white：painting4/13，scratch2/17。正像素数平均保留1.512%接近每64取1的抽样比例，**不能解读成98.5%缺陷样本消失**。应关注6个完全消失以及指标排名是否受抽样影响。

## 存储组合实测
6类k2、每族首个变体，共24unique masks、3臂72行。full448 FP32 AP .482461、full896 FP32 .561155、full896 memory-only INT8 roundtrip .561235；最后微小正差视为数值扰动，不称精度增益。

- lowres bank：3,145,728 bytes。
- highres FP32：12,582,912 bytes。
- highres INT8含scale：3,148,800 bytes，为lowres 1.000977倍。
- 解码后highres bank仍12,582,912 bytes，解码array峰值15,731,712 bytes。

可把高分辨率持久bank压回接近低分辨率字节预算，不能降低编码成本或据此声称运行memory/延迟相等。该组合只做合成量化一致性验证，不代表它已修复真实pixelAP问题。

## 双尺度扩类
六类：highres AP .5353、lowres bilinear64 .4922、concat .5488、late mean .5470；concat−late仅 .001806。收益主要已被简单late均值体现，不建议以concat为独立机制创新主线。

## 最后阶段
停止GPU方法扩展，只做现有结果的定位/跨图排名分解、独立完整性审计、文件索引与中文总报告。06:45前完成收尾，07:00前释放本次资源并暂停续跑。保留原论文和冻结A1。
