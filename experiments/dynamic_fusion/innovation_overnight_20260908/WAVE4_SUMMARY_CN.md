# 第四轮汇总（2026-09-08 03:57 北京）

## 高分辨率扩类结果
新增四类、48个unique synthetic masks、同脚本fresh full448/full896：平均AP .441014→.522257（+ .081243）。四类thin_scratch分别+ .136177 / + .090636 / + .236097 / + .180818；cutpaste分别− .007590 / − .022858 / − .008313 / + .044976。连同旧两类共6类72mask，.463093→.535253（+ .072160）。这是support-synthetic结果；尚未说明真实test效果，也未与A1双分支融合公平比较。

编码成本：本机正式full448约76.85ms、full896约368.52ms，约4.80倍；内存/磁盘bank行数为4倍。不可只报精度而忽略成本。

## 固定双尺度匹配
首2类24mask、同64网格与原mask后处理：highres .5612、lowres bilinear feature .5261、等权feature concat .5854、late score mean .5817。concat较最佳单支+ .0241，但较简单late mean仅约+ .0037。故不能忽略late控制并宣称新特征融合创新；小信号待其余4类验证。

## 第五轮（已派发）
- 冻结full448/full896的6类真实MPDD test诊断：normal support memory固定k2，不拟合选参；同后处理，优先全448像素AP/AUROC与image-AUROC，逐类/type。只与DINO lowres比较，不声称胜过A1融合。
- 双尺度同参数扩其余4类，复用features无GPU重提取。
- 分辨率×存储组合：full896 bank INT8是否以接近full448 FP32的持久化字节保住highres收益。解码内存/编码时间仍变大，不能称整体等成本。
- 独立协议与指标对齐审计。

每项均有界，06:30后不新增重计算，07:00停止。剩余时间优先确认当前最有价值入口，不再对已失败空间/高频/图选择机制扫参。
