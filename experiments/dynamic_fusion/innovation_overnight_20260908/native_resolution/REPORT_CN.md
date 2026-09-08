# 原像素mask评价：薄划痕并非没有信号

六类MPDD正常support，seed0 k2，留一图为query、另一图clean为memory。每族只用预先指定的第0个合成实例，各12个query；没有test图，没有拟合。DINO+CLIP各分支归一化后等权拼接，16网格为32网格平均池化。两路都将预测双线性上采样到原1024 mask计算Pixel-AP，不缩小真值mask。

|合成族|16网格Pixel-AP|32网格Pixel-AP|32−16|
|---|---:|---:|---:|
|cutpaste|0.757421|0.717909|−0.039511|
|local_erasure|0.904036|0.951238|+0.047203|
|thin_scratch|0.090067|0.213655|+0.123587|

**发现**：薄划痕在旧网格真值上不可评，但冻结特征在原像素指标上已有可分信号；32相对16池化保留了更多细划痕信号。这支持重新审视细小缺陷探针，而不支持“所有旧方法失败因为模型无信息”的解释。

**取舍**：32在cutpaste上损失约3.95个百分点。因此不能据此推广为“分辨率越高越好”，不能按真实缺陷类型选择网格。下一步应固定预算比较原图裁块重编码与全图；若进一步研究多尺度选择，需用正常support定义冻结规则。

**边界**：这是合成探索，只有一个seed和每族一个变体序号；support图之间不完全独立。不是冻结A1正式推理的复现（这里使用cosine距离、双线性map，没有主方法全部map后处理），也不是方法改善的真实测试证据。绝对AP不可与历史grid-level AP或论文主表相减。后续检查应扩充预定变体/shot，而不是挑选阳性类别。

复现：`.venv-anomalyclip/Scripts/python.exe scripts/innovation_overnight_20260908/probe_native_resolution.py`

逐图数据、正像素数、面积随机AP参考值及AP/面积比见RESULTS.json。k2/k4的mask可见性审计见相邻resolution_audit/。
