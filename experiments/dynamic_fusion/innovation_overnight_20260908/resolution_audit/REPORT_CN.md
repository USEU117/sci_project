# 分辨率与合成小缺陷可评价性实测

只读取normal support合成mask；没有拟合，没有test图。各shot有重复support，不视为独立样本。

|shot|族|网格|样本|原图非空|旧阈值可见|多数覆盖可见|任意覆盖可见|
|---|---|---|---|---|---|---|---|
|2|cutpaste|16|36|36|36|36|36|
|2|cutpaste|32|36|36|36|36|36|
|2|cutpaste|64|36|36|36|36|36|
|2|local_erasure|16|36|36|34|34|36|
|2|local_erasure|32|36|36|36|36|36|
|2|local_erasure|64|36|36|36|36|36|
|2|thin_scratch|16|36|36|0|0|36|
|2|thin_scratch|32|36|36|0|0|36|
|2|thin_scratch|64|36|36|0|0|36|
|4|cutpaste|16|72|72|72|72|72|
|4|cutpaste|32|72|72|72|72|72|
|4|cutpaste|64|72|72|72|72|72|
|4|local_erasure|16|72|72|70|70|72|
|4|local_erasure|32|72|72|72|72|72|
|4|local_erasure|64|72|72|72|72|72|
|4|thin_scratch|16|72|72|0|0|72|
|4|thin_scratch|32|72|72|0|0|72|
|4|thin_scratch|64|72|72|0|0|72|

结论边界：mask在网格上消失意味着该口径不能检验此类缺陷，不能据此判断提取器是否含有信号。
任意覆盖会改变目标定义，不能直接替换旧指标并宣称提升；后续应在原始/统一像素mask上评价上采样预测，并单列细小缺陷。
创新入口：在匹配前做图像裁块重编码，是否比仅插值预测/特征真正增加可分性。需相同输入预算对照。

复现：`.venv-anomalyclip/Scripts/python.exe scripts/innovation_overnight_20260908/audit_resolution.py`
