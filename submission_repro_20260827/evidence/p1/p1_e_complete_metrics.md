# P1-E A1 完整指标投稿表

输入为已存在的36份 complete-metrics reports；本步骤只聚合，不重新运行模型。图像分数为 anomaly map max-pool，像素指标 stride=8。

验收：36/36 reports，72 method-config rows，六项指标完整；相对 P0 Pixel-AP 最大绝对差 `3e-06`。

## 按数据集汇总（9配置 mean ± std）

| dataset | role | method | Image AUROC | Image AP | Image F1-max | Pixel AUROC | Pixel AP | Pixel AUPRO |
|---|---|---|---:|---:|---:|---:|---:|---:|
| btad | external_frozen_validation | A1_concat | 0.9254 ± 0.0120 | 0.9185 ± 0.0212 | 0.8706 ± 0.0131 | 0.9735 ± 0.0012 | 0.6455 ± 0.0124 | 0.7608 ± 0.0102 |
| btad | external_frozen_validation | feature_DINO_only | 0.9214 ± 0.0175 | 0.9316 ± 0.0268 | 0.8943 ± 0.0189 | 0.9708 ± 0.0015 | 0.6206 ± 0.0183 | 0.7574 ± 0.0092 |
| mpdd | development | A1_concat | 0.7750 ± 0.0519 | 0.7948 ± 0.0490 | 0.8280 ± 0.0167 | 0.9646 ± 0.0065 | 0.3562 ± 0.0350 | 0.9001 ± 0.0208 |
| mpdd | development | feature_DINO_only | 0.7460 ± 0.0538 | 0.7656 ± 0.0474 | 0.8213 ± 0.0205 | 0.9547 ± 0.0076 | 0.3304 ± 0.0345 | 0.8805 ± 0.0232 |
| mvtec | external_frozen_validation | A1_concat | 0.9583 ± 0.0141 | 0.9777 ± 0.0069 | 0.9638 ± 0.0078 | 0.9663 ± 0.0043 | 0.5546 ± 0.0196 | 0.9199 ± 0.0082 |
| mvtec | external_frozen_validation | feature_DINO_only | 0.9465 ± 0.0172 | 0.9703 ± 0.0092 | 0.9548 ± 0.0098 | 0.9548 ± 0.0057 | 0.5226 ± 0.0184 | 0.9065 ± 0.0097 |
| visa | in_domain_frozen_validation | A1_concat | 0.9046 ± 0.0196 | 0.9103 ± 0.0190 | 0.8710 ± 0.0138 | 0.9757 ± 0.0046 | 0.3725 ± 0.0246 | 0.9125 ± 0.0117 |
| visa | in_domain_frozen_validation | feature_DINO_only | 0.8730 ± 0.0303 | 0.8779 ± 0.0303 | 0.8512 ± 0.0149 | 0.9662 ± 0.0070 | 0.3201 ± 0.0278 | 0.8933 ± 0.0167 |

## A1 − feature-DINO-only（9配置均值之差）

| dataset | Image AUROC | Image AP | Image F1-max | Pixel AUROC | Pixel AP | Pixel AUPRO |
|---|---:|---:|---:|---:|---:|---:|
| mpdd | +0.0290 | +0.0291 | +0.0067 | +0.0099 | +0.0258 | +0.0196 |
| btad | +0.0039 | -0.0131 | -0.0237 | +0.0027 | +0.0249 | +0.0035 |
| visa | +0.0317 | +0.0324 | +0.0198 | +0.0094 | +0.0524 | +0.0192 |
| mvtec | +0.0118 | +0.0074 | +0.0090 | +0.0115 | +0.0320 | +0.0134 |

边界：四数据集稳定正结论针对 Pixel-AP。BTAD 的 Image-AP 与 Image-F1-max 为负，不得写成所有检测/定位指标全面提升。

注意：9配置是同一测试集上的3 seed×3 shot参考采样，不是9个独立数据集。VisA为in-domain frozen validation。