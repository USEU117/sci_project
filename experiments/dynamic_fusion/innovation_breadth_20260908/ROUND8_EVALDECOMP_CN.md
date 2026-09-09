# ROUND8 评估-诊断（2026-09-09 夜）— 缺陷尺寸轴分解：关闭"尺度/规模"假设

性质：**分析而非门测机制**（同 R3 归因 / D6 defect-type 诊断谱系）。冻结 A1 控制 map（448，parity 精确重放），
GT 448 mask 8-连通域按面积分桶 [1-9][10-31][32-99][100-315][316-999][1000+]，
每桶 Pixel-AP/AUROC 用（桶内异常像素 ∪ 该类全部正常像素）计算，桶间可比；
另报全局全分辨率 AP/AUROC 与图像级 AUROC。脚本 `analyze_evaldecomp.py`，结果 `round8_evaldecomp/EVALDECOMP_s0_k{2,4}.json`。

## 逐类关键读数（图像 AUROC imAUC / 全局 Pixel-AUROC glAUC / 主导尺寸桶）

| 类 | k2 imAUC | k4 imAUC | 缺陷连通域构成 | 逐桶 Pixel-AUROC | 解读 |
|---|---|---|---|---|---|
| metal_plate | 1.00 | 1.00 | 89/104 个 ≥1000px（中位 2.1 万 px） | 0.975–0.995 | 大缺陷饱和，非尺度问题 |
| tubes | 0.966 | 0.963 | 80/82 个 ≥1000px | 0.925–0.991 | 大缺陷饱和 |
| connector | 0.793 | 0.898 | 14 个全部 ≥1000px（中位 ~3600px） | 0.969–0.975 | 大缺陷主导；k2→k4 图像级提升 |
| bracket_white | 0.827 | 0.824 | 全在 ≤315px（52 个 ≤9px） | 0.983–0.997 | 桶 AP 低是**正像素稀疏稀释**（163–2167px vs ~15M 正常池），排序质量极好 |
| bracket_black | **0.461** | 0.797 | 97 个，10–999px 混合 | 0.92–0.99 | k2 图像级低于随机但像素 AUROC 好 → **k2 两图采样不足**；k4 全局 AP 0.0125→0.157（12.6×） |
| bracket_brown | 0.571 | 0.573 | 跨全尺寸，含 17 个 ≥1000px（5 万 px） | 大桶仅 **0.886–0.889** | **唯一跨尺寸一致的排序缺陷类**：异常区域视觉正常（上下文/错位类，D6 同源） |

## 三个结论

1. **"缺陷太小/需要更粗匹配"的假设不成立**（定量关闭）：强类（metal/tubes/connector）本就以大缺陷为主且像素级饱和；
   bracket_white/black 的小缺陷像素 AUROC 均在 0.92–0.997（排序很好），其 AP 低主要来自 **AP 指标在高正负不平衡下的稀释**
   （几十~几千正像素对 ~千万正常像素），并非排错。这与 R7 MRS 粗匹配深负互为印证：尺度轴上没有可供机制恢复的缺口。
2. **bracket_black 的 k2 困境是采样（sampling）现象**：k2 图像 AUROC 0.461（<0.5 随机）vs k4 0.797、全局 AP 12.6×跳升——
   两图参考无法覆盖该类正常纹理变异性；离线无任何机制能补 k2 采样。对论文：该类的 k2 数字应报告为**支持集采样敏感性**而非方法缺陷，
   并建议 k2 上按"参考多样性/多 seed"注明。
3. **bracket_brown 是唯一真实的表示（representation）缺口**：跨全尺寸、两 shot 均像素/图像级排序不足（大桶 AUROC ~0.89、imAUC ~0.57），
   与其 parts_mismatch/错位缺陷（异常区视觉正常）一致 → 唯一可行的未来方向仍是**跨图像结构先验/对齐/监督或训练**（GPU/授权候选），
   单图全局距离的 28 个机制族（含本轮的匹配尺度/通道/密度）均无法触碰它。

## 文件
- 脚本：`scripts/innovation_breadth_20260908/analyze_evaldecomp.py`
- 结果：`experiments/dynamic_fusion/innovation_breadth_20260908/round8_evaldecomp/EVALDECOMP_s0_k2.json` / `EVALDECOMP_s0_k4.json`
