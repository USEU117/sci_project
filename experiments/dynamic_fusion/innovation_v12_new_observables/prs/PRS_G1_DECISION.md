# PRS — Perturbation-Response Spectroscopy G1 机制门决策（2026-09-04）

协议：doc 22 §5.3 g1（预注册机制门）；doc 26 §4.2 第一步（2 干预族 × 3 强度，过门才做 GPU 导出）
脚本：`scripts/innovation_v12_new_observables/run_r3_prs_g1.py`（CPU，.venv-patchcore）
数据：既有 NTOF 导出 `outputs/dynamic_fusion/ntof_features_{dino,clip}_s0_k1/{cat}.npz`
      —— 15 fit 变体 = 5 族 × 3 强度，g1 只用 exposure(0.70/1.15/1.40) 与 gamma(0.80/1.20/1.50)，
      两族在图像亮度序上都单调增强；K=1 ref（normal）图，全 normal-only，无 GT/bad/MVTec AD 2。
范围：MPDD development seed0 shot1 × 6 类；dino vitb14→32×32，clip ViT-L/14@336→37×37。
输出：`prs/PRS_G1.json`（逐类 + summary）

## 度量

每 patch 响应 `r(a_j) = ||f(T_{a_j} x) − f(x)||_2`（768-d）；g1 = 对 3 点
Spearman(已知强度序 [0,1,2], r) ≥ 0.6 的 patch 占比 ≥ 0.80（跨 6 类 median，每 branch × family）。

| branch×family | median frac(≥0.6) | 门 ≥0.80 | 方向无关 frac(|corr|≥0.6) |
|---|---:|---:|---:|
| dino exposure | **0.1733** | FAIL | 0.1987 |
| dino gamma | **0.2637** | FAIL | 0.2754 |
| clip exposure | **0.0723** | FAIL | 0.0928 |
| clip gamma | **0.2359** | FAIL | 0.2586 |

逐类全数见 `PRS_G1.json`；每类每 branch 两个 family 均 ≪ 门值（最高单类 0.445，metal_plate dino gamma）。

## 方向无关诊断（排除"轴存在但反号"）

|corr| ≥ 0.6 的比例只比正向比例高 ~0.02–0.03（如 dino exposure 0.173→0.199，
clip exposure 0.072→0.093），即 ~75–93% 的 normal patch 的响应**在任意方向上都非单调**，
不是"单调轴反号"的混淆。门失败不是符号约定造成的。

## 幅度诊断（排除数值退化）

平均 |r| 非零且大（跨类 dino 1.8–11.8、clip 0.98–3.5），但按亮度序呈 **V 形/非单调**：
中间强度（exposure 1.15、gamma 1.20，均最接近恒等变换）响应最小，最强强度（1.40/1.50）响应最大。
例：dino bracket_black exposure [4.69, 2.98, 10.86]、clip [2.29, 1.31, 3.09]；
dino tubes gamma [2.26, 1.92, 3.74]。6 类全部一致（见 PRS_G1.json `*_mean_resp`）。

## 结论与判定：PRS G1 **FAIL → 机制门归档（负）**

- 冻结 DINOv2/CLIP 编码器下，patch 特征位移对全局光度变换**不随亮度序单调**，
  而是以"离恒等变换的距离"为主（近恒等扰动被吸收/抑制，强扰动才穿透）。
  响应谱假说所依赖的"normal support 上的已知强度轴"对这两个族不存在。
- |corr| 诊断证明不是反号轴；幅度诊断证明不是数值退化。全 6 类、双编码器、双族一致 → 结论稳健。
- 按 doc 22 §13 / doc 26 §4.2 "未过即停止"：**PRS 在此机制门归档，不做任何 GPU 阶梯导出、
  不跑合成/真实异常门**，避免浪费算力与违反 pre-registration 纪律。
- 备注（不构成重开）：若把轴重定义为 |强度−恒等| 距离，聚合响应近似单调——但这等价于
  "响应 ∝ 像素级扰动量"的平滑性平凡性质，且重定义门后通过违反预注册纪律；
  任何复活需新 doc + 新机制假说 + 新预注册。
- 下一步：第三条突破线（doc 26 §4.3 融合前空间细节恢复），PRS 与 NTOF/CL-RPF 同为归档负结果。
