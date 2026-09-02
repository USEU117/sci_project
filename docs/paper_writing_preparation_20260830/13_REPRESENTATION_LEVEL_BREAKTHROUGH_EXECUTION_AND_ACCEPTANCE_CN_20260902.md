# 表征层算法突破执行与验收任务书（A3 Breakthrough Program）

版本：2026-09-02  
执行对象：下一位负责修复审计、算法实现、实验运行和结果归档的 AI 助手  
任务性质：A2 负结果后的独立新计划；不是继续给 A2 追加参数  
当前状态：**仅完成方案设计，A3 尚未实现、尚未产生结果，任何候选都不能写成论文贡献。**

---

## 0. 给执行 AI 的直接结论与指令

A2 已完成 27 个预注册候选的 MPDD seed0 × shot {1,2,4} 小门，但没有产生可升级方法：

- LNDC、DSAM、NCPRA、FAGR 明显或稳定低于 A1；
- CEQA 最佳三-shot mean ΔPixel-AP 为 `+0.002799`，没有超过 `+0.003` 小门，而且没有胜过机制控制；
- DEVA 变化约为零；
- 因此 A1 目前仍是唯一可用于论文的冻结方法。

但不得把 A2 写成“六条科学假设全部被彻底证伪”。独立复核发现 DEVA 和 NCPRA 存在实现有效性问题，见第 2 节。先完成一次**限定范围的纠错复核**，然后把主要研发资源转移到两个真正改变信息入口的方向：

1. **主路线 G：CASF（Complementarity-Aware Synthetic Anomaly Fusion）**  
   从 DINOv2 与 AnomalyCLIP 的多个中间层提取特征，只用正常参考图构造受控伪异常，训练一个小型、确定性的双分支融合评分头。核心不是“再做动态权重”，而是用跨分支非对称伪异常显式监督何时、在哪个尺度利用两种表征。
2. **独立备选路线 H：DC-SZoom（Dual-Cue Sparse Zoom Memory）**  
   用全局双分支异常证据提出少量候选窗口，在原图上重新提取高分辨率局部特征，并与相同归一化位置的正常参考窗口建立局部记忆。目标是补回全图缩放时丢失的小缺陷，而不是继续改现有 32×32 score map 的后处理。

正确流程：

```text
A2 D/E 限定纠错（只回答旧结论是否有效）
    ↓
多层特征导出与身份回归
    ↓
路线 G、H 分别做 MPDD 小门（不得组合）
    ↓ 只有单路线通过才进入完整 MPDD
3 seeds × 3 shots 完整 MPDD + 训练随机性复验
    ↓ 最多选择一个 winner
冻结算法、配置和代码哈希
    ↓
BTAD + MVTec AD + VisA 一次性验证
    ↓
通过才允许修改论文 Method/Title/Contributions
```

如果 G/H 都失败，停止 A3，保留 A1 论文。不得在外部验证集上救参数，不得临时堆叠失败模块。

---

## 1. 为什么必须换到“表征层/监督信号层”突破

### 1.1 A2 已经回答了什么

| 路线 | 最佳 MPDD 小门 mean ΔPixel-AP | 当前能下的结论 |
|---|---:|---|
| LNDC | −0.085044 | 对现有末层 A1 距离做局部密度除法明显有害 |
| DSAM | −0.011989 | 强制局部位置窗口弱于 A1 全局 KNN；窗口放大后只是回到 A1 |
| CEQA | +0.002799 | query shift 有极小正效应，但没有证明双编码器共识机制优于简单 A1-rank 控制 |
| DEVA | +0.000024 | 当前实现未产生有效变化；另有几何实现错误，结论只能暂记为 provisional |
| NCPRA | −0.005215 | 当前训练实现和残差公式未胜出；早停权重与种子问题使其不能作为严格复现结论 |
| FAGR | −0.005183 | 在末层 score map 上做亲和平滑与普通平滑近似，且轻微退化 |

这些路线共同特点是：绝大部分工作发生在**已经导出的最深层 768+768 特征之后**。结果说明当前 A1 的末层拼接和全局 KNN 已接近该固定表征下的局部最优；继续改距离公式、平滑或密度，容易只得到千分位变化。

### 1.2 新计划改变的变量

A3 不再把主要创新放在末端 score engineering，而是改变两个上游变量：

- **可见信息**：增加早/中/深层特征和高分辨率局部视图，使微小纹理、边缘与结构异常不在进入 KNN 前就消失；
- **学习信号**：用正常样本生成有 mask 的、跨分支非对称伪异常，让小型融合头得到明确监督，同时仍不读取真实测试异常标签。

近邻工作已说明这两个方向具有合理依据，但不能据此声称本项目方法必然有效：DCP-SFR 强调深层传播中的 defect cue fading；RadioCore 使用多尺度 foundation features；RealNet 和 SimpleNet 说明合成异常/特征异常可用于训练；高分辨率 tiled ensemble 说明全图缩放会损害小缺陷；FastRef 说明 query-aware prototype refinement 是强近邻路线。A3 必须通过自己的消融证明“跨编码器非对称伪异常”或“双线索稀疏放大”的增量，而不能把多层、合成异常或切块本身写成首次提出。

主要近邻来源：

- [DCP-SFR, CVPR 2026](https://openaccess.thecvf.com/content/CVPR2026/html/Jiang_Defect_Cue-Preserved_Structural_Feature_Refinement_for_Few-Shot_Anomaly_Detection_CVPR_2026_paper.html)
- [RadioCore, CVPRW 2026](https://openaccess.thecvf.com/content/CVPR2026W/VISION26/html/Ali_RadioCore_Few-Shot_Industrial_Anomaly_Segmentation_with_Multi-Scale_Radio_ViT_Features_CVPRW_2026_paper.html)
- [RealNet, CVPR 2024](https://openaccess.thecvf.com/content/CVPR2024/html/Zhang_RealNet_A_Feature_Selection_Network_with_Realistic_Synthetic_Anomaly_for_CVPR_2024_paper.html)
- [High-Resolution Tiled Ensemble, CVPRW 2024](https://openaccess.thecvf.com/content/CVPR2024W/VAND/html/Rolih_Divide_and_Conquer_High-Resolution_Industrial_Anomaly_Detection_via_Memory_Efficient_CVPRW_2024_paper.html)
- [FastRef, CVPR 2026](https://openaccess.thecvf.com/content/CVPR2026/html/Li_FastRef_Fast_Prototype_Refinement_for_Few-shot_Industrial_Anomaly_Detection_CVPR_2026_paper.html)
- [AnomalyDINO, WACV 2025](https://openaccess.thecvf.com/content/WACV2025/html/Damm_AnomalyDINO_Boosting_Patch-Based_Few-Shot_Anomaly_Detection_with_DINOv2_WACV_2025_paper.html)

---

## 2. Wave 0：先修正 A2 的两个有效性问题

这一波不是给 A2 增加新候选，只是确认旧报告是否由实现错误造成。所有纠错报告写入新目录：

`experiments/dynamic_fusion/innovation_v2_post_audit_20260902/`

不得覆盖 `experiments/dynamic_fusion/innovation_v2/` 的历史结果。

### 2.1 DEVA：几何变换与反向映射有实证错误

涉及文件：

- `src/industrial_ad/innovation_v2/equivariant_augmentation.py`
- `scripts/innovation_v2/export_deva_references.py`

已复核到的事实：

1. `apply_transform()` 只要 `contrast is not None` 就直接返回光度结果，因此 `combined` 实际等于 photometric，没有执行平移或旋转；
2. `Transform.M_at()` 在有旋转时会覆盖先前平移矩阵，不是旋转和平移的组合；
3. 导出脚本把 `invertAffineTransform(M)` 传给 `inverse_warp_features()`。合成 impulse 测试显示当前方向把 `(6,7)` 的峰进一步移到 `(8,11)`，而传入 forward `M` 才能恢复到 `(6,7)`，有效区域 MAE 为 0。

执行要求：

- 明确统一坐标语义：变量名必须区分 `M_src_to_aug` 与 `map_original_to_aug`；
- 图像增强先做几何 warp，再做光度变换；combined 必须两者都执行；
- rotation、translation 用 3×3 齐次矩阵组合，最后取 2×3；
- inverse feature warp 对每个 original-grid 输出点采样对应的 augmented-grid 坐标；
- 新增 impulse、棋盘格、identity、translation、rotation、combined 五类 round-trip 单测；
- identity 最大绝对误差 `<1e-6`；整数平移内部有效区误差 `<1e-6`；双线性旋转有效区 MAE `<0.03`；
- 修复后只重导 MPDD seed0 shot {1,2,4} 的 geometry 和 combined；photometric 历史结果可保留；
- 旧 cache 与 marker 不得复用，报告记录新 code/config/cache SHA256。

纠错后的 DEVA 仍按原 A2 小门判断，不扩大 tau、变换包或候选预算。若仍失败，标记 `CORRECTED_ARCHIVE`；若意外通过，只能进入原 A2 Full MPDD，不得与 A3 混合。

### 2.2 NCPRA：最佳权重没有冻结，初始化种子未真正固定

涉及文件：

- `src/industrial_ad/innovation_v2/predictive_adapter.py`

已复核到的事实：

- `best = (g_d2c.state_dict(), g_c2d.state_dict())` 保存的是共享底层 storage 的 tensor 引用；后续 epoch 更新模型时，“best”也一起变化。因此最后加载的不是报告所称的最佳 epoch；
- `rng_valid` 固定只控制 NumPy view，函数内部没有在模型初始化前固定 Python/NumPy/PyTorch/CUDA seeds；
- 现有单测 42/42 通过，但没有覆盖这两个科研复现条件。

执行要求：

- 使用 `copy.deepcopy(model.state_dict())` 保存最佳权重；
- 每次 candidate/category/shot 训练前从稳定哈希生成 seed，并固定 `random`、NumPy、PyTorch、CUDA；
- 打开 deterministic 算法；若某算子不支持，必须报错，不得静默退回非确定模式；
- 保存 `init_state_sha256`、`best_state_sha256`、`last_state_sha256`；除非最佳 epoch 恰为最后一轮，否则 best 与 last 必须不同；
- 同一命令连续运行两次，逐 patch score 最大绝对误差 `<1e-7`；
- 重跑原 4 个候选，不新增 r、lambda、epoch 或学习率；
- 仍按原 A2 小门判断并单独归档。

### 2.3 Wave 0 验收文件

必须提交：

- `POST_AUDIT_REPORT.md`
- `POST_AUDIT_REPORT.json`
- `deva_roundtrip_test_report.json`
- `ncpra_repeatability_report.json`
- 修复前/后的有限结果对照表；
- `pytest tests/innovation_v2 -q` 与新增 post-audit tests 的日志；
- 明确写出 D/E 是 `CORRECTED_ARCHIVE`、`PROMOTE_TO_ORIGINAL_FULL_GATE` 或 `BLOCKED`。

Wave 0 不能用 BTAD、MVTec、VisA。

---

## 3. A3 共用协议与目录

### 3.1 保护边界

不得修改：

- `submission_repro_20260827/`
- `experiments/dynamic_fusion/freeze/`
- A1、RCEC、A2 历史报告与特征 cache
- 四数据集既有 A1 主表

新内容只能放到：

```text
configs/innovation_v3/
src/industrial_ad/innovation_v3/
scripts/innovation_v3/
tests/innovation_v3/
outputs/dynamic_fusion/innovation_v3/          # 大缓存，gitignored
experiments/dynamic_fusion/innovation_v3/      # 小报告、表格和证据，可提交
```

### 3.2 数据集角色

| 数据集 | 角色 | 允许行为 |
|---|---|---|
| MPDD | development | 允许选择 G/H、候选和固定阈值 |
| BTAD | external frozen validation | 最终唯一 winner 一次性运行 |
| MVTec AD | external frozen validation | 最终唯一 winner 一次性运行 |
| VisA | in-domain frozen validation | 最终唯一 winner 一次性运行，并披露 AnomalyCLIP checkpoint 域关系 |

方法训练、早停、候选窗口、伪异常难度和归一化不得读取任何真实测试 label、mask、异常比例或全测试集统计。evaluator 可在 score 完全写盘后读取标签计算指标。

### 3.3 主指标和稳定性

- 主要指标：dataset-level pooled Pixel-AP；
- 同时强制报告 Pixel-AUROC、Pixel-AUPRO、Image-AUROC、Image-AP、Image-F1-max；
- 所有 trainable 方法至少运行 3 个独立训练种子；
- 论文表格报告训练种子均值；配置正负判断以 3 个训练种子均值为准；
- 必须报告训练种子标准差和最差训练种子，不得只挑最好 checkpoint；
- 早停只看正常样本伪任务验证损失，不看真实异常指标。

### 3.4 泄漏硬门

每份报告必须包含并全部为 `false`：

```json
{
  "test_labels_used_by_method": false,
  "test_masks_used_by_method": false,
  "test_distribution_used_for_calibration": false,
  "validation_dataset_used_for_tuning": false,
  "category_specific_test_rules_used": false,
  "real_anomaly_used_for_training": false
}
```

训练进程不得 import evaluator 的 GT/mask loader。建议通过不同进程和不同路径权限实现物理隔离，而不是只写布尔声明。

---

## 4. Wave 1：多层特征导出与基础可行性探针

### 4.1 导出内容

当前 AnomalyCLIP 导出器已经请求 `[6,12,18,24]`，但只保存最后一层。A3 必须保存全部请求层。DINOv2 保存三个预注册层，优先使用 backbone 总 block 数的 `{1/2, 3/4, 1}` 对应层；开始前打印实际 block 数并在配置中冻结具体 index，不能实验后换层。

每层必须保存：

- patch feature `[N,H,W,D]`；
- `sample_ids`、`ref_ids`、grid、input size；
- model/checkpoint/preprocess/code SHA256；
- 每 patch L2 norm 统计；
- 对 A1 deepest layer 的逐值回归误差。

不允许简单把所有层 raw concat 成超大向量后宣称创新。各层先单独 L2，独立构建 memory 和参考 LOO 标准化。

### 4.2 身份回归

- DINO deepest 与现有 A1 DINO cache 对齐后 max abs error `<1e-5`；
- CLIP deepest 与现有 A1 CLIP cache 对齐后 max abs error `<1e-5`；
- `sample_ids/ref_ids` 完全一致；
- 所有层无 NaN/Inf；
- clean normal reference patch 的 LOO 分数可复现；
- 对 CLIP 37×37→DINO 32×32 的 resize 只实现一个权威函数，并继承 A1 对齐测试。

### 4.3 只用于理解的 layer probe

在 MPDD seed0 × shot {1,2,4} 上报告各单层 DINO-only、CLIP-only Pixel-AP，以及固定等权的同尺度双分支分数。layer probe 不参与挑选任意层组合；组合已在配置里预注册。

进入主路线 G 的最低条件：至少一个非 deepest 层在 18 个 `category×shot` 单元中有 6 个以上超过对应 deepest 单层，或者其与 deepest 的逐 patch error/rank 相关均值 `<0.95`。否则说明多层没有足够非冗余信息，G 只运行 deepest-only smoke，不投入完整训练。

---

## 5. 主路线 G：CASF——互补性感知合成异常融合

### 5.1 科学假设

A1 证明两个编码器整体互补，但无监督动态 router 和 CEQA 没有证明能够识别“何时该相信哪一个”。根本缺口是缺少局部监督。

CASF 的假设是：**在正常 patch manifold 附近构造难度受控、分支非对称的伪异常，可以教会一个极小评分头识别“某一分支或某一尺度率先偏离正常”的模式；这种训练信号比无监督可靠性公式更能转化为真实缺陷增益。**

### 5.2 输入统计量

对每个 patch、每个预注册层 `l`、分支 `b∈{D,C}`：

1. 计算到该层正常 memory 的 1-NN cosine distance `d_b^l`；
2. 由正常 reference LOO 的 median/MAD 得到 `z_b^l`，裁剪到 `[-5,10]`；
3. 计算同尺度双分支差异 `a^l = |z_D^l-z_C^l|`；
4. 保留冻结 A1 score `s_A1`；
5. 可选 3×3 局部稳健对比 `c_b^l = z_b^l - median_3x3(z_b^l)`。

最终输入通道只由这些可解释统计量组成，不把 768 维 raw token 直接交给大 MLP。这样限制参数量并降低对具体类别纹理的记忆。

### 5.3 伪异常生成器

只从 normal reference feature maps 生成。每次复制一个 clean map，采样连续 mask `B`，并在 mask 内执行以下预注册家族：

1. **Tangent noise**：对单位特征加入与原向量正交的噪声后重新 L2；
2. **Distant transplant**：用同类别、不同参考图或距离当前位置足够远的正常 patch block 替换，制造结构错位；
3. **Cross-layer break**：只替换早层或深层，制造尺度不一致；
4. **Cross-branch asymmetric break**：只扰动 DINO、只扰动 CLIP、同时扰动，比例固定为 `1:1:1`。

mask 采用矩形与不规则连通块各 50%，面积覆盖 patch map 的 `[0.5%, 15%]`，按 log-uniform 采样。异常强度不按真实测试指标选择；用 reference LOO 分位数控制，使伪异常的 A1 score 主要落在 clean LOO 的 `P70–P99.5`，避免一眼可分的巨大噪声。

必须保存 64 个固定伪异常样例的 mask、类型、强度和各分支 z-map，供人工检查。伪异常只能保存 feature/mask，不复制数据集原图到 git。

### 5.4 融合头

预注册结构：

```text
input statistics
  → 1×1 Linear(C, 32) + GELU
  → depthwise 3×3 context（可在候选中关闭）
  → Linear(32, 1)
  → bounded residual
s_final = s_A1 + rho_ref * tanh(residual)
```

- 参数量上限 `50,000`；超过即拒绝；
- backbone 完全冻结；
- `rho_ref` 由正常 LOO score 的 IQR 固定，不能看测试分数；
- clean patch 上加入 A1-preservation loss，防止模型任意改变正常区域；
- 伪异常 mask 使用 focal BCE + soft Dice；
- 加 branch-dropout，使训练不能退化为永远使用一个分支；
- 训练上限 100 epoch，patience 10；epoch 只由 held-out normal/pseudo episode loss 选择；
- shot≥2 按 reference image 留一验证；shot=1 按增强 episode 和 corruption family 留一，不把同一 base patch 的不同 view 分到 train/val 两边。

总损失固定为：

\[
L = L_{focal} + L_{dice} + 0.2L_{clean\_preserve} + 0.1L_{branch\_drop}.
\]

系数不允许在真实 MPDD 异常上调。

### 5.5 预注册候选和控制

最多三个候选：

| ID | 层 | context | 用途 |
|---|---|---|---|
| G0 | deepest only | 无 | 判断收益是否只来自合成监督 |
| G1 | 预注册多层 | 无 | 判断多层统计是否增益 |
| G2 | 预注册多层 | depthwise 3×3 | 完整 CASF |

强制控制：

- `CTRL-A1`：原始 A1；
- `CTRL-SYM`：参数量相同，但所有伪异常永远同时扰动两分支，去掉非对称互补监督；
- `CTRL-NODIS`：去掉 `|z_D-z_C|` 通道；
- `CTRL-DINO`：只用 DINO 多层统计，参数量匹配；
- `CTRL-SHUFFLE`：训练 mask 与 corruption 位置随机错位，必须接近失败；
- `CTRL-BIG` 禁止：不能通过扩大网络容量替代机制证明。

路线成立至少要求最终候选同时胜过 A1、CTRL-SYM 和 CTRL-NODIS；只胜 A1 不足以证明“互补性感知”创新。

### 5.6 CASF 单元与技术验收

- synthetic mask 与被修改 patch 精确一致；
- tangent noise 后 L2 norm 误差 `<1e-5`，且扰动与原向量点积绝对值 `<1e-4`；
- clean 输入、`rho=0` 时逐值回归 A1；
- train/eval 同 seed 两次 score max abs error `<1e-7`；
- 改变 train seed 应改变 checkpoint hash，但不改变数据 split；
- CTRL-SHUFFLE 在合成验证集 Dice 不得高于正确标签模型；
- 训练代码无法访问真实 test masks；
- 50k 参数上限、峰值 VRAM、训练时间自动记录；
- 至少生成一张门控残差/分支差异/最终 heatmap 的 debug 图，确认 resize 与方向正确。

---

## 6. 独立路线 H：DC-SZoom——双线索稀疏高分辨率记忆

### 6.1 科学假设

现有 A1 把整张图缩放到固定输入后产生 32×32 patch grid。若缺陷很小，异常线索可能在进入末层特征前被平均掉；任何后处理都不能恢复已丢失的信息。

DC-SZoom 的假设是：**A1、DINO-only、CLIP-only 的全局低分辨率证据虽然不足以精确分割，但足以提出少量可疑窗口；只在这些窗口重提高分辨率双分支特征，可用可控算力找回微小缺陷。**

### 6.2 算法

1. 用原始 A1 流程得到全局 `s_A1`、`z_D`、`z_C`；
2. proposal map 固定为 `max(z_A1, z_D, z_C, |z_D-z_C|)`，各 z 只由 reference LOO 校准；
3. 在 proposal map 上 NMS，选择最多 M 个窗口；不足 M 时用预注册均匀窗口补齐，保证固定计算量；
4. 从原始 query 图裁剪窗口，重新 resize 到 backbone 输入大小并提取双分支 patch；
5. 对每个 query window，在每张正常 reference 原图的**相同归一化坐标**裁剪窗口，构造 window-specific normal memory；
6. 使用冻结 A1 concat+KNN 得到 local high-resolution map；
7. inverse paste 回全图，只在窗口内部用 reference-only 标定后的 `max(global, local)` 合并；重叠处取 max；
8. 未选区域严格保持原 A1。

不得根据测试 mask 移动窗口；不得让 GT 决定 M、窗口大小或 NMS。

### 6.3 预注册候选

| ID | M | 窗口边长（相对短边） | NMS IoU |
|---|---:|---:|---:|
| H0 | 2 | 0.50 | 0.25 |
| H1 | 4 | 0.50 | 0.25 |
| H2 | 4 | 0.625 | 0.25 |

不得追加 M=3/5/6 或连续扫窗口比例。

### 6.4 强制 compute-matched controls

- `CTRL-UNIFORM`：相同 M、相同窗口大小的固定均匀窗口；
- `CTRL-A1TOP`：只用 A1 map 提 proposal，不用双线索 max/disagreement；
- `CTRL-RESIZE`：全图提高输入分辨率到近似相同 FLOPs（若显存允许）；
- 报告每图平均 backbone forward 次数、秒数、VRAM 和 memory patch 数。

路线 H 必须胜过 compute-matched uniform control，才能把“双线索稀疏 proposal”写成机制贡献。若只胜 A1 但不胜 uniform，它只是高分辨率带来的工程增益。

### 6.5 几何验收

- crop→resize→patch map→paste 的合成棋盘格 round trip，无 off-by-one；
- window 边界坐标统一为 half-open `[x0,x1)`；
- query 与 reference 使用相同归一化窗口；
- 原图尺寸/长宽比不同的 reference 必须通过归一化坐标映射；
- 未选区域 `s_final == s_A1`，max abs error `<1e-7`；
- 同一窗口重算两次 bitwise 或 `<1e-7` 一致；
- 至少人工检查 12 张 debug montage：原图、proposal、窗口、local map、paste 后 map；人工检查只查几何，不用于选参数。

---

## 7. MPDD 小门

### 7.1 路线 G 小门

数据：MPDD split seed0 × shot {1,2,4} × train seed {0,1,2}，所有 6 类。

每个候选以训练种子均值汇总 shot delta，必须同时满足：

- 三-shot mean ΔPixel-AP vs A1 `≥ +0.005`；
- 3/3 shot 均值为正；
- worst shot mean delta `≥ −0.003`；
- train-seed Pixel-AP 标准差均值 `≤0.005`；
- mean ΔPixel-AUROC `≥ −0.002`；
- mean ΔImage-AP `≥ −0.010`；
- 相对 `CTRL-SYM` 的 mean Pixel-AP `≥ +0.002`；
- 相对 `CTRL-NODIS` 的 mean Pixel-AP `≥ +0.001`；
- 无泄漏、NaN、错位和复现失败。

G0/G1/G2 最多保留一个；字典序：mean ΔPixel-AP、worst shot、较少参数。

### 7.2 路线 H 小门

数据：MPDD seed0 × shot {1,2,4}，所有 6 类。

必须同时满足：

- 三-shot mean ΔPixel-AP vs A1 `≥ +0.006`；
- 至少 2/3 shot 为正；
- worst shot delta `≥ −0.005`；
- 相对同 compute 的 `CTRL-UNIFORM` mean Pixel-AP `≥ +0.002`；
- 相对 `CTRL-A1TOP` mean Pixel-AP `≥ +0.001`；
- 小面积缺陷分层（只作为预注册分析，不参与总门槛）Pixel-AP 不得更差；
- 推理时间不超过 A1 的 6 倍，峰值 VRAM不超过现有机器；
- 无泄漏、NaN、几何错位。

H0/H1/H2 最多保留一个。

### 7.3 多路线防止试到成功

- G 与 H 是两个不同科学假设，可以分别过小门；
- 小门报告必须公布全部候选和控制，不得只保留最好结果；
- 小门失败不得增加候选；
- 不得此时组合 G+H；
- 外部三个数据集在小门/Full MPDD 选择完成前不得运行 A3。

---

## 8. Full MPDD Gate 与唯一 winner

小门每条路线最多一个候选进入 MPDD 3 split seeds × 3 shots。G 还要包含 3 train seeds。

进入最终选择必须满足：

- 9 个 split/shot 配置 mean ΔPixel-AP vs A1 `≥ +0.008`；
- 至少 8/9 配置为正；
- worst config delta `≥ −0.010`；
- 至少 5/6 MPDD 类别均值为正；
- worst category mean delta `≥ −0.015`；
- mean ΔPixel-AUROC `≥ −0.002`；
- mean ΔPixel-AUPRO `≥ 0`；
- mean ΔImage-AP `≥ −0.005`；
- 机制 control 继续满足各路线要求；
- category bootstrap 95% CI 和 per-image bootstrap 全部报告，不强制显著，但不得隐瞒跨零；
- G 的 3 train-seed std 与 H 的运行复现均达标。

若 G/H 都通过，只按以下字典序选择一个：

1. Full MPDD mean ΔPixel-AP；
2. 差 `<0.002` 时选 worst category 更高者；
3. 再平局选 Pixel-AUPRO 更高者；
4. 再平局选零训练的 H；
5. 再平局选推理成本更低者。

不得测试 G+H 组合，除非两条路线都通过 Full Gate、逐图 error 相关 `<0.80`，并且用户或导师另行明确批准一个只含该组合的新预注册任务。当前任务书不授权组合。

---

## 9. 冻结与一次性外部验证

唯一 winner 产生后必须先创建：

- `FROZEN_METHOD_SPEC.md`
- `FROZEN_CONFIG.yaml`
- `SOURCE_COMMIT.txt`
- `MODEL_OR_HEAD_SHA256.json`
- `INPUT_AND_SPLIT_HASHES.json`
- `ENVIRONMENT_LOCK.txt`
- `NO_LEAKAGE_DECLARATION.json`
- `MPDD_SELECTION_REPORT.md/json`

然后锁定 Git commit。之后才能对 BTAD、MVTec AD、VisA 各运行一次完整 3 seeds × 3 shots。

验证升级最低条件：

- 三个验证数据集的 mean ΔPixel-AP 中至少 2 个为正；
- 三数据集合并 mean ΔPixel-AP `≥ +0.005`；
- 任一数据集 mean ΔPixel-AP 不低于 `−0.010`；
- 27 个 dataset×seed×shot 配置至少 19 个为正；
- worst category mean delta `≥ −0.025`；
- 主要机制 control 在至少 2/3 数据集仍优于其对应控制；
- 完整六指标、效率、失败类别和训练/推理成本全部披露。

若失败：A3 归档，不能拿验证集结果回 MPDD 改参数，也不能再验证另一路线。A1 继续投稿。

---

## 10. 执行波次与停止点

| Wave | 内容 | 通过标准 | 失败动作 |
|---|---|---|---|
| 0 | A2 D/E 限定纠错 | 几何与复现测试全过 | 标记 BLOCKED，不影响 A3 G/H |
| 1 | 多层特征导出 | deepest 回归、ID、finite 全过 | 不得训练 G |
| 2 | G/H 单元与 smoke | 技术验收全过 | 修技术问题，不看真实指标调公式 |
| 3 | MPDD 小门 | 第 7 节 | 该路线 ARCHIVE |
| 4 | Full MPDD | 第 8 节 | 该路线 ARCHIVE |
| 5 | 唯一 winner 冻结 | 哈希/配置/环境齐全 | 禁止外部验证 |
| 6 | 三数据集一次性验证 | 第 9 节 | A3 ARCHIVE，保留 A1 |
| 7 | 论文材料更新 | 主张与证据一一对应 | 不得提前改贡献 |

每个 Wave 完成后都要写 `WAVE_<n>_DECISION.md/json`。没有前一波 PASS 文件不得进入后一波。

---

## 11. 必须实现的测试

至少包含：

1. A1 deepest feature identity；
2. multi-layer sample/ref ID alignment；
3. DINO/CLIP grid resize identity；
4. reference LOO 排除自身和同图规则；
5. synthetic mask 与 corruption 区域一致；
6. tangent projection 和 L2 norm；
7. corruption family train/val group isolation；
8. deterministic model init、best checkpoint deepcopy、重复评分；
9. CASF `rho=0` 回归 A1；
10. 50k 参数硬限制；
11. branch dropout 与 CTRL-NODIS 通道契约；
12. crop/paste half-open 坐标；
13. translation/rotation/combined round trip；
14. 未 zoom 区域严格等于 A1；
15. report/config/code/input hash 绑定；
16. stale marker 拒绝；
17. evaluator 与训练 loader 隔离；
18. 六指标聚合与旧 A1 报告 `<1e-6` 回归；
19. 缺失类别、空 memory、NaN/Inf fail-fast；
20. 外部验证只能接受冻结 config hash。

项目测试命令：

```powershell
.\.venv-patchcore\Scripts\python.exe -m pytest tests -q
```

第三方 `methods/` 上游测试不属于本项目回归范围，不要用无边界 pytest 收集它们。

---

## 12. 报告与交付物

最终至少交付：

```text
experiments/dynamic_fusion/innovation_v3/
├── 00_input_audit/
├── 01_layer_probe/
├── 02_smoke/
├── 03_small_gates/
│   ├── G_CASF/
│   └── H_DCSZOOM/
├── 04_full_mpdd/
├── 05_frozen_winner/
├── 06_external_validation/
├── 07_ablations/
├── 08_efficiency/
├── 09_figures/
├── FINAL_DECISION.md
└── FINAL_DECISION.json
```

每个 candidate 报告都要记录：

- 公式/候选 ID；
- 数据集角色、split seed、shot、train seed；
- 六指标和 ΔA1；
- 每类别与总体结果；
- config/code/input/checkpoint SHA256；
- 参数量、时间、VRAM、memory 大小；
- control 对照；
- leakage flags；
- marker 与生成时间；
- 失败时的具体门槛项。

报告必须有 CSV/JSON 机器文件和一份中文 Markdown 人类摘要。全部候选都要保留，禁止删除负结果。

---

## 13. 论文口径

### 13.1 A3 未通过前

论文仍只能说：

- A1 是 frozen, fixed, equal-weight dual-visual feature fusion；
- 0 trainable parameters；
- A2 是探索性负结果，不是贡献；
- 不能在标题、摘要或方法图加入 CASF/DC-SZoom。

### 13.2 CASF 通过后可以怎样写

方法定位必须改为 `lightweight normal-only adaptation`，不能再说整个方法零训练。可候选贡献最多三点：

1. normal-only cross-encoder asymmetric feature corruption；
2. multi-level, reference-calibrated complementarity statistics；
3. bounded residual fusion with controlled cross-dataset validation。

不能说“首次使用合成异常”“首次多尺度融合”“首次动态融合”。

### 13.3 DC-SZoom 通过后可以怎样写

仍可保持 backbone frozen/training-free，但推理成本上升。可候选贡献：

1. dual-cue sparse proposal；
2. query-window-conditioned normal reference memory；
3. compute-matched evidence that sparse zoom, not generic tiling, provides增益。

不能把切块、高分辨率或 KNN 本身当作创新。

---

## 14. 禁止事项

- 不复活 LNDC、DSAM、CEQA、DEVA、NCPRA、FAGR 的参数搜索；Wave 0 只做纠错复核；
- 不重新扫 A1 branch weight；
- 不把多个 layer 任意组合后只报最好组合；
- 不用真实异常 mask 训练、早停、选伪异常强度或选窗口；
- 不从 BTAD/MVTec/VisA 结果回头选择 G/H；
- 不在 shot/类别上手写 fallback；
- 不把 test-image patches 加进 normal memory 导致 self-match；
- 不用大网络容量掩盖机制无效；
- 不把“tests passed”误写成“算法有效”；
- 不把 corrected D/E 的结果和 A3 混成一个贡献；
- 不静默覆盖历史报告或冻结复现包；
- 不在结果不满足门槛时降低门槛、改主指标或只报有利类别。

---

## 15. 最终验收问题

执行 AI 在结束前必须逐条回答：

1. A2 D/E 的旧结论哪些有效、哪些因实现问题改为 provisional？
2. 多层特征是否真的提供非冗余线索，还是只是增加维度？
3. CASF 是否超过参数量匹配的 symmetric/no-disagreement controls？
4. DC-SZoom 是否超过相同计算量的 uniform tiles？
5. 增益是否跨 shot、split seed、train seed 和类别稳定？
6. 是否完全没有使用真实异常标签进行方法决策？
7. 外部验证是否只运行了冻结后的唯一 winner？
8. Pixel-AP 增益是否伴随 Pixel-AUPRO、Image-AP 或效率代价？
9. 论文主张是否由现有证据支持，而不是由方案名称支持？
10. 若 A3 失败，是否诚实保留 A1 并停止？

只有这十项有明确证据，任务才算完成。

---

## 16. 当前正式决策

- A2 当前结果：**不升级**；A1 继续为论文主方法。
- A2 D/E：结果数值保留，但科学结论暂标为 **provisional pending bounded correction audit**。
- A3：批准进入实现，但只允许本任务书中的 G/H 两条独立路线和 Wave 0 限定纠错。
- 在 Full MPDD 与冻结外部验证完成前，CASF、DC-SZoom 均只是研究假设，不是论文创新点。

