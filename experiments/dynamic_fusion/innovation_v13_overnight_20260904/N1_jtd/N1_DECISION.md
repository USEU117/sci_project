# N1 / JTD 决策：R1 失败，归档（doc27 §5）

日期：2026-09-04（夜间机时轮）。实现：`scripts/innovation_v13_overnight_20260904/run_n1_jtd.py`。
CPU-only（faiss/sklearn），支持集 LOO 校准，normal-only 拟合，GT 仅评估。

## 1. 实现要点与冻结规则（执行前冻结）

- u_D,u_C = 单分支末层最近邻 1-cos 距离的 normal-only 经验 CDF（support LOO 池）。
- 8×8 二维直方图，Dirichlet 平滑 ALPHA=1.0（64 单元上均匀先验质量 =1）。
- R = max(0, log(p_D·p_C / p_joint))，仅 max(u_D,u_C)>0.9 激活（GATE=0.9），cap=5。
- candidate = rankF + 0.1·R；rankF = 冻结 A1 融合距离的 normal-only CDF（support LOO）。
- 对照：a1_raw（同管线原始融合距离，A1 锚）、a1_rank（rank-only）、u_sum/u_max（独立尾部）、
  shuf（打乱 D/C 配对、保边缘）、no_gate（无高尾门二维稀有度，诊断）。
- 超范围处理（预注册）：经验 CDF 带边界斜率线性尾部延伸，严格保序、不 tie；记录于 docstring。

## 2. 关键 bug 修复（保留失败记录）

初版（修复前）RESULTS.json 无效：`rankdata(vals)` 返回原始顺序的秩，而 `np.interp` 的 fp 需与
排序后的 xp 对齐 → 变换非单调（metal k2 上 Spearman(FQ, rankF) = -0.26），a1_rank AP 崩塌为
缺陷先验（metal 0.0649 而非 0.878）。修复为对排序数组计算秩。修复后 Spearman = 1.0，
原生 32 网格 AP(a1_rank) ≡ AP(a1_raw)（逐类相等，例如 metal k2 均 0.7573）。

注：56 网格 pooled AP 中 a1_rank 仍 ≤ a1_raw（高 AP 类更明显，如 bracket_white k2 0.0167 vs
0.1060）。原因：rankF 是 A1 距离的非线性单调变换，dists2map 的线性上采样+高斯模糊与值域缩放
交互，改变跨图合并排序。32 网格逐像素顺序完全一致（AP 相同），故 a1_raw 为 A1 的忠实锚，
a1_rank 为同 scale 族的 rank-only 控制——二者都是计划内对照，不视为机制问题。

## 3. R0（toy + 校准块稳定性）

- 配对可识别性（2D 高斯 r=0.8 vs 独立，尾部 |R−R_shuf| 均值）：dep=0.117 vs ind=0.024 → 可识别。
- 单调缩放不变性（cbrt/sqrt 后再量化）：|ΔR| 均值 = 0.0 → 通过。
- 校准分块稳定性（各半拟合 vs 全量）：逐类 corner_R 均 = 0，半块 cell 差 ≤0.054（block 字段），
  无单块崩溃 → R0 作为机制 sanity 通过（不构成成功结论，见 doc27 §5）。

## 4. R1（真实配对，k2/k4，六类 pooled Pixel-AP@56）宏平均

| score | k2 macro | k4 macro |
|---|---|---|
| a1_raw（A1 锚） | 0.3437 | 0.3883 |
| a1_rank（rank-only 控制） | 0.3087 | 0.3351 |
| u_sum（独立尾部） | 0.3237 | 0.3402 |
| u_max（独立尾部） | 0.2871 | 0.3196 |
| shuf（打乱配对） | 0.3067 | 0.3331 |
| **cand（JTD）** | **0.2074** | **0.2496** |
| no_gate（诊断） | 0.2026 | 0.2443 |

门判定（doc27 §5 R1：cand − max(独立尾部, rank-only) ≥ +0.003 且 cand − shuf ≥ +0.003）：

- cand − 最强独立尾部/rank-only：k2 = 0.2074 − 0.3237 = **−0.116**；k4 = 0.2496 − 0.3402 = **−0.091**。
- cand − shuf：k2 = **−0.099**；k4 = **−0.084**。
- 12 个 (cat,shot) 中 candidate 无一胜过 a1_rank；tubes 上真实配对远差于打乱（k2 0.196 vs 0.643）。

→ R1 两项均不满足，方向为负。按 doc27 §5「只胜原始 A1、不胜 rank-only，立即归档」归档，
不进入 §11 性能门，不进入胜者复验。

## 5. 归因诊断（为什么负）

1. 校准联合密度在 (高,高) corner 无负依赖：逐类 block.corner_R_full = 0 → 正常数据双分支高尾
   组合并不比独立更罕见，R 在想要的 corner 恒为 0。
2. R>0 的单元（正常联合稀疏、p_joint < p_D·p_C 的离角单元）在真实测试集上对应大量正常/背景
   像素 → R 主要在 FP 上激活，cand 系统性低于 rankF（tubes 最极端）。
3. u_sum/u_max（独立尾部，无联合模型）介于中间且已优于 JTD——说明信息在边缘分位里，
   8×8 联合稀有序量没有提供额外可分性。

## 6. 归档与后续

- 状态：**FAIL（机制负向）**，不保留候选。R0 toy 与 block 稳定性存 `TOY.json`/RESULTS 内。
- 观察（不作本轮声明）：metal 类 u_max 高于 a1_raw（k2 0.9097 vs 0.8784；k4 0.9240 vs 0.8850），
  属"独立尾部 max"融合规则的旁路观察；改变融合权重/规则需另轮预注册，本夜不包装为新候选。
- 产物：TOY.json、RESULTS.json（含 diag/block）、本决策。
