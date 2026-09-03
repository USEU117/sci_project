# V11 BC-MCR — R0 结构合成门决策（2026-09-03）

协议：`R0_PROTOCOL.json`（doc 21 §6.3）
脚本：`scripts/innovation_v11_regret_router/run_r1_bcmcr_structural_gate.py`
数据：MPDD seed0 shot2 正常材料（2 refs 训练；16 test/good 网格按 D2 同参扰动），
**全程无 GT、无 bad 图**。模型：每类盲中心 7×7−中心3×3（40 cell）上下文 + top-16
support center tokens（context-mean 检索）→ 2 层 cross-attention → 预测中心，
score = 1−cos(pred, actual)。对照：CTRL_COPY（中心可见）/ CTRL_CTX（无 support）/
CTRL_POS（旧 D2 非参数 ring→center 检索，无训练）/ CTRL_SHUFFLE（配对打乱）。

## 结果（patch-AUROC，6 类均值）

| 扰动 | A1 | CTRL_POS(旧) | **FULL** | CTRL_COPY | CTRL_CTX | CTRL_SHUFFLE |
|---|---|---:|---:|---:|---:|---:|
| permutation | 0.502 | 0.719 | **0.720** | 0.540 | 0.725 | 0.502 |
| missing | 0.582 | 0.520 | **0.405** | 0.417 | 0.406 | 0.345 |
| duplicate | 0.499 | 0.715 | **0.716** | 0.540 | 0.714 | 0.509 |

逐类 missing：bracket_black 0.351 / brown 0.449 / white 0.439 / connector 0.394 /
metal_plate 0.464 / tubes 0.336 —— 全 6 类 FULL 均低于 A1（0.44–0.65）与 CTRL_POS。

## 门判定（全部 FAIL）
- g1（三类均 ≥0.75）：FAIL（perm 0.72 / missing 0.41 / dup 0.72）。
- g2（三类均 ≥ A1+0.10）：FAIL——missing 为 **−0.177**。
- g3（missing ≥ CTRL_POS+0.15）：FAIL——FULL 0.405 **低于**旧方法 0.520。
- g4（FULL ≥ 每个对照于每类）：FAIL——permutation/duplicate 上 CTRL_CTX ≥ FULL。
- g5（FULL − best control 均值 ≥+0.05）：FAIL。

## 机制结论：盲中心余弦残差无法检测"结构抹除为均值"（doc 6.1 前提被证伪）
三类扰动的对照行为证明机制本身有效且可信：
- 盲性必要：CTRL_COPY（中心可见）三类型全部掉到 ~0.54（接近 chance 0.5）；
- 配对监督必要：CTRL_SHUFFLE 全部 ~0.35–0.51（打乱配对即失去结构信号）；
- **但** permutation/duplicate 上 FULL(0.72) ≈ CTRL_POS(0.72) ≈ CTRL_CTX(0.72)：
  训练式盲中心修复相对**旧的非参数 ring 检索没有增益**，support 条件也几乎不贡献
  （CTX 追平 FULL）；
- missing 上 FULL 反比 A1 更差：D2 式 missing 扰动把块抹成**自身空间均值**，该均值
  恰是上下文可预测的中心方向 → 余弦残差在缺失块上**反向更低**。doc 21 预期的
  "训练可修复 old retrieval 的 missing 失败"在本代理下被证伪（旧方法 0.520、训练
  0.405、A1 0.582）。

## 结论：ARCHIVED（按 doc 21 §6.3 stop："不过门不接触真实异常"）
真实异常 MPDD 门（§6.4）**不启动**。BC-MCR 若想复活，需要：(a) 一个真正
"缺件→背景纹理"的 missing 代理（而非块均值抹除），或 (b) 非余弦的分数
（如预测方差/不确定性方向），或 (c) 证明比旧 ring-retrieval 有不可解释的增益。
按当前证据这三者均未建立。

## 复现备注
- 全量 6 类运行使用 `hash(cat)`（进程盐值）做扰动种子，跨进程 bit 不可复现；已在
  脚本中改为稳定 `sum(ord)` 映射（`_cat_seed`）。单类两次运行（salted vs 修正前）
  数值差异 ≤0.03、门结论不变。
- torch non-writable numpy 警告无害（score_grid 内 broadcast 视图只读）。
