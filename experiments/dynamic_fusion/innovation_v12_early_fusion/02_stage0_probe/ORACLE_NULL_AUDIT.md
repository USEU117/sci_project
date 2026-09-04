# ORACLE NULL-CONTROL AUDIT — Stage0 headroom 解读修订（doc 25 §5）

date: 2026-09-04
authority: docs/.../25_OVERNIGHT_VALIDATION_CODE_AUDIT_AND_NEXT_STEPS_CN_20260904.md §5
supersedes: `STAGE0_DECISION.md` §1 中"oracle headroom +0.3885 ⇒ 存在约 0.39 可学习互补增益"的因果解读。
机器：CPU（.venv-patchcore）逐类计算层 map 后复用与归档 probe 完全一致的
per-GT-连通域选择规则（bce + (1-AP) + mean(normal)），加三类空信息对照。
产物：ORACLE_NULL_AUDIT_k{1,2,4}.json。

## 1. 对照设计（doc 25 §5）

| 名称 | 专家池 | 含义 |
|---|---|---|
| real | 7 个单分支层 map（D6/9/11, C6/12/18/24） | 复现归档 oracle（校验 harness） |
| a1copy | {A1} | 替换 = 复制 A1 自身 → 应 Δ=0（程序健全性） |
| scale | {c·A1, c∈0.25…8} | 纯单调缩放、零新信息 → 测尺度差/GT 特权是否单独产生 headroom |
| shuffle | 7 个真实层 map，每连通域随机指认专家（3 seed） | 打乱“哪条缺陷该用哪层”的对应 |

## 2. 结果（6 类宏平均 pooled Pixel-AP @56，MPDD s0）

| shot | a1 | real Δ | a1copy Δ | **scale Δ** | shuffle Δ (3-seed mean) |
|---|---|---|---|---|---|
| k1 | 0.309212 | +0.399753 | 0.0 | **+0.690780** | −0.011746 |
| k2 | 0.343706 | +0.375420 | 0.0 | **+0.656249** | −0.023092 |
| k4 | 0.388328 | +0.390378 | 0.0 | **+0.611638** | −0.051471 |
| mean | — | +0.388517 | 0.0 | **+0.652889** | −0.028770 |

- real Δ 与归档 STAGE0_DECISION/ORACLE_HEADROOM 逐位一致（0.399753 / 0.375420 / 0.390378）→ harness 校验通过。
- a1copy Δ = 0.0：替换机制本身不造假增益（程序健全）。
- **scale Δ ≈ +0.61~0.69，比 real oracle（+0.39）还高**：只允许"A1 的单调缩放 ×
  GT 缺陷连通域内粘贴"，零新信息，就能造出 ≥ 整个归档 headroom 的增益。
- shuffle Δ ≈ 0/负：随机层对应无增益（也说明不是"任意替换都涨"）。

## 3. 结论（修订后解读）

1. **+0.3885 不能用作“存在约 0.39 可学习互补增益”的证据**（doc 25 §5 的最小反例
   在此得到全量确认）：纯尺度差 + GT 边界特权（scale 对照）即可单独产生 +0.65 宏增益，
   超过整个 real oracle。headroom 的主要成分是 scale/privilege，而非跨分支层排序信息。
2. 因此 Stage0 oracle 只支持弱结论："该数据上存在使 Pixel-AP 上升的专家分数操纵空间"，
   不支持强结论："某可学习模块能兑现该增益"。
3. 按 doc 25 §5："若 null control 也有巨大 headroom，先停止用该 oracle 做入场判断，
   不训练 router"——**oracle 入场判断口径废止**；后续如需复用 oracle 概念，必须先做
   专家同规则校准（同一 normal calibration）+ 无标签候选区域 + 整区域（含正常像素）替换
   + 训练/验证按图像/类别分离，并单独预注册。
4. 与 03_scaif_small_gate/CORRECTION_NOTES.md §5（Phase B：main correction == static2、
   门全关）一致：学习式路线的两条入场/继续依据（oracle headroom、训练可超越 static 基线）
   现均被证伪或削弱 → 缓存级 SCAIF 停止，不进入 Stage 2。
