# 20. 多路线 R0 执行总结（v10 Portfolio，2026-09-03）

任务书：`19_MULTI_ROUTE_INNOVATION_EXPLORATION_HANDOFF_CN_20260903.md`。
执行范围：路线 A(CRAM)/C(CAPM)/E(STR) 完整 R0 + 强对照；B(MESP) 几何审计 + dino-only
promise 探针；D(NORC) 设计 + 最小实现 + 单测。所有协议在结果前预注册并落盘。

## 结果总表（全部在 MPDD development 集；BTAD/MVTec/VisA 未触碰）

| 路线 | R0 结论 | 关键数字 | 决定性失败原因 | 产物 |
|---|---|---|---|---|
| A CRAM | FAIL/ARCHIVED | A1 两-shot 均值 −0.0028；A2 −0.0076 | 均值负；connector −0.029；shuffled 对照 不比 real 差 | cram/{R0_RESULT,R0_DECISION,FAILURE_ANALYSIS}.md/json |
| B MESP | ARCHIVED（未跑全融合 R0） | dino B1 +0.0034(5/6正)；错位对照 +0.0036 | 强对照 g4 失败：真实 vs 错位 −0.0002；增益是平滑不是等变 | mesp/{R0_GEOMETRY_AUDIT.json,R0_PROMISE_PROBE.json,R0_DECISION.md,FAILURE_ANALYSIS.md} |
| C CAPM | 可行性 PASS → 像素 R0 FAIL/ARCHIVED | 全 6 类 mean ΔAP −0.0275，0/6 正 | 位姿差致 diffuse 抬升；random 对齐对照不差于真实 | capm/{R0_RESULT.json(可行性),R0_PIXEL_EVAL_RESULT.json,R0_DECISION.md,FAILURE_ANALYSIS.md} |
| D NORC | 机制就绪 → ARCHIVED（无伙伴） | g0/g1 PASS by construction | 唯一伙伴 MESP 失败；K≤4 时 p_min=0.20 门永不触发 | norc/{R0_DESIGN,R0_DECISION}.md + norc.py + 单测 |
| E STR | FAIL/ARCHIVED | region info-value Δ −0.027；1/6 正 | A1 自身分数更能预测 A1 漏检；纹理类混淆 | str/{R0_RESULT.json,R0_DECISION.md,FAILURE_ANALYSIS.md} |
| F SPRG | 可行性门1 FAIL/ARCHIVED | 跨样本节点匹配率 36%（门≥90%） | 外观匹配余弦高(0.87-0.97)但跨参考位置/链稳定性崩；MPDD 无稳定部件对应 | sprg/{R0_PROTOCOL,R0_RESULT}.json + R0_DECISION.md |

## 质量与防泄漏
- A0 identity 与冻结 A1 像素级 bit-exact（max diff 0.0），A1 mean AP 复现
  0.3437(k2)/0.3883(k4) 与冻结矩阵完全一致。
- 强对照体系全部执行：CRAM(重复参考/打乱参考/med-only)、CAPM(随机 homography)、
  STR(错位/随机相位/梯度)、MESP(错位视图)。
- 无泄漏单测 16/16 通过（AST 扫描评分模块无 gt 键；"改 GT 不改图" 确定性测试）。
- 所有校准统计仅用正常参考（LOO）；方法函数签名无 gt_masks/gt_labels。

## 跨路线统一结论（为什么 A1 之外没有稳定增益）
四条失败路线共用同一个根因：**任何把 A1 分数场做平滑单调变换的机制（加正项、
位置约束、多视图平均、光谱残差叠加）在 MPDD 上都移动整体分数分布，无法把少数
真缺陷像素与远大于它的正常像素群分离**。K-shot A1 在融合特征上已经吃掉了可用
的正常参考结构；新增机制缺少独立、稀疏、region-selective 的证据源。

## Scenario 判定与后续建议
按任务书 §14 → **Scenario E：全部优先路线负结果归档，停止算法搜索**。
论文继续以 A1 为主线：补统计（bootstrap CI 逐类）、复现核查、图表与写作。
真正能推进的下一个动作需要用户决策（不自动执行）：
1. 新外部验证集授权（Real-IAD / AeBAD 下载与许可）——目前冻结 A1 已 4 数据集
   验证，新集仅用于论文外部证据；
2. SPRG 已在 MPDD 完成可行性探针并失败（门1 36%<90%），不再推进；MVTec LOCO 与
   许可无需启动。
复用资产：per-reference 记忆库分解（CRAM）、GT-free RANSAC 对齐管线（CAPM）、
NORC 校准单元设计——留档备未来数据使用。

## X 补充探索（用户指令追加）：LLSE 记忆局部线性重构残差

背景：A–F 全部归档后，用户要求再想一条"机制上更创新"的路线并实测。提出
**LLSE**：query 的分数 = 到其 top-8 记忆近邻张成的局部线性流形的最小二乘重构
残差（r²=‖q‖²−wᵀNᵀq），与 A1 的 1-NN 机制正交。因 AP 对分数单调变换不变，
ΔAP≠0 证明发生了非单调重排，可排除"平滑变换"类失败根因。

| seed | A1 mean AP | LLSE mean ΔAP | n 正类 | 最差类 | random-8 对照 Δ | true−random |
|---|---:|---:|---:|---:|---:|---:|
| s0 | 0.3092 | **+0.0089** | 4/6 | bracket_black −0.0028 | −0.179 | +0.188 |
| s1 | 0.3425 | **−0.0028** | 3/6 | bracket_black −0.0565 | −0.204 | +0.201 |
| s2 | 0.3159 | **+0.0014** | 2/6 | connector −0.0205 | −0.188 | +0.189 |

判定：g0/g3 通过（3/3 seed 对照一致：破坏邻域局部性后分数场完全失去分离力，
证明增益确为 locality 驱动）；**g1/g2 多 seed 失败**（s1 均值转负、s2 最差 −0.021、
s1 bracket_black −0.056）。按纪律归档为 exploration，不进候选主线。
留存 per-category 观察：**tubes/metal_plate 三 seed 一致 +0.013~+0.040**（缺陷区域
连贯、A1 本已较高的类），可作为未来大尺度缺陷/ensemble 场景的补充证据源。
产物：`explore_llse/{R0_PROTOCOL,R0_RESULT,R0_CONFIRM_s1,R0_CONFIRM_s2}.json +
R0_DECISION.md`。
结论不变：**Scenario E（A1 主线收尾），算法搜索停止**。

### X2 CSS 图内上下文自一致性（reference-free，第二轮追加）

用户再次追问后测试的**唯一种类完全不同的机制**：score(q)=q 与其同图内 24 个空间
邻域的 mean(1−cos)，完全不使用参考图。s0/k1 结果：

| 指标 | 值 |
|---|---|
| CSS-alone mean AP | 0.040（weak 但真信号；bracket_black 0.0144 > A1 0.0105）|
| fused-sum ΔAP（预注册门规则）| −0.042（3/6 正，worst metal_plate −0.365）FAIL |
| fused-max ΔAP（amendment 次级）| −0.025（3/6 正，worst −0.292）FAIL |
| corr 诊断 | defect 像素上 CSS 与 A1 弱/负相关（metal_plate −0.38、tubes −0.23），normal 上强正（0.45–0.56）|

判定：CSS 拯救了 A1 最弱类（bracket_* +0.078~+0.160），但大块连贯缺陷内部自一致
（CSS 只响边界环）使其在 A1 强类上破坏融合。同一根因再加一类独立证据：**MPDD 六类
缺陷形态差异大，任何单一固定证据源都无法全局稳定超越 A1**；A1 弱类（bracket_black）
恰是参考图代表性差的类，构成类别级观察留存。
产物：`explore_css/{R0_PROTOCOL,R0_RESULT}.json + R0_DECISION.md`。
Scenario E 维持。
