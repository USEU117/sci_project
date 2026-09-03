# V12 MTCOA — FAILURE_ANALYSIS（2026-09-03）

## 机制层面为什么失败
MTCOA 一次性修正审计失败不是因为校准/损失实现（这些已修好并被逐项对照验证），而是
专家池的**宏观互补上限**不足：

1. **A1 强类上无任何专家可补**：metal_plate/tubes 的 A1 AP（0.73–0.89）已接近该类可达
   上界；text/LLSE/CSS 在这些类的校准后缺陷响应都弱于 A1。即使 Oracle 拥有每个 GT 组件
   的精确几何并按 doc21 区域损失逐组件挑专家，替换 A1 仍净损 −0.07~−0.13（跨 shot、跨
   两种 loss 解释一致）。k4 中该亏损主要落在 >64-cell 大块层。
2. **正 headroom 单类集中**：bracket_white（k4 +0.138）占全部正 headroom 的 77%，违反
   g4 的"不超 50%"；该类的 32 个 GT 组件全部落在 1–4 cells 层（唯一的小缺陷类）。
3. **LOO 负 drop 的读数**：k2/k4 去掉 text/LLSE 反而提升 oracle 均值 Δ——说明在这些 shot
   上非 A1 专家在 A1 强类造成的净损大于其在弱类的净益，component-macro 选择份额
   （E1 33%/E2 24%）是"选择多"而非"选得对"。

## 校准/实现层面已排除的干扰（记录以便后续方向复用）
- 顶端饱和 CDF 会破坏 identity（最初 cal AP 0.001 vs raw 0.0105）→ 改为无饱和严格单调
  插值 CDF + 线性外推（slope cap 10），identity 精确至 ≤1e-4。
- support-only 校准池在低 AP 类退化（ref-test 域差使 A1 正常 FP 被推到 ~1）→ 校准池并入
  本类 test/good 正常图（无 bad/mask），跨专家尺标可比。
- 整图 0/1 BCE 使安静背景专家（text）在 A1 完美类上系统性胜出（metal_plate oracle
  −0.255，>64 层）→ 按 doc21 语义改区域像素 BCE hit + FP 项，strong-class 净损从 −0.26
  收窄到 −0.10 但仍为负。

## 禁止继续的方向（按 doc 22 §12 + 本审计）
- 不得再训练任何 {A1,text,LLSE,CSS} pseudo-regret router；
- 不得用 per-category 条件或类别名作为 RouterInput 挽救本结论（§11.4 泄漏条款）；
- 不得把 bracket_white/小缺陷的单类 headroom 包装成宏观融合成功。
