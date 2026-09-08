# Support 图级 facility coverage 选择：预定协议

版本：`support_selection_v1`；建立时间：2026-09-08 夜间；本轮只写入 `support_selection/` 与对应的新 probe，不改任何历史实验产物。

## 目的和边界

测试一个与 Track-4 patch coreset 不同的 memory policy：以整张 normal support image 为选择单位，在 k 4 的四张 support 图中，每折留出一张作 query，只允许剩余三张 candidate 参与选择；被选图的全部 feature cells 都保留。该 probe 只测 support-only synthetic proxy 和 normal nuisance 稳定性，不能作为论文创新确认。

## 冻结设置

- 数据：v14 P1 frozen support cache，seed 0，shot k 4，六类 `bracket_black`、`bracket_brown`、`bracket_white`、`connector`、`metal_plate`、`tubes`。
- 每类四折：第 `h` 张 support image 是 held-out query，候选是另外三张 clean normal image；候选顺序保持 v14 cache 的 `ref_rel` 顺序。
- 分支：DINO 与 CLIP 分别逐行 L2 normalize；CLIP 的 37 × 37 grid 先双线性 resize 到 32 × 32；两分支分别做 2 × 2 平均池化到 16 × 16，再按 A1 `0.5 / 0.5` 拼接并整体逐行 L2 normalize。每张被选图贡献 256 个 fused memory cells。
- facility coverage：对候选池的全部 `3 × 256` cells，计算其到所选 image subset 全部 cells 的最小 `1 − cosine` 距离并取平均。k 1、k 2 分别在三种组合中取最小值；相同值按组合的 candidate index 字典序选择。held-out feature、mask 和 synthetic query 不进入该分数。
- 对照：相同预算的全部组合均值（`C(3,k)` 个组合的精确均值，作为 uniform random-combination expectation）、固定候选首图 / 前两图，以及候选全部三图。all-3 只作上限参照，不与 k 1 / k 2 预算等同。
- synthetic：v14 的 `cutpaste`、`local_erasure`、`thin_scratch` 三族，每族固定使用 seed 0、1、2 三个变体；顺序和 mask 来自 cache。若原像素 mask 没有正像素，记录 invalid 和正像素数，绝不因 16-grid 不可见而删除 thin scratch episode。
- mask 评价：16 × 16 score grid 双线性 resize 到原始 1024 × 1024，再和原始 binary mask 计算 Pixel-AP；mask 不下采样、不阈值化为 grid。
- normal 稳定性：使用 held-out clean 与 15 个预先存在的 photometric nuisance 变体，每张 16 × 16 grid 固定取行列步长 4 的 16 cells，记录 nuisance-vs-clean 的 image-independent patch score AUC、均值分数和差值。该指标只作稳定性诊断，不作按结果选图。
- 资源：CPU，PyTorch 线程 2，OpenCV 线程 1；脚本设置 `CUDA_VISIBLE_DEVICES` 不使用 GPU。硬截止为 `2026-09-07 23:00:00 UTC`（北京时间 2026-09-08 07:00），每完成一折写 partial checkpoint。

## 输出和判定

每折记录 held-out path、三条 candidate path、facility 选择 path、覆盖分数、每个预算的有效 memory cells、每个组合和策略的 per-family / per-seed AP、normal nuisance stability 和选择耗时。所有结果只标为 exploratory signal / negative result；不设“论文通过”门，不依据真实 MPDD test 选策略，也不使用 `/test/` 图像或 mask。

复现命令：

```powershell
$env:CUDA_VISIBLE_DEVICES = ""
$env:OMP_NUM_THREADS = "2"
$env:MKL_NUM_THREADS = "2"
python scripts/innovation_overnight_20260908/probe_support_selection.py
```

预期产物：`RESULTS.json`、`PARTIAL_RESULTS.json`、`DECISION_CN.md`、`RUN_MANIFEST.json`，均位于本目录；输入 cache 只读。
