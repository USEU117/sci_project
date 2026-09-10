# 精确三角下界检索探针协议（2026-09-08）

## 目的

本轮只做已知的 pivot / 三角不等式下界精确近邻检索工程探针。目标是测量：在高分辨率 `64 × 64` DINO patch memory 上，normal-only pivot 下界能否减少候选并降低端到端 CPU 检索时间，同时保持与完整 F32 精确 1-NN 相同的距离和近邻结果。该方法本身不是原创算法；结果只用于本项目的资源与精确性判断。

## 旧代码审计

在 `scripts/`、`src/`、`tests/` 中检索 `triangle`、`pivot`、`lower bound`、`exact NN`、`IndexFlatL2` 等关键词。旧代码只有完整 FAISS / NumPy 近邻、coreset 和普通 pandas `pivot` 表操作，未见 pivot + triangle-bound 的精确剪枝实现。因此本脚本不复用或覆盖旧检索实现。

## 冻结输入与选择

- 输入只读现有 full896 缓存，不重新提取模型，不使用 GPU：
  - `outputs/dynamic_fusion/overnight_20260908_full896/features/bracket_black.npz`
  - `outputs/dynamic_fusion/overnight_20260908_full896_extension/features/connector.npz`
- 两个预先写死的类别为 `bracket_black`、`connector`。每类把两张 clean support 图的 `64 × 64 × 768` 特征全部展平为 memory（共 `8192` 行），只用 support memory 选 pivot。
- 每类把 synthetic cache 的全部 query 行按展平顺序合并，再用固定 `np.linspace` 行号取 `512` 个 query；不读取或使用 mask、family、缺陷标签来选 pivot 或 query。两类合计 `1024` 个 query，全部保留 `768` 维并逐行 F32 L2 归一化。
- 固定 pivot 数为 `32`。从 clean support memory 的第 `0` 行开始，按欧氏距离的确定性 farthest-point greedy 选择，其余候选只来自 normal memory；不看 query 标签。

## 两个检索臂

1. **full F32 exact**：对每个 unit query 与全部 `8192` 个 unit memory 行计算欧氏距离，取最小值。
2. **pivot bound exact**：先计算 query 到 `32` 个 pivot 的欧氏距离；对每个 memory 行使用
   `LB(q,x)=max_p |d(q,p)-d(x,p)|`，只有 `LB ≤ 当前 pivot 上界 + 1e-6` 的行进入精确重排。当前上界是 query 到 pivot 的最小欧氏距离。候选保留后仍对所有保留行做完整欧氏重排，因而不把下界当近似距离。

距离计算直接使用 unit F32 行的欧氏范数；只有得到最小欧氏距离后，额外记录 `d² / 2` 作为与旧 DINO `1 − cosine` 口径可读的返回值。三角界和候选判定使用欧氏距离，不以 cosine 距离作三角界。

## 计时与精确性

- `OMP_NUM_THREADS=2`、`MKL_NUM_THREADS=2`、`OPENBLAS_NUM_THREADS=2`，脚本也调用 `torch.set_num_threads(2)`（若 torch 可用）；不初始化 CUDA。
- 预热 exact 和 bound 各两次；正式 exact 与 bound 各三次，使用同一 query 顺序。bound 计时包括 query-pivot 距离、下界计算、候选精确重排和结果归约；另行报告 pivot 选择与 bank-pivot 预计算成本，并给出把一次性 pivot 成本摊到本轮 query 的时间。
- 报告每类及合计的最大最小距离差、近邻索引一致性和并列容忍集合一致性（`1e-6` 欧氏距离）；报告候选行数和保留率的均值、中位数、p95、最小最大值。
- 结果和代码不读取 test 标签，也不按 AP 或标签选择 pivot、query、阈值或参数。脚本设置 `15` 分钟运行上限并在阶段边界保存 partial；本轮不做任何 GPU 训练或重新提取。

## 解释边界

实测时间包含下界计算，不能由三角不等式理论直接声称加速。高维特征集中可能导致几乎不剪枝，或者 Python/NumPy 候选管理使 bound 更慢；两种情况均原样归档。即使近邻完全一致，也只能说明本次输入上的精确性，不能证明更大 memory、其他类别或其他硬件上的普遍加速。

