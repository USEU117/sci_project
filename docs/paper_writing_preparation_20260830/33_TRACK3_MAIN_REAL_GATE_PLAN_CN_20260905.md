# Track-3 主线立项：真实 MPDD 门验证 A1+50% coreset（doc32 探针通过后的落地验证）

日期：2026-09-05　上游：doc32 Probe-E1 **通过**（T1=50% 棋盘 coreset 三门全过，合成代理）；A1 冻结配置 `freeze/a1_mpdd_w05`（concat pca0/whiten0/w0.5，真实 MPDD 主矩阵 seed0_k2 mean pixel AP=0.3437，`v3_direction_a/a1_matrix_20260817/seed0_k2/concat_pca0_whiten0_w0.5_report.json`）。
性质：**真实门验证（只读评估，不训练、不调参）**——把合成代理上通过的无损压缩候选放到真实 MPDD 六类 test 上，验证效率主张可落地且不损失判别力。

## 1. 验证对象与口径
- **配置**：A1 冻结 concat（dino 32×32 + clip 37→32 resize，逐分支 L2，w=0.5 加权 concat 1536-D；无 PCA；`s = L2(dists(feat, ref))/…`，faiss IndexFlatL2 k=1，dists2map → 448 map；Pixel AP stride=8）。
- **对照**：`A1 full memory`（ref = K 张 support 图全部 1024 patch/图）vs `A1 + T1 coreset`（ref 每图 32×32 网格按棋盘 `(i+j)%2==0` 保留 50% patch = 512/图）；**只压 memory 端，查询 patch 全量**。
- **范围（预注册 first-shot）**：seed0，k2 先行；k2 通过后补 k4 同 seed0 确认；不扩展其它 seed（真实门不按结果挑 seed，通过即结案，失败即归档）。
- 数据：复用 A1 冻结特征缓存（`outputs/dynamic_fusion/v3_direction_a/features_vitb14_s{seed}_k{shot}/*/` 与 `features_s{seed}_k{shot}/*/`）；只读 test 特征做评价；ref 棋盘掩码为确定性几何操作，不含任何 test 信息、无拟合。
- 纪律：不读 /test/good 进 memory；不用真实缺陷训练/调参；结果如实，失败即归档。

## 2. 门（预注册，按结果不调）
- **G-R1（无损保持，主门）**：六类宏 Pixel-AP(coreset50) − Pixel-AP(full) ≥ **−0.01**（k2 或 k4 任一 shot 成立即算无损窗口成立；两 shot 都成立更强）。
- **G-R2（无灾难类别）**：任一类别 coreset50 相对 full 的 ΔPixel-AP ≥ −0.03（防止单类崩溃被宏掩盖）。
- **G-R3（效率落地，结构性）**：memory patch 数 = 0.5×（棋盘确定性减半），查询端与 map 生成流程完全不变 → 端到端检索成本 ≈0.5×。
- 通过 → Track-3 主线结案：真实 MPDD 六类（k2/k4 seed0）上 A1+50% coreset 无损，可作为论文 Fig07 效率主张的实验支撑；失败 → 归档（合成代理结论不向真实泛化）。

## 3. 产物
- `scripts/innovation_t3_efficiency_20260905/run_real_gate_coreset.py`
- `experiments/dynamic_fusion/innovation_t3_efficiency_20260905/REAL_GATE_{shot}.json`、`REAL_GATE_DECISION.md`

## 4. 执行提示
> 跑 run_real_gate_coreset：对每 cat 加载 dino/clip 冻结缓存 → 对齐/拼接/加权 → (full | 棋盘50% ref) 两路 faiss KNN → 448 map → Pixel-AP/AUROC；输出逐类 + 宏 Δ；按 G-R1/G-R2 判定，记录 memory 规模比。
