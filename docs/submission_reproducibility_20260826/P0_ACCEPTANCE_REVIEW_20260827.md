# P0 投稿复现工作验收（2026-08-27）

## 1. 验收结论

结论分成两层，禁止合并表述：

- **P0 研究数值重建：PASS。** 四数据集 A1 concat vs matched feature-DINO-only 的 36 个配置全部重建，数值均在历史结果绝对误差 `5e-4` 内。
- **P0 可发布投稿复现包：CONDITIONAL / 尚未最终通过。** 当前包的内容哈希完整，但它仍是“指标与环境证据包”，不是能在隔离机器上从 compact 预测重新计算像素指标的独立复现包。

机器审计：`P0_ACCEPTANCE_AUDIT_20260827.json`。其中：

- `research_rebuild_complete = true`；
- `submission_repro_package_complete = false`。

## 2. 已通过证据

### 2.1 四数据集缓存矩阵

严格计数为 648 个 NPZ：

| 数据集 | DINO | CLIP | 合计 |
|---|---:|---:|---:|
| MPDD | 54 | 54 | 108 |
| BTAD | 27 | 27 | 54 |
| VisA | 108 | 108 | 216 |
| MVTec AD | 135 | 135 | 270 |
| 总计 | 324 | 324 | 648 |

四个数据集的 `--validate-only` 均返回 9 jobs passed。72 份 branch export report（每数据集 18）均为 `status=passed`。

### 2.2 一类一图 smoke

`outputs/p0_2_smoke/smoke_report.json` 证明：

- DINO grid `32×32×768`；
- AnomalyCLIP image-tower grid `37×37×768`；
- 对齐后 concat 实际维度为 **1536**；旧文档中的 1152 明确错误；
- sample ID 与参考 ID 对齐；
- 无 NaN/Inf；
- 重复计算完全一致；
- 五项泄漏 flag 全 false；
- 峰值显存约 DINO 375 MiB、CLIP 2073 MiB。

### 2.3 四数据集重建结果

| 数据集 | 重建 ΔPixel-AP | 历史值 | 绝对误差 | 配置均值为正 |
|---|---:|---:|---:|---:|
| MPDD | +0.025829 | +0.025830 | 0.000001 | 9/9 |
| BTAD | +0.024895 | +0.024895 | 0 | 9/9 |
| VisA | +0.052353 | +0.052353 | 0 | 9/9 |
| MVTec AD | +0.031962 | +0.031962 | 0 | 9/9 |

36 份 per-config JSON 数量、数据集、类别数和 matched baseline source 均通过结构复核。

### 2.4 测试与包完整性

- CPU 回归：`81 passed in 6.29s`；仅有 Windows 临时目录清理 warning，不影响测试结果。
- `submission_repro_20260827/SHA256SUMS`：83 项全部校验通过。
- 包体约 0.331 MiB；未包含数据集和第三方权重。

## 3. 未通过问题

### P0-H：所谓 `predictions_compact` 不是预测

该目录当前只有 4 个 summary JSON 与一份 acceptance JSON，没有逐图 `sample_id`、A1/DINO patch-distance map、图像分数或其他可重放预测。因此它只能重算“汇总数字的汇总”，不能独立重新计算 Pixel-AP/AUROC/AUPRO。

修复要求：每个 `dataset × seed × shot × category` 至少保存：

- `sample_ids`；
- concat 与 matched DINO-only 的低分辨率 patch anomaly maps（建议 float16/float32 压缩）；
- map grid/resize/stride 元数据；
- reference IDs；
- dataset/category/seed/shot；
- 生成代码、split、checkpoint hash。

不要打包 GT mask；包内重算脚本应从用户按许可准备的数据根读取 mask，并按 sample ID 对齐。

### P0-H：没有包内 CPU 重算脚本

README 指向仓库根目录的 `scripts/p0_3_evaluate_a1_rebuild.py`，但 compact 包内部没有脚本，也没有固定的源码版本。因此将包复制到新目录后不能执行所宣称的 CPU 重算。

修复要求：新增包内 `recompute_tables.py`，读取 compact patch maps + 用户数据 mask，重新计算逐类、逐配置和四数据集论文表；另提供 `--verify-only` 检查 sample ID、数量、shape、hash 和泄漏契约。

### P0-I：没有源码提交标识，所有 P0 新内容仍未跟踪

当前 `HEAD=9cb9869`，但 smoke、P0-3 evaluator、恢复的 `src/utils.py`、重建报告和 compact 包均为 untracked。包内也没有 `SOURCE_COMMIT.txt`。因此第三方无法获得产生这些结果的确定源码状态。

修复要求：

1. 审阅所有新增文件；
2. 排除不应提交的大缓存；
3. 提交代码、文档、轻量 JSON 与 compact 预测；
4. 在包中写入最终 commit SHA、dirty=false、Python/依赖和生成命令；
5. 提交后重新生成 `SHA256SUMS`。

### 历史冻结与本次重建被混淆

旧 `freeze_a1_mpdd.py --verify` 当前仍失败：历史 manifest 声明 216 个缓存，其中 108 个 legacy v2 baseline cache 缺失，108 个重建 feature cache 与旧压缩文件 size 不同。重建数值等价不代表历史文件 byte-identical。

修复要求：保留旧 freeze manifest 作为历史证据，禁止覆盖；为本次重建单独创建 `rebuild_manifest_v2.json`，记录 648 个当前特征缓存或最终 compact 预测的 hash，并明确 `numerically_equivalent_to_historical=true`、`byte_identical_to_historical=false`。

### 方法文档仍残留 1152 与 multimodal 命名

历史 `METHOD_CARD.md` 仍写 concat 1152，并使用 `Multimodal`/`anomalyclip_text` 容易让论文误称文本融合。

修复要求：不要静默篡改历史 frozen 文件；新增当前投稿版 `METHOD_SPEC_V2.md`，明确两个 image encoder、768+768=1536，并说明 `anomalyclip_text` 仅是历史目录名，不代表文本特征参与 A1 推理。

### 许可证说明仍需正式化

当前许可证文档只有概括，没有每个数据集/权重的精确官方 URL、访问日期、许可证文件/hash 和本仓库自研代码 LICENSE。不能使用“默认研究用途”代替明确许可证。

## 4. 修复后的最终 P0 门禁

P0 最终通过必须同时满足：

1. 648 个四数据集特征缓存结构审计通过；
2. 36 个重建报告与历史值容差通过；
3. smoke 证明 1536、对齐、确定性和无泄漏；
4. compact 包 SHA256 全通过；
5. compact 包包含可重放的逐图 patch maps；
6. 包内 CPU 脚本能从 patch maps + 合法数据 mask 重算所有论文指标；
7. 有独立 rebuild manifest，不冒充旧 freeze byte identity；
8. 有最终源码 commit SHA 且工作树干净；
9. 投稿版方法说明修正 1536/双视觉语义；
10. 数据、权重和代码许可证清单完整。

满足以上条件后，才能把 `submission_repro_package_complete` 改为 true。补齐这些工作不需要重新导出四数据集 DINO/CLIP 特征。

## 5. P0 之后的顺序

1. P1-A：从 compact patch maps 做 paired bootstrap / category bootstrap 置信区间。
2. P1-B：整理 shot-wise mean±std、worst category、负增益类别和失败图。
3. P1-C：效率表（推理时间、峰值 VRAM/RAM、memory bank 与包大小）。
4. P1-D：公平性对照表，分开不同训练/适配协议。
5. P2：按双编码器视觉固定融合主线重写论文。
6. P3：实时核验 SCI 四区候选期刊后适配格式。

当前不需要新增动态融合、视觉—文本路由、SubspaceAD 或新 backbone 实验。MVTec AD 2 对四区投稿是可选增强项，不是 P0/P1 前置条件。
