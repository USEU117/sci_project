# V2 真实 bank 量化 QA 结论（FP16 / INT8 存储 roundtrip）

状态：`complete`（s0 k2 与 k4）；决策：`REALBANK_QUANT_QA_PASS`（工程级存储选项，非新颖算法，不改冻结 A1）。

范围：冻结 A1 concat（pca0/whiten0/w0.5，1−max-cos，448 map，Pixel-AP stride8）在**真实 MPDD test**（6 类全量、seed0 k2/k4）上的 memory-only 存储量化 QA；query 保持 F32；未运行低精度 runtime matmul；无 test 标签拟合/选择。
脚本：`scripts/validation_followup_20260908/probe_realbank_quant.py`；结果：`RESULTS_s0_k2.json`、`RESULTS_s0_k4.json`。

## 对账（control 必须复现冻结宏）
| shot | control macro Pixel-AP（实测/冻结） | parity |
|---|---:|---|
| k2 | 0.343698 / 0.343706 | 通过 |
| k4 | 0.388328 / 0.388328 | 通过 |

## 结果（真实 bank，六类宏）
| shot | 候选 | macro ΔAP | worst 类 ΔAP | NN 一致率 | 距离 err p99（最差类） | 存储比 |
|---|---|---|---:|---:|---:|---:|---:|
| k2 | FP16 | +7.6e-06 | −6.0e-07 | 0.99985 | 1.3e-05 | 0.500 |
| k2 | INT8(pc) | +2.1e-05 | −0.00012（bracket_white） | 0.99563 | 3.0e-04 | 0.250 |
| k4 | FP16 | +1.1e-07 | −1.4e-06 | 0.99981 | 1.1e-05 | 0.500 |
| k4 | INT8(pc) | −2.2e-05 | −0.00016（bracket_white） | 0.99526 | 2.7e-04 | 0.250 |

逐类明细见 `RESULTS_s0_k{2,4}.json`。

## 门判定（预注册，两 shot 均须过）
| 门 | FP16 | INT8 |
|---|---|---|
| macro ΔAP ≥ −0.0005 / −0.002 | 通过 | 通过 |
| worst 类 ΔAP ≥ −0.002 / −0.01 | 通过 | 通过 |
| NN 一致率 ≥ 0.999 / 0.99 | 通过 | 通过 |
| 存储比 ≤ 0.51 / 0.30 | 通过（0.500） | 通过（0.250） |

## 诚实边界
1. **这是"持久化存储"压缩，不是运行时压缩**：解码后仍是 F32，resident bank 内存不变；decode/scale 有一次性小成本（prepare < 数秒/类）。低精度检索未运行，不宣称加速。
2. **与真实排序相关的质量损失可忽略**：最差类别 ΔAP ≤ −1.6e-4（INT8，bracket_white 两 shot 一致），NN 身份 ≥ 99.5%，距离误差 p99 ~1e-3 量级；真实 QA 未复现任何类别的合成探针最差折（−0.02）那样的崩塌。
3. **是否采用**：作为可选项记录——若未来把持久化 memory bank 纳入复现包/部署（现仓库不存 bank、`*.npz` 与 `outputs/` 均不入库），FP16（2×）或 INT8（4×）都是安全的存储表示；不改变冻结 A1 数字与 manifest。
4. 未触碰 test 标签、未调门、未按结果选择类别。

## 复现
```powershell
.\.venv-patchcore\Scripts\python.exe scripts\validation_followup_20260908\probe_realbank_quant.py --shot 2
.\.venv-patchcore\Scripts\python.exe scripts\validation_followup_20260908\probe_realbank_quant.py --shot 4
```
