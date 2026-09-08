# 既有探针重复性审计

本方向不是既有脚本的完整重复：

- `scripts/innovation_overnight_20260908/probe_precision.py` 只在 v14 k2、16 × 16 pooled cell 上测 FP16/INT8，并包含 runtime 变体；本轮固定 k4、原生 32 × 32，且只做 memory-only roundtrip。
- `scripts/innovation_overnight_20260908/probe_native_resolution.py` 只审计 k2 的 16 vs 32 grid AP，没有 precision、INT8 scale、NN 一致率或距离误差。
- `scripts/innovation_overnight_20260908/probe_support_selection.py` 只评估 support clean 图选择，未做 memory quantization。

因此本轮只复核相同量化定义在更细 native32、k4、heldout/remaining3 条件下的稳定性，不扩展到真实 test 或 runtime 优化。
