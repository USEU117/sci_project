# 06. Table and Figure Plan

## 1. Main-paper figures

### Figure 1. Method overview

Show one query/reference flow:

`normal references + query → frozen DINOv2 / frozen AnomalyCLIP image tower → patch-grid alignment → branch L2 → fixed concat → global L2 → normal memory k-NN → anomaly map`.

Required visual labels:

- `image tower only; no text at inference`
- DINO input 448, CLIP input 518
- 768-D + 768-D → 1536-D
- CLIP grid resized to DINO grid
- k=1, distance/2, sigma=4, map 448
- trainable parameters = 0

Do not use text-prompt icons, router/gate graphics, or the word multimodal.

### Figure 2. Experimental protocol and dataset roles

A compact timeline/flow showing:

- MPDD development and method freeze;
- BTAD/MVTec external frozen validation;
- VisA in-domain frozen validation;
- 3 seeds × 1/2/4 normal shots;
- matched DINO-only control with identical references and evaluator.

This figure helps make the leakage contract immediately visible.

### Figure 3. Configuration-level Pixel-AP gains

Preferred form: four small panels, x-axis = 1/2/4 shots grouped by seed, y-axis = A1 − DINO Pixel-AP, with a horizontal zero line. Every point should be above zero. Use exact values from `p1_a_bootstrap_ci.json`, not rounded manuscript numbers.

Caption must say that configurations within a dataset share the same test set.

### Figure 4. Category-level gain/loss map

Horizontal diverging bar chart or heatmap for mean category ΔPixel-AP. Highlight:

- positive: `visa@pcb1`, `visa@cashew`, `mvtec@toothbrush`, `mpdd@metal_plate`;
- negative: `mvtec@leather`, `visa@chewinggum`, `mvtec@hazelnut`, `visa@candle`, `mpdd@bracket_brown`.

Use all categories in the supplementary version; main paper can show the strongest and weakest groups with a clear selection rule.

### Figure 5. Qualitative successes and failures

For each example show: input, ground truth, DINO-only map, A1 map, and optionally difference map. Use at least two successes and two failures from different datasets.

Selection must be deterministic and declared before layout work, for example:

- select categories by cross-configuration mean ΔPixel-AP;
- within each selected category, use the median per-image improvement/loss among anomalous images, not the most visually dramatic sample;
- use the same seed/shot convention for all panels.

Candidate IDs are available in `p1_b_failure_samples.csv`; success sample IDs still need an equivalent deterministic extraction table.

### Figure 6. Accuracy–cost summary (optional)

Plot DINO-only, CLIP-only where available, and A1 by Pixel-AP versus steady-state latency or memory-bank size. Only include methods measured on the same hardware and code path.

## 2. Main-paper tables

### Table 1. Protocol-aware related work comparison

Columns: method, backbone, normal reference use, auxiliary/source training, target-normal tuning, test-time adaptation, text at inference, learned fusion. Source: `p1_d_fairness_table` plus verified 2025–2026 literature.

### Table 2. Dataset and split statistics

Columns: dataset, role, categories, normal reference pool, test normal, test anomalous, annotation type, official license/source. Regenerate counts from local manifests; do not type them from memory.

### Table 3. Main matched-control results

Rows: four datasets; columns: A1 Pixel-AP mean ± std, DINO Pixel-AP mean ± std, Δ, positive configurations. This is the headline table.

### Table 4. Complete metrics

Report the six metrics for A1 and DINO. If page space is limited, keep deltas in the main paper and move full mean ± std rows to supplementary material. BTAD negative image metrics must remain visible.

### Table 5. Shot-wise stability

Dataset × shot rows, mean ΔPixel-AP ± seed std. Source: `p1_a_bootstrap_ci.md`.

### Table 6. Efficiency and memory

Zero trainable parameters, per-stage latency, total throughput, peak VRAM/RAM, and memory bank sizes for each dataset/shot. Clearly label hardware and measured scenario.

## 3. Supplementary tables

- Full 36-configuration metrics.
- Full category-level gain table.
- Per-configuration bootstrap confidence intervals.
- Failure sample IDs and per-image deltas.
- Baseline fairness table with exact implementation source.
- Checkpoint and split hashes.
- Dataset license and access table.
- Historical dynamic-route negative results, with invalid/test-selected experiments marked ineligible.

## 4. Existing sources

| Planned artifact | Existing source | Status |
|---|---|---|
| Main result table | `submission_repro_20260827/evidence/p1/p1_e_complete_metrics.*` | ready |
| Method overview | `METHOD_SPEC_V2.md` | complete: `figures_20260830/Fig01` |
| Dataset-role diagram | split manifests + package README | complete: `figures_20260830/Fig02` |
| Configuration-gain figure | `p1_a_bootstrap_ci.json` | complete: `figures_20260830/Fig03` |
| Category-gain figure | `p1_b_failure_boundaries.*` | complete: `figures_20260830/Fig04` and `FigS01` |
| Complete-metric delta heatmap | `p1_e_complete_metrics.json` | complete: `figures_20260830/Fig05` |
| Three-way CLIP/DINO/A1 control | P1-E + `clip_only_controls_20260830` | complete: `figures_20260830/Fig06` |
| Efficiency and memory figure | `p1_c_efficiency.*` | complete: `figures_20260830/Fig07` |
| Success qualitative figure | compact maps + frozen R4 manifest + local data | complete: `figures_20260830/Fig08` |
| Failure qualitative figure | compact maps + frozen R4 manifest + local data | complete: `figures_20260830/Fig09` |
| Shot-wise stability figure | `p1_a_bootstrap_ci.json` | complete: `figures_20260830/FigS02` |
| Efficiency table | `p1_c_efficiency.*` | ready |
| Fairness/related-work table | `p1_d_fairness_table.*` | ready; add new close neighbors |

## 5. Figure integrity rules

- Never normalize DINO and A1 heatmaps independently in a way that visually exaggerates one method; use a declared shared scale per sample or provide raw score ranges.
- Do not crop away difficult background regions without saying so.
- Ground-truth masks are for evaluation/visualization only, never method selection.
- Keep exact sample IDs in supplementary material.
- Do not redistribute source images if dataset terms prohibit it; the manuscript may display research examples under the publisher workflow, while the reproducibility package should retain IDs rather than copies.
