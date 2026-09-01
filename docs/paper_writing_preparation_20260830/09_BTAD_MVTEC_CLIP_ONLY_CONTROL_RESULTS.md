# 09. BTAD and MVTec CLIP-Image-Only Controls

## Purpose and status

This post-freeze control completes the three-way comparison on BTAD and MVTec AD:

1. `feature_DINO_only`: the matched DINOv2 branch;
2. `CLIP_image_only`: the AnomalyCLIP image tower without text;
3. `A1_concat`: the frozen, fixed-weight concatenation of the two visual branches.

The control does not alter A1 selection. It uses the same normal-reference identities, 3 seeds × 1/2/4 shots, frozen feature caches, nearest-neighbour rule, anomaly-map reconstruction, and evaluator as the frozen study.

## Mean results over nine reference configurations

Values are mean ± standard deviation across the nine reference-sampling configurations. These configurations share a test set and must not be described as nine independent datasets.

### BTAD — external frozen validation

| Method | Image AUROC | Image AP | Image F1-max | Pixel AUROC | Pixel AP | Pixel AUPRO |
|---|---:|---:|---:|---:|---:|---:|
| DINO-only | 0.9214 ± 0.0175 | **0.9316 ± 0.0268** | **0.8943 ± 0.0189** | 0.9708 ± 0.0015 | 0.6206 ± 0.0183 | 0.7574 ± 0.0092 |
| CLIP-image-only | 0.8660 ± 0.0212 | 0.7532 ± 0.0405 | 0.7648 ± 0.0288 | 0.9540 ± 0.0038 | 0.4006 ± 0.0277 | 0.7284 ± 0.0119 |
| A1 fixed concat | **0.9254 ± 0.0120** | 0.9185 ± 0.0212 | 0.8706 ± 0.0131 | **0.9735 ± 0.0012** | **0.6455 ± 0.0124** | **0.7608 ± 0.0102** |

For Pixel AP, A1 exceeds DINO-only by `+0.0249` and CLIP-image-only by `+0.2449`. A1 also exceeds CLIP-image-only in all 9/9 reference configurations. However, DINO-only remains better than A1 on Image AP and Image F1-max. BTAD therefore supports a localization-oriented complementarity claim, not an all-metric dominance claim.

### MVTec AD — external frozen validation

| Method | Image AUROC | Image AP | Image F1-max | Pixel AUROC | Pixel AP | Pixel AUPRO |
|---|---:|---:|---:|---:|---:|---:|---:|
| DINO-only | 0.9465 ± 0.0172 | 0.9703 ± 0.0092 | 0.9548 ± 0.0098 | 0.9548 ± 0.0057 | 0.5226 ± 0.0184 | 0.9065 ± 0.0097 |
| CLIP-image-only | 0.9200 ± 0.0163 | 0.9566 ± 0.0070 | 0.9341 ± 0.0068 | 0.9539 ± 0.0046 | 0.4654 ± 0.0180 | 0.8732 ± 0.0111 |
| A1 fixed concat | **0.9583 ± 0.0141** | **0.9777 ± 0.0069** | **0.9638 ± 0.0078** | **0.9663 ± 0.0043** | **0.5546 ± 0.0196** | **0.9199 ± 0.0082** |

For Pixel AP, A1 exceeds DINO-only by `+0.0320` and CLIP-image-only by `+0.0892`. A1 exceeds CLIP-image-only in all 9/9 reference configurations and has the best nine-configuration mean for all six metrics.

## Paper-ready interpretation

> The image tower inherited from AnomalyCLIP was not competitive as an isolated nearest-neighbour representation under the matched protocol. Nevertheless, its fixed fusion with DINOv2 improved mean Pixel AP from 0.6206 to 0.6455 on BTAD and from 0.5226 to 0.5546 on MVTec AD. The fused model also exceeded the CLIP-image-only control in every reference configuration on both datasets (18/18 comparisons). These results are consistent with complementary information being contributed by the weaker branch, rather than the fusion merely selecting the stronger standalone representation. The BTAD image-level AP and F1-max exceptions further show that this complementarity is metric-dependent.

Use “consistent with complementary information” rather than “proves feature complementarity”: the control excludes a simple branch-replacement explanation, but does not identify a causal feature mechanism.

## Reproducibility and cross-check

- Machine-readable summary: `experiments/dynamic_fusion/v3_direction_a/clip_only_controls_20260830/summary.json`
- Per-configuration table: `experiments/dynamic_fusion/v3_direction_a/clip_only_controls_20260830/per_config.csv`
- Dataset summary: `experiments/dynamic_fusion/v3_direction_a/clip_only_controls_20260830/dataset_summary.csv`
- Evaluator: `scripts/evaluate_clip_only_complete_metrics.py`
- Frozen A1/DINO source: `submission_repro_20260827/evidence/p1/p1_e_complete_metrics.md`

All 18 requested reports are present. For MVTec, eight of nine rebuilt pixel summaries agree with the historical CLIP-only reports within `5e-6`. The remaining `seed0/4-shot` report differs only in Pixel AUPRO by `1e-5`, while Pixel AUROC and Pixel AP are identical. All nine are well inside the frozen package's established `5e-4` reconstruction tolerance.

