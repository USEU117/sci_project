# Manuscript Figure Package

This directory contains Word-ready manuscript figures generated from frozen project evidence.

## Recommended Word use

- Prefer the SVG file in modern Microsoft Word because labels remain sharp and editable.
- If the journal system or Word version mishandles SVG, use the 600-dpi PNG.
- Insert at the intended final width; do not enlarge beyond 17.8 cm (7.0 in).
- Keep aspect ratio locked. Do not crop axes, legends, color bars, or panel labels.
- PDF files are archival/vector alternatives and are usually best for the final publisher upload.
- Quantitative files are publication-ready. `qualitative_local_only/` contains benchmark images and must not be copied into the public reproducibility package.

## Figure captions

**Figure 1. Overview of the frozen dual-encoder visual feature fusion pipeline.** DINOv2 and the AnomalyCLIP image tower independently extract 768-D patch descriptors. After spatial alignment and branch-wise normalization, the descriptors are concatenated with fixed equal scaling and stored in a category-specific normal memory. Nearest-neighbour distance produces the anomaly map. No text embedding or trainable parameter is used.

**Figure 2. Development and frozen-validation protocol.** MPDD is used for method development before the A1 configuration is frozen. BTAD and MVTec AD provide external frozen validation, whereas VisA is reported as in-domain frozen validation because of the AnomalyCLIP checkpoint lineage. All comparisons use identical normal-reference identities and evaluation code without test-label-based selection.

**Figure 3. Configuration-level Pixel-AP gains over the matched DINO-only control.** Each point is one reference-sampling configuration; lines connect the 1-, 2-, and 4-shot values for the same seed. All 36 configurations have positive ΔPixel AP. Configurations within a dataset share the same test set and are not independent datasets.

**Figure 4. Category-level heterogeneity of fusion gains.** Bars show mean category ΔPixel AP across the nine reference configurations. The panel contains the ten lowest- and ten highest-gain categories according to a fixed ranking; the full 36-category version is Figure S1. Blue and vermilion indicate positive and negative mean changes, respectively.

**Figure 5. Complete-metric change relative to the matched DINO-only control.** Cells report the difference between nine-configuration means. BTAD improves in all pixel-level metrics but loses Image AP and Image F1-max, preventing an all-metric dominance claim.

**Figure 6. Three-way control on BTAD and MVTec AD.** Bars show mean ± standard deviation across the nine reference configurations. The AnomalyCLIP image tower is weaker in isolation, whereas fixed fusion exceeds both single-encoder controls in Pixel AP.

**Figure 7. Runtime composition and reference-memory scaling.** Runtime is measured in steady state on MVTec bottle (seed 0, one shot; three warm-up passes and 30 repetitions). Memory values are float32 normal-reference patch banks and exclude test features. The hardware-specific CLIP stage dominates latency.

**Figure 8. Representative localization improvements.** Cases follow the frozen R4 manifest. Heatmaps are overlaid on the input image; DINO and A1 use the same raw anomaly-score scale within each row. Ground-truth contours are shown only for evaluation and visualization.

**Figure 9. Representative negative-transfer cases.** Cases follow the frozen R4 manifest. Shared per-row score scales prevent independent heatmap normalization from exaggerating differences. These examples illustrate that average complementarity does not eliminate category- or image-level failures.

**Figure S1. Full category-level gain/loss distribution.** All 36 dataset-category units are ordered by mean ΔPixel AP across nine reference configurations.

**Figure S2. Shot-wise gain stability.** Points show the mean ΔPixel AP across three reference seeds and error bars show the corresponding descriptive standard deviation. The plot does not imply monotonic improvement with increasing shot count.

## Integrity notes

- Figure values are read from the frozen P1 evidence and the post-freeze BTAD/MVTec CLIP-image-only control.
- Error bars represent descriptive standard deviation across reference-sampling configurations, not independent-dataset uncertainty.
- Qualitative heatmaps are never normalized independently between methods within a sample.
- Sample identities and deterministic selection rules are retained in `source_data/Fig08_Fig09_qualitative_cases.csv`.
