# Figure Package QA Report

Checked on 2026-08-30 after two visual revision rounds.

## Package completeness

- 9 quantitative/diagram figures × 3 formats: 9 SVG, 9 PDF and 9 PNG files.
- 2 qualitative figures as high-resolution PNG files.
- 1 contact sheet for rapid visual review.
- 8 source-data CSV files and one JSON file manifest with SHA256 values.
- Captions, Word insertion guidance and integrity notes are included in `README.md`.

## Automated file checks

- All 9 quantitative PNG files decode successfully.
- Every quantitative PNG records 600 × 600 dpi.
- Quantitative PNG dimensions range from 3,349 × 1,272 to 4,248 × 1,650 pixels for normal full-width figures; taller supplementary figures reach 4,707 pixels in height.
- All 9 SVG files parse as valid XML.
- `pdfinfo` reports exactly one page for each of the 9 PDF files.
- No vector file is unexpectedly small; the smallest PDF is larger than 27 kB.

## Numerical and provenance checks

- Figure 3 reads all 36 frozen configurations; 36/36 ΔPixel-AP values are positive.
- Figures 4 and S1 contain 36 unique dataset-category units.
- Figure 5 is generated from the frozen six-metric dataset deltas and retains the two BTAD image-level losses.
- Figure 6 reads the 18-row BTAD/MVTec CLIP-image-only control summary and the frozen A1/DINO summary.
- Figures 8 and 9 use four sample IDs already recorded in the frozen qualitative manifest:
  - `visa/cashew/Data/Images/Anomaly/085.JPG`
  - `mvtec/toothbrush/test/defective/004.png`
  - `visa/chewinggum/Data/Images/Anomaly/037.JPG`
  - `mvtec/leather/test/poke/016.png`
- DINO and A1 qualitative overlays use the same raw anomaly-score minimum and maximum within each sample row. They are not independently normalized.

## Manual visual checks

The contact sheet and full-resolution versions of Figures 1, 2, 7, 8 and 9 were inspected directly.

- Revision round 1 found text overflow in Figures 1 and 2 and color-bar overlap in Figures 8 and 9.
- Revision round 2 corrected box spacing, line wrapping, title clearance, dedicated color-bar columns and qualitative row spacing.
- Final inspection found no clipped labels, overlapping legends, cropped axes, ambiguous color scales or method-flow contradictions.
- Color choices are redundant with sign, position, label or marker shape; interpretation does not depend on red/green discrimination alone.

## Word suitability

- Default intended width: 17.8 cm, with aspect ratio locked.
- SVG is preferred in recent Word versions; 600-dpi PNG is the compatibility fallback.
- All essential labels remain readable on the package contact sheet at reduced scale.
- Quantitative figures use a white background and embedded/outlined vector text settings compatible with normal publisher conversion.

## Distribution boundary

Figures 8 and 9 contain local benchmark images. They are suitable for manuscript drafting and publisher submission subject to the dataset and publisher terms, but they must not be copied into the public compact reproducibility package. Quantitative figures and diagrams do not contain dataset images.
