# CLIP image-tower-only complete-metric controls

- Method: AnomalyCLIP ViT-L/14@336 image tower only; no text; frozen patch features; native CLIP grid; branch L2; FAISS IndexFlatL2 k=1; distance/2; dists2map 448; stride=8; image score=max-pool
- This is a post-freeze control and does not change A1 selection.

| dataset | configs | Image AUROC | Image AP | Image F1-max | Pixel AUROC | Pixel AP | Pixel AUPRO |
|---|---:|---:|---:|---:|---:|---:|---:|
| btad | 9 | 0.8660 +/- 0.0212 | 0.7532 +/- 0.0405 | 0.7648 +/- 0.0288 | 0.9539 +/- 0.0038 | 0.4006 +/- 0.0277 | 0.7284 +/- 0.0119 |
| mvtec | 9 | 0.9200 +/- 0.0163 | 0.9566 +/- 0.0070 | 0.9341 +/- 0.0068 | 0.9539 +/- 0.0046 | 0.4654 +/- 0.0180 | 0.8732 +/- 0.0111 |

Note: 3 seeds x 3 shots are reference-sampling configurations on shared test sets, not independent datasets.

Validation: all 18 requested reports are present. For the nine MVTec reports, eight historical pixel-summary cross-checks are within 5e-6. The remaining seed0/k4 case differs only in Pixel AUPRO by 1e-5 (Pixel AUROC and Pixel AP deltas are zero), well inside the frozen project's 5e-4 reconstruction tolerance.
