# Route E — STR: R0 Decision (region information value, MPDD seed0 k4)

Date: 2026-09-03 · Status: **FAIL → ARCHIVED (unified route)**

## What was measured
Diagnostic only (no fusion). STR residual = 2-level Haar robust z (gray + R-G + B-Y,
median/MAD calibrated on the 4 normal references). Task: rank A1-MISSED GT-defect
pixels (GT components where <20% of pixels exceed the reference-LOO fused d_min
95th-percentile θ) above normal pixels. Predictors compared at Pixel-AP.

## Results (per category)

| category | missed px | AP_STR | AP_A1 | Δ(STR−A1) |
|---|---|---:|---:|---:|
| bracket_black | 6029 | 0.1147 | 0.2407 | **−0.1260** |
| bracket_brown | 308 | 0.0440 | 0.0029 | +0.0411 |
| bracket_white | 14 | 0.0001 | 0.0003 | −0.0002 |
| connector | 0 | — | — | — (no missed comps) |
| metal_plate | 0 | — | — | — (no missed comps) |
| tubes | 934 | 0.0098 | 0.0343 | −0.0244 |

Mean Δ = **−0.027** · positive classes 1/6.

## Gate evaluation (pre-registered, task book §8.4)

| gate | threshold | result |
|---|---|---|
| g1 region info-value AP | STR ≥ A1 + 0.05 AND ≥4/6 positive | **FAIL** (mean −0.027; best class +0.041 < +0.05; only 1/6 positive) |
| g2 true vs misaligned ≥ +0.02 | — | +0.030 passes but moot (g1 failed) |
| g3 runtime < 20% A1 | — | plausible, moot |

## Decision
**FAIL.** STR does not add information where A1 is wrong, at any usable level:
its best single class (bracket_brown +0.041) is below the +0.05 bar and is the only
positive class. On bracket_black (the biggest miss set) A1's own score identifies
its misses far better than the spectral residual (−0.126). Per task book §8.4
("如果只在一个纹理类别有效，归档为类别专项观察") STR is archived as a unified
route. bracket_brown (+0.041, coarse-bracket texture/scratches) is retained as a
per-category observation only.
