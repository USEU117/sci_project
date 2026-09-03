# Route C — CAPM: R0 Decision (MPDD seed0 k1)

Date: 2026-09-03 · Status: **FAIL → ARCHIVED** (after feasibility PASS)

## Phase 1 — feasibility statistics (PASS)
DINO-vitb14 mutual-NN + RANSAC affine, query vs the k1 reference:
median inlier ratio per class 0.478–0.737; **100% of images** inlier ≥ 0.3 in every
class; mean inlier reprojection error < 1 patch unit. Alignment is technically
feasible on MPDD (rigid product images). This alone does NOT rescue the route.

## Phase 2 — pixel evaluation (FAIL, pre-registered post-pass gates)
score = d_global + 0.25·relu(d_pos − d_global), d_pos = fused NN restricted to the
aligned radius-2 neighbourhood (inverse-affine coordinates); identity fallback if
RANSAC inlier < 0.30 (all 458 test images were "reliable" → fallback never fired).

| gate | threshold | result |
|---|---|---|
| mean Δ Pixel-AP (6 classes) | ≥ +0.003 | **−0.0275 FAIL** |
| ≥ 4/6 positive | — | **0/6 FAIL** |
| worst category | ≥ −0.015 | tubes **−0.0909 FAIL** |
| real vs random-homography | ≥ +0.003 | **−0.0033 FAIL** |

Per class ΔAP: bracket_black −0.0016 · bracket_brown −0.0060 · bracket_white −0.0114 ·
connector −0.0164 · metal_plate −0.0385 · tubes −0.0909.
Random-transform control is not below real alignment — on bracket_white random even
beats real (+0.0143 vs −0.0114) → the effect is noise, not alignment value.
Identity fallback max|diff| = 0 (bit-exact by construction) — verified.

## Decision
Feasibility ≠ value: with a single reference per class, pose/lighting differences
between the reference and each query make the position-restricted distance
`d_pos > d_global` for most normal patches (aligned-neighbourhood texture simply
does not match), so the penalty inflates scores diffusely across normal regions —
the same failure mode as CRAM. CAPM is archived; do NOT re-tune (per-category
exceptions and reliability thresholds are precluded by the task book).
