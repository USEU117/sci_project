# Route B — MESP: R0 Decision (geometry audit + dino-only promise probe)

Date: 2026-09-03 · Status: **ARCHIVED** (promise probe + mandatory contrast failed;
full fused R0 not executed — cost not justified)

## Phase 1 — geometry audit (PASS, 10 normal refs, DINO-vitb14@448)
| view | inverse RMSE (px) | interior feature cos |
|---|---|---:|---:|
| hflip | 0.00 | 1.0000 |
| bright ×0.9 / ×1.1 | 0.95 / 1.91 | 0.9966 / 0.9964 |
| rot −5° / +5° | 2.64 / 2.64 | 0.9797 / 0.9785 |

Cost: +0.055 s per extra view per image, peak VRAM 0.39 GB. Transform chain is
faithful; interpolation does NOT manufacture perfect stability (rotation retains a
real 2.6 px residual and cos < 1.0). Mechanism is geometrically implementable.

## Phase 2 — promise probe (DINO-only maps, seed0 k1, all 6 classes)
Explicitly NOT the pre-registered fused gate — a cheap decision probe.

| candidate | mean ΔAP | positive classes |
|---|---:|---:|
| B1 median over 6 views | +0.0034 | 5/6 (metal_plate +0.0090, tubes +0.0077) |
| B2 stability-scaled base | +0.0003 | 4/6 |
| B1 with rot+5 misaligned (control) | **+0.0036** | — |

## Gate-relevant observations
- Pre-registered g4 contrast ("true multi-view beats misaligned-view control by
  ≥ +0.002"): real (+0.0034) does NOT beat the misaligned control (+0.0036) —
  difference −0.0002. **Control FAIL.** Median-across-views is robust to a 14 px
  displacement of one view; the gain comes from smoothing/diversity, not from
  equivariant persistence of defect hotspots.
- Pre-registered g2 (mean ≥ +0.005, 3/3 shots positive) not reachable at the
  probe level (mean +0.0034, and misaligned control matches it, so any fused
  version would have to beat BOTH bars plus the contrast).
- B2 (stability suppression) is near-zero (+0.0003): unstable-hotspot suppression
  does nothing measurable on MPDD dino maps.

## Decision
The single most important MESP claim — that cross-view persistence identifies real
defects while misalignment does not — fails its mandatory control at the cheapest
possible level. The residual smoothing effect (+0.003~0.004 on dino maps) is below
the +0.005 bar and is not attributable to geometry. Executing the full fused R0
(dino+clip, 6 views × 3 shots, ~1–2 h GPU) is therefore not justified: the
mechanism's only observable effect would inherit the same control failure. MESP is
ARCHIVED per task book §12 (contrast failure → archive; do not tune).
