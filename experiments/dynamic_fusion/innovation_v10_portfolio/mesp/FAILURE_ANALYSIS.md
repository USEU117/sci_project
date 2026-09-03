# Route B — MESP: Failure Analysis

## Hypothesis
A true defect persists across mild lighting/flip/small-geometry views after inverse
transform; interpolation or single-lucky-match hotspots do not. Cross-view
agreement should therefore sharpen the anomaly map (median across views, or
stability-weighted base).

## Evidence against it
1. Views are individually faithful (audit: flip exact, rotation 2.6 px residual
   after inverse, brightness ~1–2 px). So the transform chain is not the problem.
2. On DINO-only maps (seed0 k1, 6 classes) the median-over-views gain is small
   (+0.0034 mean, 5/6 positive) and concentrated on the two classes that are
   already strong (metal_plate +0.0090, tubes +0.0077) — i.e. it is a small
   smoothing gain on strong classes, not a rescue of weak classes.
3. **The misalignment control is the decisive negative**: displacing the rot+5°
   view by 14 px before the median does not degrade the result (+0.0036 vs
   +0.0034). If cross-view persistence were doing real work, an unaligned view
   would disagree with the true geometry and the median would worsen. It does not,
   because after gaussian smoothing (sigma=4) + strided metrics, individual views
   are largely redundant; any monotone combination of them inherits the base map's
   ranking. This mirrors the doc's own "monotone-copy baseline" concern — the
   median of highly correlated views is nearly a monotone function of the base.
4. B2 (stability-weighted suppression) is ~0 (+0.0003): MAD across views carries no
   usable hotspot-vs-defect separation at MPDD's patch scale, again consistent with
   views being near-monotone copies after smoothing.

## Lesson (same root cause as CRAM/CAPM/STR, stated differently)
Any module whose effect is a smooth, approximately-monotone transformation of the
A1 map (add a positive term — CRAM/CAPM; average across correlated views — MESP)
moves the whole score field and cannot separate the few true-defect pixels from the
much larger normal population on MPDD. Modules that were *suppressive and sparse*
(B2) fail because the auxiliary evidence source they need (unstable hotspots that
are genuinely separable) does not exist at MPDD's scale. Multi-view augmentation is
cheap and can still be useful as a test-time *robustness* tool (reporting), but not
as an anomaly-detection mechanism here.
