# Route A — CRAM: Failure Analysis

## Scientific question asked
For K-shot normal references, are true defects far from EVERY reference image while normal variation
is only far from SOME? If so, penalizing the gap between the lucky-minimum distance and the
cross-reference median (or the MAD across references) should raise defect scores.

## What the evidence says (MPDD seed0, k2+k4, fused A1 space)
The mechanism is real but the direction of the effect is the OPPOSITE of the hypothesis on the classes
that matter:

1. **Strong-signal classes are hurt.** connector (AP 0.30→0.38) drops −0.013 (k2) and −0.045 (k4) under
   the A1 gap penalty; tubes −0.004/−0.008; metal_plate −0.0016 (mean). For real defects on these
   objects, one of the K references genuinely IS the best match (same object, similar pose/lighting),
   so `d_med − d_min` is not an "accidental lucky match" — it is real within-class variation. Penalizing
   it suppresses true positives more than it removes false positives.
2. **The gap penalty mostly raises scores in already-normal regions** on medium/hard classes
   (Pixel-AUROC loss −0.0014 for A1), i.e. the correction adds diffuse false positives instead of
   targeted true positives.
3. **Where the effect is positive (bracket_black +0.0071, bracket_white +0.0122 mean), it is not
   causally tied to reference identity**: the shuffled-reference control (same patch pool, attribution
   destroyed) still yields positive gains on these classes. Most of the gain is therefore an artifact of
   splitting the pool into per-block minima, not of modelling "agreement across reference IMAGES".
4. **K-scaling does not rescue the route**: k2 mean gain −0.0008 (A1); the per-shot claim that "K4-only"
   modules exist cannot be supported because the only category with consistent 2-shot positive sign
   (bracket_white) is one class, and its shuffled control is also positive.
5. **A2 (MAD calibrator) is strictly worse** (k4 mean −0.0151; connector −0.053, tubes −0.034): MAD over
   4 references is dominated by a single dissimilar reference, and normal-side mad95 (reference LOO)
    does not separate defect disagreement from normal variation at the region level.
6. Instrumentation is trustworthy: A0 is bit-identical to the frozen A1 pooled map and reproduces the
   frozen A1 mean AP exactly (0.3437 / 0.3883); duplicate-reference control leaves A0 unchanged to 5e-7;
   med-only pooling is worse than A0, ruling out a trivial "switch the aggregator" explanation.

## Why not to keep tuning (per task book §1.3)
- "只差一个统计门 + 均值/最差类均安全 → 允许一次确认" does not apply: the mean is negative and the
  worst class (−0.029) is far past −0.015.
- Per-category exception rules (only use on brackets) are explicitly forbidden as after-the-fact overfitting.
- The negative result is consistent with the accumulated project lesson: in-domain A1 KNN on fused
  features already exploits most of the usable normal-reference structure; adding explicit statistical
  structure on top of the nearest-neighbour distance does not transfer into Pixel-AP.

## What was learned (usable elsewhere)
- Per-image reference decomposition of the A1 memory bank is implemented and verified (bit-exact with
  pooled scoring) → reusable for NORC region calibration and any future per-reference module.
- Reference LOO calibration on fused features is cheap and GT-free → reusable in NORC.
- Hard-class (low AP) false positives are "lucky patch matches" in a way that a small gap correction can
  touch, but the effect is too weak/class-specific to be a route on MPDD.
