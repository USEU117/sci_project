# Route C — CAPM: Failure Analysis

## Question
Global patch memory matches any-position patches and misses "locally normal at the
wrong position" anomalies. Can query-reference geometry be canonicalized (mutual
NN + RANSAC affine on frozen DINO) so a position-restricted search adds signal?

## Facts
1. Alignment itself is excellent on MPDD (median inlier 0.48–0.74, reproj < 1
   patch unit) — the geometry problem is NOT the blocker.
2. The position-restricted distance hurts every class and hurts the most on the
   classes A1 is best at (tubes −0.091, metal_plate −0.039, connector −0.016).
3. Root cause: with K=1 the reference is one specific physical sample. Across a
   class, query images vary in pose/lighting at the ~1–3 patch scale (inlier
   reprojection < 1 patch is the *consensus* error; per-patch residuals reach the
   radius bound). A true-normal query patch whose aligned neighbourhood in the
   reference has no similar patch gets d_pos ≫ d_global and is penalized like an
   anomaly. The penalty therefore acts as a diffuse *pose-change detector*, not a
   wrong-position-defect detector.
4. Random-homography control shows no real-vs-random separation (−0.0033, and
   bracket_white random > real): any geometric coupling we add only costs.
5. MPDD's defect set is dominated by appearance anomalies (scratch, dent,
   deformation, contamination), not by "correct local appearance at wrong
   position"; logical/misplacement anomalies are rare here. This is a
   dataset/route mismatch, not just an implementation issue.

## Lessons
- Registration feasibility must never be used as a proxy for anomaly-detection
  value; position-conditional memory needs a dataset with real misplacement
  anomalies and more than one reference per pose to build a position-conditional
  normal distribution. Both are absent on MPDD.
- Diffuse score inflation is the recurring killer: CRAM (median gap), CAPM
  (position penalty) both add a smooth positive term over normal regions. Any
  future "raise-where-uncertain" module must demonstrate sparse, region-selective
  activation on normal images before an AP claim.
- The reusable asset from this route is the alignment pipeline (mutual NN + RANSAC
  affine, all GT-free) — potentially useful for MESP's transform audit or future
  datasets, not for position-conditioning A1 on MPDD.
