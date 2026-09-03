# Route D — NORC: R0 Design (calibration unit, effective sample size, exchangeability)

Date: 2026-09-03 · Scope: task book 19 §7 (risk-controlled auxiliary gating)

## Purpose
NORC is a SAFETY module, not an AP route. Every failed route (CRAM, CAPM) added a
diffuse positive score term; NORC's job is to decide, per candidate region and with
an auditable p-value, whether ANY auxiliary module may modify the frozen A1 map.
Value metric: normal-image FPR / normal-pixel high-score rate and explainable
identity fallback — explicitly NOT raw AP (§7.3).

## Calibration unit (definition)
- **Unit = one normal reference IMAGE** (the K-shot train/good images of a class).
- Calibration protocol: leave-one-reference-out (LOO). For reference image r,
  build the frozen A1 memory bank from the other K−1 references, run r as a normal
  pseudo-query, and record its anomaly-map statistics. This yields exactly K
  calibration units per class at a given (seed, shot).
- **Patches are NOT units.** Patches within one pseudo-query are spatially
  correlated (nearest-neighbour fields, blurred maps). Counting patch samples as
  independent would inflate the effective sample size; NORC never does that.

## Effective sample size & augmentation policy
- Base effective sample size per class = K (LOO images). With MPDD K=1 there is no
  LOO — calibration is IMPOSSIBLE at k1; NORC therefore activates only for K ≥ 2
  (seed0: k4 → 4 units).
- Optional deterministic mild augmentations (hflip, brightness ×0.9/×1.1,
  rot ±2°) are allowed ONLY when the class geometry is stable; they are a second
  independent pseudo-image per reference, still one unit each. No random crops, no
  per-class tuning. Not used in the minimal implementation below (deferred).
- Reported sample size is ALWAYS the unit count, never the region/patch count.

## Exchangeability limits (written down, no overclaiming)
- Exchangeability holds within the in-domain dev set (normal references vs normal
  regions of the same class under the same imaging). Cross-domain (BTAD/MVTec)
  coverage claims are NOT made; the protocol measures cross-seed nominal-coverage
  deviation instead (gate g3: ≤ 5 pp).
- Because n is small (K=4 → n=4 per class per seed), region p-values are
  conservative (p = (1 + #{calib ≥ score})/(n+1), finite-sample conformal rank);
  the 5%-level gate is therefore far more conservative than nominal in practice.

## Non-conformity & region definition
- Region = connected component of {fused A1 d_min map > θ} on the A1 grid, where
  θ = per-class 95th percentile of the reference-LOO per-patch d_min distribution
  (reference-only, GT-free; identical statistic used in STR R0).
- Region score (non-conformity) = region max of the A1 d_min map.
- AUX modification is applied only inside regions whose finite-sample p ≤ 0.05;
  everywhere else the output is EXACTLY A1 (identity).

## Minimal implementation (this R0)
`norc.py` provides (GT-free signatures):
- `loo_theta(fused_ref_blocks)` — θ from LOO per-patch d_min;
- `region_max_scores(map_grid, theta)` — labeled components + region max d_min;
- `conformal_p(score, calib_scores)` — finite-sample p;
- `gate_delta(delta_map_grid, a1_grid, theta, calib_region_maxs, alpha=0.05)`
  — returns gated delta map (0 outside p≤0.05 regions) and stats
  (regions, activated regions, identity ratio);
- `identity == no auxiliary` (pass-through) — trivial exact A1.

## Gates (from task book §7.4)
- g0 identity: gated output with no auxiliary == A1 exactly (max|diff| = 0).
- g1 AP floor: MPDD seed0 gated output ≥ A1 − 0.002 (by construction ≥ A1 when the
  gated delta is applied inside p≤0.05 anomaly regions; verified numerically).
- g2 FPR: normal-image modified-pixel rate of the gated aux ≥ 20% lower than the
  uncalibrated aux.
- g3 cross-seed nominal coverage deviation ≤ 5 pp (measured when a partner route
  passes R0 and NORC integrates at seed0; seeds 1/2 then confirm).
g2/g3 require a real auxiliary module that passed its own R0 (CRAM/STR/CAPM all
FAILED; MESP is the last candidate). This R0 therefore delivers g0/g1 evidence and
the gating machinery; g2/g3 are deferred and will be reported against whichever
module is integrated (MESP or none).

## Leakage
All calibration statistics (θ, region max distribution) come from reference images
only. gating functions never receive gt_masks/gt_labels; unit tests scan for
forbidden keys and verify GT-independence of the maps.
