# Route D — NORC: R0 Decision

Date: 2026-09-03 · Status: **IMPLEMENTED (design + minimal machinery); partner-gated**

## Deliverables produced
- `R0_DESIGN.md` — calibration unit (normal reference image, LOO), effective sample
  size (units = images, never patches), augmentation policy, exchangeability
  limits, region non-conformity & conformal p, gating rule (p ≤ 0.05 → modify A1,
  else exact identity).
- `norc.py` — GT-free machinery: `loo_theta`, `region_max_scores`, `conformal_p`,
  `significant_region_mask`, `gate_delta`.
- Unit tests (in `tests/innovation_v10_portfolio/test_norc.py`): conformal rank,
  region extraction, gate identity when nothing is significant, gating restricted
  to significant regions, K<2 guard, no-gt signatures.

## Gate evaluation
| gate | result |
|---|---|
| g0 identity (no aux / nothing significant → output == A1) | **PASS** (exact, verified by construction + tests) |
| g1 AP floor (gated output ≥ A1 − 0.002) | **PASS by construction** (gated delta ≥ 0 only inside significant regions; equals A1 elsewhere) |
| g2 normal-FPR reduction ≥ 20% vs uncalibrated aux | **DEFERRED** — needs an auxiliary module that passed its own R0. CRAM/STR/CAPM all FAILED; MESP is the last candidate |
| g3 cross-seed nominal coverage deviation ≤ 5 pp | **DEFERRED** — requires the same integrated partner at seed0, then seed1/2 confirmation |

## Important operational fact
With MPDD per-class calibration at K=4, n=4 units → the smallest finite-sample
conformal p is 1/5 = 0.20 > 0.05, so the gate can NEVER fire at dev-set K=4. NORC
is thus fully conservative (identity) at K ≤ 4 unless calibration units are
enlarged (e.g. more references, or deterministic augmentations as additional
units — deferred by design). This is safe but means NORC cannot improve ANY metric
on MPDD at K≤4; its role is confined to (a) datasets with more normal references
or (b) guarding a partner module whose activation is region-conditional.

## Standalone verdict
Not an AP route by design (task book §7.3). Cannot claim risk improvement until an
auxiliary module passes R0. **Final status (2026-09-03): ARCHIVED.** MESP (the last
candidate partner) failed its mandatory contrast; with CRAM/STR/CAPM also failed,
there is no auxiliary module that NORC could gate on MPDD. Design + machinery +
tests are retained in this directory as part of the negative-result archive
(Scenario E). NORC remains the correct framework to reuse if a future dataset
provides (a) a passing evidence source and (b) ≥ ~20 calibration units per class
(so the p ≤ 0.05 gate is reachable).
