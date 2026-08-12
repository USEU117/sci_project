# DynamicFusion V3.1 next actions

Updated: 2026-08-12 10:25 Asia/Shanghai

This is the authoritative next-action list for V3.1. The root NEXT_ACTIONS.md
contains legacy mojibake text and is not rewritten to avoid destroying history.

## Completed

- Gate A1 and Gate A2 completed; Gate A2 scientifically failed.
- Counterfactual, leakage and provenance audits passed.
- V3 tests pass 19/19; the project CPU suite passes 73/73.
- AdaptCLIP MPDD seed0/K1 inference completed in
  `adaptclip_gate_a_v2`: six categories and 458 samples.  The cache acceptance
  audit passed: exact visual-branch sample-ID alignment, finite scores/maps,
  matching map/mask shapes, no runner traceback/OOM, and all router leakage
  flags false.
- Gate A1 was recomputed with AdaptCLIP (not the historical AnomalyCLIP cache)
  and passed: oracle headroom is positive in 5/6 image categories and 6/6
  pixel categories; AdaptCLIP is stronger in 204/407 anomaly regions.
- A separate six-image K=1 normal-reference inference was used to fit the
  AdaptCLIP calibration.  It contains no test predictions, labels, masks, or
  test-set statistics.
- Gate A2 was recomputed with that calibration and failed: the safest candidate
  still has mean pixel AP delta -0.07369, AUROC delta -0.00769, and 0/6
  categories with positive AP.  This is a valid negative result, not a cache
  or GPU failure.

## Frozen decision

V3 is **not frozen as a successful dynamic-fusion method**.  No AdaptCLIP
multi-seed/multi-shot GPU matrix, no BTAD access, and no paper claim of stable
dynamic rescue may be started from this branch.  Preserve the A1/A2 reports as
negative-result evidence.  Any future V3 revision must introduce a new,
pre-registered label-free reliability signal and restart from Gate A1.
