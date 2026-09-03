# Route A — CRAM: R0 Decision (MPDD seed0, k2+k4)

Date: 2026-09-03 · Status: **FAIL → ARCHIVED (unified route)**

## Instrumentation sanity (PASS)

| check | result |
|---|---|
| g1 identity: pooled-A1 vs per-image-ref min (max\|diff\|) | **0.0** (bit-exact across 12 category×shot runs) |
| A0 mean Pixel-AP vs frozen A1 matrix | k2 0.3437 == 0.3437 · k4 0.3883 == 0.3883 (exact) |
| c1 duplicate refs: A0 AP unchanged (max \|Δ\|) | ≤ 5e-7 |
| c3 med-only control mean AP | k2 0.3407 < A0 · k4 0.3484 < A0 → gain is not just a pooling swap |
| no-leakage unit tests | 6/6 pass (AST scan + determinism + identical-refs degeneracy) |

## Candidate gates (MPDD seed0, mean over k2+k4)

| gate | threshold | A1 (median-gap) | A2 (MAD-calibrated) |
|---|---|---|---|
| g2 mean Δ Pixel-AP (2 shots) | ≥ +0.005 | **−0.0028 FAIL** | **−0.0076 FAIL** |
| g3 positive categories | ≥ 4/6 | 3/6 FAIL | 3/6 FAIL |
| g4 worst category | ≥ −0.015 | connector **−0.0291 FAIL** | connector −0.0266 FAIL |
| g5 mean Pixel-AUROC loss | ≥ −0.002 | −0.0014 PASS | −0.0004 PASS |

Per-shot mean ΔAP: A1 k2 **−0.0008** · k4 **−0.0048**; A2 k4 **−0.0151** (k2 degenerate identity by design).
Category-mean ΔAP (A1): bracket_black +0.0071 · bracket_brown +0.0005 · bracket_white +0.0122 ·
connector **−0.0291** · metal_plate −0.0016 · tubes −0.0061.

## Controls

- c1 duplicate refs: gains ≈ real gains, never inflated (e.g. k4 bracket_black dup +0.0149 == real +0.0149) → mechanism is not "more refs are better".
- c2 shuffled refs: real-reference mean gain (−0.0028) does NOT beat shuffled control mean gain (−0.0037) by ≥ +0.002 → g6 FAIL. (On the three bracket classes real refs do beat shuffled refs by +0.005~+0.009, but shuffled attribution alone still yields positive gains there, i.e. most of the bracket effect comes from block-splitting, not from genuine per-image structure.)

## Decision

FAIL on g2/g3/g4 for both pre-registered candidates (plus g6). Mean gains are negative; only the three
very-hard bracket classes are helped, and that help is not robustly attributable to per-reference
agreement (shuffled control also positive). Per task book §4.5 ("若只有 k=4 有效、k=2 失败…" and
"精度均值为负 → 立即停止"), CRAM is **not** promoted to R1 and is archived as a unified route.

Preserved per-category observation (NOT a route): bracket_white is positive at both shots (k2 +0.0115,
k4 +0.0129) — for extremely low-signal classes a small median-gap correction can de-suppress
lucky-match hotspots. Revisit only if a future dataset has ≥3 such classes with the same sign; do not
tune per category now.
