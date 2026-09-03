# Route F — SPRG: R0 Decision (part-discovery feasibility gate 1)

Date: 2026-09-03 · Status: **FAIL → ARCHIVED (feasibility gate 1 failed; immediate stop)**

## What was measured
For each of the 6 MPDD classes (seed0, 4 train/good references): per-image part
discovery by spatial-constrained KMeans (k=6, on DINO-vitb14 patch features +
spatial coords) and chain matching of nodes across the 4 reference images
(Hungarian on appearance cosine + geometry). Gate 1 (task book §9.4): cross-normal
matching success ≥ 90% (node must match on ALL 3 links with centroid cos ≥ 0.85
AND normalized position displacement ≤ 0.15).

## Result
| category | chain rate | link cos (matched) |
|---|---:|---|
| bracket_black | 0% | 0.74–0.90 |
| bracket_brown | 50% | 0.87–0.94 |
| bracket_white | 50% | 0.88–0.92 |
| connector | 67% | 0.93–0.96 |
| metal_plate | 50% | 0.86–0.92 |
| tubes | 0% | 0.92–0.97 |
| **mean** | **36.1%** | — |

## Gate evaluation
g1 (≥ 0.90): **FAIL (0.36)**. Per doc §9.4 ("若部件发现不稳定，立即停止；不进入真实异常
评估"), no relation-score stability or counterfactual work was done. Route archived.

## Root cause
MPDD normal samples within a class differ in pose/lighting/occlusion enough that no
stable 4–16-part correspondence exists across the K-shot normal references at DINO
patch granularity. Appearance centroids match well when geometry agrees (cos
0.87–0.97), but position constraints and full-chain consistency collapse on every
class; tubes (rotationally symmetric) are structurally ambiguous for unsupervised
spatial clustering. Unsupervised part discovery from K normal samples is not stable
on this data — consistent with the task book's warning that MPDD is not an ideal
logical-anomaly dataset.
