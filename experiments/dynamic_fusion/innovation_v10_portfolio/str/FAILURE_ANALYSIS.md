# Route E — STR: Failure Analysis

## Hypothesis
DINO/CLIP deep patches are structurally strong but could be blind to fine periodic
texture / scratches / high-frequency change; a cheap reference-calibrated spectral
residual should be informative exactly where A1 misses.

## Evidence against it (MPDD seed0 k4)
1. **A1's own score is a better predictor of A1's misses than STR is** on the
   largest miss set (bracket_black: AP_A1 0.241 vs AP_STR 0.115). A1-missed defect
   components are not "score-invisible": they sit in the upper tail of the normal
   distance distribution but below the strict component-level flag threshold θ. A
   spectral residual cannot beat the distance signal there because the residual is
   only weakly coupled to A1's failure mode.
2. **Most classes have almost no A1-missed area to rescue** (connector, metal_plate:
   0 missed pixels; bracket_white: 14 px). The classes where A1 is weak at the
   component level (bracket_black) are precisely the ones where simple texture
   energy is confounded with normal surface roughness — STR is at chance
   (AP 0.11 on a 1.6% positive-rate problem is near the base-rate ceiling of
   ranking).
3. The only positive class (bracket_brown, +0.041) is below the pre-registered
   +0.05 bar and is not reproduced on any sibling class (bracket_black −0.126,
   bracket_white ≈0) → not a mechanism, a category artefact. Consistent with the
   protocol: archive as per-category observation, never promote.
4. g2 (true vs misaligned +0.030) passes — spatial structure in the STR map is
   real — but that was never in doubt; the residual simply does not point at
   A1-missed defects.

## Lesson
On MPDD, A1's few-shot fused memory bank already captures texture-scale structure
(CLIP/DINO operate on 14-16 px patches; the fusion raises exactly the textured
classes). The residual blind spot STR targets is either already covered by A1 or
too confounded with surface roughness to be usable as an independent evidence
channel. Spectral residual ideas are closed for this data and setting unless a
new dataset has a class whose defects are sub-patch and texture-pure.
