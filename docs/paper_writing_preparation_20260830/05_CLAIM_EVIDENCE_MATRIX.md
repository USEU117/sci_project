# 05. Claim–Evidence Matrix

主稿中的每个定量或方法论断都应能回到这里。路径相对于仓库根目录。

## 1. Method and protocol claims

| Claim | Allowed wording | Primary evidence | Caveat |
|---|---|---|---|
| A1 uses two frozen visual encoders | “frozen DINOv2 ViT-B/14 and the AnomalyCLIP ViT-L/14@336 image tower” | `submission_repro_20260827/METHOD_SPEC_V2.md` | Do not call the inference pipeline vision-language |
| No explicit text at inference | “no text embeddings or language scores are computed” | `METHOD_SPEC_V2.md`; package README | Historical folder `anomalyclip_text` is misleading |
| No trainable parameters | “zero trainable parameters; only a normal reference memory is constructed” | `evidence/p1/p1_c_efficiency.md` | Building a memory bank is still data-dependent inference preparation |
| Concatenated dimension | “1536-D after aligning two 768-D branches” | `METHOD_SPEC_V2.md`; P0-2 smoke report | 1152-D in historical files is wrong |
| Fixed fusion | “branch-wise L2, fixed 0.5/0.5 concatenation, global L2” | `METHOD_SPEC_V2.md` | Do not say learned or adaptive |
| k-NN scoring | “FAISS IndexFlatL2, k=1, distance/2” | `METHOD_SPEC_V2.md` | State unit normalization before interpreting distance |
| Leakage-safe normal-only protocol | “no test labels, masks or test statistics are used for selection” | method spec; P0 live audit; split manifests | Qualitative figures are selected after evaluation and must not be described as tuning data |
| Dataset roles | MPDD development; BTAD/MVTec external frozen; VisA in-domain frozen | package README and method spec | Never call VisA external validation |

## 2. Primary result claims

| Claim | Exact value | Primary evidence | Safe interpretation |
|---|---:|---|---|
| MPDD mean Pixel-AP gain | +0.0258 (0.3562 vs 0.3304) | `evidence/p1/p1_e_complete_metrics.md` | positive in 9/9 configurations |
| BTAD mean Pixel-AP gain | +0.0249 (0.6455 vs 0.6206) | same | positive in 9/9 configurations |
| VisA mean Pixel-AP gain | +0.0524 (0.3725 vs 0.3201) | same | in-domain frozen validation |
| MVTec mean Pixel-AP gain | +0.0320 (0.5546 vs 0.5226) | same | external frozen validation |
| All configuration directions | 36/36 positive Pixel-AP deltas | `evidence/p1/p1_a_bootstrap_ci.md`; package README | configurations share test sets; not 36 independent datasets |
| Full metric completeness | 36 reports × 2 methods × 6 metrics | `evidence/p1/p1_e_complete_metrics.*`; `p1_acceptance.json` | primary cross-dataset consistency claim remains Pixel-AP |
| BTAD/MVTec A1 vs CLIP-image-only | A1 Pixel AP is higher in 18/18 configurations; means: BTAD 0.6455 vs 0.4006, MVTec 0.5546 vs 0.4654 | `experiments/dynamic_fusion/v3_direction_a/clip_only_controls_20260830/`; `09_BTAD_MVTEC_CLIP_ONLY_CONTROL_RESULTS.md` | supports, but does not causally prove, complementary information; BTAD is not all-metric dominant over DINO-only |

## 3. Statistical claims

| Claim | Evidence | Required wording |
|---|---|---|
| Bootstrap procedure | `evidence/p1/p1_a_bootstrap_ci.md/json` | Paired image and category bootstrap, B=2000, seed=20260827 |
| Dataset-wise stability | same file | Report mean ± std across nine reference configurations as descriptive stability, not independent-sample significance |
| Per-configuration uncertainty | same file | Some category or image CIs include zero; never claim every interval excludes zero |
| Rebuilt values match historical table | P1-A sanity + P0-3 | Maximum allowed comparison tolerance is 5e-4; report rounded values in paper |

## 4. Failure and limitation claims

| Claim | Exact value | Evidence | Interpretation boundary |
|---|---:|---|---|
| MVTec leather persistent loss | -0.0428; 9 negative configs | `evidence/p1/p1_b_failure_boundaries.md` | category-level negative transfer |
| VisA chewinggum persistent loss | -0.0386; 9 negative configs | same | category-level negative transfer |
| MVTec hazelnut persistent loss | -0.0297; 9 negative configs | same | category-level negative transfer |
| VisA candle persistent loss | -0.0198; 9 negative configs | same | category-level negative transfer |
| MPDD bracket_brown persistent loss | -0.0051; 9 negative configs | same | small but consistent loss |
| BTAD Image-AP loss | -0.0131 | `evidence/p1/p1_e_complete_metrics.md` | prevents all-metric claim |
| BTAD Image-F1-max loss | -0.0237 | same | prevents all-metric claim |
| Ten dataset@category units have at least one negative config | 65 negative configuration occurrences across 10 unique units | `p1_b_failure_boundaries.md` | do not confuse occurrence count with category count |

## 5. Efficiency and reproducibility claims

| Claim | Exact value | Evidence | Caveat |
|---|---:|---|---|
| Steady-state latency | 0.4146 s/image | `evidence/p1/p1_c_efficiency.md/json` | bottle s0/k1; warm-up 3, repeated 30 |
| Throughput | 2.412 image/s | same | hardware- and implementation-specific |
| DINO extraction | 0.0626 s/image | same | steady-state GPU stage |
| CLIP extraction | 0.3049 s/image | same | dominant cost |
| Alignment + concat + k-NN | 0.0471 s/image | same | CPU stage in measured setup |
| Peak GPU memory | approx. 2073 MB | same | sequential branch execution |
| Peak process RAM | 3980.9 MB | same | measured bottle s0/k1 process maximum |
| Compact package | 186.8 MB, 324 float16 maps | same; package README | contains no dataset images or third-party weights |
| P1 completeness | all 21 listed checks true; P1 complete | `evidence/p1/p1_acceptance.json` | describe exact deliverables rather than saying “fully reproducible” without qualification |

## 6. Broader baseline claims

| Claim | Evidence | Safe wording |
|---|---|---|
| A1 is not universal SOTA | `evidence/p1/p1_r3_baseline_comparison.csv`; Chinese draft | “AnomalyDINO remains stronger on several MVTec/VisA macro metrics under the local comparison.” |
| Protocols differ across baselines | `evidence/p1/p1_d_fairness_table.md` | Explicitly label zero-shot, target-normal tuning, source-domain training, test-time adaptation and training-free methods |
| Historical dynamic routing did not justify the final method | `docs/CURRENT_DYNAMIC_FUSION_STATUS.md`; negative result index | Put in Discussion/Appendix; do not use the test-mask-selected V3.3 route as valid evidence |

## 7. Claims that still lack final evidence

- Exact target-journal page/word format.
- Public archive URL/DOI and final distribution notice for the project's own reproducibility package. BTAD itself is now verified as CC BY-SA 4.0 in `BTAD_LICENSE_EVIDENCE.md`.
- Public release URL or DOI for the reproducibility package.
- Final author list, affiliations, funding, conflict-of-interest and data/code availability wording.
- Final method diagram and pre-registered qualitative figure selection rule.
- A mechanistic proof of why specific feature families help or hurt a category. Current explanations are hypotheses supported by patterns, not causal demonstrations.
