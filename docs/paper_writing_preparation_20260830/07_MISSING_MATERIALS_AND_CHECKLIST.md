# 07. Missing Materials and Pre-submission Checklist

## 1. Priority A — required before a complete English draft

- [ ] Select one target journal and download its current author guide/template.
- [ ] Decide final title and whether the paper is framed as a method paper or controlled empirical study. Recommendation: controlled empirical study.
- [ ] Confirm author order, affiliations, corresponding author, ORCID, funding and acknowledgments.
- [ ] Confirm whether the compact reproducibility package will be public at submission, after acceptance, or available on request.
- [ ] Verify every BibTeX entry against an official publisher/CVF/OpenReview page; remove placeholder notes.
- [x] Resolve the original BTAD dataset license/redistribution terms. The original author repository labels the dataset CC-BY-SA and links that label to the CC BY-SA 4.0 legal code; evidence is recorded in `BTAD_LICENSE_EVIDENCE.md`.
- [ ] Regenerate a dataset statistics table from the exact split manifests used in the paper.
- [x] Produce the final method diagram with the correct image-only dual-encoder path and 1536-D concat.
- [x] Produce deterministic qualitative success/failure panels with shared per-sample score scales.
- [ ] Convert the Chinese detailed draft into an English main manuscript using the current method vocabulary, not the old DOCX storyline.

## 2. Priority B — high-value scientific strengthening

These items are not necessary to establish the existing matched-control claim, but they would reduce likely reviewer objections if resources allow.

### Complete CLIP-only control on all four datasets — BTAD/MVTec completed

The missing BTAD and MVTec controls were completed on 2026-08-30 with six metrics for all 3 seeds × 1/2/4 shots. Together with the existing MPDD/VisA fairness evidence, this completes the intended three-way comparison:

`DINO-only vs CLIP-only vs fixed concat`.

The controls use the same reference identities and evaluator. A1 exceeds CLIP-image-only in Pixel AP for all 18 BTAD/MVTec configurations. See `09_BTAD_MVTEC_CLIP_ONLY_CONTROL_RESULTS.md`. Before claiming an all-four-dataset six-metric table, normalize the historical MPDD/VisA CLIP-only outputs into this same complete-metric schema.

### Development-only weight sensitivity summary

The fixed 0.5/0.5 choice should be justified only from MPDD development evidence or as a parameter-free symmetry choice. Consolidate existing weight scans into one plot/table, then show that the chosen weight was frozen before BTAD/VisA/MVTec evaluation. Do not re-select weights on validation test masks.

### Strong simple baseline context

SubspaceAD and FastRef are 2026 strong neighbors. If exact local reproduction is impractical, include them in a protocol-aware related-work table and avoid direct numerical superiority claims. If a local run is attempted, freeze the evaluation plan before seeing target test outcomes.

### Dimension objection

Reviewers may ask whether concat helps merely because it doubles dimension. A useful analytical note is that concatenating a normalized DINO vector with an identical copy and globally normalizing preserves pairwise cosine geometry; therefore a duplicated-DINO control is mathematically redundant under the stated score. A short derivation can answer this without a new experiment. A random-projection/dimension-matched control is optional, not essential.

## 3. Priority C — writing and presentation

- [ ] Write a 200-word structured and an unstructured abstract variant.
- [ ] Standardize terminology: `normal reference`, `shot`, `patch feature`, `frozen image encoder`, `matched control`, `in-domain frozen validation`.
- [ ] Define all six metrics once and use consistent capitalization.
- [ ] Choose one rounding policy, preferably four decimals for metrics and deltas.
- [ ] Add a limitations subsection rather than hiding limitations in the conclusion.
- [ ] Prepare a protocol/fairness paragraph for every broader baseline table.
- [ ] Add Data Availability, Code Availability, Ethics/Competing Interests, Funding and Author Contributions sections as required by the journal.
- [ ] Run a final language pass for articles, tense, acronym introduction and claim strength.

## 4. Dataset and license status

| Asset | Current status | Required action |
|---|---|---|
| MVTec AD | official page states CC BY-NC-SA 4.0 | cite official page and dataset paper |
| VisA | AWS Open Data entry states CC BY 4.0 | cite AWS entry and SPot-the-Difference paper |
| MPDD | official GitHub repository contains a CC BY-NC-SA 4.0 license file | update manuscript license table; do not edit frozen package silently |
| BTAD | original author repository identifies CC BY-SA 4.0; local archive README corroborates CC-BY-SA | cite the author repository and keep original images outside the compact package |
| DINOv2 / AnomalyCLIP weights | package has source/hash records; redistribution excluded | verify upstream license notice in final release |
| Dataset images | not included in compact package | keep this policy |

Official sources checked in this preparation round:

- [MVTec AD official dataset page](https://www.mvtec.com/research-teaching/datasets/mvtec-ad)
- [VisA on the Registry of Open Data on AWS](https://registry.opendata.aws/visa/)
- [MPDD official repository](https://github.com/stepanje/MPDD) and [license](https://github.com/stepanje/MPDD/blob/main/LICENSE)
- [BTAD / VT-ADL original author repository](https://github.com/pankajmishra000/VT-ADL) and [CC BY-SA 4.0 legal code](https://creativecommons.org/licenses/by-sa/4.0/legalcode)

## 5. Submission-readiness gates

### Scientific gate

- [x] Final A1 method frozen.
- [x] Matched DINO-only control available.
- [x] Four datasets × nine configurations rebuilt.
- [x] Six metrics complete.
- [x] Bootstrap and failure-boundary analysis complete.
- [x] Efficiency measurement complete.
- [x] Leakage/protocol audit complete.
- [ ] Final broader-baseline narrative updated for 2026 close neighbors.
- [x] Run the missing BTAD/MVTec CLIP-image-only controls (18 reports; six metrics).

### Reproducibility gate

- [x] Compact prediction maps and reconstruction script.
- [x] Method V2 specification.
- [x] Split/checkpoint hashes and manifests.
- [x] P1 acceptance complete.
- [ ] Reconcile the already modified `VERSIONED_EVIDENCE.sha256` before creating a release tag.
- [ ] Assign archive URL/DOI and release license notices.

### Manuscript gate

- [x] Chinese detailed draft exists.
- [x] Corrected English Introduction V0.1 exists in this directory.
- [x] Related-work gap and claim boundary prepared.
- [ ] Full English Method/Experiments/Results draft.
- [x] Final scientific figure package (11 figures; SVG/PDF/600-dpi PNG; QA complete).
- [ ] Journal-formatted final tables.
- [ ] Target-journal style, declarations and cover letter.
- [ ] Reference manager consistency check.

## 6. Recommended immediate sequence

1. User selects the target journal or provides 2–3 candidates.
2. Lock the journal template and manuscript length.
3. Decide whether to consolidate MPDD/VisA CLIP-only outputs and MPDD-only weight sensitivity into final paper tables.
4. Generate figures and dataset table from machine-readable evidence.
5. Draft Method, Experimental Setup and Results in English.
6. Revise the Introduction after exact table/figure numbering is fixed.
7. Complete declarations, bibliography audit and release package.
