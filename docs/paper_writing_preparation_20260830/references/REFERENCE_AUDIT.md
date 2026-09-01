# Reference Audit

## 1. Purpose

`curated_references.bib` merges the two older Introduction bibliographies and adds the most important 2025–2026 neighbors discovered during this preparation round. It is a working database, not yet the target journal's final reference list.

## 2. Core references required by the current Introduction

| Key | Role | Verification source | Status |
|---|---|---|---|
| `bergmann2019mvtec` | MVTec AD | official CVF/MVTec page | ready |
| `zou2022spot` | VisA/SPot-the-Difference | official ECCV/AWS entry | ready |
| `jezek2021mpdd` | MPDD | IEEE DOI + official repo | ready |
| `mishra2021vtadl` | BTAD/VT-ADL | IEEE DOI + arXiv + original author repository | ready; dataset CC BY-SA 4.0 verified |
| `roth2022patchcore` | patch memory | official CVF | ready |
| `oquab2024dinov2` | DINOv2 backbone | TMLR/OpenReview | ready |
| `radford2021clip` | CLIP | PMLR | ready |
| `jeong2023winclip` | CLIP anomaly baseline | official CVF | ready |
| `zhou2024anomalyclip` | AnomalyCLIP | official ICLR proceedings | ready |
| `li2024promptad` | normal-only prompt learning | official CVF | ready |
| `damm2025anomalydino` | few-shot frozen DINO | official CVF | ready |
| `ma2025rempad` | multimodal few-shot neighbor | official CVF/IEEE DOI | ready |
| `guo2026seaclip` | closest CLIP+DINO few-shot neighbor | official CVF | ready |
| `lendering2026subspacead` | training-free DINO/PCA neighbor | official CVF | ready |
| `li2026fastref` | prototype refinement neighbor | official CVF | ready |
| `ma2026papl` | CLIP+DINO zero-shot neighbor | Pattern Recognition/DOI | ready |
| `jiang2025clipdino` | trained CLIP+DINO zero-shot neighbor | Electronics/DOI | ready |

## 3. Dataset license evidence

This is a publication-preparation note, not legal advice.

- MVTec AD: official site states CC BY-NC-SA 4.0.
- VisA: AWS Open Data entry states CC BY 4.0.
- MPDD: the official repository's `LICENSE` file is CC BY-NC-SA 4.0.
- BTAD: the original VT-ADL author repository labels BTAD `CC-BY-SA`; its license hyperlink resolves to the CC BY-SA 4.0 legal code. The repository's root MIT file applies to software, not the dataset. See `../BTAD_LICENSE_EVIDENCE.md`.

## 4. Metadata quality levels

- **Ready**: title, authors, venue/year, pages or DOI, and official URL checked.
- **Imported**: copied from the earlier project BibTeX and useful for Related Work, but abbreviated `others` author lists should be expanded by the reference manager before submission.
- **Lead only**: mentioned in notes but not included until official metadata is checked.

Entries marked by comments in the BibTeX as “imported” must not be considered final solely because the file parses.

## 5. Final bibliography checklist

- [ ] Expand every `and others` author list if the journal style requires complete authors.
- [ ] Normalize venue names to the journal's abbreviation policy.
- [ ] Add DOI to CVF papers where the journal requires it.
- [ ] Check capitalization protection for `DINOv2`, `CLIP`, `MVTec`, `VisA`, `BTAD`, `MPDD`.
- [ ] Remove uncited imported entries from the final manuscript database.
- [ ] Verify all 2026 volume/issue/page metadata again immediately before submission.
- [ ] Use one reference manager or BibTeX processor to detect duplicate DOI/title entries.
- [ ] Add access dates only for dataset web pages if required by the target journal.

## 6. Important novelty implication

The verified record establishes that Sea-CLIP, PAPL, and the Electronics 2025 method already combine CLIP- and DINO-family representations. Therefore the manuscript must not claim the first CLIP–DINO fusion. These references directly shape the controlled-study positioning used in the new Introduction.
