# 08. Project Structure and Authority Index

## 1. Recommended logical structure

No physical relocation is performed because experiment scripts, manifests and hashes may depend on existing paths. Use this map as the organizational layer.

```text
sci_project/
├─ docs/
│  ├─ paper_writing_preparation_20260830/   # current writing hub (incl. task books 11–21, R0 summaries)
│  ├─ introduction_research_20260825/       # literature archive; partly outdated framing
│  ├─ PAPER_DETAILED_CHINESE_DRAFT_*.md     # current Chinese manuscript source
│  └─ historical status/audit documents
├─ submission_repro_20260827/               # authoritative compact evidence package
├─ experiments/
│  ├─ dynamic_fusion/main_results_*         # main A1 result history
│  ├─ dynamic_fusion/v3_direction_a/        # A1 reconstruction and ablations
│  ├─ dynamic_fusion/rcec_v1/               # 2026-09-02 RCEC development-only negative result
│  ├─ dynamic_fusion/v3_3, v3_5, v4...      # negative/closed exploratory routes
│  ├─ dynamic_fusion/innovation_v8_tcrr_probe/    # TCRR region text evidence (MPDD pos, external neg)
│  ├─ dynamic_fusion/innovation_v9_ncsafe_tcrr/   # NC-safe TCRR (archived)
│  ├─ dynamic_fusion/innovation_v10_portfolio/    # A–F + LLSE + CSS R0s (all archived) + PORTFOLIO_LEDGER.md
│  └─ dynamic_fusion/innovation_v11_regret_router/  # doc 21: RSR Oracle + BC-MCR (archived) + PORTFOLIO_LEDGER.md
├─ outputs/                                 # large raw and unified method outputs
├─ configs/                                 # runnable/frozen configurations
├─ scripts/                                 # evaluation, audit and table-generation scripts
├─ src/                                     # project source code
├─ tests/                                   # regression tests
├─ methods/                                 # third-party method checkouts
├─ data/                                    # local datasets; never package for redistribution
├─ patches/                                 # third-party/project patch records
└─ result/                                  # miscellaneous result artifacts; not an authority source
```

## 2. Authority classes

### Class A — manuscript-authoritative

- `submission_repro_20260827/METHOD_SPEC_V2.md`
- `submission_repro_20260827/evidence/p1/`
- `submission_repro_20260827/evidence/paper_tables/`
- `submission_repro_20260827/config/`
- `docs/PRE_MANUSCRIPT_READINESS_AUDIT_20260827.md`
- this preparation directory

Use these for method equations, result numbers, roles, limits and captions.

### Class B — supporting and traceable

- `docs/PAPER_DETAILED_CHINESE_DRAFT_20260827.md`
- `docs/CURRENT_DYNAMIC_FUSION_STATUS.md`
- `docs/PAPER_SUBMISSION_HANDOFF_AND_REPRODUCIBILITY_PLAN_20260826.md`
- `experiments/dynamic_fusion/main_results_20260818/`
- `experiments/dynamic_fusion/v3_direction_a/p0_rebuild_20260826/`
- `experiments/dynamic_fusion/rcec_v1/`（RCEC 工程与 MPDD 早停证据；只支持负结果论断）
- `experiments/summaries/`

Use for explanation and cross-checking; where values differ, Class A wins.

### Class C — literature and writing archive

- `docs/introduction_research_20260825/`
- `outputs/paper_draft_20260810/`
- literature screening notes and source lists
- old project overview/progress DOCX files

Use for reusable background, bibliography leads and process history. Recheck all method descriptions and citations.

### Class D — historical/invalid for final claims

- V3.3 route selected with test masks.
- failed or closed dynamic routing, D1 predictability, V4 and SubspaceAD-gate experiments.
- RCEC v1 after the MPDD small-gate failure; retain as an explicitly labeled development-only negative result.
- `innovation_v8_tcrr_probe` / `innovation_v9_ncsafe_tcrr` / `innovation_v10_portfolio`
  (routes A–F, LLSE, CSS) / `innovation_v11_regret_router` (RSR region-oracle + BC-MCR
  structural gate): all development-only negative results archived with per-route
  pre-registered `R0_PROTOCOL.json → R0_RESULT.json → R0_DECISION.md` and a
  `PORTFOLIO_LEDGER.md` status table. Each route may be cited only as an explicitly
  labeled negative result; none is a paper contribution.
- old `METHOD_CARD.md` where concat dimension is 1152.
- old draft claims about text evidence, uncertainty routing or universal improvement.

These may be cited only as explicitly labeled negative results or historical decisions.

## 3. Directory-specific handling rules

| Directory | Keep | Avoid |
|---|---|---|
| `data/` | local licensed evaluation | copying data into docs/release |
| `methods/` | upstream code provenance | treating third-party README claims as local results |
| `outputs/` | raw audit trail | manual result selection from old runs |
| `experiments/` | ablations and historical provenance | confusing exploratory selection with frozen validation |
| `submission_repro_20260827/` | final evidence and replay | silent edits that invalidate hashes |
| `docs/` | narrative and planning | multiple competing “current” method descriptions |
| `.tmp_*`, caches | local tooling state | citing or committing as scientific evidence |

## 4. Current manuscript source-of-truth set

For day-to-day writing, keep only these files open:

1. `docs/paper_writing_preparation_20260830/README.md`
2. `submission_repro_20260827/METHOD_SPEC_V2.md`
3. `submission_repro_20260827/evidence/p1/p1_e_complete_metrics.md`
4. `submission_repro_20260827/evidence/p1/p1_a_bootstrap_ci.md`
5. `submission_repro_20260827/evidence/p1/p1_b_failure_boundaries.md`
6. `submission_repro_20260827/evidence/p1/p1_c_efficiency.md`
7. `submission_repro_20260827/evidence/p1/p1_d_fairness_table.md`
8. `docs/PAPER_DETAILED_CHINESE_DRAFT_20260827.md`
9. `docs/paper_writing_preparation_20260830/references/curated_references.bib`

## 5. Naming recommendations for new artifacts

- Main manuscript: `manuscript_en_v0_1_YYYYMMDD.docx` or journal LaTeX source after venue selection.
- Figures: `fig01_method_overview.*`, `fig02_protocol.*`, `fig03_config_delta_pixel_ap.*`.
- Tables: generate from scripts into `paper_artifacts/tables/`, not by hand.
- Supplement: `supplement_en_v0_1_YYYYMMDD.*`.
- Do not create another top-level “final” directory until the target journal package is assembled.

## 6. Deferred cleanup

The repository contains large historical outputs, multiple virtual environments and temporary render directories. Cleanup is deliberately deferred because it is destructive and not required for paper writing. A later storage-cleanup task should first produce a checked manifest of reproducible, regenerable and irreplaceable assets, then archive or delete only with explicit approval.
