# Project Status

Updated: 2026-07-24

## Confirmed

- The project root was initially empty; no prior code or Git history was present.
- Python 3.10.11 and Git 2.54.0 are available; Conda is not installed.
- Hardware: NVIDIA RTX 3060 Laptop, 6 GB VRAM; driver 591.86.
- The full phase-one plan is in `PLAN.md`.
- Official sources for AnomalyCLIP, PatchCore, PromptAD, AnomalyDINO, ReMP-AD and AdaptCLIP have been recorded.
- MVTec AD requires the official download form/license flow; VisA is available from AWS Open Data.
- `scripts/verify_system.ps1` has run successfully. PowerShell execution requires:
  `powershell -ExecutionPolicy Bypass -File scripts/verify_system.ps1`.
- `scripts/prepare_splits.py` and `scripts/validate_dataset.py` pass Python 3.10 syntax checks.
- A synthetic directory test confirmed deterministic nested 1/2/4-shot manifest generation.

## In progress

- Obtaining the AnomalyCLIP source code. Git clone and codeload are unusually slow on
  the current Windows network connection.

## Not started

- Dataset download and validation.
- Per-method virtual environments.
- Checkpoint download.
- The AnomalyCLIP `bottle` smoke test.
- Full-category and unified 1/2/4-shot experiments.

## Constraints and risks

- The official AnomalyCLIP experiments used an RTX 3090 with 24 GB; this machine will
  first run pretrained-checkpoint inference with batch size 1.
- Windows GPU FAISS may be unstable; PatchCore and AnomalyDINO will use CPU FAISS first.
- The current base Python has no `torch`, `numpy` or `scikit-learn`. Dependencies will
  be installed only after the upstream entry points are inspected.
- An incomplete `methods/anomalyclip-main.zip` remains from a timed-out codeload
  attempt; it is not valid source and is not tracked by Git.

## Next action

1. Finish a verifiable AnomalyCLIP source download and record its commit/hash.
2. Inspect README, requirements, test scripts, dataset JSON generation and checkpoint links.
3. Create `.venv-anomalyclip`.
4. Obtain MVTec AD/VisA and run the `bottle` smoke test.

