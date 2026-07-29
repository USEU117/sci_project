# ReMP-AD and AdaptCLIP source audit

Audit date: 2026-07-29

## ReMP-AD

- Official repository: `https://github.com/cshcma/ReMP-AD`
- Local source commit: `d3fbc46adfd91406859b90dece65c221343096c7`
- The repository contains separate MVTec and VisA shell entry points:
  `run_mvtec.sh` and `run_visa.sh`.
- The official scripts train once and then test 4-, 2- and 1-shot settings.
- The README expects a Python 3.8 environment and an MVTec/VisA directory
  layout with a method-specific `meta.json`.
- No downloadable pretrained checkpoint is documented in the README; the
  default workflow trains the method before testing.
- A Gate A run is not started yet. Before running it, the official loader must
  be adapted to the project's frozen manifest and the method's training output
  must be converted to the common NPZ schema.

## AdaptCLIP

- Official repository: `https://github.com/gaobb/AdaptCLIP`
- Local source commit: `354d9e3332ec5348b3d8e4439111d34f8e94c0a9`
- The official test script expects pretrained checkpoints under
  `adaptclip_checkpoints/`, downloaded from the author's Hugging Face model
  page.
- The documented checkpoint configuration uses ViT-L/14 at 518px, with
  `features_list 6 12 18 24`, three adapters, and batch size 8.
- This configuration is not directly safe on the local 6 GB GPU. A Gate A
  smoke must first use batch size 1 and check whether the published checkpoint
  can load without changing the model or image resolution.
- The official shell script contains a duplicated “MVTec-trained” block whose
  `test_dataset` is still set to `visa`; this is recorded as a reproducibility
  issue and must be corrected in a local patch before any matrix run.
- No checkpoint has been downloaded and no Gate A run has started yet.

## Decision

PromptAD's resumable VisA Gate B queue has priority while the GPU is occupied.
After that queue reaches a stable checkpoint, ReMP-AD and AdaptCLIP will each
receive an isolated environment, a one-category Gate A, NPZ conversion and
schema/metric validation before any full matrix is attempted.
