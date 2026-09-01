# 02. English Manuscript Blueprint

This blueprint assumes an applied computer-vision journal paper of roughly 7,000–9,000 words. Exact limits must be adapted after the target journal is selected.

## Front matter

### Working title

**Do Frozen Visual Encoders Provide Complementary Evidence for Few-Shot Industrial Anomaly Detection? A Controlled Four-Dataset Study**

### Abstract logic (180–250 words)

Use five moves:

1. Problem: few-shot industrial inspection has few normal references and heterogeneous defects.
2. Gap: learned fusion can obscure whether two frozen visual representations are intrinsically complementary; simple combinations are rarely evaluated with matched controls and failure boundaries.
3. Method: frozen DINOv2 + AnomalyCLIP image tower, aligned patch features, branch normalization, fixed concatenation, k-NN normal memory; zero trainable parameters.
4. Results: positive Pixel-AP gain in all 36 configurations; dataset-level gains of +0.0258 to +0.0524; mention non-universal image-level/category effects.
5. Significance: a reproducible reference point showing both value and limits of frozen representation fusion.

Do not put “multimodal,” “language evidence,” “adaptive,” “SOTA,” or “significant improvement” in the abstract.

### Keywords

Industrial anomaly detection; few-shot learning; visual foundation models; feature fusion; anomaly localization; normal memory bank; reproducibility.

## 1. Introduction (900–1,200 words)

Recommended eight-paragraph argument:

1. Industrial importance and defect-label scarcity.
2. Why normal-only few-shot methods and patch memories are practical.
3. Why a single representation can be insufficient across texture, geometry, pose and semantics.
4. Foundation-model landscape: DINOv2-based visual methods and CLIP-derived anomaly methods.
5. Closest CLIP–DINO combinations and why they do not answer the controlled, training-free question.
6. Research question and deliberately minimal A1 design.
7. Compact quantitative preview plus limitations.
8. Three contributions.

An English working draft is provided in [03_INTRODUCTION_WORKING_DRAFT_EN.md](03_INTRODUCTION_WORKING_DRAFT_EN.md).

## 2. Related Work (1,000–1,300 words)

### 2.1 Normal-memory industrial anomaly detection

Cover PaDiM, PatchCore and representative reconstruction/distillation methods briefly. End with the distinction between conventional many-normal-image training and 1/2/4-shot normal references.

### 2.2 Frozen visual foundation models for few-shot anomaly detection

Discuss DINOv2, AnomalyDINO, UniVAD, FastRef and SubspaceAD. Emphasize whether each method trains, stores a memory bank, fits a subspace, refines prototypes at test time, or uses auxiliary data.

### 2.3 CLIP-based and cross-encoder anomaly detection

Discuss WinCLIP, AnomalyCLIP, PromptAD, ReMP-AD, Sea-CLIP, PAPL and the CLIP–DINOv2 DMA/SAP journal method. Separate:

- CLIP text/image scoring;
- prompt learning;
- learned fusion of CLIP and DINO;
- the present work's image-only, fixed patch fusion.

### 2.4 Position of this study

Conclude with one explicit novelty paragraph: the work does not claim a new encoder pair, but isolates the effect of adding a second frozen visual representation under a matched normal-memory pipeline and reports reproducibility and negative transfer.

Use the comparison map in [04_RELATED_WORK_AND_NOVELTY_MAP.md](04_RELATED_WORK_AND_NOVELTY_MAP.md).

## 3. Method (1,200–1,500 words)

### 3.1 Problem formulation

Define a category-specific normal reference set \(R_c=\{x_i\}_{i=1}^{K}\), \(K\in\{1,2,4\}\), a query image \(x\), and pixel anomaly map \(S(x)\). State that no abnormal training image, test label, or test mask is available to the method.

### 3.2 Frozen patch encoders

- DINOv2 ViT-B/14, 448 input, 768-D patches.
- AnomalyCLIP ViT-L/14@336 image tower, 518 input, 768-D patches extracted through the DAPM path.
- Clarify that no text embedding is computed at inference.

### 3.3 Spatial alignment and normalized concatenation

Let \(F_D\in\mathbb{R}^{H_D\times W_D\times768}\) and \(F_C\in\mathbb{R}^{H_C\times W_C\times768}\). Bilinearly resize \(F_C\) to the DINO grid, L2-normalize each branch, concatenate the two equally scaled branches, and L2-normalize again:

\[
F_A = \operatorname{norm}\!\left([0.5\operatorname{norm}(F_D);\;0.5\operatorname{norm}(\mathcal{R}(F_C))]\right).
\]

Explain that the fixed coefficients are part of a frozen protocol, not learned gates.

### 3.4 Normal memory and anomaly score

Build one category-specific FAISS `IndexFlatL2` memory from reference patches. Query with k=1. Because unit-normalized squared Euclidean distance equals twice cosine distance, use distance/2 as the patch anomaly score. Apply Gaussian smoothing with sigma 4 and bilinear upsampling to 448 × 448.

### 3.5 Matched control

Define the DINO-only pipeline. List exactly what is shared: reference IDs, input/evaluation resolution, k-NN, distance convention, smoothing, upsampling, stride and metrics.

### 3.6 Leakage contract

State that test labels/masks/statistics are not used for weights, thresholds, rules or sample selection. Distinguish the data roles of MPDD, BTAD, VisA and MVTec AD.

## 4. Experimental Setup (1,000–1,300 words)

### 4.1 Datasets and roles

- MPDD: development.
- BTAD: external frozen validation.
- VisA: in-domain frozen validation because of the AnomalyCLIP checkpoint lineage.
- MVTec AD: external frozen validation.

Include category counts, train/test image counts used locally, and official citations/licenses only after the dataset table is regenerated from manifests.

### 4.2 Few-shot protocol

Explain 3 seeds × 1/2/4 shots per dataset, category-wise sampling of normal references, and why these are reference-sampling configurations rather than independent datasets.

### 4.3 Metrics

Report Image-AUROC, Image-AP, Image-F1-max, Pixel-AUROC, Pixel-AP and Pixel-AUPRO. State 448 evaluation size and stride 8. Make Pixel-AP the primary endpoint because localization under class imbalance is central to the study.

### 4.4 Baselines and fairness

Use the P1-D fairness table. Separate locally reproduced methods, protocol-matched comparisons and literature-only context. Never mix incomparable published numbers into the main matched-control claim.

### 4.5 Statistical analysis

Describe paired image bootstrap and category bootstrap, B=2000, seed 20260827, and the distinction between anomalous-image and category resampling. Do not treat the nine seed/shot configurations as independent samples.

### 4.6 Implementation and hardware

Copy exact library versions, GPU/CPU and checkpoint hashes from the reproducibility package. Include zero trainable parameters and memory-bank sizes.

## 5. Results (1,300–1,700 words)

### 5.1 Main matched-control result

Lead with the four-dataset Pixel-AP table and 36/36 positive configuration direction. Then report bootstrap intervals without implying all per-configuration category CIs exclude zero.

### 5.2 Complete metrics

Show all six metrics. Explicitly discuss BTAD image-level degradation. This prevents selective-reporting criticism.

### 5.3 Shot-wise stability

Report 1/2/4-shot means and variation across seeds. Avoid claiming monotonic scaling because the incremental gain does not strictly grow with shot count.

### 5.4 Category-level heterogeneity

Contrast strong gains (`visa@pcb1`, `visa@cashew`, `mvtec@toothbrush`, `mpdd@metal_plate`) with persistent losses (`mvtec@leather`, `visa@chewinggum`, `mvtec@hazelnut`, `visa@candle`).

### 5.5 Comparison with broader baselines

State honestly that AnomalyDINO remains stronger on several MVTec/VisA macro metrics. Frame A1 as evidence of complementarity, not SOTA.

### 5.6 Efficiency

Report steady-state latency, throughput, VRAM, RAM and memory-bank scaling. The dominant CLIP cost should be visible.

## 6. Discussion (900–1,200 words)

Suggested subsections:

- **Why simple fusion helps:** complementary invariances and local representations.
- **Why it fails:** texture-sensitive categories, spatial interpolation, branch imbalance and nearest-neighbor geometry.
- **Why dynamic routing was excluded:** no leakage-safe, stable improvement over fixed fusion; one historical route used test masks and is scientifically ineligible.
- **Deployment implications:** zero training and transparency versus throughput and bank-size cost.
- **Limits of generalization:** four benchmarks, one pair of backbones, one fixed weight, VisA domain relationship, no production drift study.

## 7. Conclusion (250–350 words)

Answer the research question directly: frozen visual representations show reproducible average localization complementarity, but the effect is neither universal nor free. End with future work on label-free reliability estimation or per-category calibration learned only from normal data—not with a claim that the present method already solves routing.

## Appendix / Supplementary Material

- Full per-category/per-seed/per-shot tables.
- Bootstrap details.
- Reference image manifests and hashes.
- Complete failure sample IDs and qualitative panels.
- Efficiency and memory-bank calculation.
- Historical negative routing results, clearly labeled as exploratory and excluded from model selection.
- Reproduction commands and package manifest.

## Recommended writing order

1. Method and leakage contract.
2. Experimental setup and exact tables.
3. Results and failure analysis.
4. Discussion.
5. Introduction and related work.
6. Abstract, title and conclusion.

