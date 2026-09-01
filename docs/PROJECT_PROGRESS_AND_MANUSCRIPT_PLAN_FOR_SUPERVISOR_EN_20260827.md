# Research Progress and Manuscript Plan

## Tentative Paper Title

**Frozen Dual-Encoder Visual Feature Fusion for Few-Shot Industrial Anomaly Localization: Cross-Dataset Validation and Failure Boundaries**

## 1. Project Overview

This project investigates few-shot industrial anomaly detection under a normal-only setting. The practical motivation is that normal product images are usually easy to collect, whereas defective samples are scarce, diverse, and often unavailable during system development. The objective is therefore to detect and localize previously unseen defects using only one, two, or four normal reference images for each category.

The project initially focused on dynamic fusion between visual and language-related anomaly evidence. We expected that different branches might be reliable for different objects or defect regions and that a dynamic router could select or reweight them accordingly. During the project, we implemented and evaluated several fusion variants, calibration strategies, reliability indicators, and routing mechanisms. However, the experiments showed that the more complicated dynamic approaches did not provide stable improvements over a simple fixed fusion. One early calibration variant also used test masks and was therefore identified as data leakage; it has been excluded from all paper-eligible results.

These findings led us to refine the research question. Instead of claiming that dynamic visual–text routing is effective, the final study asks a narrower and more defensible question: can two heterogeneous frozen visual patch representations provide reproducible complementary information under a strict few-shot normal-reference protocol?

## 2. Final Method

The final method, referred to as A1, combines patch-level features from two frozen visual encoders:

1. DINOv2 ViT-B/14, which produces 768-dimensional patch features and provides strong self-supervised structural and texture representations;
2. the image tower of AnomalyCLIP ViT-L/14@336, which also produces 768-dimensional patch features but originates from a different pretraining framework.

The two feature grids have different spatial resolutions. We first resize the AnomalyCLIP visual feature grid to match the DINOv2 grid. Each branch is independently L2-normalized, and the aligned features are concatenated with a fixed equal weight of 0.5/0.5. The resulting 1536-dimensional feature is normalized again.

For each industrial category, patch features from the K normal reference images are stored in a normal memory bank. At inference time, every test patch is matched to its nearest normal patch using KNN with k=1. Half of the squared nearest-neighbor distance is used as the local anomaly score. The patch-level score map is then smoothed and resized to the image resolution to produce the final anomaly localization map.

The method has no trainable parameters. It does not use a learned router, category-specific weights, test-set calibration, or explicit text features during final inference. Although the AnomalyCLIP encoder originates from a vision–language model, the final system should be described as **frozen dual-encoder visual feature fusion**, rather than visual–text fusion.

## 3. Experimental Development and Validation

The experimental process was organized in several stages.

First, we reproduced and audited the main baselines, including PatchCore, WinCLIP+, AnomalyDINO, PromptAD, AnomalyCLIP, ReMP-AD, and AdaptCLIP. Their protocols were recorded separately because some methods are zero-shot, some use target-normal prompt tuning, and others rely on source-domain training. These results are used for method-level context, while the main controlled comparison is performed against a matched DINO-only KNN pipeline.

Second, we developed the fusion method on MPDD. A weight scan indicated that a DINO weight of 0.4 produced only approximately 0.0009 higher Pixel-AP than the symmetric 0.5 setting. Because this difference was within the observed experimental variation, we fixed the weight at 0.5 to reduce hyperparameter freedom and avoid overfitting the development dataset.

Third, after freezing the method, we evaluated it on VisA, BTAD, and MVTec AD. MPDD is treated as the development dataset, VisA as in-domain frozen validation because of the source of the AnomalyCLIP checkpoint, and BTAD and MVTec AD as external frozen validation datasets.

All four datasets were evaluated using 1-, 2-, and 4-shot normal references with three random seeds. This produced 36 dataset–seed–shot configurations. The same reference images, KNN procedure, post-processing, and evaluator were used for A1 and the matched feature-DINO-only control. Therefore, the difference between these two methods isolates the contribution of adding the second visual representation.

## 4. Main Results

The central result is a stable improvement in pixel-level localization relative to the matched DINO-only control. Averaged over the nine seed/shot configurations for each dataset, the Pixel-AP improvements were:

| Dataset | A1 Pixel-AP | DINO-only Pixel-AP | Improvement |
|---|---:|---:|---:|
| MPDD | 0.3562 | 0.3304 | +0.0258 |
| BTAD | 0.6455 | 0.6206 | +0.0249 |
| VisA | 0.3725 | 0.3201 | +0.0524 |
| MVTec AD | 0.5546 | 0.5226 | +0.0320 |

The dataset-level Pixel-AP improvement was positive in all nine seed/shot configurations on each dataset. The full evaluation includes Image-AUROC, Image-AP, Image-F1-max, Pixel-AUROC, Pixel-AP, and Pixel-AUPRO. A1 improved all three pixel-level metrics on all four datasets.

The results also reveal an important limitation. On BTAD, Image-AP decreased by approximately 0.0131 and Image-F1-max decreased by approximately 0.0237, despite the improvement in pixel localization. Consequently, the paper should claim robust improvement in **pixel-level anomaly localization**, rather than universal improvement across all detection and localization metrics.

In the broader baseline comparison, AnomalyDINO remains stronger on most macro-averaged metrics on MVTec AD and VisA. A1 is competitive and generally stronger than several other locally evaluated baselines, but it is not a new overall state of the art. Its main scientific value is the controlled demonstration of complementary frozen representations and the rigor of the validation procedure.

## 5. Negative Results and Failure Analysis

The negative findings are an important part of the study. More complex dynamic fusion and routing methods did not consistently outperform the fixed equal-weight fusion. The best dynamic variant improved the development average by only approximately 0.0009 and won in fewer than half of the seed/shot configurations. A later predictability test also showed that the available unlabeled reliability features could not reliably determine when one branch should correct the other. We therefore stopped this direction instead of presenting an unstable method as the main contribution.

The fixed fusion itself also has category-level failure cases. Persistent negative transfer was observed for categories such as MVTec leather, VisA chewinggum, and MPDD bracket_brown. In contrast, categories such as VisA cashew, MVTec toothbrush, and MPDD metal_plate showed clear improvements. Seven fixed qualitative cases have been prepared to illustrate both successful and unsuccessful localization. These results support a balanced discussion: the two representations are complementary on average, but their combination is not universally beneficial for every texture or defect type.

## 6. Reproducibility and Current Completion Status

The experimental stage is now essentially complete. The project contains:

- all 36 A1 configurations and complete six-metric reports;
- paired bootstrap confidence intervals and three-seed mean/standard-deviation tables;
- category-level negative-transfer statistics and per-image failure samples;
- a final protocol-aware baseline comparison table;
- qualitative success and failure figures with fixed sample-selection rules;
- a steady-state efficiency benchmark and memory-bank analysis;
- a leakage audit and frozen method specification;
- a compact reproducibility package containing 324 replayable patch maps, configuration and split hashes, a standalone CPU recomputation script, and SHA256 integrity checks.

The final method uses zero trainable parameters. In the current hardware benchmark, steady-state inference requires approximately 0.4146 seconds per image, corresponding to 2.412 images per second. Peak GPU memory is approximately 2.1 GB, and peak process RAM is approximately 4.0 GB. The project tests, P0 reproducibility gates, and P1 statistical, efficiency, fairness, failure-analysis, and complete-metric gates have all passed.

The remaining tasks are primarily manuscript preparation rather than additional model development. Before public release, the usage and redistribution terms of MPDD and BTAD should be manually confirmed, and the final bibliography metadata should be checked against the official publications. These points do not require rerunning the main experiments.

## 7. Proposed Manuscript Structure

We plan to write the paper around the following argument.

The Introduction will first explain the practical need for normal-only few-shot anomaly detection and accurate defect localization. It will then review normal-memory methods, frozen visual foundation models, and vision–language anomaly detection. The research gap will be framed as the insufficiently characterized reproducibility and failure boundary of combining heterogeneous frozen patch representations under a strict normal-only protocol.

The Related Work section will cover three groups: memory-bank and nearest-neighbor anomaly detection; frozen visual foundation models such as DINOv2 and AnomalyDINO; and CLIP-based or multimodal anomaly detection. It will explicitly distinguish our image-feature fusion from methods that use text prompts or learned multimodal routing during inference.

The Method section will describe the two frozen encoders, feature-grid alignment, branch-wise normalization, fixed equal-weight concatenation, normal memory-bank construction, KNN scoring, and leakage constraints. The matched DINO-only control will be described as part of the method design because it is essential for isolating the fusion contribution.

The Experiments section will present the dataset roles, 1/2/4-shot and three-seed protocol, evaluation metrics, baseline protocol differences, statistical analysis, and implementation details. The main table will report all six metrics for A1 and matched DINO-only across four datasets. A separate table will provide the protocol-aware comparison with existing methods.

The Results and Discussion sections will emphasize three conclusions: fixed fusion consistently improves Pixel-AP across the four datasets; the improvement does not extend to every image-level metric or every category; and dynamic routing provides no reliable advantage over the simpler fixed design. Qualitative success and failure cases, efficiency, limitations, and reproducibility will be included to support these conclusions.

The Conclusion will state that heterogeneous frozen visual representations can provide reproducible complementary information for few-shot industrial anomaly localization, but that this benefit should be established through strict controls and failure analysis rather than assumed from model complexity.

## 8. Intended Contribution and Positioning

The planned paper is not positioned as a new state-of-the-art architecture. Its contribution is a simple and reproducible method supported by unusually complete experimental evidence. The strongest defensible claims are:

1. a zero-training dual-encoder visual patch fusion method for few-shot anomaly localization;
2. stable Pixel-AP improvement over a matched DINO-only control across four datasets, three shot settings, and three reference seeds;
3. explicit evidence showing where the fusion fails and why dynamic routing was rejected;
4. a leakage-audited and replayable evaluation package.

For an applied SCI journal, particularly a lower-quartile venue, this provides a coherent manuscript: the method is technically simple, but the experimental design, cross-dataset validation, negative-result analysis, and reproducibility package make the conclusions clear and defensible.

