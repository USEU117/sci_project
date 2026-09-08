# DCFnet: Few-Shot Industrial Anomaly Localization via a Dual-Encoder Calibrated Fusion Network

## Abstract

Industrial inspection often starts with only a few defect-free images, while defects remain diverse and spatially small. A single pretrained representation may therefore provide incomplete evidence of abnormality. We present DCFnet, a Dual-Encoder Calibrated Fusion Network for normal-only few-shot anomaly localization. Frozen DINOv2 and Contrastive Language–Image Pre-training (CLIP) visual encoders supply heterogeneous patch descriptors. Deterministic grid alignment and branch-wise normalization make these descriptors compatible before fixed equal-weight concatenation and joint normalization. A memory constructed from normal support patches provides one-nearest-neighbor anomaly scores without target-domain parameter optimization or text-based inference. Across MPDD, BTAD, VisA, and MVTec AD, with 1, 2, or 4 normal references and three sampling seeds, DCFnet improves mean pixel-level average precision over a matched DINO-only control by 0.0258, 0.0249, 0.0524, and 0.0320, respectively. All 36 dataset-level reference configurations show positive gains. However, individual categories can deteriorate, two image-level metrics decrease on BTAD, and the second encoder increases inference cost. VisA is treated conservatively as validation within the AnomalyCLIP source-protocol lineage. The results support a reproducible localization benefit of fixed visual-feature fusion under the evaluated protocol. Code is available at https://github.com/USEU117/sci_project.

Keywords: industrial anomaly detection; few-shot learning; anomaly localization; dual-encoder fusion; normal memory bank

## 1 Introduction

Visual quality inspection must identify defects whose appearance and location cannot be fully specified before deployment. Small surface flaws, missing components, and contamination may be difficult to distinguish from acceptable variation in material, pose, or illumination. Supervised inspection requires representative defect examples and appropriate annotations, which may be unavailable for a new product or an unexpected failure mode. Industrial anomaly detection instead models normal appearance and identifies deviations at test time. Recent work on one-class industrial inspection, such as SLSG, illustrates this approach by improving normal feature embeddings and learning to distinguish normal from simulated abnormal patterns [1]. For practical inspection, the detector must also localize suspicious regions so that an image-level alarm can be examined.

The data constraint becomes more severe when even normal examples are scarce. During production-line start-up or a change of product, collecting a large category-specific reference set may delay deployment. This cold-start problem motivates recent few-shot industrial detectors such as DMMGNet [2]. In the normal-only setting considered here, the detector receives K verified normal images and must identify previously unseen abnormalities without abnormal target-domain training samples. A patch memory supports this task by comparing local query descriptors with normal reference descriptors. AnomalyDINO demonstrates that frozen DINOv2 features can make this simple comparison effective in few-shot anomaly detection [3]. Such results motivate examining the information available in pretrained visual representations before introducing additional trainable components.

A remaining difficulty is that industrial defects require sensitivity to different visual cues. A thin scratch alters local texture, whereas a missing or displaced part may require broader structural context. PGAD addresses this distinction by combining global invariant features with multiscale local features and calibrating the resulting anomaly scores [4]. This suggests that combining representations deserves investigation, but it does not establish that any pair of encoders will be complementary. In particular, self-supervised visual pretraining and image-text contrastive pretraining provide different learning signals. Whether their frozen image descriptors contribute complementary information to the same normal-memory detector must therefore be tested, rather than assumed.

Using two encoders is not, by itself, a new contribution. Sea-CLIP already combines CLIP and DINOv2 representations through semantic-aware patch matching and a learned anomaly-matching decoder [5]. Its results evaluate the complete adapted system and do not directly isolate the effect of a second frozen visual descriptor in an unchanged nearest-neighbor detector. We therefore ask a narrower question: can deterministic visual-feature fusion improve localization when the normal references and downstream scorer are held constant?

Three technical difficulties must be addressed. First, heterogeneous encoders produce patch grids at different input resolutions, so spatially corresponding descriptors cannot be combined without explicit alignment. Second, the feature spaces have different scales and pretraining biases; uncalibrated concatenation can allow one branch to dominate the distance metric. Third, average benchmark scores may conceal reference-sampling instability, category-specific negative transfer, or additional inference cost. A credible solution should therefore use a clearly specified alignment and normalization procedure, freeze all fusion decisions before external evaluation, compare against a genuinely matched single-encoder control, and report both successful and adverse cases.

DCFnet addresses the first two issues through spatial calibration and magnitude calibration before fixed feature fusion. It aligns CLIP-derived patches to the DINOv2 grid, normalizes each branch, concatenates equal-scaled descriptors, and normalizes the result. The term calibrated refers to these deterministic operations; it does not denote probability calibration or a learned gate. Normal support descriptors form a category memory, and query patches are scored by exact nearest-neighbor search. The evaluation addresses the third issue by preserving reference identities and the scoring pipeline across controls, then examining localization gains, adverse cases, and resource use.

The contributions are threefold. First, we formulate a target-domain training-free dual-visual-encoder detector with an explicit shared support/query representation and normal-memory interface. Second, we specify grid alignment and branch-wise magnitude normalization as a deterministic calibration path, and derive the resulting joint distance to clarify what fixed fusion actually measures. Third, we evaluate this design using matched single-encoder controls, multiple reference configurations, complete image- and pixel-level metrics, category-level failure analysis, and resource measurements. The contribution is the specified integration and its controlled evaluation; interpolation and concatenation are not claimed as new mathematical operators.

## 2 Related Work

Industrial anomaly detection can be organized into reconstruction and one-class learning, pretrained feature embeddings with normal memories, and foundation-model or hybrid approaches. The boundaries overlap: foundation features can also be used inside a memory detector. This organization highlights what each method learns, which data it uses, and how it produces an anomaly score.

### 2.1 Reconstruction and One-Class Learning

Reconstruction methods learn to reproduce normal images or features and interpret residuals as abnormality. One-class approaches instead learn a boundary around normal representations, sometimes using artificial negative examples. Both strategies can adapt features to industrial inspection, but fitting a category-dependent model from very few normal images can be unstable; a flexible reconstruction model may also reproduce anomalous content.

Yang et al. introduce SLSG, which combines generative pretraining, anomaly simulation, and graph reasoning to improve one-class embeddings [1]. Zhang et al. propose RealNet, using realistic synthetic anomalies and feature selection to make the representation sensitive to defects [6]. Bai et al. develop dual-path frequency discriminators to expose abnormalities through frequency-sensitive learning [7]. Luo et al. design DMMGNet with augmented training branches, discrimination mapping, and memory-bank mean guidance for few-shot learning [2]. These methods offer task-specific discrimination, but their optimized components and simulated anomaly assumptions differ from a frozen normal-reference detector. DCFnet retains pretrained representations and studies a deterministic combination without fitting a new target-category network.

### 2.2 Feature Embedding and Normal-Memory Methods

Embedding-based methods avoid image reconstruction and instead measure deviations in a pretrained feature space. PaDiM models the distribution of normal patch embeddings with multivariate Gaussian statistics [8]. PatchCore stores a coreset of normal local features and scores query patches by nearest-neighbor distance, demonstrating that a simple memory can achieve strong localization [9]. This paradigm is especially compatible with normal-only inspection because it neither requires defect labels nor assumes a closed anomaly taxonomy. Its weakness is equally direct: when normal references are few, the memory may inadequately cover acceptable appearance variation, and the distance metric is only as informative as the underlying feature representation.

Few-shot work has consequently focused on expanding normal coverage or improving representation quality. FEAD transforms support features conditionally and constructs multi-frequency descriptors to compensate for sparse normal patterns [10]. The K-NG framework formulates few-shot online detection and updates a Neural Gas representation from an unlabeled stream while attempting to avoid contamination by anomalies [11]. PGAD combines global invariant and multiscale local features and explicitly calibrates differently distributed anomaly scores [4]. FastRef refines normal prototypes at test time and uses optimal transport to reduce the risk of absorbing anomalous query features [12]. These methods address reference scarcity through transformation, updating, multiscale modeling, or transductive refinement. DCFnet instead keeps the normal memory fixed and asks whether a second frozen representation provides a reproducible gain without query-conditioned adaptation.

### 2.3 Foundation Models and Hybrid Approaches

Self-supervised visual foundation models have substantially changed the quality of transferable local descriptors. DINOv2 learns robust visual features without labels and transfers well across recognition and dense-prediction tasks. AnomalyDINO applies frozen DINOv2 patches to few-shot anomaly detection and shows that straightforward patch comparison is already highly competitive [3]. UniVAD develops a training-free unified model that reasons over reference images across industrial, logical, and medical anomalies [13]. SubspaceAD fits a principal subspace to frozen DINOv2 patches and scores reconstruction residuals, providing a strong training-free alternative to direct nearest-neighbor memories [14]. DCP-SFR argues that deep structural representations may suppress shallow defect cues and refines them to preserve subtle anomalies [15]. Collectively, these studies show that feature selection, normal-space modeling, and cue preservation can be as important as the downstream anomaly scorer. They also establish a demanding visual-only baseline against which fusion must be assessed.

Vision-language pretraining offers another route to transfer. CLIP learns aligned image and text embeddings from large-scale image-text supervision. WinCLIP uses prompt ensembles, window-level visual features, and optional normal references for zero- and few-shot anomaly classification and segmentation [16]. AnomalyCLIP replaces object-specific descriptions with object-agnostic prompts learned for normality and abnormality [17], whereas PromptAD learns prompts using only normal target samples [18]. InCTRL performs in-context residual learning with few-shot sample prompts [19]. AA-CLIP introduces anomaly-aware text anchors and patch-level visual alignment [20], and FAPrompt decomposes coarse abnormality descriptions into fine-grained prompt components [21]. These methods improve semantic transfer, but most retain text-conditioned scoring or learned prompt parameters. DCFnet uses only the frozen DPAM-modified CLIP image encoder at inference; it is therefore a dual-visual-encoder method rather than a vision-text inference system.

Hybrid approaches are the closest context for DCFnet. Ma et al. combine retrieval-enhanced references and multimodal prompt fusion in ReMP-AD [22]. Xu et al. study cooperative refinement of multimodal features for few-shot detection [23]. Guo et al. use DINOv2-guided patch matching and a learned decoder in Sea-CLIP [5]. PAPL introduces particle-based adaptive prompt learning for zero-shot inspection [24], while Jiang et al. combine CLIP and DINOv2 with multimodal fusion and stabilized attention pooling [25]. These studies preclude a claim that pairing the two encoder families is itself novel. They also motivate separating representation fusion from adaptation when interpreting results.

Across these three groups, stronger features and better normality models have reduced the need for complex target-category training, but they have not established that fixed heterogeneous fusion helps uniformly. DCFnet focuses on this limited question using spatial and magnitude calibration within a shared normal-memory pipeline. Its value must be assessed through matched controls and disclosed failure boundaries, rather than by the number of encoders or the complexity of a diagram.

## 3 Detailed DCFnet

### 3.1 Problem Statement

Let c denote a product category and let the support set contain K verified normal images. A query image x can be normal or anomalous. DCFnet returns a pixel-level anomaly map and an image-level score; it does not predict named defect classes. The normal support set is

[Editable equation 1 in DOCX]

Here K ∈ {1, 2, 4}. Labels and masks are evaluation annotations, not inputs to feature extraction, reference-memory construction, or external-validation parameter selection. All target-category model parameters remain fixed. Building a memory from normal images is a data-dependent preparation step, even though it performs no gradient optimization.

### 3.2 Overview Structure

Figure 1 separates support-memory construction from query inference. Both paths share frozen dual-encoder extraction, CLIP-to-DINO grid alignment, branch-wise normalization, equal-scaled concatenation, and joint normalization. Support descriptors are stored once for each category and reference configuration. Query descriptors are compared with this fixed memory, then patch scores are resized and smoothed. The query is never added to the normal memory. The encoder and control details are expanded in Supplementary Figures S1 and S2.

Figure 1 DCFnet overview. Normal supports build a category memory; the query uses the same frozen encoders and fusion path, then reads the fixed memory. Squared ℓ2 / 2 denotes half the squared Euclidean distance. Example images are from MPDD; the heatmap is an inference example, not a ground-truth label.

### 3.3 Frozen Dual-Encoder Visual Representation

The first branch is a frozen DINOv2 ViT-B/14 encoder [26]. An input is resized according to the fixed pipeline with a target short-side resolution of 448 pixels and cropped so that each spatial dimension is divisible by the patch size of 14. The resulting grid is denoted by h_D x w_D; its shorter dimension is 32 patches, and each patch descriptor has dimension d_D = 768. DINOv2 is selected because its self-supervised objective produces transferable dense visual descriptors and because DINO-based patch comparison is a strong few-shot baseline [3]. The branch is not fine-tuned on any of the four target datasets.

The second branch is a frozen CLIP ViT-L/14@336 visual encoder configured with AnomalyCLIP's diagonally prominent attention map (DPAM) modification [27] [17]. It receives a 518 x 518 input and produces a 37 x 37 grid of 768-dimensional local descriptors. The extractor requests the outputs of visual layers 6, 12, 18, and 24 with DPAM_layer set to 20 and retains the final returned patch tensor, corresponding to layer 24. Only image features are retained. No text prompt, text-encoder output, normal-versus-abnormal language similarity, or dynamic vision-language routing is computed during DCFnet inference. The branch is used to test whether CLIP-pretrained local descriptors processed by a fixed DPAM attention path provide information complementary to self-supervised DINOv2 descriptors. This motivation does not assume that complementarity holds for every category; that question is evaluated rather than built into the method.

Both backbones use 14-pixel patches. Their grid mismatch is caused by the different preprocessing resolutions and aspect-ratio handling, not different patch sizes. The CLIP transformer has hidden width 1024; its final local features are projected to 768 dimensions. The class token is excluded from the memory descriptors. The export script loads the AnomalyCLIP prompt checkpoint for its inherited setup, but the retained feature path calls only the image encoder and does not consume the loaded prompts.

### 3.4 Spatial Calibration and Fixed Feature Fusion

Let F_D and F_C be the DINOv2 and CLIP patch tensors. As shown in Figure 2, a bilinear operator R maps F_C onto the h_D × w_D reference lattice with align_corners = false:

[Editable equation 2 in DOCX]

This operation matches grid dimensions; it does not guarantee exact receptive-field or object-part correspondence. After resizing, we normalize each patch descriptor independently, using ε to protect the denominator:

[Editable equation 3 in DOCX]

The calibrated vectors are concatenated using the frozen coefficient α = 0.5 and jointly normalized to form a 1536-dimensional descriptor. Here Norm divides a vector by the maximum of its L2 norm and ε:

[Editable equation 4 in DOCX]

Figure 2 Deterministic grid and magnitude calibration. The CLIP grid is resized to the DINO lattice before branch-wise normalization, equal scaling, concatenation, and joint normalization. Grid correspondence is not learned semantic matching.

For nonzero branch descriptors, equal scaling implies f_A,p = [g_D,p; g_C,p]/√2. Consequently, the distance between a query patch p and a support patch q decomposes as

[Editable equation 5 in DOCX]

Thus, fusion averages two branch distances for the same candidate support patch before the nearest neighbor is selected. It is generally different from averaging two anomaly maps whose nearest neighbors were selected independently. A support patch that matches only one branch can become less competitive under the joint metric. This explains a possible benefit of joint evidence, while also exposing a failure mode: an unhelpful second descriptor can worsen the ranking. Equation (5) describes the implemented geometry, not proof that the two branches encode complementary defect causes.

### 3.5 Normal Reference Memory and Anomaly Scoring

For category c, all fused descriptors from its K normal references form the memory M_c (Figure 3). We retain all reference patches without a coreset or principal-component projection:

[Editable equation 6 in DOCX]

Here G_D(x_i) denotes the DINO patch lattice of reference image x_i. Descriptors are stored in a FAISS IndexFlatL2 index [28]. For query patch p, the raw anomaly score is one half of the minimum squared Euclidean distance:

[Editable equation 7 in DOCX]

FAISS returns squared distances. For unit descriptors, one half of squared Euclidean distance equals cosine distance. We arrange patch scores on the DINO grid as a score matrix a, bilinearly resize to 448 × 448, and apply Gaussian smoothing with σ = 4 pixels at that output resolution. The image score is the maximum response:

[Editable equation 8 in DOCX]

Figure 3 Normal-memory construction and exact nearest-neighbor scoring. All normal support patches are retained. A query does not update the memory; its scores are bilinearly resized and Gaussian-smoothed.

### 3.6 Matched Controls and Computational Complexity

The matched DINO-only control removes the DPAM-modified CLIP branch and the concatenation step but retains the identical DINOv2 descriptors, support-image identities, K values, random seeds, exact one-nearest-neighbor search, distance convention, upsampling, Gaussian smoothing, and evaluation resolution. This design is essential: comparing DCFnet only with published numbers from methods using different checkpoints, data roles, or post-processing would not isolate the effect of feature fusion. The matched control instead changes a single factor: whether the calibrated second visual representation is present.

The CLIP-image-only control likewise uses its visual descriptors without text, with matched reference identities and the same downstream metric convention. If a category has N_c stored patches and a query contains P patches, memory storage is O(N_c d) and exact retrieval costs O(P N_c d), where d is 1536 for DCFnet and 768 for a single branch. For a common grid, N_c = KP. The fused descriptor doubles bank storage relative to matched DINO-only, and an additional encoder must execute. No optimization cost does not imply negligible inference cost.

### 3.7 Model Configuration

Table 1 specifies the configuration used throughout the study. The spatial-alignment, normalization, concatenation, and nearest-neighbor stages have no trainable weights. Both visual backbones remain fixed; learning rate, optimizer, training epochs, and target-category training batch size are therefore not applicable. Support-memory construction is repeated when category, seed, or K changes.

Table 1 Frozen model configuration

| Component | Configuration | Output / trainable parameters |

|---|---|---|

| DINOv2 | ViT-B/14; 12 blocks; width 768; 12 heads; short side 448 | h_D × w_D × 768; 0 |

| CLIP visual | ViT-L/14@336; 24 blocks; width 1024; 16 heads; input 518 × 518 | 37 × 37 × 768 after projection; 0 |

| DPAM / layer output | DPAM_layer = 20; request 6/12/18/24; retain layer 24, exclude CLS | Frozen attention path; 0 |

| Grid calibration | Bilinear CLIP-to-DINO resize; align_corners = false | h_D × w_D × 768; 0 |

| Fusion | Per-patch branch L2; α = 0.5; concat; joint L2 | h_D × w_D × 1536; 0 |

| Normal memory | K = 1/2/4; seeds 0/1/2; all normal patches; no PCA/whitening | N_c × 1536; 0 |

| Scoring | FAISS IndexFlatL2; 1-NN; squared Euclidean distance / 2 | h_D × w_D scores; 0 |

| Post-processing | Bilinear resize, then Gaussian σ = 4; image maximum | 448 × 448 map and scalar; 0 |

| Target optimization | Optimizer, learning rate, epochs, training batch: not applicable | 0 trainable parameters |

The DINOv2 checkpoint contains 86,580,480 parameter elements. The CLIP visual architecture contains 304,293,888 parameters at its native 336-pixel position lattice; DPAM replacement preserves this count. The implementation interpolates the CLIP positional embedding from 24 × 24 to 37 × 37 at a 518-pixel input, increasing its stored position values by 812,032 without learning new weights. These counts exclude the unused text tower and prompt learner. All target-domain trainable-parameter counts are zero.

## 4 Experimental Evaluations

### 4.1 Experimental Design

#### 4.1.1 Datasets and Reference Protocol

We evaluate MPDD [29], BTAD [30], MVTec AD [31], and VisA [32]. Table 2 lists the evaluated category and test-image counts from the frozen project splits, rather than implying that every benchmark uses its full original training set. Every category is supplied with K ∈ {1, 2, 4} normal references, sampled using seeds 0, 1, and 2. Reference identities are shared by DCFnet and its matched controls. This yields nine reference configurations per dataset and 36 dataset-level comparisons.

Table 2 Evaluated datasets and their roles

| Dataset | Categories | Test images | Role |

|---|---|---|---|

| MPDD | 6 | 458 | Development |

| BTAD | 3 | 741 | External frozen validation |

| VisA | 12 | 2162 | In-domain frozen validation |

| MVTec AD | 15 | 1725 | External frozen validation |

MPDD is the development dataset. Fusion and post-processing decisions are frozen before BTAD and MVTec AD evaluation. VisA is reported conservatively as in-domain frozen validation because the inherited AnomalyCLIP setup uses a VisA-trained prompt checkpoint, even though the retained visual path does not consume those prompts. This role assignment avoids presenting VisA as independently established external-transfer evidence. Test labels are used for metric computation and subsequent analysis, not for fitting the validation detector. Development-set comparisons informed the decision to retain a symmetric fusion coefficient; they must not be described as label-free method selection.

#### 4.1.2 Baselines and Ablation Scope

The primary comparison is DCFnet against matched DINO-only. Additional CLIP-image-only results cover all nine configurations on BTAD and MVTec AD. We also report local evaluations of PatchCore [9], WinCLIP+ [16], AnomalyDINO [3], and PromptAD [18], with supplementary context from zero-shot AnomalyCLIP [17] and source-trained ReMP-AD [22]. These external methods differ in backbone, adaptation, and sometimes shot coverage; their numbers provide context rather than an isolated estimate of fusion effectiveness. Recent methods discussed in Section 2, including Sea-CLIP and SubspaceAD, are literature comparisons where a verified common-protocol numerical run is unavailable.

The validated ablations concern encoder presence, reference count, and reference sampling. We do not have a complete factorial ablation independently removing spatial interpolation, branch normalization, joint normalization, and DPAM under the same final protocol. Accordingly, the study establishes the effect of the complete fusion path, not a separately measured gain for every calibration operation.

#### 4.1.3 Evaluation Metrics and Uncertainty

Image-level metrics are the area under the receiver operating characteristic curve (I-AUROC), average precision (I-AP), and maximum F1 over evaluation thresholds (I-F1max). Pixel-level metrics are P-AUROC, P-AP, and the normalized area under the per-region overlap curve up to a false-positive rate of 0.30 (P-AUPRO). All values are reported on the 0–1 scale. I-F1max is an evaluation summary selected with ground-truth labels; it is not a deployment threshold obtained from normal references.

For pixel metrics, the frozen evaluator samples each 448 × 448 prediction and mask at stride 8. Test pixels are pooled within each category before its metric is computed, and category metrics are then averaged with equal weight. The nine-configuration mean and sample standard deviation describe sensitivity to reference selection and shot count. These configurations share the same test set; they are not nine independent datasets. Full-resolution scores with a different pixel-sampling rule are not directly interchangeable with these results.

Archived uncertainty analyses use 2,000 bootstrap resamples and percentile 95% intervals, with base random seed 20260827. The category bootstrap resamples paired category-level P-AP differences and corresponds to the macro-average statistic. A separate image bootstrap resamples paired per-image P-AP differences only for images retaining at least one positive mask pixel after stride sampling; it is not category-stratified and does not estimate the same statistic as the main table. We therefore emphasize category intervals when interpreting macro-average gains. Replay comparisons were checked against the archived configuration results within a tolerance of 5 × 10⁻⁴.

#### 4.1.4 Implementation and Computing Environment

Feature extraction uses PyTorch with frozen pretrained backbones; exact nearest-neighbor retrieval uses FAISS on the CPU. The archived environment records Windows, Python 3.10.11, PyTorch 2.0.0 with CUDA 11.8 for the CLIP branch, and an NVIDIA GeForce RTX 3060 Laptop GPU with 6,144 MiB VRAM. The workstation inventory checked during manuscript preparation identifies an Intel Core i9-12900H CPU and approximately 16 GB installed RAM (15.8 GiB reported as physical memory). CPU/RAM inventory was completed retrospectively, whereas GPU and software details are present in the experiment archive.

Efficiency measurements use the MVTec bottle category, seed 0, and one normal reference, with three warm-up passes and 30 timed repetitions per stage. The two extraction stages execute on the GPU in separate processes; alignment, concatenation, and retrieval are measured on the CPU from cached features. The reported aggregate is the sum of stage means and excludes one-time model loading. It should not be interpreted as a separately instrumented application-wide latency including interprocess orchestration, image acquisition, or all transfer costs.

### 4.2 Results and Analysis

#### 4.2.1 Main Results against the Matched Control

Table 3 reports all six metrics for the matched comparison. Mean P-AP increases from 0.3304 to 0.3562 on MPDD, from 0.6206 to 0.6455 on BTAD, from 0.3201 to 0.3725 on VisA, and from 0.5226 to 0.5546 on MVTec AD. The mean gains computed from the unrounded paired records are 0.0258, 0.0249, 0.0524, and 0.0320. All three pixel-level metric means increase on each dataset. The strongest supported conclusion is a localization benefit relative to the same DINO-only pipeline.

Table 3 Complete matched results across four datasets

| Dataset | Method | I-AUROC | I-AP | I-F1max | P-AUROC | P-AP | P-AUPRO |

|---|---|---|---|---|---|---|---|

| MPDD | DINO-only | 0.7460 ± 0.0538 | 0.7656 ± 0.0474 | 0.8213 ± 0.0205 | 0.9547 ± 0.0076 | 0.3304 ± 0.0345 | 0.8805 ± 0.0232 |

| MPDD | DCFnet | 0.7750 ± 0.0519 | 0.7948 ± 0.0490 | 0.8280 ± 0.0167 | 0.9646 ± 0.0065 | 0.3562 ± 0.0350 | 0.9001 ± 0.0208 |

| BTAD | DINO-only | 0.9214 ± 0.0175 | 0.9316 ± 0.0268 | 0.8943 ± 0.0189 | 0.9708 ± 0.0015 | 0.6206 ± 0.0183 | 0.7574 ± 0.0092 |

| BTAD | DCFnet | 0.9254 ± 0.0120 | 0.9185 ± 0.0212 | 0.8706 ± 0.0131 | 0.9735 ± 0.0012 | 0.6455 ± 0.0124 | 0.7608 ± 0.0102 |

| VisA | DINO-only | 0.8730 ± 0.0303 | 0.8779 ± 0.0303 | 0.8512 ± 0.0149 | 0.9662 ± 0.0070 | 0.3201 ± 0.0278 | 0.8933 ± 0.0167 |

| VisA | DCFnet | 0.9046 ± 0.0196 | 0.9103 ± 0.0190 | 0.8710 ± 0.0138 | 0.9757 ± 0.0046 | 0.3725 ± 0.0246 | 0.9125 ± 0.0117 |

| MVTec AD | DINO-only | 0.9465 ± 0.0172 | 0.9703 ± 0.0092 | 0.9548 ± 0.0098 | 0.9548 ± 0.0057 | 0.5226 ± 0.0184 | 0.9065 ± 0.0097 |

| MVTec AD | DCFnet | 0.9583 ± 0.0141 | 0.9777 ± 0.0069 | 0.9638 ± 0.0078 | 0.9663 ± 0.0043 | 0.5546 ± 0.0196 | 0.9199 ± 0.0082 |

Image-level improvements are less uniform. On BTAD, I-AP decreases from 0.9316 to 0.9185 and I-F1max from 0.8943 to 0.8706, despite improved pixel localization. Maximum pooling compresses the map to one score, so improved pixel ranking need not improve the ordering of entire images. The results do not establish all-metric superiority and should be interpreted in relation to the intended localization task.

#### 4.2.2 Encoder Controls and Reference Stability

Table 4 completes the three-way P-AP comparison on BTAD and MVTec AD. CLIP-image-only is weaker than matched DINO-only, but its fusion with DINOv2 improves the mean beyond both single branches. DCFnet exceeds CLIP-image-only in all 18 configurations across the two datasets. This excludes the simple explanation that fusion merely substitutes a stronger standalone CLIP branch, and is consistent with complementary information under the joint metric. It does not identify the particular features responsible for the gain.

Table 4 Matched three-way P-AP comparison

| Dataset | DINO-only | CLIP-image-only | DCFnet |

|---|---|---|---|

| BTAD | 0.6206 ± 0.0183 | 0.4006 ± 0.0277 | 0.6455 ± 0.0124 |

| MVTec AD | 0.5226 ± 0.0184 | 0.4654 ± 0.0180 | 0.5546 ± 0.0196 |

Every dataset-level seed/shot configuration has a positive P-AP difference against matched DINO-only (Figure 4). Table 5 groups those differences by shot count. MPDD benefits somewhat more at two and four shots than at one shot, while the VisA and MVTec AD margins narrow as K increases. BTAD has similar mean margins across the three shot settings. These are descriptive patterns within the evaluated references, not evidence of a universal monotonic relation between K and fusion gain.

Table 5 P-AP gain grouped by normal-reference count

| Dataset | 1 shot | 2 shots | 4 shots |

|---|---|---|---|

| MPDD | 0.0210 ± 0.0122 | 0.0277 ± 0.0024 | 0.0288 ± 0.0025 |

| BTAD | 0.0258 ± 0.0092 | 0.0239 ± 0.0091 | 0.0250 ± 0.0073 |

| VisA | 0.0558 ± 0.0063 | 0.0523 ± 0.0033 | 0.0490 ± 0.0019 |

| MVTec AD | 0.0348 ± 0.0056 | 0.0311 ± 0.0054 | 0.0300 ± 0.0051 |

Figure 4 P-AP differences between DCFnet (A1) and matched DINO-only for every seed/shot configuration. All 36 dataset-level point estimates are positive. The panels use different vertical ranges; connecting lines aid reading and do not establish a fitted trend.

Positive point estimates do not mean that every confidence interval excludes zero. For MPDD at seed 1 and one shot, the category-bootstrap interval is [−0.0374, 0.0450] around a point gain of approximately 0.0070. For BTAD at seed 2 and two shots, it is [−0.0085, 0.0320] around 0.0140. The archived category intervals are above zero for all nine configurations of VisA and MVTec AD, but MPDD and BTAD contain intervals spanning zero. In particular, category resampling from only three BTAD categories has limited resolution. No independent-sample significance claim is based on the 36 shared-test configurations.

#### 4.2.3 Comparison with Broader Baselines

Table 6 places DCFnet alongside verified local baseline results. DCFnet exceeds the reported PatchCore, WinCLIP+, and PromptAD means on both MVTec AD and VisA, but remains below the project AnomalyDINO run: 0.5546 versus 0.5710 on MVTec AD and 0.3725 versus 0.4117 on VisA. That AnomalyDINO run uses ViT-S/14 and its own pipeline; it is distinct from the matched ViT-B/14 DINO-only control. A gain against the matched control therefore cannot be rewritten as a gain against AnomalyDINO.

Table 6 Broader local baseline context in P-AP

| Method | Protocol / backbone | MVTec AD | VisA |

|---|---|---|---|

| PatchCore | Frozen WRN-50; normal coreset | 0.3944 ± 0.0378 | 0.2553 ± 0.0305 |

| WinCLIP+ | Frozen OpenCLIP ViT-B/16+; text and references | 0.2892 ± 0.0071 | 0.1080 ± 0.0040 |

| AnomalyDINO | Frozen DINOv2 ViT-S/14; own pipeline | 0.5710 ± 0.0096 | 0.4117 ± 0.0192 |

| PromptAD | CLIP ViT-L/14; target-normal prompt tuning | 0.5202 ± 0.0153 | 0.3002 ± 0.0184 |

| DCFnet | Frozen DINOv2-B + CLIP-L; no target optimization | 0.5546 ± 0.0196 | 0.3725 ± 0.0246 |

On MVTec AD, the archived zero-shot AnomalyCLIP result is 0.4454 P-AP for one run, while source-trained ReMP-AD reaches 0.5790 ± 0.0296 across its three shot evaluations. These protocols have different sample counts and adaptation assumptions from the nine-configuration DCFnet study and are not pooled with its stability estimate. Together with the stronger AnomalyDINO result, they preclude a state-of-the-art claim. Wider evaluation on larger, multi-view benchmarks such as Real-IAD [33] would also be needed to establish broader industrial coverage.

#### 4.2.4 Category Boundaries and Qualitative Results

Dataset averages conceal persistent negative transfer. Table 7 includes both high-gain and adverse categories. MVTec leather, VisA chewinggum, MVTec hazelnut, VisA candle, and MPDD bracket_brown all lose P-AP in every reference configuration. Across the four datasets, ten dataset–category pairs have at least one negative configuration, accounting for 65 negative category/configuration occurrences. These counts distinguish occasional losses from persistent category-level deterioration.

Table 7 Selected category-level gains and losses

| Dataset | Category | Mean ΔP-AP | Negative configs / 9 |

|---|---|---|---|

| VisA | pcb1 | +0.1148 | 0 |

| VisA | cashew | +0.1145 | 0 |

| MVTec AD | toothbrush | +0.1080 | 0 |

| MPDD | metal_plate | +0.0917 | 0 |

| MVTec AD | leather | −0.0428 | 9 |

| VisA | chewinggum | −0.0386 | 9 |

| MVTec AD | hazelnut | −0.0297 | 9 |

| VisA | candle | −0.0198 | 9 |

| MPDD | bracket_brown | −0.0051 | 9 |

Figures 5 and 6 show fixed one-shot, seed-0 success and failure cases. Ground-truth defect boundaries are marked in cyan, and each case uses a shared color scale for the two anomaly maps. The displayed P-AP is the per-image stride-sampled metric, not the category-pooled value in Table 3. Successes were selected for the largest per-image gain within the specified category; the displayed failures are category extremes recorded in the fixed case manifest. They illustrate behavior and are not an unbiased random sample of test images.

Figure 5 Selected success cases at seed 0 and one shot: VisA cashew 085 and MVTec toothbrush 004. A1 denotes DCFnet. Cyan marks ground-truth boundaries; scores share a color scale within each row. P-AP is per image. Cases maximize the recorded per-image gain in their respective categories and are illustrative extremes.

Figure 6 Selected negative-transfer cases at seed 0 and one shot: VisA chewinggum 037 and MVTec leather poke 016. A1 denotes DCFnet. These category-extreme failures show stronger DINO-only localization being degraded by fusion; they do not establish the cause of the degradation.

The cashew and toothbrush examples show improved ranking of defect regions relative to background responses. Conversely, chewinggum and leather show that a stronger DINO-only localization can be diluted by the second branch. Texture sensitivity, receptive-field differences, and imperfect grid correspondence are plausible explanations, but the present controls do not separate them causally. A dedicated intervention study would be required before attributing the losses to a specific semantic or texture mechanism.

#### 4.2.5 Efficiency and Memory Cost

Table 8 reports the measured stage costs. CLIP extraction dominates the stage-summed mean of 0.4146 seconds per image; its reciprocal is approximately 2.412 images per second under the stated setup. This is an aggregate estimate from separate stage measurements, not an independently measured production throughput. The experiment establishes that zero target-domain training does not remove the runtime cost of the additional backbone.

Table 8 Stage timing for the one-shot bottle benchmark

| Stage | Device | Mean seconds | Std seconds |

|---|---|---|---|

| DINOv2 extraction | GPU | 0.0626 | 0.0010 |

| CLIP extraction | GPU | 0.3049 | 0.0010 |

| Alignment + fusion + retrieval | CPU | 0.0471 | 0.0012 |

| Sum of stage means | Mixed | 0.4146 | Not estimated |

Archived extraction measurements report peak allocated GPU memory of 374.58 MB for DINOv2 and 2,072.81 MB for CLIP. The maximum recorded working set across the separate benchmark processes is 3,980.9 MB. These are stage/process maxima; they do not establish the peak of an implementation holding both models and all cached features simultaneously. Dataset-level bank totals in Table 9 show the exact dimension-driven storage increase. They sum the category banks at each K and exclude model parameters, query caches, and allocator overhead.

Table 9 Aggregate normal-memory storage in float32 MB

| Dataset | Method | 1 shot | 2 shots | 4 shots |

|---|---|---|---|---|

| MPDD | DINO-only | 18.87 | 37.75 | 75.50 |

| MPDD | DCFnet | 37.75 | 75.50 | 150.99 |

| BTAD | DINO-only | 10.42 | 20.84 | 41.68 |

| BTAD | DCFnet | 20.84 | 41.68 | 83.36 |

| VisA | DINO-only | 49.94 | 99.88 | 199.75 |

| VisA | DCFnet | 99.88 | 199.75 | 399.51 |

| MVTec AD | DINO-only | 47.19 | 94.37 | 188.74 |

| MVTec AD | DCFnet | 94.37 | 188.74 | 377.49 |

#### 4.2.6 Reproducibility and Remaining Limitations

The accompanying repository records the frozen configuration, reference identities, split and checkpoint hashes, per-configuration metrics, and a CPU replay path. The compact prediction package contains 324 category/configuration files with float16 patch-score arrays, occupying approximately 186.8 MB; each file can contain multiple test images. Dataset images and third-party weights are obtained separately. Replaying quantized maps verifies the archived evaluation within its stated tolerance, whereas a full reproduction also requires the recorded checkpoints, preprocessing, and legally obtained datasets.

Several limitations remain. The study validates the complete fusion path but not the isolated contribution of each calibration operation. Deterministic grid resizing is not semantic correspondence. Category-specific losses show that equal fusion is not universally suitable. The pixel evaluator uses stride sampling, so tiny defects can be missed by the evaluation lattice. VisA retains a conservative in-domain role, the tested benchmarks do not cover all industrial settings, and the stage-based runtime audit is narrower than a deployed application benchmark. These limitations bound the present claim and define concrete targets for further experiments.

## 5 Conclusion

We presented DCFnet, a normal-only few-shot anomaly-localization pipeline that combines frozen DINOv2 and DPAM-configured CLIP visual descriptors through deterministic spatial and magnitude calibration. Its joint normal-memory distance uses both representations without learning a fusion network or invoking text at inference. Matched experiments across four datasets, three reference seeds, and three shot levels show positive dataset-level P-AP gains in all 36 configurations. Additional single-branch controls on BTAD and MVTec AD support a benefit beyond either isolated visual branch under that protocol.

The benefit is qualified by persistent category losses, lower BTAD image-level AP and maximum F1, stronger results from some broader baselines, and a material inference-cost increase. DCFnet is therefore a transparent fixed-fusion reference rather than a universal best-performing detector. Future work should isolate calibration components, test controlled alternatives to grid interpolation, validate any selective fusion rule without external-test tuning, and measure an integrated application pipeline. Such studies are necessary before assigning a causal mechanism to the observed complementarity or claiming wider deployment suitability.

## Data and Code Availability

The implementation and reproducibility materials are available at https://github.com/USEU117/sci_project. The evaluated data are MPDD [29], BTAD [30], MVTec AD [31], and VisA [32], distributed by their respective providers. The repository records the required dataset layout and checkpoint identifiers without redistributing the original images or third-party model weights.

## References

[1] M. Yang, J. Liu, Z. Yang, and Z. Wu, 'SLSG: Industrial image anomaly detection with improved feature embeddings and one-class classification,' Pattern Recognit., vol. 156, Art. no. 110862, 2024, doi: 10.1016/j.patcog.2024.110862.

[2] A. Luo, G. Wen, Y. Cheng, S. Mei, H. Dong, and X. Liu, 'DMMGNet: A discrimination mapping and memory bank mean guidance-based network for high-performance few-shot industrial anomaly detection,' Neurocomputing, vol. 610, Art. no. 128622, 2024, doi: 10.1016/j.neucom.2024.128622.

[3] S. Damm, M. Laszkiewicz, J. Lederer, and A. Fischer, 'AnomalyDINO: Boosting patch-based few-shot anomaly detection with DINOv2,' in Proc. IEEE/CVF Winter Conf. Appl. Comput. Vis. (WACV), 2025, pp. 1319-1329.

[4] J. Zhou, W. Wong, and F. Liao, 'One-shot unsupervised industrial anomaly detection: Enhanced performance under extreme data scarcity,' Pattern Recognit., vol. 173, Art. no. 112759, 2026, doi: 10.1016/j.patcog.2025.112759.

[5] X. Guo, Z. Chen, C. D. Castillo, H. Wang, and X. Liu, 'Sea-CLIP: Mining semantic-aware representations for few-shot anomaly detection with CLIP,' in Proc. IEEE/CVF Winter Conf. Appl. Comput. Vis. (WACV), 2026, pp. 3689-3699.

[6] X. Zhang, M. Xu, and X. Zhou, 'RealNet: A feature selection network with realistic synthetic anomaly for anomaly detection,' in Proc. IEEE/CVF Conf. Comput. Vis. Pattern Recognit. (CVPR), 2024, pp. 16699-16708.

[7] Y. Bai, J. Zhang, Z. Chen, Y. Dong, Y. Cao, and G. Tian, 'Dual-path frequency discriminators for few-shot anomaly detection,' Knowl.-Based Syst., vol. 302, Art. no. 112397, 2024, doi: 10.1016/j.knosys.2024.112397.

[8] T. Defard, A. Setkov, A. Loesch, and R. Audigier, 'PaDiM: A patch distribution modeling framework for anomaly detection and localization,' in Pattern Recognition. ICPR International Workshops and Challenges, 2021, pp. 475-489, doi: 10.1007/978-3-030-68799-1_35.

[9] K. Roth, L. Pemula, J. Zepeda, B. Schölkopf, T. Brox, and P. Gehler, 'Towards total recall in industrial anomaly detection,' in Proc. IEEE/CVF Conf. Comput. Vis. Pattern Recognit. (CVPR), 2022, pp. 14318-14328.

[10] Z. Hu, X. Zeng, Y. Li, Z. Yin, E. Meng, L. Zhu, and X. Kong, 'Few-shot anomaly detection with adaptive feature transformation and descriptor construction,' Chin. J. Aeronaut., vol. 38, no. 3, Art. no. 103098, 2025, doi: 10.1016/j.cja.2024.06.007.

[11] S. Wei, X. Wei, Z. Ma, S. Dong, S. Zhang, and Y. Gong, 'Few-shot online anomaly detection and segmentation,' Knowl.-Based Syst., vol. 300, Art. no. 112168, 2024, doi: 10.1016/j.knosys.2024.112168.

[12] Y. Li, L. Tian, Y. Dai, W. Chen, L. Bao, and X. Liu, 'FastRef: Fast prototype refinement for few-shot industrial anomaly detection,' in Proc. IEEE/CVF Conf. Comput. Vis. Pattern Recognit. (CVPR), 2026, pp. 43040-43049.

[13] Z. Gu, B. Zhu, G. Zhu, Y. Chen, M. Tang, and J. Wang, 'UniVAD: A training-free unified model for few-shot visual anomaly detection,' in Proc. IEEE/CVF Conf. Comput. Vis. Pattern Recognit. (CVPR), 2025, pp. 15194-15203.

[14] C. Lendering, E. Akdag, and E. Bondarev, 'SubspaceAD: Training-free few-shot anomaly detection via subspace modeling,' in Proc. IEEE/CVF Conf. Comput. Vis. Pattern Recognit. (CVPR), 2026, pp. 28557-28566.

[15] L. Jiang, Y. Huang, Z. Xu, Y. Xu, H.-S. Wong, and S. Wu, 'Defect cue-preserved structural feature refinement for few-shot anomaly detection,' in Proc. IEEE/CVF Conf. Comput. Vis. Pattern Recognit. (CVPR), 2026, pp. 35607-35616.

[16] J. Jeong, Y. Zou, T. Kim, D. Zhang, A. Ravichandran, and O. Dabeer, 'WinCLIP: Zero-/few-shot anomaly classification and segmentation,' in Proc. IEEE/CVF Conf. Comput. Vis. Pattern Recognit. (CVPR), 2023, pp. 19606-19616.

[17] Q. Zhou, G. Pang, Y. Tian, S. He, and J. Chen, 'AnomalyCLIP: Object-agnostic prompt learning for zero-shot anomaly detection,' in Proc. Int. Conf. Learn. Represent. (ICLR), 2024.

[18] X. Li, Z. Zhang, X. Tan, C. Chen, Y. Qu, Y. Xie, and L. Ma, 'PromptAD: Learning prompts with only normal samples for few-shot anomaly detection,' in Proc. IEEE/CVF Conf. Comput. Vis. Pattern Recognit. (CVPR), 2024, pp. 16838-16848.

[19] J. Zhu and G. Pang, 'Toward generalist anomaly detection via in-context residual learning with few-shot sample prompts,' in Proc. IEEE/CVF Conf. Comput. Vis. Pattern Recognit. (CVPR), 2024, pp. 17826-17836.

[20] W. Ma et al., 'AA-CLIP: Enhancing zero-shot anomaly detection via anomaly-aware CLIP,' in Proc. IEEE/CVF Conf. Comput. Vis. Pattern Recognit. (CVPR), 2025, pp. 4744-4754.

[21] J. Zhu, Y.-S. Ong, C. Shen, and G. Pang, 'Fine-grained abnormality prompt learning for zero-shot anomaly detection,' in Proc. IEEE/CVF Int. Conf. Comput. Vis. (ICCV), 2025, pp. 22241-22251.

[22] H. Ma, G. Yang, D. Zhao, Y. Ji, and W. Zuo, 'ReMP-AD: Retrieval-enhanced multi-modal prompt fusion for few-shot industrial visual anomaly detection,' in Proc. IEEE/CVF Int. Conf. Comput. Vis. (ICCV), 2025, pp. 20425-20434, doi: 10.1109/ICCV51701.2025.01899.

[23] L. Xu, D. Han, G. Li, M. Zhou, J. Wan, and M. Li, 'Multimodal feature cooperative refinement for few-shot anomaly detection,' Adv. Eng. Inform., vol. 68, pt. C, Art. no. 103792, 2025, doi: 10.1016/j.aei.2025.103792.

[24] R. Ma, C. Li, J. Chen, Y. Feng, and J. Xie, 'PAPL: Particle-based adaptive prompt learning for zero-shot industrial anomaly detection,' Pattern Recognit., vol. 178, Art. no. 113489, 2026, doi: 10.1016/j.patcog.2026.113489.

[25] J. Jiang, Z. He, A. Wan, K. AL-Bukhaiti, and K. Wang, 'Zero-shot industrial anomaly detection via CLIP-DINOv2 multimodal fusion and stabilized attention pooling,' Electronics, vol. 14, no. 24, Art. no. 4785, 2025, doi: 10.3390/electronics14244785.

[26] M. Oquab et al., 'DINOv2: Learning robust visual features without supervision,' Trans. Mach. Learn. Res., 2024.

[27] A. Radford et al., 'Learning transferable visual models from natural language supervision,' in Proc. 38th Int. Conf. Mach. Learn., vol. 139, 2021, pp. 8748-8763.

[28] J. Johnson, M. Douze, and H. Jegou, 'Billion-scale similarity search with GPUs,' IEEE Trans. Big Data, vol. 7, no. 3, pp. 535-547, 2021, doi: 10.1109/TBDATA.2019.2921572.

[29] S. Jezek, M. Jonak, R. Burget, P. Dvorak, and M. Skotak, 'Deep learning-based defect detection of metal parts: Evaluating current methods in complex conditions,' in Proc. 13th Int. Congr. Ultra Modern Telecommun. Control Syst. Workshops, 2021, pp. 66-71, doi: 10.1109/ICUMT54235.2021.9631567.

[30] P. Mishra, R. Verk, D. Fornasier, C. Piciarelli, and G. L. Foresti, 'VT-ADL: A vision transformer network for image anomaly detection and localization,' in Proc. IEEE 30th Int. Symp. Ind. Electron., 2021, pp. 1-6, doi: 10.1109/ISIE45552.2021.9576231.

[31] P. Bergmann, M. Fauser, D. Sattlegger, and C. Steger, 'MVTec AD - A comprehensive real-world dataset for unsupervised anomaly detection,' in Proc. IEEE/CVF Conf. Comput. Vis. Pattern Recognit. (CVPR), 2019, pp. 9592-9600.

[32] Y. Zou, J. Jeong, L. Pemula, D. Zhang, and O. Dabeer, 'SPot-the-Difference self-supervised pre-training for anomaly detection and segmentation,' in Proc. Eur. Conf. Comput. Vis. (ECCV), 2022, pp. 392-408.

[33] C. Wang et al., 'Real-IAD: A real-world multi-view dataset for benchmarking versatile industrial anomaly detection,' in Proc. IEEE/CVF Conf. Comput. Vis. Pattern Recognit. (CVPR), 2024, pp. 22883-22892.

## Supplementary Method Figures

Figure S1 Frozen visual encoders and native patch grids. Both patch sizes are 14 pixels; the grid difference arises from preprocessing. CLIP hidden features are projected to 768 dimensions, and only the final requested local tensor is retained.

Figure S2 Matched ablation design. DINO-only and DCFnet share reference identities and downstream scoring, while each builds a memory in its own descriptor space. The CLIP-image-only control is additionally evaluated on BTAD and MVTec AD.