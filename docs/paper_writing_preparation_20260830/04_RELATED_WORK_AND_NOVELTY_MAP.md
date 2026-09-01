# 04. Related Work and Novelty Map

## 1. 近邻工作比较

| Method | Setting | Main representation | Target normal references | Learned component | Text at inference | Relation to A1 |
|---|---|---|---|---|---|---|
| PatchCore (CVPR 2022) | unsupervised/normal-only | ImageNet CNN patches | many or sampled normal images | none; coreset memory | no | establishes nearest-neighbor normal memory |
| WinCLIP (CVPR 2023) | zero/few-shot | CLIP image-text | optional few-shot normal references | none | yes | CLIP anomaly baseline, but not image-only dual encoder |
| AnomalyCLIP (ICLR 2024) | zero-shot | anomaly-adapted CLIP | no target references | learned prompts during source training | yes in original method | A1 reuses only its frozen image tower |
| PromptAD (CVPR 2024) | normal-only few-shot | CLIP image-text | 1/2/4-shot | target-normal prompt learning | yes | stronger adaptation but not training-free |
| AnomalyDINO (WACV 2025) | few-shot | frozen DINOv2 patches | few-shot normal memory | none | no | conceptual and matched visual baseline family |
| UniVAD (CVPR 2025) | unified few-shot | pretrained visual features + reference reasoning | few-shot | training-free pipeline | no/varies by component | broad training-free context, different architecture |
| ReMP-AD (ICCV 2025) | few-shot | CLIP + retrieved textual priors | 1/2/4-shot | source-trained modules | yes | normal-reference multimodal fusion, not fixed image-only fusion |
| CLIP–DINOv2 DMA/SAP (Electronics 2025) | zero-shot transfer | CLIP + DINOv2 | no target references | auxiliary-data training, attention and pooling | framework is multimodal | direct encoder-pair overlap, different protocol and learned fusion |
| Sea-CLIP (WACV 2026) | few-shot | CLIP + DINOv2 | few-shot references | learnable decoder, trainable prompts, synthetic anomalies | yes | closest encoder-pair competitor; substantially more learned machinery |
| PAPL (Pattern Recognition 2026) | zero-shot transfer | CLIP + DINO hierarchical features | no target references | particle prompt learning, hierarchical attention, adaptive pooling | yes | direct feature-family overlap, but zero-shot and trained |
| FastRef (CVPR 2026) | few-shot | plugs into PatchCore/WinCLIP/AnomalyDINO | 1/2/4-shot | test-time closed-form prototype refinement | depends on base | query-adaptive prototype method, not frozen fixed memory |
| SubspaceAD (CVPR 2026) | few-shot | frozen DINOv2 | few-shot normal patches | PCA fit, no neural training | no | strong simple visual baseline; models normal subspace instead of encoder complementarity |
| **A1 (this study)** | normal-only few-shot | frozen DINOv2 + AnomalyCLIP image patches | 1/2/4-shot; 3 seeds | none; fixed concat + k-NN | **no** | isolates representation complementarity with matched control and failure audit |

## 2. Novelty boundary

### What is not novel

- Using DINOv2 for anomaly detection.
- Using a CLIP-derived image encoder for anomaly detection.
- Combining CLIP and DINO features in general.
- Constructing a patch memory and applying nearest-neighbor scoring.
- Concatenating normalized features as a generic operation.

### What can be defended

The defensible novelty is the **study design and evidence package**:

1. The second frozen image representation is added to an otherwise matched DINO normal-memory pipeline, enabling a clean attribution of the observed difference.
2. The fusion rule is intentionally fixed and training-free, separating intrinsic representation complementarity from the capacity of a learned fusion network.
3. The protocol covers four datasets, three reference seeds and three shot levels, with explicit dataset roles and no test-label selection.
4. The paper reports complete metrics and stable category-level negative transfer, not only favorable averages.
5. Reconstructable compact maps, hashes, scripts, statistical intervals and efficiency measurements make the empirical claim auditable.

## 3. Recommended Related Work closing paragraph

> Existing studies show that DINO-based visual features, CLIP-based anomaly priors, and learned CLIP–DINO integration can all be effective. Our goal is different from proposing another high-capacity fusion architecture. We ask whether the addition of a heterogeneous frozen image representation provides measurable complementarity when the anomaly detector, normal references, scoring rule, and evaluation protocol are held fixed. This controlled perspective also requires reporting where simple fusion degrades performance and the computational cost of obtaining the additional representation.

## 4. Literature sources verified in this preparation round

- [Sea-CLIP, WACV 2026](https://openaccess.thecvf.com/content/WACV2026/html/Guo_Sea-CLIP_Mining_Semantic-Aware_Representations_for_Few-Shot_Anomaly_Detection_with_CLIP_WACV_2026_paper.html) / [paper PDF](https://openaccess.thecvf.com/content/WACV2026/papers/Guo_Sea-CLIP_Mining_Semantic-Aware_Representations_for_Few-Shot_Anomaly_Detection_with_CLIP_WACV_2026_paper.pdf)
- [ReMP-AD, ICCV 2025](https://openaccess.thecvf.com/content/ICCV2025/html/Ma_ReMP-AD_Retrieval-enhanced_Multi-modal_Prompt_Fusion_for_Few-Shot_Industrial_Visual_Anomaly_ICCV_2025_paper.html)
- [FastRef, CVPR 2026](https://openaccess.thecvf.com/content/CVPR2026/html/Li_FastRef_Fast_Prototype_Refinement_for_Few-shot_Industrial_Anomaly_Detection_CVPR_2026_paper.html)
- [SubspaceAD, CVPR 2026](https://openaccess.thecvf.com/content/CVPR2026/html/Lendering_SubspaceAD_Training-Free_Few-Shot_Anomaly_Detection_via_Subspace_Modeling_CVPR_2026_paper.html)
- [PAPL, Pattern Recognition 2026](https://www.sciencedirect.com/science/article/pii/S0031320326004553)
- [Zero-Shot Industrial Anomaly Detection via CLIP-DINOv2 Multimodal Fusion and Stabilized Attention Pooling, Electronics 2025](https://www.mdpi.com/2079-9292/14/24/4785)

## 5. Related Work writing cautions

- 不要用 “multimodal” 描述 A1；可以用它描述 ReMP-AD、PAPL 等原方法。
- “AnomalyCLIP image tower” 要与 “AnomalyCLIP method” 区分；A1 没有执行其原始文本打分流程。
- 不要把 target-normal prompt tuning、source-domain training、zero-shot 和 training-free memory methods 放在一张主结果表里直接定胜负；先给协议表，再给结果。
- 2026 论文的卷期、页码、DOI 仍应在投稿前通过出版社/CVF 页面再次核验。
- 对非官方聚合网站只作检索线索，不作为最终引用元数据来源。

