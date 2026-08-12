"""Direction B: Defect-word-enhanced prompt ensemble for AnomalyCLIP text branch.

Strategy:
  1. Use standard CLIP model (design_details=None) -- text encode_text works,
     visual encoder returns full token output (CLS + patches).
  2. Replace the learnable prompt with multiple hand-crafted defect word
     variants, ensemble at feature level.
  3. Compute single-layer anomaly maps from CLIP patch tokens.
  4. Export to NPZ for downstream V3.3 fusion evaluation.

Requires GPU.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
from PIL import Image
from scipy.ndimage import gaussian_filter

ROOT = Path(__file__).resolve().parents[1]
METHOD_ROOT = ROOT / "methods" / "AnomalyCLIP-main"
sys.path.insert(0, str(METHOD_ROOT))

import AnomalyCLIP_lib
from utils import get_transform
from v2_mpdd_prediction_common import index_dataset, sha256, validate_dataset_gate_inputs
from defect_ensemble_utils import (
    build_ensemble_text_features,
    CATEGORY_OBJECT_NAMES,
    DEFECT_VARIANTS,
    DEFECT_VARIANTS_FAST,
)


def setup_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def main() -> int:
    parser = argparse.ArgumentParser(description="Direction B: Defect-word prompt ensemble")
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--shot", type=int, choices=(1, 2, 4), required=True)
    parser.add_argument("--dataset", choices=("mpdd", "btad"), default="mpdd")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--image-size", type=int, default=518)
    parser.add_argument("--sigma", type=float, default=4.0)
    parser.add_argument("--model-seed", type=int, default=111)
    parser.add_argument("--fast", action="store_true",
                        help="Use reduced defect variant set (6 instead of 18)")
    args = parser.parse_args()

    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    validate_dataset_gate_inputs(args.dataset, args.data_root, manifest, args.seed, args.shot)

    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise SystemExit("CUDA requested but unavailable")

    setup_seed(args.model_seed)

    # Write progress to file (sandbox truncates stdout)
    progress_path = args.output_dir / "_progress.txt"
    def log(msg: str) -> None:
        print(msg, flush=True)
        with progress_path.open("a", encoding="utf-8") as f:
            f.write(msg + "\n")

    log("Step 0: Starting...")

    # ---- Phase 0: Load standard CLIP model ----
    # design_details=None gives us standard VisionTransformer with normal
    # encode_image / encode_text. No DAPM_replace needed.
    log("Step 1: Loading model...")
    model, _ = AnomalyCLIP_lib.load("ViT-L/14@336px", device=args.device, design_details=None)
    model.eval()
    log("  Model loaded OK")

    log("  Loading transform...")
    preprocess, _ = get_transform(SimpleNamespace(image_size=args.image_size))
    log("  Indexing dataset...")
    indexed = index_dataset(args.dataset, args.data_root)
    log(f"  Indexed {len(indexed)} categories")
    args.output_dir.mkdir(parents=True, exist_ok=True)

    # ---- Phase 1: Pre-compute text features ----
    n_variants = 6 if args.fast else 18
    log(f"Phase 1: Pre-computing text features ({n_variants} variants)...")
    category_text_features = {}
    for category in sorted(indexed.keys()):
        object_name = CATEGORY_OBJECT_NAMES.get(category, category)
        log(f"  [{category}] object='{object_name}' building...")
        text_features = build_ensemble_text_features(model, object_name, args.device, fast=args.fast)
        category_text_features[category] = text_features
        log(f"  [{category}] -> shape={text_features.shape}")

    # ---- Phase 2: Run image inference ----
    rows = []
    n_patches_per_side = args.image_size // 14  # ViT-L/14 patch size
    log(f"Phase 2: Image inference ({n_patches_per_side}x{n_patches_per_side} patches)...")

    for category, samples in sorted(indexed.items()):
        text_features = category_text_features[category]  # [1, 2, embed_dim]
        log(f"  [{category}] {len(samples)} samples...")

        scores, maps, masks, labels, sample_ids = [], [], [], [], []

        for sample in samples:
            with Image.open(sample.image_path) as opened:
                tensor = preprocess(opened.convert("RGB"))
            image = tensor.reshape(1, 3, args.image_size, args.image_size).to(args.device)

            with torch.inference_mode():
                # Standard CLIP encode_image returns [1, N_tokens, embed_dim]
                # N_tokens = 1 (CLS) + n_patches
                visual_output = model.encode_image(image)  # [1, N_tokens, 1024]

                # Split CLS and patch tokens
                cls_feat = visual_output[:, 0, :]       # [1, 1024]
                patch_feat = visual_output[:, 1:, :]    # [1, n_patches, 1024]

                # Normalize
                cls_feat = cls_feat / cls_feat.norm(dim=-1, keepdim=True)
                patch_feat = patch_feat / patch_feat.norm(dim=-1, keepdim=True)

                # Image-level anomaly score
                logits = (cls_feat @ text_features.permute(0, 2, 1).squeeze(0)) / 0.07  # [1, 2]
                probabilities = logits.softmax(dim=-1)  # [1, 2]

                # Pixel-level anomaly map via patch-text similarity
                # similarity = patch @ text.T -> [1, n_patches, 2]
                similarity = patch_feat @ text_features[0].T  # [1, n_patches, 2]
                similarity = (similarity / 0.07).softmax(dim=-1)  # [1, n_patches, 2]

                # Reshape to spatial map
                sim_map = similarity.reshape(1, n_patches_per_side, n_patches_per_side, 2)
                sim_map = torch.nn.functional.interpolate(
                    sim_map.permute(0, 3, 1, 2),  # [1, 2, H, W]
                    (args.image_size, args.image_size),
                    mode='bilinear'
                ).permute(0, 2, 3, 1)  # [1, H, W, 2]

                anomaly_map = (sim_map[..., 1] + 1 - sim_map[..., 0]) / 2
                anomaly_map = anomaly_map[0].cpu().numpy()

            if sample.mask_path is None:
                mask = np.zeros((args.image_size, args.image_size), dtype=np.uint8)
            else:
                with Image.open(sample.mask_path) as opened:
                    mask = np.asarray(
                        opened.convert("L").resize(
                            (args.image_size, args.image_size), Image.Resampling.NEAREST
                        )
                    )
                mask = (mask > 0).astype(np.uint8)

            scores.append(float(probabilities[0, 1].cpu().item()))
            maps.append(gaussian_filter(anomaly_map, sigma=args.sigma).astype(np.float32))
            masks.append(mask)
            labels.append(sample.label)
            sample_ids.append(sample.sample_id)

        output = args.output_dir / f"{category}.npz"
        np.savez_compressed(
            output,
            gt_sp=np.asarray(labels, dtype=np.int64),
            pr_sp=np.asarray(scores, dtype=np.float32),
            imgs_masks=np.asarray(masks, dtype=np.uint8),
            anomaly_maps=np.asarray(maps, dtype=np.float32),
            sample_ids=np.asarray(sample_ids),
            dataset=np.asarray(args.dataset),
            dataset_role=np.asarray("development"),
            branch=np.asarray("anomalyclip_text_defect_ensemble"),
            seed=np.asarray(args.seed),
            shot=np.asarray(args.shot),
            score_direction=np.asarray("higher_is_more_anomalous"),
            defect_variants=np.asarray(
                DEFECT_VARIANTS_FAST if args.fast else DEFECT_VARIANTS
            ),
        )
        rows.append({
            "category": category,
            "samples": len(samples),
            "output": str(output.resolve()),
            "sha256": sha256(output),
        })
        log(f"  -> saved {output}")

    # ---- Phase 3: Export report ----
    log("Phase 3: Writing report...")
    report = {
        "schema_version": 1,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": "passed",
        "direction": "B_defect_word_prompt_ensemble",
        "dataset": args.dataset,
        "seed": args.seed,
        "shot": args.shot,
        "n_defect_variants": 6 if args.fast else 18,
        "model_type": "standard_clip_no_dapm",
        "categories": rows,
    }
    (args.output_dir / "export_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    log("Done!")
    return 0


if __name__ == "__main__":
    import traceback
    try:
        raise SystemExit(main())
    except Exception:
        err = traceback.format_exc()
        print(err, flush=True)
        # Also write to progress file if possible
        try:
            from pathlib import Path
            for p in Path("outputs/dynamic_fusion/v3_5_defect_ensemble").rglob("_progress.txt"):
                with p.open("a", encoding="utf-8") as f:
                    f.write(f"\nERROR:\n{err}\n")
        except Exception:
            pass
        raise SystemExit(1)
