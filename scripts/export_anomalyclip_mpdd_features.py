"""Export raw CLIP patch features for Direction A (feature-level fusion).

Persists the per-patch tokens (projected via ln_post @ proj) produced by
AnomalyCLIP's encode_image, so the fusion stage can align them with DINO tokens.
We keep the last requested intermediate layer (deepest) and drop the [CLS] token.

For MPDD images are 1024x1024 and the preprocessing uses image_size=518, so the
ViT-L/14@336px backbone yields a fixed 37x37 grid of embed_dim tokens.
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

ROOT = Path(__file__).resolve().parents[1]
METHOD_ROOT = ROOT / "methods" / "AnomalyCLIP-main"
sys.path.insert(0, str(METHOD_ROOT))

import AnomalyCLIP_lib
from prompt_ensemble import AnomalyCLIP_PromptLearner
from utils import get_transform
from v2_mpdd_prediction_common import index_dataset, sha256, validate_dataset_gate_inputs


def setup_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--dataset", choices=("mpdd", "btad"), default="mpdd")
    parser.add_argument("--dataset-role", choices=("development", "holdout"), default="development")
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--shot", type=int, choices=(1, 2, 4), required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--image-size", type=int, default=518)
    parser.add_argument("--features-list", type=int, nargs="+", default=[6, 12, 18, 24])
    parser.add_argument("--depth", type=int, default=9)
    parser.add_argument("--n-ctx", type=int, default=12)
    parser.add_argument("--text-n-ctx", type=int, default=4)
    parser.add_argument("--model-seed", type=int, default=111)
    parser.add_argument("--validate-only", action="store_true")
    args = parser.parse_args()

    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    counts = validate_dataset_gate_inputs(args.dataset, args.data_root, manifest, args.seed, args.shot)
    if not args.checkpoint.is_file():
        raise SystemExit(f"checkpoint missing: {args.checkpoint}")
    if args.validate_only:
        print(json.dumps({"status": "passed", "mode": "validate_only",
                          "checkpoint_sha256": sha256(args.checkpoint), **counts}))
        return 0
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise SystemExit("CUDA requested but unavailable")

    setup_seed(args.model_seed)
    design = {
        "Prompt_length": args.n_ctx,
        "learnabel_text_embedding_depth": args.depth,
        "learnabel_text_embedding_length": args.text_n_ctx,
    }
    model, _ = AnomalyCLIP_lib.load("ViT-L/14@336px", device=args.device, design_details=design)
    model.eval()
    preprocess, _ = get_transform(SimpleNamespace(image_size=args.image_size))
    prompt_learner = AnomalyCLIP_PromptLearner(model.to("cpu"), design)
    checkpoint = torch.load(args.checkpoint, map_location="cpu")
    prompt_learner.load_state_dict(checkpoint["prompt_learner"])
    prompt_learner.to(args.device)
    model.to(args.device)
    model.visual.DAPM_replace(DPAM_layer=20)

    indexed = index_dataset(args.dataset, args.data_root)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    rows = []

    def extract_patch_features(tensor: torch.Tensor) -> tuple[np.ndarray, tuple[int, int]]:
        image = tensor.reshape(1, 3, args.image_size, args.image_size).to(args.device)
        with torch.inference_mode():
            _, patch_features = model.encode_image(image, args.features_list, DPAM_layer=20)
        # Take the deepest requested layer, drop [CLS], keep HxW patch tokens.
        patch_token = patch_features[-1][0, 1:, :].float().cpu().numpy()
        seq_len = patch_token.shape[0]
        side = int(round(seq_len ** 0.5))
        if side * side != seq_len:
            raise RuntimeError(f"non-square patch sequence: {seq_len}")
        return patch_token.reshape(side, side, -1), (side, side)

    with torch.inference_mode():
        for category, samples in sorted(indexed.items()):
            references = manifest["categories"][category][str(args.seed)][str(args.shot)]

            ref_blocks = []
            ref_grid = None
            for relative in references:
                path = args.data_root / relative
                with Image.open(path) as opened:
                    tensor = preprocess(opened.convert("RGB"))
                patches, grid = extract_patch_features(tensor)
                ref_blocks.append(patches)
                ref_grid = grid

            feat_blocks = []
            masks, labels, sample_ids = [], [], []
            test_grid = None
            for sample in samples:
                with Image.open(sample.image_path) as opened:
                    tensor = preprocess(opened.convert("RGB"))
                patches, grid = extract_patch_features(tensor)
                feat_blocks.append(patches)
                if test_grid is None:
                    test_grid = grid
                elif test_grid != grid:
                    raise RuntimeError(f"inconsistent test grid in {category}: {test_grid} vs {grid}")

                if sample.mask_path is None:
                    mask = np.zeros((args.image_size, args.image_size), dtype=np.uint8)
                else:
                    with Image.open(sample.mask_path) as opened:
                        mask = np.asarray(opened.convert("L").resize(
                            (args.image_size, args.image_size), Image.Resampling.NEAREST))
                    mask = (mask > 0).astype(np.uint8)
                masks.append(mask)
                labels.append(sample.label)
                sample_ids.append(sample.sample_id)

            if ref_grid != test_grid:
                raise RuntimeError(f"ref/test grid mismatch in {category}: {ref_grid} vs {test_grid}")

            output = args.output_dir / f"{category}.npz"
            np.savez_compressed(
                output,
                patch_features=np.asarray(feat_blocks, dtype=np.float32),
                ref_patch_features=np.asarray(ref_blocks, dtype=np.float32),
                gt_sp=np.asarray(labels, dtype=np.int64),
                imgs_masks=np.asarray(masks, dtype=np.uint8),
                sample_ids=np.asarray(sample_ids),
                grid_size=np.asarray(test_grid, dtype=np.int64),
                dataset=np.asarray(args.dataset),
                dataset_role=np.asarray(args.dataset_role),
                branch=np.asarray("anomalyclip_text"),
                seed=np.asarray(args.seed),
                shot=np.asarray(args.shot),
                score_direction=np.asarray("higher_is_more_anomalous"),
            )
            rows.append({"category": category, "samples": len(samples),
                         "references": len(references), "grid": list(test_grid),
                         "output": str(output.resolve()), "sha256": sha256(output)})
            print(f"wrote {output}", flush=True)

    report = {
        "schema_version": 1,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": "passed",
        "kind": "raw_patch_features",
        "dataset": args.dataset,
        "dataset_role": args.dataset_role,
        "branch": "anomalyclip_text",
        "checkpoint": str(args.checkpoint.resolve()),
        "checkpoint_sha256": sha256(args.checkpoint),
        "seed": args.seed,
        "shot": args.shot,
        "manifest": str(args.manifest.resolve()),
        "manifest_sha256": sha256(args.manifest),
        "test_predictions_used_for_parameter_fit": False,
        "test_labels_used_for_parameter_fit": False,
        "test_set_statistics_used_for_calibration": False,
        "categories": rows,
    }
    (args.output_dir / "export_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
