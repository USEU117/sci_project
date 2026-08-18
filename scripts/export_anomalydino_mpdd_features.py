"""Export raw DINO patch features for Direction A (feature-level fusion).

Unlike export_anomalydino_mpdd_predictions.py (which collapses patches into a
scalar anomaly map), this script persists the raw per-patch tokens so the
feature-level fusion stage can align, concatenate, project and score them.

For MPDD images are 1024x1024, so DINOv2 (patch 14, smaller_edge 448) yields a
fixed 32x32 grid of 384-dim tokens.
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import cv2
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
METHOD_ROOT = ROOT / "methods" / "anomalydino"
sys.path.insert(0, str(METHOD_ROOT))

from src.backbones import get_model
from v2_mpdd_prediction_common import index_dataset, sha256, validate_dataset_gate_inputs


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--dataset", choices=("mpdd", "btad"), default="mpdd")
    parser.add_argument("--dataset-role", choices=("development", "holdout"), default="development")
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--shot", type=int, choices=(1, 2, 4), required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--model-name", default="dinov2_vits14")
    parser.add_argument("--resolution", type=int, default=448)
    parser.add_argument("--map-size", type=int, default=448)
    parser.add_argument("--validate-only", action="store_true")
    args = parser.parse_args()

    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    counts = validate_dataset_gate_inputs(args.dataset, args.data_root, manifest, args.seed, args.shot)
    if args.validate_only:
        print(json.dumps({"status": "passed", "mode": "validate_only", **counts}))
        return 0
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise SystemExit("CUDA requested but unavailable")

    indexed = index_dataset(args.dataset, args.data_root)
    model = get_model(args.model_name, args.device, smaller_edge_size=args.resolution)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    rows = []

    def extract_patch_features(image_rgb: np.ndarray) -> tuple[np.ndarray, tuple[int, int]]:
        tensor, grid_size = model.prepare_image(image_rgb)
        tokens = model.extract_features(tensor).astype(np.float32)
        # DINOv2 get_intermediate_layers returns pure patch tokens (no [CLS]).
        expected = grid_size[0] * grid_size[1]
        if tokens.shape[0] != expected:
            raise RuntimeError(f"patch count {tokens.shape[0]} != grid {grid_size}")
        return tokens.reshape(grid_size[0], grid_size[1], -1), grid_size

    with torch.inference_mode():
        for category, samples in sorted(indexed.items()):
            references = manifest["categories"][category][str(args.seed)][str(args.shot)]

            ref_blocks = []
            ref_grid = None
            for relative in references:
                image = cv2.cvtColor(cv2.imread(str(args.data_root / relative)), cv2.COLOR_BGR2RGB)
                patches, grid = extract_patch_features(image)
                ref_blocks.append(patches)
                ref_grid = grid

            feat_blocks = []
            masks, labels, sample_ids = [], [], []
            test_grid = None
            for sample in samples:
                image = cv2.cvtColor(cv2.imread(str(sample.image_path)), cv2.COLOR_BGR2RGB)
                patches, grid = extract_patch_features(image)
                feat_blocks.append(patches)
                if test_grid is None:
                    test_grid = grid
                elif test_grid != grid:
                    raise RuntimeError(f"inconsistent test grid in {category}: {test_grid} vs {grid}")

                if sample.mask_path is None:
                    mask = np.zeros((args.map_size, args.map_size), dtype=np.uint8)
                else:
                    mask = cv2.imread(str(sample.mask_path), cv2.IMREAD_GRAYSCALE)
                    mask = cv2.resize(mask, (args.map_size, args.map_size), interpolation=cv2.INTER_NEAREST)
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
                branch=np.asarray("anomalydino_visual"),
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
        "branch": "anomalydino_visual",
        "model": args.model_name,
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
