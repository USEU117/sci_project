"""Export full MPDD development predictions from AnomalyDINO for V2 Gate A."""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import cv2
import faiss
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
METHOD_ROOT = ROOT / "methods" / "anomalydino"
sys.path.insert(0, str(METHOD_ROOT))

from src.backbones import get_model
from src.post_eval import mean_top1p
from src.utils import dists2map
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
    with torch.inference_mode():
        for category, samples in sorted(indexed.items()):
            references = manifest["categories"][category][str(args.seed)][str(args.shot)]
            blocks = []
            for relative in references:
                image = cv2.cvtColor(cv2.imread(str(args.data_root / relative)), cv2.COLOR_BGR2RGB)
                tensor, _ = model.prepare_image(image)
                blocks.append(model.extract_features(tensor).astype(np.float32))
            memory = np.concatenate(blocks, axis=0)
            faiss.normalize_L2(memory)
            index = faiss.IndexFlatL2(memory.shape[1])
            index.add(memory)
            scores, maps, masks, labels, sample_ids = [], [], [], [], []
            for sample in samples:
                image = cv2.cvtColor(cv2.imread(str(sample.image_path)), cv2.COLOR_BGR2RGB)
                tensor, grid_size = model.prepare_image(image)
                features = model.extract_features(tensor).astype(np.float32)
                faiss.normalize_L2(features)
                distances, _ = index.search(features, k=1)
                distances = distances[:, 0] / 2.0
                anomaly_map = dists2map(distances.reshape(grid_size), (args.map_size, args.map_size)).astype(np.float32)
                if sample.mask_path is None:
                    mask = np.zeros((args.map_size, args.map_size), dtype=np.uint8)
                else:
                    mask = cv2.imread(str(sample.mask_path), cv2.IMREAD_GRAYSCALE)
                    mask = cv2.resize(mask, (args.map_size, args.map_size), interpolation=cv2.INTER_NEAREST)
                    mask = (mask > 0).astype(np.uint8)
                scores.append(float(mean_top1p(distances)))
                maps.append(anomaly_map)
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
                dataset_role=np.asarray(args.dataset_role),
                branch=np.asarray("anomalydino_visual"),
                seed=np.asarray(args.seed),
                shot=np.asarray(args.shot),
                score_direction=np.asarray("higher_is_more_anomalous"),
            )
            rows.append({"category": category, "samples": len(samples), "output": str(output.resolve()), "sha256": sha256(output)})
            print(f"wrote {output}", flush=True)
    report = {
        "schema_version": 1,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": "passed",
        "dataset": args.dataset,
        "dataset_role": args.dataset_role,
        "branch": "anomalydino_visual",
        "seed": args.seed,
        "shot": args.shot,
        "manifest": str(args.manifest.resolve()),
        "manifest_sha256": sha256(args.manifest),
        "test_predictions_used_for_parameter_fit": False,
        "test_labels_used_for_parameter_fit": False,
        "test_set_statistics_used_for_calibration": False,
        "categories": rows,
    }
    (args.output_dir / "export_report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
