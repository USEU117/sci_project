"""Export AnomalyDINO scores for deterministic normal-reference views."""

from __future__ import annotations

import argparse
import hashlib
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
from src.utils import dists2map, get_dataset_info


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--views-json", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--dataset", default="VisA")
    parser.add_argument("--model-name", default="dinov2_vits14")
    parser.add_argument("--resolution", type=int, default=448)
    parser.add_argument("--map-max-edge", type=int, default=448)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument(
        "--validate-only",
        action="store_true",
        help="Validate frozen inputs and reference paths without loading the model.",
    )
    args = parser.parse_args()

    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise SystemExit("CUDA requested but unavailable")
    views = json.loads(args.views_json.read_text(encoding="utf-8"))
    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    if views["status"] != "passed":
        raise SystemExit("reference-view manifest is not passed")
    if views["test_images_used"] or views["test_labels_used"]:
        raise SystemExit("reference-view manifest used forbidden test data")

    _, _, masking_default, _ = get_dataset_info(
        args.dataset, "agnostic", data_path=str(args.data_root)
    )
    grouped: dict[str, list[dict[str, object]]] = {}
    for item in views["items"]:
        grouped.setdefault(str(item["category"]), []).append(item)
    missing: list[str] = []
    for category in sorted(grouped):
        if category not in manifest["categories"]:
            raise SystemExit(f"category missing from manifest: {category}")
        selected = manifest["categories"][category][str(views["seed"])][
            str(views["shot"])
        ]
        for relative_path in selected:
            reference_path = args.data_root / Path(relative_path)
            if not reference_path.is_file():
                missing.append(str(reference_path))
    if missing:
        raise SystemExit(f"missing {len(missing)} normal references; first: {missing[0]}")
    if args.validate_only:
        print(
            json.dumps(
                {
                    "status": "passed",
                    "mode": "validate_only",
                    "dataset": views["dataset"],
                    "categories": len(grouped),
                    "normal_references": sum(
                        len(manifest["categories"][category][str(views["seed"])][str(views["shot"])])
                        for category in grouped
                    ),
                    "test_predictions_used": False,
                    "test_labels_used": False,
                }
            )
        )
        return 0

    model = get_model(
        args.model_name, args.device, smaller_edge_size=args.resolution
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    category_rows: list[dict[str, object]] = []

    with torch.inference_mode():
        for category, items in sorted(grouped.items()):
            selected = manifest["categories"][category][str(views["seed"])][
                str(views["shot"])
            ]
            reference_features: list[np.ndarray] = []
            for relative_path in selected:
                reference_path = args.data_root / Path(relative_path)
                if not reference_path.is_file():
                    raise FileNotFoundError(reference_path)
                image = cv2.cvtColor(
                    cv2.imread(str(reference_path), cv2.IMREAD_COLOR),
                    cv2.COLOR_BGR2RGB,
                )
                tensor, grid_size = model.prepare_image(image)
                features = model.extract_features(tensor)
                mask = model.compute_background_mask(
                    features,
                    grid_size,
                    threshold=10,
                    masking_type=False,
                )
                reference_features.append(features[mask])
            memory = np.concatenate(reference_features, axis=0).astype(np.float32)
            faiss.normalize_L2(memory)
            index = faiss.IndexFlatL2(memory.shape[1])
            index.add(memory)

            image_scores: list[float] = []
            pixel_maps: list[np.ndarray] = []
            for item in items:
                image = cv2.cvtColor(
                    cv2.imread(str(item["view_path"]), cv2.IMREAD_COLOR),
                    cv2.COLOR_BGR2RGB,
                )
                tensor, grid_size = model.prepare_image(image)
                features = model.extract_features(tensor)
                if masking_default[category]:
                    mask = model.compute_background_mask(
                        features, grid_size, threshold=10, masking_type=True
                    )
                else:
                    mask = np.ones(features.shape[0], dtype=bool)
                selected_features = features[mask].astype(np.float32)
                faiss.normalize_L2(selected_features)
                distances, _ = index.search(selected_features, k=1)
                distances = distances / 2.0
                output_distances = np.zeros_like(mask, dtype=np.float32)
                output_distances[mask] = distances.squeeze()
                image_scores.append(float(mean_top1p(output_distances)))
                map_shape = image.shape[:2]
                if max(map_shape) > args.map_max_edge:
                    scale = args.map_max_edge / max(map_shape)
                    map_shape = (
                        max(1, round(map_shape[0] * scale)),
                        max(1, round(map_shape[1] * scale)),
                    )
                pixel_maps.append(
                    dists2map(output_distances.reshape(grid_size), map_shape).astype(
                        np.float32
                    )
                )
            output = args.output_dir / f"{category}.npz"
            np.savez_compressed(
                output,
                sample_ids=np.asarray([item["sample_id"] for item in items]),
                source_ids=np.asarray([item["source_id"] for item in items]),
                augmentation_ids=np.asarray(
                    [item["augmentation_id"] for item in items]
                ),
                image_scores=np.asarray(image_scores, dtype=np.float32),
                pixel_maps=np.asarray(pixel_maps, dtype=np.float32),
                dataset=np.asarray(views["dataset"]),
                branch=np.asarray("anomalydino_visual"),
                category=np.asarray(category),
                seed=np.asarray(views["seed"]),
                shot=np.asarray(views["shot"]),
                score_direction=np.asarray("higher_is_more_anomalous"),
            )
            category_rows.append(
                {
                    "category": category,
                    "views": len(items),
                    "output": str(output.resolve()),
                    "sha256": sha256(output),
                }
            )
            print(f"wrote {output}", flush=True)

    report = {
        "schema_version": 1,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": "passed",
        "dataset": views["dataset"],
        "seed": views["seed"],
        "shot": views["shot"],
        "branch": "anomalydino_visual",
        "model": args.model_name,
        "views_json": str(args.views_json.resolve()),
        "views_json_sha256": sha256(args.views_json),
        "manifest": str(args.manifest.resolve()),
        "manifest_sha256": sha256(args.manifest),
        "device": args.device,
        "test_predictions_used": False,
        "test_labels_used": False,
        "categories": category_rows,
    }
    (args.output_dir / "export_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
