"""Export AnomalyCLIP scores for deterministic normal-reference views."""

from __future__ import annotations

import argparse
import hashlib
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
from prompt_ensemble import AnomalyCLIP_PromptLearner
from utils import get_transform


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


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
    parser.add_argument("--views-json", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--image-size", type=int, default=518)
    parser.add_argument("--features-list", type=int, nargs="+", default=[6, 12, 18, 24])
    parser.add_argument("--feature-map-layer", type=int, default=0)
    parser.add_argument("--depth", type=int, default=9)
    parser.add_argument("--n-ctx", type=int, default=12)
    parser.add_argument("--text-n-ctx", type=int, default=4)
    parser.add_argument("--sigma", type=float, default=4.0)
    parser.add_argument("--seed", type=int, default=111)
    parser.add_argument(
        "--validate-only",
        action="store_true",
        help="Validate frozen inputs and checkpoint without loading the model.",
    )
    args = parser.parse_args()

    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise SystemExit("CUDA requested but unavailable")
    setup_seed(args.seed)
    views = json.loads(args.views_json.read_text(encoding="utf-8"))
    if views["status"] != "passed":
        raise SystemExit("reference-view manifest is not passed")
    if views["test_images_used"] or views["test_labels_used"]:
        raise SystemExit("reference-view manifest used forbidden test data")
    if not args.checkpoint.is_file():
        raise SystemExit(f"checkpoint not found: {args.checkpoint}")
    missing_views = [
        str(item["view_path"])
        for item in views["items"]
        if not Path(str(item["view_path"])).is_file()
    ]
    if missing_views:
        raise SystemExit(
            f"missing {len(missing_views)} reference views; first: {missing_views[0]}"
        )
    if args.validate_only:
        print(
            json.dumps(
                {
                    "status": "passed",
                    "mode": "validate_only",
                    "dataset": views["dataset"],
                    "views": len(views["items"]),
                    "checkpoint": str(args.checkpoint.resolve()),
                    "checkpoint_sha256": sha256(args.checkpoint),
                    "test_predictions_used": False,
                    "test_labels_used": False,
                }
            )
        )
        return 0

    design = {
        "Prompt_length": args.n_ctx,
        "learnabel_text_embedding_depth": args.depth,
        "learnabel_text_embedding_length": args.text_n_ctx,
    }
    model, _ = AnomalyCLIP_lib.load(
        "ViT-L/14@336px", device=args.device, design_details=design
    )
    model.eval()
    preprocess, _ = get_transform(SimpleNamespace(image_size=args.image_size))
    prompt_learner = AnomalyCLIP_PromptLearner(model.to("cpu"), design)
    checkpoint = torch.load(args.checkpoint, map_location="cpu")
    prompt_learner.load_state_dict(checkpoint["prompt_learner"])
    prompt_learner.to(args.device)
    model.to(args.device)
    model.visual.DAPM_replace(DPAM_layer=20)
    prompts, tokens, compound = prompt_learner(cls_id=None)
    with torch.inference_mode():
        text_features = model.encode_text_learn(prompts, tokens, compound).float()
        text_features = torch.stack(
            torch.chunk(text_features, dim=0, chunks=2), dim=1
        )
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)

    grouped: dict[str, list[dict[str, object]]] = {}
    for item in views["items"]:
        grouped.setdefault(str(item["category"]), []).append(item)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    category_rows: list[dict[str, object]] = []
    for category, items in sorted(grouped.items()):
        image_scores: list[float] = []
        pixel_maps: list[np.ndarray] = []
        for item in items:
            with Image.open(str(item["view_path"])) as opened:
                tensor = preprocess(opened.convert("RGB"))
            image = tensor.reshape(1, 3, args.image_size, args.image_size).to(
                args.device
            )
            with torch.inference_mode():
                image_features, patch_features = model.encode_image(
                    image, args.features_list, DPAM_layer=20
                )
                image_features = image_features / image_features.norm(
                    dim=-1, keepdim=True
                )
                probabilities = (
                    (image_features @ text_features.permute(0, 2, 1)) / 0.07
                ).softmax(-1)
                image_scores.append(float(probabilities[:, 0, 1].cpu().item()))
                map_layers = []
                for index, patch_feature in enumerate(patch_features):
                    if index < args.feature_map_layer:
                        continue
                    patch_feature = patch_feature / patch_feature.norm(
                        dim=-1, keepdim=True
                    )
                    similarity, _ = AnomalyCLIP_lib.compute_similarity(
                        patch_feature, text_features[0]
                    )
                    similarity_map = AnomalyCLIP_lib.get_similarity_map(
                        similarity[:, 1:, :], args.image_size
                    )
                    map_layers.append(
                        (similarity_map[..., 1] + 1 - similarity_map[..., 0]) / 2
                    )
                anomaly_map = torch.stack(map_layers).sum(dim=0)[0].cpu().numpy()
                pixel_maps.append(
                    gaussian_filter(anomaly_map, sigma=args.sigma).astype(np.float32)
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
            branch=np.asarray("anomalyclip_text"),
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
        "branch": "anomalyclip_text",
        "checkpoint": str(args.checkpoint.resolve()),
        "checkpoint_sha256": sha256(args.checkpoint),
        "views_json": str(args.views_json.resolve()),
        "views_json_sha256": sha256(args.views_json),
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
