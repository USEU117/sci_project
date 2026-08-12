"""Export full MPDD development predictions from AnomalyCLIP for V2 Gate A."""

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
    parser.add_argument("--feature-map-layer", type=int, default=0)
    parser.add_argument("--depth", type=int, default=9)
    parser.add_argument("--n-ctx", type=int, default=12)
    parser.add_argument("--text-n-ctx", type=int, default=4)
    parser.add_argument("--sigma", type=float, default=4.0)
    parser.add_argument("--model-seed", type=int, default=111)
    parser.add_argument("--validate-only", action="store_true")
    args = parser.parse_args()

    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    counts = validate_dataset_gate_inputs(args.dataset, args.data_root, manifest, args.seed, args.shot)
    if not args.checkpoint.is_file():
        raise SystemExit(f"checkpoint missing: {args.checkpoint}")
    if args.validate_only:
        print(
            json.dumps(
                {
                    "status": "passed",
                    "mode": "validate_only",
                    "checkpoint_sha256": sha256(args.checkpoint),
                    **counts,
                }
            )
        )
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
    prompts, tokens, compound = prompt_learner(cls_id=None)
    with torch.inference_mode():
        text_features = model.encode_text_learn(prompts, tokens, compound).float()
        text_features = torch.stack(torch.chunk(text_features, dim=0, chunks=2), dim=1)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)

    indexed = index_dataset(args.dataset, args.data_root)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for category, samples in sorted(indexed.items()):
        scores, maps, masks, labels, sample_ids = [], [], [], [], []
        for sample in samples:
            with Image.open(sample.image_path) as opened:
                tensor = preprocess(opened.convert("RGB"))
            image = tensor.reshape(1, 3, args.image_size, args.image_size).to(args.device)
            with torch.inference_mode():
                image_features, patch_features = model.encode_image(image, args.features_list, DPAM_layer=20)
                image_features = image_features / image_features.norm(dim=-1, keepdim=True)
                probabilities = ((image_features @ text_features.permute(0, 2, 1)) / 0.07).softmax(-1)
                map_layers = []
                for index, patch_feature in enumerate(patch_features):
                    if index < args.feature_map_layer:
                        continue
                    patch_feature = patch_feature / patch_feature.norm(dim=-1, keepdim=True)
                    similarity, _ = AnomalyCLIP_lib.compute_similarity(patch_feature, text_features[0])
                    similarity_map = AnomalyCLIP_lib.get_similarity_map(similarity[:, 1:, :], args.image_size)
                    map_layers.append((similarity_map[..., 1] + 1 - similarity_map[..., 0]) / 2)
                anomaly_map = torch.stack(map_layers).sum(dim=0)[0].cpu().numpy()
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
            scores.append(float(probabilities[:, 0, 1].cpu().item()))
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
            dataset_role=np.asarray(args.dataset_role),
            branch=np.asarray("anomalyclip_text"),
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
        "branch": "anomalyclip_text",
        "seed": args.seed,
        "shot": args.shot,
        "checkpoint": str(args.checkpoint.resolve()),
        "checkpoint_sha256": sha256(args.checkpoint),
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
