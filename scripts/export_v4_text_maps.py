"""G3 explicit-text-branch export: AnomalyCLIP text-conditioned anomaly maps.

Unlike A1 (which only used AnomalyCLIP.encode_image patch features), this script
produces the *text-conditioned* anomaly map where the learned prompt context
(normal "object" vs abnormal "damaged object") actually produces text embeddings
and each image patch is scored by its image-text similarity. This is the T0
text branch required by the V4 plan (section 5.2).

Pipeline (mirrors methods/AnomalyCLIP-main/test.py, minus the official dataset
loader, plus the project manifest/seed/shot protocol):

  prompt learner -> normal/abnormal text embeddings (encode_text_learn)
  image -> CLIP image patch features (encode_image)
  per patch: softmax image-text similarity -> (P_abnormal + 1 - P_normal)/2
  -> sum over requested layers -> gaussian_filter(sigma)

The K normal references from the manifest are validated for protocol integrity
but are NOT used to score this zero-shot text branch (no memory bank, no
calibration on references).

Physical-isolation contract (V4 section 4.3): prediction NPZ contains NO ground
truth; gt_sp/imgs_masks are written to a separate `<cat>_targets.npz` for the
evaluator only.

Swap test: `--swap-prompts` exchanges the normal/abnormal text embeddings before
scoring; on a working text branch the output map must flip direction
(map -> 1 - map approximately).
"""

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

import AnomalyCLIP_lib  # noqa: E402
from prompt_ensemble import AnomalyCLIP_PromptLearner  # noqa: E402
from utils import get_transform  # noqa: E402
from v2_mpdd_prediction_common import index_dataset, sha256, validate_dataset_gate_inputs  # noqa: E402


def setup_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def tensor_sha256(tensor: torch.Tensor) -> str:
    return hashlib.sha256(tensor.detach().float().cpu().numpy().tobytes()).hexdigest()


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
    parser.add_argument("--feature-map-layer", type=int, nargs="+", default=[0])
    parser.add_argument("--depth", type=int, default=9)
    parser.add_argument("--n-ctx", type=int, default=12)
    parser.add_argument("--text-n-ctx", type=int, default=4)
    parser.add_argument("--model-seed", type=int, default=111)
    parser.add_argument("--sigma", type=float, default=4)
    parser.add_argument("--swap-prompts", action="store_true",
                        help="swap normal/abnormal text embeddings (direction test)")
    parser.add_argument("--categories", type=str, nargs="*", default=None,
                        help="optional subset of categories to export")
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

    # Explicit text branch: prompt learner -> normal/abnormal text embeddings.
    prompts, tokenized_prompts, compound_prompts_text = prompt_learner(cls_id=None)
    with torch.inference_mode():
        text_features = model.encode_text_learn(prompts, tokenized_prompts, compound_prompts_text).float()
    text_features = torch.stack(torch.chunk(text_features, dim=0, chunks=2), dim=1)
    text_features = text_features / text_features.norm(dim=-1, keepdim=True)
    if args.swap_prompts:
        text_features = text_features[:, [1, 0], :]
    text_features = text_features.to(args.device)

    indexed = index_dataset(args.dataset, args.data_root)
    if args.categories is not None:
        unknown = sorted(set(args.categories).difference(indexed))
        if unknown:
            raise SystemExit(f"unknown categories: {unknown}")
        indexed = {c: indexed[c] for c in args.categories}

    args.output_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    text_embedding_hash = tensor_sha256(text_features)

    with torch.inference_mode():
        for category, samples in sorted(indexed.items()):
            references = manifest["categories"][category][str(args.seed)][str(args.shot)]
            _ = references  # zero-shot text branch does not build a memory bank

            pr_sp, masks, sample_ids = [], [], []
            map_blocks = []
            for sample in samples:
                with Image.open(sample.image_path) as opened:
                    tensor = preprocess(opened.convert("RGB"))
                image = tensor.reshape(1, 3, args.image_size, args.image_size).to(args.device)
                image_features, patch_features = model.encode_image(image, args.features_list, DPAM_layer=20)
                image_features = image_features / image_features.norm(dim=-1, keepdim=True)
                text_probs = image_features @ text_features.permute(0, 2, 1)
                text_probs = (text_probs / 0.07).softmax(-1)
                abnormal_prob = float(text_probs[:, 0, 1].item())

                anomaly_map_list = []
                for idx, patch_feature in enumerate(patch_features):
                    if idx >= args.feature_map_layer[0]:
                        patch_feature = patch_feature / patch_feature.norm(dim=-1, keepdim=True)
                        similarity, _ = AnomalyCLIP_lib.compute_similarity(patch_feature, text_features[0])
                        similarity_map = AnomalyCLIP_lib.get_similarity_map(similarity[:, 1:, :], args.image_size)
                        anomaly_map = (similarity_map[..., 1] + 1 - similarity_map[..., 0]) / 2.0
                        anomaly_map_list.append(anomaly_map)
                anomaly_map = torch.stack(anomaly_map_list).sum(dim=0)
                anomaly_map = torch.stack(
                    [torch.from_numpy(gaussian_filter(i, sigma=args.sigma)) for i in anomaly_map.detach().cpu()],
                    dim=0,
                )

                if sample.mask_path is None:
                    mask = np.zeros((args.image_size, args.image_size), dtype=np.uint8)
                else:
                    with Image.open(sample.mask_path) as opened:
                        mask = np.asarray(opened.convert("L").resize(
                            (args.image_size, args.image_size), Image.Resampling.NEAREST))
                    mask = (mask > 0).astype(np.uint8)

                pr_sp.append(abnormal_prob)
                map_blocks.append(anomaly_map.squeeze(0).numpy().astype(np.float32))
                masks.append(mask)
                sample_ids.append(sample.sample_id)

            pred = np.asarray(map_blocks, dtype=np.float32)
            pred_sha = hashlib.sha256(pred.tobytes()).hexdigest()

            pred_out = args.output_dir / f"{category}.npz"
            np.savez_compressed(
                pred_out,
                pr_sp=np.asarray(pr_sp, dtype=np.float32),
                anomaly_maps=pred,
                sample_ids=np.asarray(sample_ids),
                dataset=np.asarray(args.dataset),
                dataset_role=np.asarray(args.dataset_role),
                branch=np.asarray("anomalyclip_text"),
                score_direction=np.asarray("higher_is_more_anomalous"),
            )
            target_out = args.output_dir / f"{category}_targets.npz"
            np.savez_compressed(
                target_out,
                gt_sp=np.asarray([s.label for s in samples], dtype=np.int64),
                imgs_masks=np.asarray(masks, dtype=np.uint8),
                sample_ids=np.asarray(sample_ids),
            )
            rows.append({"category": category, "samples": len(samples),
                         "anomaly_map_sha256": pred_sha,
                         "prediction": str(pred_out.resolve()),
                         "evaluation_target": str(target_out.resolve())})
            print(f"wrote {pred_out} (n={len(samples)}, text_map sha={pred_sha[:12]})", flush=True)

    report = {
        "schema_version": 1,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": "passed",
        "branch": "anomalyclip_text",
        "kind": "explicit_text_conditioned_maps",
        "dataset": args.dataset,
        "dataset_role": args.dataset_role,
        "seed": args.seed,
        "shot": args.shot,
        "checkpoint": str(args.checkpoint.resolve()),
        "checkpoint_sha256": sha256(args.checkpoint),
        "text_embedding_sha256": text_embedding_hash,
        "prompts_swapped": bool(args.swap_prompts),
        "manifest": str(args.manifest.resolve()),
        "manifest_sha256": sha256(args.manifest),
        "image_size": args.image_size,
        "sigma": args.sigma,
        "feature_map_layer": args.feature_map_layer,
        "test_predictions_used_for_parameter_fit": False,
        "test_labels_used_for_parameter_fit": False,
        "test_masks_used_for_parameter_fit": False,
        "test_dataset_statistics_used_for_calibration": False,
        "test_normal_selection_used": False,
        "categories": rows,
    }
    (args.output_dir / "export_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"text_embedding_sha256": text_embedding_hash,
                      "categories": len(rows)}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
