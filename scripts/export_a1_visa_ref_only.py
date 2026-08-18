"""A1 reference-features-only export for VisA (post-freeze validation).

Rationale (validated on MPDD): A1 test patch features depend only on the fixed
test images, NOT on (seed, shot). The s0/k1 full caches already contain the
complete test patch features; for every other (seed, shot) combo we only need
NEW reference patch features (the memory-bank source images change).

New standalone script: does NOT touch the frozen A1 scripts in freeze_manifest.json.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from datetime import datetime, timezone
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "src"))

from v2_mpdd_prediction_common import index_dataset, sha256, validate_dataset_gate_inputs


def setup_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    import torch

    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def load_base_cache(path: Path) -> dict:
    with np.load(path, allow_pickle=False) as data:
        return {
            "patch_features": np.asarray(data["patch_features"], dtype=np.float32),
            "gt_sp": np.asarray(data["gt_sp"], dtype=np.int64),
            "imgs_masks": np.asarray(data["imgs_masks"], dtype=np.uint8),
            "sample_ids": np.asarray(data["sample_ids"]),
            "grid_size": np.asarray(data["grid_size"], dtype=np.int64),
            "dataset": np.asarray(data["dataset"]),
            "dataset_role": np.asarray(data["dataset_role"]),
            "branch": np.asarray(data["branch"]),
            "score_direction": np.asarray(data["score_direction"]),
        }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--dataset", choices=("mpdd", "btad", "visa", "mvtec"), required=True)
    parser.add_argument("--dataset-role", choices=("development", "holdout"), default="holdout")
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--base-cache", type=Path, required=True,
                        help="existing s0/k1 full cache dir (same branch) whose test features are reused")
    parser.add_argument("--branch", choices=("dino", "clip"), required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--shot", type=int, choices=(1, 2, 4), required=True)
    parser.add_argument("--device", default="cuda:0")
    # dino
    parser.add_argument("--model-name", default="dinov2_vitb14")
    parser.add_argument("--resolution", type=int, default=448)
    # clip
    parser.add_argument("--checkpoint", type=Path, default=None)
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
    if not args.base_cache.is_dir():
        raise SystemExit(f"base cache missing: {args.base_cache}")
    if args.validate_only:
        print(json.dumps({"status": "passed", "mode": "validate_only", **counts}))
        return 0

    import torch

    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise SystemExit("CUDA requested but unavailable")

    indexed = index_dataset(args.dataset, args.data_root)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    setup_seed(args.model_seed)

    if args.branch == "dino":
        sys.path.insert(0, str(ROOT / "methods" / "anomalydino"))
        from src.backbones import get_model

        model = get_model(args.model_name, args.device, smaller_edge_size=args.resolution)

        def extract_ref(relative: str) -> tuple[np.ndarray, tuple[int, int]]:
            image = cv2.cvtColor(cv2.imread(str(args.data_root / relative)), cv2.COLOR_BGR2RGB)
            tensor, grid = model.prepare_image(image)
            tokens = model.extract_features(tensor).astype(np.float32)
            expected = grid[0] * grid[1]
            if tokens.shape[0] != expected:
                raise RuntimeError(f"patch count {tokens.shape[0]} != grid {grid}")
            return tokens.reshape(grid[0], grid[1], -1), grid

        model_label = args.model_name
    else:  # clip
        sys.path.insert(0, str(ROOT / "methods" / "AnomalyCLIP-main"))
        if args.checkpoint is None or not args.checkpoint.is_file():
            raise SystemExit(f"checkpoint missing: {args.checkpoint}")
        from types import SimpleNamespace

        from PIL import Image

        import AnomalyCLIP_lib
        from prompt_ensemble import AnomalyCLIP_PromptLearner
        from utils import get_transform

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

        def extract_ref(relative: str) -> tuple[np.ndarray, tuple[int, int]]:
            with Image.open(str(args.data_root / relative)) as opened:
                tensor = preprocess(opened.convert("RGB"))
            image = tensor.reshape(1, 3, args.image_size, args.image_size).to(args.device)
            with torch.inference_mode():
                _, patch_features = model.encode_image(image, args.features_list, DPAM_layer=20)
            patch_token = patch_features[-1][0, 1:, :].float().cpu().numpy()
            seq_len = patch_token.shape[0]
            side = int(round(seq_len**0.5))
            if side * side != seq_len:
                raise RuntimeError(f"non-square patch sequence: {seq_len}")
            return patch_token.reshape(side, side, -1), (side, side)

        model_label = "AnomalyCLIP_ViT-L/14@336px"

    rows = []
    with torch.inference_mode():
        for category, samples in sorted(indexed.items()):
            references = manifest["categories"][category][str(args.seed)][str(args.shot)]
            ref_blocks = []
            ref_grid = None
            for relative in references:
                patches, grid = extract_ref(relative)
                ref_blocks.append(patches)
                ref_grid = grid

            base_path = args.base_cache / f"{category}.npz"
            if not base_path.is_file():
                raise SystemExit(f"base cache missing category: {base_path}")
            base = load_base_cache(base_path)
            test_grid = tuple(int(v) for v in base["grid_size"])
            if ref_grid != test_grid:
                raise RuntimeError(f"ref/test grid mismatch in {category}: {ref_grid} vs {test_grid}")

            output = args.output_dir / f"{category}.npz"
            np.savez_compressed(
                output,
                patch_features=base["patch_features"],
                ref_patch_features=np.asarray(ref_blocks, dtype=np.float32),
                gt_sp=base["gt_sp"],
                imgs_masks=base["imgs_masks"],
                sample_ids=base["sample_ids"],
                grid_size=base["grid_size"],
                dataset=base["dataset"],
                dataset_role=base["dataset_role"],
                branch=base["branch"],
                seed=np.asarray(args.seed, dtype=np.int64),
                shot=np.asarray(args.shot, dtype=np.int64),
                score_direction=base["score_direction"],
            )
            rows.append(
                {
                    "category": category,
                    "test_samples": len(samples),
                    "references": len(references),
                    "grid": list(test_grid),
                    "output": str(output.resolve()),
                    "sha256": sha256(output),
                }
            )
            print(f"wrote {output}", flush=True)

    report = {
        "schema_version": 1,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": "passed",
        "kind": "raw_patch_features_ref_only",
        "dataset": args.dataset,
        "dataset_role": args.dataset_role,
        "branch": args.branch,
        "model": model_label,
        "seed": args.seed,
        "shot": args.shot,
        "test_features_source": str(args.base_cache.resolve()),
        "manifest": str(args.manifest.resolve()),
        "manifest_sha256": sha256(args.manifest),
        "test_predictions_used_for_parameter_fit": False,
        "test_labels_used_for_parameter_fit": False,
        "test_set_statistics_used_for_calibration": False,
        "categories": rows,
    }
    (args.output_dir / "export_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps({"status": "passed", "n_categories": len(rows), "branch": args.branch}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
