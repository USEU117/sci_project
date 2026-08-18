"""A1 feature-level export for VisA (post-freeze validation, 阶段七).

New standalone script: does NOT touch the frozen A1 scripts in
freeze_manifest.json. Reuses v2_mpdd_prediction_common.index_visa for the
official VisA test split (meta.json).

--branch dino -> DINOv2 vitb14 patch features (run in .venv-patchcore)
--branch clip -> AnomalyCLIP ViT-L/14@336 patch features (run in .venv-anomalyclip)
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


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--dataset", choices=("mpdd", "btad", "visa", "mvtec"), required=True)
    parser.add_argument("--dataset-role", choices=("development", "holdout"), default="holdout")
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--branch", choices=("dino", "clip"), required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--shot", type=int, choices=(1, 2, 4), required=True)
    parser.add_argument("--device", default="cuda:0")
    # dino
    parser.add_argument("--model-name", default="dinov2_vitb14")
    parser.add_argument("--resolution", type=int, default=448)
    parser.add_argument("--map-size", type=int, default=448)
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
    if args.validate_only:
        print(json.dumps({"status": "passed", "mode": "validate_only", **counts}))
        return 0
    if args.device.startswith("cuda") and not torch_cuda_available():
        raise SystemExit("CUDA requested but unavailable")

    import torch

    indexed = index_dataset(args.dataset, args.data_root)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    setup_seed(args.model_seed)

    if args.branch == "dino":
        sys.path.insert(0, str(ROOT / "methods" / "anomalydino"))
        from src.backbones import get_model

        model = get_model(args.model_name, args.device, smaller_edge_size=args.resolution)

        def extract(image_rgb: np.ndarray) -> tuple[np.ndarray, tuple[int, int]]:
            tensor, grid = model.prepare_image(image_rgb)
            tokens = model.extract_features(tensor).astype(np.float32)
            expected = grid[0] * grid[1]
            if tokens.shape[0] != expected:
                raise RuntimeError(f"patch count {tokens.shape[0]} != grid {grid}")
            return tokens.reshape(grid[0], grid[1], -1), grid

        def read_image(relative: str) -> np.ndarray:
            return cv2.cvtColor(cv2.imread(str(args.data_root / relative)), cv2.COLOR_BGR2RGB)

        def read_mask(relative: str, size: int) -> np.ndarray:
            mask = cv2.imread(str(args.data_root / relative), cv2.IMREAD_GRAYSCALE)
            mask = cv2.resize(mask, (size, size), interpolation=cv2.INTER_NEAREST)
            return (mask > 0).astype(np.uint8)

        mask_size = args.map_size
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

        def extract(image_rgb: np.ndarray) -> tuple[np.ndarray, tuple[int, int]]:
            with Image.fromarray(image_rgb) as opened:
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

        def read_image(relative: str) -> np.ndarray:
            with Image.open(str(args.data_root / relative)) as opened:
                return np.asarray(opened.convert("RGB"))

        def read_mask(relative: str, size: int) -> np.ndarray:
            with Image.open(str(args.data_root / relative)) as opened:
                mask = np.asarray(opened.convert("L").resize((size, size), Image.Resampling.NEAREST))
            return (mask > 0).astype(np.uint8)

        mask_size = args.image_size
        model_label = "AnomalyCLIP_ViT-L/14@336px"

    rows = []
    with torch.inference_mode():
        for category, samples in sorted(indexed.items()):
            references = manifest["categories"][category][str(args.seed)][str(args.shot)]

            ref_blocks = []
            ref_grid = None
            for relative in references:
                patches, grid = extract(read_image(relative))
                ref_blocks.append(patches)
                ref_grid = grid

            feat_blocks = []
            masks, labels, sample_ids = [], [], []
            test_grid = None
            for sample in samples:
                patches, grid = extract(read_image(sample.image_path))
                feat_blocks.append(patches)
                if test_grid is None:
                    test_grid = grid
                elif test_grid != grid:
                    raise RuntimeError(f"inconsistent test grid in {category}: {test_grid} vs {grid}")
                if sample.mask_path is None:
                    mask = np.zeros((mask_size, mask_size), dtype=np.uint8)
                else:
                    mask = read_mask(sample.mask_path, mask_size)
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
                branch=np.asarray("anomalydino_visual" if args.branch == "dino" else "anomalyclip_text"),
                seed=np.asarray(args.seed),
                shot=np.asarray(args.shot),
                score_direction=np.asarray("higher_is_more_anomalous"),
            )
            rows.append(
                {
                    "category": category,
                    "samples": len(samples),
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
        "kind": "raw_patch_features",
        "dataset": args.dataset,
        "dataset_role": args.dataset_role,
        "branch": args.branch,
        "model": model_label,
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
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps({"status": "passed", "n_categories": len(rows), "branch": args.branch}))
    return 0


def torch_cuda_available() -> bool:
    try:
        import torch

        return torch.cuda.is_available()
    except Exception:
        return False


if __name__ == "__main__":
    raise SystemExit(main())
