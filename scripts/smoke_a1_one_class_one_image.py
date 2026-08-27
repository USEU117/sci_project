"""P0-2 smoke: one-category, one-reference, one-test-image end-to-end A1 check.

Standalone (does NOT modify frozen scripts under freeze_manifest.json).

Modes (one process per branch, then a merge check):
  --mode dino   (run in .venv-patchcore):  extract DINOv2 vitb14 patch features
  --mode clip   (run in .venv-anomalyclip): extract AnomalyCLIP ViT-L/14@336 patch features
  --mode check  (run in .venv-patchcore):  merge both branches, verify grid/dtype/
               sample-id/reference-id alignment, measure the real concat dimension,
               run the frozen A1 concat + KNN (k=1) pipeline, emit an anomaly map,
               smoke metrics, leakage flags and repeat-run consistency.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "src"))


def sha256(path: Path) -> str:
    import hashlib

    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


# ----------------------------------------------------------------------
# branch extraction (dino / clip)
# ----------------------------------------------------------------------
def run_branch(args: argparse.Namespace) -> int:
    from v2_mpdd_prediction_common import index_dataset

    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    indexed = index_dataset(args.dataset, args.data_root)
    category = args.category
    if category not in indexed:
        raise SystemExit(f"category not found: {category}")
    refs = manifest["categories"][category][str(args.seed)][str(args.shot)]
    ref_id = refs[0]
    # Pick the first anomalous test image so the smoke metrics have a real mask.
    sample = next((s for s in indexed[category] if s.label == 1), indexed[category][0])

    import torch

    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise SystemExit("CUDA requested but unavailable")
    torch.cuda.reset_peak_memory_stats()
    t0 = time.time()

    if args.mode == "dino":
        sys.path.insert(0, str(ROOT / "methods" / "anomalydino"))
        import cv2

        from src.backbones import get_model

        model = get_model(args.model_name, args.device, smaller_edge_size=args.resolution)

        def extract(image_rgb: np.ndarray) -> tuple[np.ndarray, tuple[int, int]]:
            tensor, grid = model.prepare_image(image_rgb)
            tokens = model.extract_features(tensor).astype(np.float32)
            return tokens.reshape(grid[0], grid[1], -1), grid

        ref_rgb = cv2.cvtColor(cv2.imread(str(args.data_root / ref_id)), cv2.COLOR_BGR2RGB)
        test_rgb = cv2.cvtColor(cv2.imread(str(sample.image_path)), cv2.COLOR_BGR2RGB)
        ref_feat, ref_grid = extract(ref_rgb)
        test_feat, test_grid = extract(test_rgb)
        branch = "anomalydino_visual"
    else:
        sys.path.insert(0, str(ROOT / "methods" / "AnomalyCLIP-main"))
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

        with Image.open(str(args.data_root / ref_id)) as opened:
            ref_rgb = np.asarray(opened.convert("RGB"))
        with Image.open(str(sample.image_path)) as opened:
            test_rgb = np.asarray(opened.convert("RGB"))
        ref_feat, ref_grid = extract(ref_rgb)
        test_feat, test_grid = extract(test_rgb)
        branch = "anomalyclip_text"

    if ref_grid != test_grid:
        raise RuntimeError(f"ref/test grid mismatch: {ref_grid} vs {test_grid}")

    # Mask resized to the A1 map resolution (448), matching the frozen exporter schema.
    mask_arr = np.zeros((448, 448), dtype=np.uint8)
    if sample.mask_path is not None:
        if args.mode == "dino":
            mask_arr = cv2.imread(str(sample.mask_path), cv2.IMREAD_GRAYSCALE)
            mask_arr = cv2.resize(mask_arr, (448, 448), interpolation=cv2.INTER_NEAREST)
        else:
            with Image.open(str(sample.mask_path)) as opened:
                mask_arr = np.asarray(opened.convert("L").resize((448, 448), Image.Resampling.NEAREST))
        mask_arr = (mask_arr > 0).astype(np.uint8)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    out = args.output_dir / f"{args.mode}_smoke.npz"
    np.savez_compressed(
        out,
        patch_features=test_feat.astype(np.float32)[None, ...],
        ref_patch_features=ref_feat.astype(np.float32)[None, ...],
        gt_sp=np.asarray([sample.label], dtype=np.int64),
        imgs_masks=mask_arr[None, ...],
        sample_ids=np.asarray([sample.sample_id]),
        grid_size=np.asarray(ref_grid, dtype=np.int64),
        branch=np.asarray(branch),
    )
    peak_mb = torch.cuda.max_memory_allocated() / (1024 * 1024) if torch.cuda.is_available() else 0.0
    meta = {
        "mode": args.mode,
        "branch": branch,
        "category": category,
        "seed": args.seed,
        "shot": args.shot,
        "sample_id": sample.sample_id,
        "ref_id": ref_id,
        "test_grid": list(test_grid),
        "ref_grid": list(ref_grid),
        "ref_feature_shape": list(ref_feat.shape),
        "test_feature_shape": list(test_feat.shape),
        "ref_dtype": str(ref_feat.dtype),
        "test_dtype": str(test_feat.dtype),
        "peak_vram_mb": round(peak_mb, 2),
        "wall_seconds": round(time.time() - t0, 3),
        "output": str(out.resolve()),
        "output_sha256": sha256(out),
    }
    (args.output_dir / f"{args.mode}_smoke_meta.json").write_text(
        json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(meta, ensure_ascii=False, indent=2))
    return 0


# ----------------------------------------------------------------------
# merge check
# ----------------------------------------------------------------------
def run_check(args: argparse.Namespace) -> int:
    from evaluate_a1_feature_fusion import compute_metrics, fuse_category, load_features

    dino_meta = json.loads((args.output_dir / "dino_smoke_meta.json").read_text(encoding="utf-8"))
    clip_meta = json.loads((args.output_dir / "clip_smoke_meta.json").read_text(encoding="utf-8"))

    dino = load_features(args.output_dir / "dino_smoke.npz")
    clip = load_features(args.output_dir / "clip_smoke.npz")

    checks = {
        "sample_id_match": dino["sample_ids"].tolist() == clip["sample_ids"].tolist(),
        "ref_match": dino_meta["ref_id"] == clip_meta["ref_id"],
        "dino_grid": list(dino["grid_size"]),
        "clip_grid": list(clip["grid_size"]),
        "dino_dim": dino["patch_features"].shape[-1],
        "clip_dim": clip["patch_features"].shape[-1],
        "dino_dtype": str(dino["patch_features"].dtype),
        "clip_dtype": str(clip["patch_features"].dtype),
        "no_nan_inf_dino": bool(np.isfinite(dino["patch_features"]).all()),
        "no_nan_inf_clip": bool(np.isfinite(clip["patch_features"]).all()),
    }
    # Resolved concat dimension: DINO_dim + CLIP_dim after grid alignment (no PCA).
    concat_dim = checks["dino_dim"] + checks["clip_dim"]
    checks["concat_dim_resolved"] = concat_dim
    checks["documented_1152_is_wrong"] = bool(concat_dim != 1152)

    map_size = (448, 448)
    maps_a = fuse_category(dino, clip, "concat", pca_dim=0, whiten=False, map_size=map_size, dino_weight=0.5)
    maps_b = fuse_category(dino, clip, "concat", pca_dim=0, whiten=False, map_size=map_size, dino_weight=0.5)
    repeat_identical = bool(np.array_equal(maps_a, maps_b))
    metrics = compute_metrics(maps_a.astype(np.float64), dino["imgs_masks"])

    leakage_flags = {
        "test_predictions_used_for_parameter_fit": False,
        "test_labels_used_for_parameter_fit": False,
        "test_masks_used_for_parameter_fit": False,
        "test_set_statistics_used_for_calibration": False,
        "test_normal_selection_used": False,
    }

    args.output_dir.mkdir(parents=True, exist_ok=True)
    map_path = args.output_dir / "smoke_anomaly_map.npz"
    np.savez_compressed(map_path, maps=maps_a.astype(np.float32))

    report = {
        "p0_gate": "P0-2_one_class_one_image_smoke",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "category": args.category,
        "seed": args.seed,
        "shot": args.shot,
        "sample_id": dino_meta["sample_id"],
        "ref_id": dino_meta["ref_id"],
        "checks": checks,
        "smoke_metrics": metrics,
        "smoke_metrics_note": "smoke only; NOT for the paper table",
        "repeat_run_identical": repeat_identical,
        "leakage_flags": leakage_flags,
        "all_leakage_flags_false": all(v is False for v in leakage_flags.values()),
        "anomaly_map": str(map_path.resolve()),
        "anomaly_map_sha256": sha256(map_path),
        "peak_vram_mb": {"dino": dino_meta["peak_vram_mb"], "clip": clip_meta["peak_vram_mb"]},
        "wall_seconds": {"dino": dino_meta["wall_seconds"], "clip": clip_meta["wall_seconds"]},
    }
    report_path = args.output_dir / "smoke_report.json"
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=("dino", "clip", "check"), required=True)
    parser.add_argument("--manifest", type=Path, default=ROOT / "data/splits/mpdd/manifest.json")
    parser.add_argument("--dataset", choices=("mpdd", "btad"), default="mpdd")
    parser.add_argument("--data-root", type=Path, default=ROOT / "data/mpdd_raw/MPDD")
    parser.add_argument("--category", default="bracket_black")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--shot", type=int, default=1)
    parser.add_argument("--output-dir", type=Path, default=ROOT / "outputs/p0_2_smoke")
    parser.add_argument("--device", default="cuda:0")
    # dino
    parser.add_argument("--model-name", default="dinov2_vitb14")
    parser.add_argument("--resolution", type=int, default=448)
    # clip
    parser.add_argument("--checkpoint", type=Path,
                        default=ROOT / "methods/AnomalyCLIP-main/checkpoints/9_12_4_multiscale_visa/epoch_15.pth")
    parser.add_argument("--image-size", type=int, default=518)
    parser.add_argument("--features-list", type=int, nargs="+", default=[6, 12, 18, 24])
    parser.add_argument("--depth", type=int, default=9)
    parser.add_argument("--n-ctx", type=int, default=12)
    parser.add_argument("--text-n-ctx", type=int, default=4)
    parser.add_argument("--model-seed", type=int, default=111)
    args = parser.parse_args()

    if args.mode in ("dino", "clip"):
        return run_branch(args)
    return run_check(args)


if __name__ == "__main__":
    raise SystemExit(main())
