"""G3 explicit-text-branch Gate: text capability + Oracle headroom vs strong visual.

Reads the explicit AnomalyCLIP text-conditioned anomaly maps (zero-shot, so
seed/shot invariant) and compares them against two visual anchors computed from
the frozen DINO raw patch cache on the MPDD 3-seed x {1,2,4}-shot matrix:

  * V0 = matched AnomalyDINO (DINO patch KNN; the plan's "must-beat" baseline)
  * V1 = `subspace_style_same_backbone` (PCA normal-subspace reconstruction
          residual, pca_ev=0.99) -- the G2 candidate.

For each (seed, shot, category) it computes, on a stride=8 grid at 448x448:
  * text-only pixel P-AP (B3 in the diagnostic matrix),
  * per-pixel label-informed Oracle = best-of-two(visual, text) selected with
    the ground-truth mask (development-only; never a router input),
  * Oracle headroom = Oracle P-AP - visual-only P-AP.

Text Gate (DYNAMIC_FUSION_NEXT_STEPS.md section 8 / G3):
  * Oracle mean P-AP headroom vs strong visual >= +0.015;
  * >= 4/6 categories headroom >= +0.005;
  * headroom must not be > 60% from a single category.
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import cv2
import faiss
import numpy as np
from scipy.ndimage import gaussian_filter
from sklearn.decomposition import PCA
from sklearn.metrics import average_precision_score, roc_auc_score

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from evaluate_unified import aupro_fast  # noqa: E402

STRIDE = 8
MAP_SIZE = 448
LEAKAGE_FLAGS = {
    "test_predictions_used_for_parameter_fit": False,
    "test_labels_used_for_parameter_fit": False,
    "test_masks_used_for_parameter_fit": False,
    "test_dataset_statistics_used_for_calibration": False,
    "test_normal_selection_used": False,
}


def dists2map(dists: np.ndarray, shape: tuple[int, int]) -> np.ndarray:
    return gaussian_filter(
        cv2.resize(dists, (shape[1], shape[0]), interpolation=cv2.INTER_LINEAR), sigma=4
    )


def load_dino(path: Path) -> dict:
    with np.load(path, allow_pickle=False) as d:
        return {
            "patch_features": np.asarray(d["patch_features"], dtype=np.float32),
            "ref_patch_features": np.asarray(d["ref_patch_features"], dtype=np.float32),
            "imgs_masks": np.asarray(d["imgs_masks"], dtype=np.uint8),
            "sample_ids": np.asarray(d["sample_ids"]),
            "grid_size": tuple(int(v) for v in d["grid_size"]),
        }


def load_text(path: Path) -> dict:
    with np.load(path, allow_pickle=False) as d:
        return {
            "anomaly_maps": np.asarray(d["anomaly_maps"], dtype=np.float32),
            "sample_ids": np.asarray(d["sample_ids"]),
        }


def load_text_targets(path: Path) -> dict:
    with np.load(path, allow_pickle=False) as d:
        return {
            "gt_sp": np.asarray(d["gt_sp"], dtype=np.int64),
            "imgs_masks": np.asarray(d["imgs_masks"], dtype=np.uint8),
            "sample_ids": np.asarray(d["sample_ids"]),
        }


def subspace_maps(feat: np.ndarray, ref: np.ndarray, pca_ev: float) -> np.ndarray:
    d = feat.shape[-1]
    feat_flat = feat.reshape(-1, d).astype(np.float64)
    ref_flat = ref.reshape(-1, d).astype(np.float64)
    pca = PCA(n_components=pca_ev, svd_solver="full", random_state=0)
    pca.fit(ref_flat)
    recon = pca.inverse_transform(pca.transform(feat_flat))
    residual = feat_flat - recon
    score = np.linalg.norm(residual, axis=1).astype(np.float32)
    n, gh, gw = feat.shape[0], feat.shape[1], feat.shape[2]
    return np.stack([dists2map(s, (MAP_SIZE, MAP_SIZE)) for s in score.reshape(n, gh, gw)])


def dino_knn_maps(feat: np.ndarray, ref: np.ndarray) -> np.ndarray:
    d = feat.shape[-1]
    feat_flat = feat.reshape(-1, d).astype(np.float32)
    ref_flat = ref.reshape(-1, d).astype(np.float32)
    faiss.normalize_L2(feat_flat)
    faiss.normalize_L2(ref_flat)
    index = faiss.IndexFlatL2(d)
    index.add(ref_flat)
    distances, _ = index.search(feat_flat, k=1)
    dists = (distances[:, 0] / 2.0).reshape(feat.shape[0], feat.shape[1], feat.shape[2])
    return np.stack([dists2map(s, (MAP_SIZE, MAP_SIZE)) for s in dists]).astype(np.float32)


def robust_map01(maps: np.ndarray) -> np.ndarray:
    maps = np.asarray(maps, dtype=np.float32)
    flat = maps.reshape(len(maps), -1)
    lower = np.quantile(flat, 0.01, axis=1).astype(np.float32)
    upper = np.quantile(flat, 0.99, axis=1).astype(np.float32)
    scale = np.maximum(upper - lower, np.float32(1e-6))
    return np.clip((maps - lower[:, None, None]) / scale[:, None, None], 0.0, 1.0)


def label_informed_oracle(labels: np.ndarray, visual: np.ndarray, text: np.ndarray) -> np.ndarray:
    truth = np.asarray(labels, dtype=bool)
    return np.where(truth, np.maximum(visual, text), np.minimum(visual, text))


def pixel_ap(labels: np.ndarray, scores: np.ndarray) -> float:
    return float(average_precision_score(labels.reshape(-1), scores.reshape(-1)))


def pixel_auroc(labels: np.ndarray, scores: np.ndarray) -> float:
    return float(roc_auc_score(labels.reshape(-1), scores.reshape(-1)))


def resize_maps(maps: np.ndarray, shape: tuple[int, int]) -> np.ndarray:
    return np.stack([cv2.resize(m, (shape[1], shape[0]), interpolation=cv2.INTER_LINEAR)
                     for m in maps]).astype(np.float32)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--text-dir", type=Path,
                        default=ROOT / "outputs" / "dynamic_fusion" / "v4_text_maps_mpdd")
    parser.add_argument("--pca-ev", type=float, choices=[0.95, 0.99], default=0.99)
    parser.add_argument("--output-dir", type=Path,
                        default=ROOT / "experiments" / "dynamic_fusion" / "v4_vision_text_20260819"
                        / "03_text_gate")
    args = parser.parse_args()

    if not args.text_dir.is_dir():
        raise SystemExit(f"text map dir missing: {args.text_dir}")

    cats = sorted(p.stem for p in args.text_dir.glob("*.npz") if not p.stem.endswith("_targets"))
    if not cats:
        raise SystemExit(f"no text npz in {args.text_dir}")
    missing_targets = [c for c in cats if not (args.text_dir / f"{c}_targets.npz").is_file()]
    if missing_targets:
        raise SystemExit(f"missing targets for: {missing_targets}")

    seeds, shots = [0, 1, 2], [1, 2, 4]
    feats_root = ROOT / "outputs" / "dynamic_fusion" / "v3_direction_a"

    # Load text maps once (zero-shot: seed/shot invariant).
    text_maps = {}
    text_ids = {}
    for cat in cats:
        t = load_text(args.text_dir / f"{cat}.npz")
        tg = load_text_targets(args.text_dir / f"{cat}_targets.npz")
        if not np.array_equal(t["sample_ids"], tg["sample_ids"]):
            raise ValueError(f"text sample_ids mismatch for {cat}")
        text_maps[cat] = resize_maps(t["anomaly_maps"], (MAP_SIZE, MAP_SIZE))
        text_ids[cat] = t["sample_ids"]

    rows = []
    for seed in seeds:
        for shot in shots:
            dino_dir = feats_root / f"features_vitb14_s{seed}_k{shot}" / "anomalydino_visual"
            for cat in cats:
                dino = load_dino(dino_dir / f"{cat}.npz")
                if not np.array_equal(dino["sample_ids"], text_ids[cat]):
                    raise ValueError(f"sample_ids mismatch {seed}/{shot}/{cat}")
                if len(dino["sample_ids"]) != text_maps[cat].shape[0]:
                    raise ValueError(f"sample count mismatch {seed}/{shot}/{cat}")

                v1 = subspace_maps(dino["patch_features"], dino["ref_patch_features"], args.pca_ev)
                dknn = dino_knn_maps(dino["patch_features"], dino["ref_patch_features"])
                txt = text_maps[cat]
                gt = dino["imgs_masks"]  # (N,448,448) uint8

                v1_s = robust_map01(v1[:, ::STRIDE, ::STRIDE])
                dknn_s = robust_map01(dknn[:, ::STRIDE, ::STRIDE])
                txt_s = robust_map01(txt[:, ::STRIDE, ::STRIDE])
                mask_s = gt[:, ::STRIDE, ::STRIDE].astype(bool)

                v1_ap = pixel_ap(mask_s, v1_s)
                dknn_ap = pixel_ap(mask_s, dknn_s)
                txt_ap = pixel_ap(mask_s, txt_s)
                ora_v1_ap = pixel_ap(mask_s, label_informed_oracle(mask_s, v1_s, txt_s))
                ora_dknn_ap = pixel_ap(mask_s, label_informed_oracle(mask_s, dknn_s, txt_s))

                rows.append({
                    "seed": seed, "shot": shot, "category": cat,
                    "text_pixel_ap": round(txt_ap, 6),
                    "v1_pixel_ap": round(v1_ap, 6),
                    "anomalydino_pixel_ap": round(dknn_ap, 6),
                    "oracle_vs_v1_pixel_ap": round(ora_v1_ap, 6),
                    "oracle_vs_dino_pixel_ap": round(ora_dknn_ap, 6),
                    "headroom_vs_v1": round(ora_v1_ap - v1_ap, 6),
                    "headroom_vs_dino": round(ora_dknn_ap - dknn_ap, 6),
                })

    cats_over_005_v1 = {}
    cats_over_005_dino = {}
    for cat in cats:
        sub = [r for r in rows if r["category"] == cat]
        cats_over_005_v1[cat] = float(np.mean([r["headroom_vs_v1"] for r in sub]))
        cats_over_005_dino[cat] = float(np.mean([r["headroom_vs_dino"] for r in sub]))

    def text_gate(headroom_by_cat: dict[str, float]) -> dict:
        mean_headroom = float(np.mean(list(headroom_by_cat.values())))
        over_005 = {c: h for c, h in headroom_by_cat.items() if h >= 0.005}
        pos_total = sum(max(h, 0.0) for h in headroom_by_cat.values())
        top_share = (max(headroom_by_cat.values()) / pos_total) if pos_total > 0 else 0.0
        passed = bool(
            mean_headroom >= 0.015
            and len(over_005) >= 4
            and top_share <= 0.60
        )
        return {
            "mean_headroom": round(mean_headroom, 6),
            "categories_over_0_005": sorted(over_005),
            "n_categories_over_0_005": len(over_005),
            "top_category_share_of_positive_headroom": round(top_share, 6),
            "gate_passed": passed,
        }

    text_mean_ap = float(np.mean([r["text_pixel_ap"] for r in rows]))
    text_ap_by_cat = {cat: round(float(np.mean([r["text_pixel_ap"] for r in rows if r["category"] == cat])), 6)
                      for cat in cats}

    report = {
        "run_id": "v4_g3_text_branch_gate",
        "pipeline": "v4_text_gate",
        "dataset": "mpdd",
        "dataset_role": "development",
        "text_branch": "anomalyclip_text (explicit text-conditioned maps, zero-shot, seed/shot invariant)",
        "text_checkpoint": "methods/AnomalyCLIP-main/checkpoints/9_12_4_multiscale_visa/epoch_15.pth",
        "map": "text 518->448 (INTER_LINEAR) then stride=8; visual 448 stride=8; robust_map01 per-image",
        "stride": STRIDE,
        "pca_ev": args.pca_ev,
        "seeds": seeds,
        "shots": shots,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "text_mean_pixel_ap": round(text_mean_ap, 6),
        "text_pixel_ap_by_category": text_ap_by_cat,
        "gate_vs_strong_visual_anomalydino": text_gate(cats_over_005_dino),
        "gate_vs_v1_subspace": text_gate(cats_over_005_v1),
        "per_category_mean_headroom_vs_dino": {c: round(v, 6) for c, v in cats_over_005_dino.items()},
        "per_category_mean_headroom_vs_v1": {c: round(v, 6) for c, v in cats_over_005_v1.items()},
        "leakage_flags": LEAKAGE_FLAGS,
        "rows": rows,
    }

    out_dir = args.output_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "text_gate_report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    gate_dino = report["gate_vs_strong_visual_anomalydino"]
    gate_v1 = report["gate_vs_v1_subspace"]
    (out_dir / "marker.json").write_text(json.dumps({
        "run_id": report["run_id"],
        "status": "passed",
        "gate_passed": {"vs_anomalydino": gate_dino["gate_passed"], "vs_v1_subspace": gate_v1["gate_passed"]},
        "paper_eligible": False,
        "dataset_role": "development",
    }, ensure_ascii=False, indent=2), encoding="utf-8")

    print(json.dumps({
        "text_mean_pixel_ap": report["text_mean_pixel_ap"],
        "text_ap_by_cat": text_ap_by_cat,
        "gate_vs_dino": gate_dino,
        "gate_vs_v1": gate_v1,
    }, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
