"""G2 visual-anchor gate matrix (CPU, read-only).

Runs the two matched branches over the full MPDD 3-seed x {1,2,4}-shot matrix
using the frozen DINO vitb14 raw patch cache:

  * V1  = `subspace_style_same_backbone`: PCA normal subspace (energy threshold
          in {0.95, 0.99}) fit on the K normal-reference patches of the current
          (seed, shot, category); anomaly = ||x - x_recon||_2.
  * matched AnomalyDINO = DINO patch KNN (normalize + faiss IndexFlatL2 k=1,
          distance/2) — the same map post-processing (dists2map: bilinear resize
          to 448 + gaussian_filter sigma=4) and evaluator as the frozen A1 DINO
          baseline.

Both branches share the identical frozen cache, manifest, grid, map and
evaluator, so the comparison is a matched-protocol ablation (V1 vs AnomalyDINO).

Gate G2 (DYNAMIC_FUSION_NEXT_STEPS.md section 8):
  * 9-config mean P-AP vs matched AnomalyDINO >= +0.010, OR
    (|P-AP delta| <= 0.003 AND AUPRO delta >= +0.010);
  * >= 7/9 configs non-negative and >= 4/6 categories positive;
  * no category mean P-AP regression < -0.020.
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


def dists2map(dists: np.ndarray, img_shape: tuple[int, int]) -> np.ndarray:
    dists = cv2.resize(dists, (img_shape[1], img_shape[0]), interpolation=cv2.INTER_LINEAR)
    return gaussian_filter(dists, sigma=4)


def load_features(path: Path) -> dict:
    with np.load(path, allow_pickle=False) as data:
        return {
            "patch_features": np.asarray(data["patch_features"], dtype=np.float32),
            "ref_patch_features": np.asarray(data["ref_patch_features"], dtype=np.float32),
            "imgs_masks": np.asarray(data["imgs_masks"], dtype=np.uint8),
            "grid_size": tuple(int(v) for v in data["grid_size"]),
        }


def compute_metrics(pixel_maps: np.ndarray, gt_masks: np.ndarray) -> dict:
    maps_strided = pixel_maps[:, ::STRIDE, ::STRIDE]
    masks_strided = gt_masks[:, ::STRIDE, ::STRIDE]
    flat_maps = maps_strided.ravel()
    flat_labels = (masks_strided.ravel() > 0.5).astype(np.int32)
    return {
        "pixel_auroc": float(roc_auc_score(flat_labels, flat_maps)),
        "pixel_ap": float(average_precision_score(flat_labels, flat_maps)),
        "pixel_aupro": float(aupro_fast(masks_strided, maps_strided)),
    }


def subspace_maps(feat: np.ndarray, ref: np.ndarray, pca_ev: float) -> np.ndarray:
    """PCA normal subspace on refs; anomaly = L2 reconstruction residual."""
    d = feat.shape[-1]
    feat_flat = feat.reshape(-1, d).astype(np.float64)
    ref_flat = ref.reshape(-1, d).astype(np.float64)
    pca = PCA(n_components=pca_ev, svd_solver="full", random_state=0)
    pca.fit(ref_flat)
    recon = pca.inverse_transform(pca.transform(feat_flat))
    residual = feat_flat - recon
    score = np.linalg.norm(residual, axis=1).astype(np.float32)
    n = feat.shape[0]
    gh, gw = feat.shape[1], feat.shape[2]
    return np.stack([dists2map(s, (MAP_SIZE, MAP_SIZE)) for s in score.reshape(n, gh, gw)])


def dino_knn_maps(feat: np.ndarray, ref: np.ndarray) -> np.ndarray:
    """Matched AnomalyDINO: L2-normalize, faiss k=1 KNN, distance/2."""
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


def evaluate_config(dino_dir: Path, seed: int, shot: int, pca_ev: float) -> list[dict]:
    cats = sorted(p.stem for p in dino_dir.glob("*.npz"))
    rows = []
    for cat in cats:
        d = load_features(dino_dir / f"{cat}.npz")
        v1 = compute_metrics(subspace_maps(d["patch_features"], d["ref_patch_features"], pca_ev),
                             d["imgs_masks"])
        dino = compute_metrics(dino_knn_maps(d["patch_features"], d["ref_patch_features"]),
                               d["imgs_masks"])
        rows.append({
            "category": cat,
            "v1": v1,
            "anomalydino": dino,
            "delta_ap": round(v1["pixel_ap"] - dino["pixel_ap"], 6),
            "delta_aupro": round(v1["pixel_aupro"] - dino["pixel_aupro"], 6),
        })
    return rows


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path,
                        default=ROOT / "experiments" / "dynamic_fusion" / "v4_vision_text_20260819"
                        / "02_visual_gate" / "g2_matrix")
    args = parser.parse_args()

    out_dir = args.output_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    seeds = [0, 1, 2]
    shots = [1, 2, 4]
    pca_evs = [0.95, 0.99]
    feats_root = ROOT / "outputs" / "dynamic_fusion" / "v3_direction_a"

    all_configs = []
    for pca_ev in pca_evs:
        for seed in seeds:
            for shot in shots:
                dino_dir = feats_root / f"features_vitb14_s{seed}_k{shot}" / "anomalydino_visual"
                if not dino_dir.is_dir():
                    print(f"[SKIP] missing {dino_dir}", flush=True)
                    continue
                rows = evaluate_config(dino_dir, seed, shot, pca_ev)
                mean_v1_ap = round(float(np.mean([r["v1"]["pixel_ap"] for r in rows])), 6)
                mean_dino_ap = round(float(np.mean([r["anomalydino"]["pixel_ap"] for r in rows])), 6)
                mean_v1_aupro = round(float(np.mean([r["v1"]["pixel_aupro"] for r in rows])), 6)
                mean_dino_aupro = round(float(np.mean([r["anomalydino"]["pixel_aupro"] for r in rows])), 6)
                delta_ap = round(mean_v1_ap - mean_dino_ap, 6)
                delta_aupro = round(mean_v1_aupro - mean_dino_aupro, 6)
                cfg = {
                    "seed": seed, "shot": shot, "pca_ev": pca_ev,
                    "mean_v1_pixel_ap": mean_v1_ap,
                    "mean_anomalydino_pixel_ap": mean_dino_ap,
                    "mean_delta_ap": delta_ap,
                    "mean_v1_aupro": mean_v1_aupro,
                    "mean_anomalydino_aupro": mean_dino_aupro,
                    "mean_delta_aupro": delta_aupro,
                    "positive_categories": sum(1 for r in rows if r["delta_ap"] > 0),
                    "min_category_delta_ap": round(min(r["delta_ap"] for r in rows), 6),
                    "rows": rows,
                }
                all_configs.append(cfg)
                print(f"pca={pca_ev} s{seed}_k{shot}: V1 AP={mean_v1_ap:.4f} "
                      f"DINO AP={mean_dino_ap:.4f} dAP={delta_ap:+.4f} "
                      f"dAUPRO={delta_aupro:+.4f} pos_cat={cfg['positive_categories']}/6",
                      flush=True)

    # Gate G2 per pca value.
    gates = {}
    for pca_ev in pca_evs:
        cfgs = [c for c in all_configs if c["pca_ev"] == pca_ev]
        mean_delta_ap = round(float(np.mean([c["mean_delta_ap"] for c in cfgs])), 6)
        mean_delta_aupro = round(float(np.mean([c["mean_delta_aupro"] for c in cfgs])), 6)
        non_neg = sum(1 for c in cfgs if c["mean_delta_ap"] >= 0)
        pos_cat_configs = sum(1 for c in cfgs if c["positive_categories"] >= 4)
        worst_cat = round(min(c["min_category_delta_ap"] for c in cfgs), 6)
        cond_a = mean_delta_ap >= 0.010
        cond_b = abs(mean_delta_ap) <= 0.003 and mean_delta_aupro >= 0.010
        hard = cond_a or cond_b
        gate = {
            "pca_ev": pca_ev,
            "n_configs": len(cfgs),
            "mean_delta_ap": mean_delta_ap,
            "mean_delta_aupro": mean_delta_aupro,
            "cond_a_delta_ap_ge_0010": cond_a,
            "cond_b_ap_flat_aupro_ge_0010": cond_b,
            "non_negative_configs": non_neg,
            "non_negative_ge_7_of_9": non_neg >= 7,
            "positive_4_of_6_cat_configs": pos_cat_configs,
            "positive_cat_ge_7_of_9": pos_cat_configs >= 7,
            "worst_category_delta_ap": worst_cat,
            "worst_category_ge_neg_0020": worst_cat >= -0.020,
            "gate_passed": bool(hard and non_neg >= 7 and pos_cat_configs >= 7 and worst_cat >= -0.020),
        }
        gates[pca_ev] = gate

    report = {
        "run_id": "v4_g2_visual_anchor_matrix",
        "pipeline": "v4_visual_anchor",
        "mode": "subspace_style_same_backbone",
        "dataset": "mpdd",
        "dataset_role": "development",
        "backbone": "dinov2_vitb14 (frozen DINO raw patch cache)",
        "map": "dists2map: bilinear resize 448 + gaussian_filter sigma=4",
        "stride": STRIDE,
        "pca_evs": pca_evs,
        "seeds": seeds,
        "shots": shots,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "configs": all_configs,
        "gate": gates,
        "leakage_flags": LEAKAGE_FLAGS,
    }
    (out_dir / "g2_matrix_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    (out_dir / "marker.json").write_text(json.dumps({
        "run_id": report["run_id"],
        "status": "passed" if any(g["gate_passed"] for g in gates.values()) else "failed",
        "gate_passed": {str(k): g["gate_passed"] for k, g in gates.items()},
        "paper_eligible": any(g["gate_passed"] for g in gates.values()),
        "dataset_role": "development",
    }, ensure_ascii=False, indent=2), encoding="utf-8")

    print(json.dumps({"gate": {str(k): v["gate_passed"] for k, v in gates.items()},
                      "mean_delta_ap": {str(k): v["mean_delta_ap"] for k, v in gates.items()}},
                     ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
