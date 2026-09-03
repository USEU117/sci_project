"""Route C — CAPM feasibility statistics (task book 19 §6): MPDD seed0 k1.

Mutual-NN patch correspondences (DINO vitb14, cosine) between each test image and
its single normal reference, then RANSAC affine. Reports inlier-ratio /
reprojection-error distributions per class and the full-6-class summary.

EARLY FAIL rule (pre-registered): majority of test images (median across classes)
have inlier ratio < 0.3, or the full-6-class aligned fraction (<inlier>=0.3) is
< 40% -> route FAILS before any pixel evaluation.

  .venv-patchcore\\Scripts\\python.exe scripts\\innovation_v10_portfolio\\run_r0_capm.py
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from industrial_ad.innovation_v10_portfolio.common import load_features

CATEGORIES = ["bracket_black", "bracket_brown", "bracket_white", "connector", "metal_plate", "tubes"]
MAX_IMAGES = 60
RANSAC_REPROJ_THRESH = 2.0  # patch units


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def mutual_matches(qf: np.ndarray, rf: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """(row_idx_in_query, row_idx_in_ref) mutual nearest neighbours (cosine)."""
    import faiss

    d = qf.shape[-1]
    index_q = faiss.IndexFlatIP(d)
    index_r = faiss.IndexFlatIP(d)
    index_q.add(qf)
    index_r.add(rf)
    _, q_to_r = index_r.search(qf, 1)  # query -> its best ref
    _, r_to_q = index_q.search(rf, 1)  # ref -> its best query
    qi = np.arange(qf.shape[0])
    ri = q_to_r[:, 0]
    mutual = r_to_q[ri, 0] == qi
    return qi[mutual], ri[mutual]


def affine_inliers(qf: np.ndarray, rf: np.ndarray, grid: tuple[int, int]):
    """RANSAC affine ref->query on patch coordinates. Returns dict of stats."""
    qi, ri = mutual_matches(qf, rf)
    n_mut = int(qi.size)
    if n_mut < 8:
        return {"n_mutual": n_mut, "inliers": 0, "inlier_ratio": 0.0, "reproj_err": float("nan"),
                "ransac_ok": False}
    qy, qx = np.unravel_index(qi, grid)
    ry, rx = np.unravel_index(ri, grid)
    src = np.stack([rx.astype(np.float32), ry.astype(np.float32)], axis=1)
    dst = np.stack([qx.astype(np.float32), qy.astype(np.float32)], axis=1)
    M, inliers = cv2.estimateAffine2D(
        src, dst, method=cv2.RANSAC, ransacReprojThreshold=RANSAC_REPROJ_THRESH,
        maxIters=2000, confidence=0.99)
    if M is None or inliers is None:
        return {"n_mutual": n_mut, "inliers": 0, "inlier_ratio": 0.0, "reproj_err": float("nan"),
                "ransac_ok": False}
    n_in = int(inliers.sum())
    # inlier mean reprojection error
    sel = inliers[:, 0] > 0
    if sel.sum() == 0:
        err = float("nan")
    else:
        pred = cv2.transform(src[sel].reshape(-1, 1, 2), M).reshape(-1, 2)
        err = float(np.sqrt(((pred - dst[sel]) ** 2).sum(axis=1)).mean())
    return {"n_mutual": n_mut, "inliers": n_in, "inlier_ratio": float(n_in / n_mut),
            "reproj_err": err, "ransac_ok": True}


def run_category(path_npz: Path, max_images: int) -> dict:
    d = load_features(path_npz)
    qf_all = d["patch_features"]                     # [N, H, W, D]
    rf = d["ref_patch_features"][0]                  # [H, W, D] single ref (k1)
    grid = d["grid_size"]
    n, h, w, dim = qf_all.shape
    rf_flat = rf.reshape(h * w, dim).astype(np.float32)
    import faiss

    faiss.normalize_L2(rf_flat)
    n_imgs = min(n, max_images)
    rows = []
    for i in range(n_imgs):
        qf = qf_all[i].reshape(h * w, dim).astype(np.float32)
        faiss.normalize_L2(qf)
        rows.append(affine_inliers(qf, rf_flat, grid))
    ratios = np.array([r["inlier_ratio"] for r in rows])
    errs = np.array([r["reproj_err"] for r in rows if np.isfinite(r["reproj_err"])])
    frac_aligned = float((ratios >= 0.3).mean())
    return {
        "category": Path(path_npz).stem,
        "n_images": n_imgs,
        "n_mutual_median": float(np.median([r["n_mutual"] for r in rows])),
        "inlier_ratio_mean": float(ratios.mean()),
        "inlier_ratio_median": float(np.median(ratios)),
        "frac_images_inlier_ge_0.3": round(frac_aligned, 4),
        "inlier_ratio_p25": float(np.percentile(ratios, 25)),
        "inlier_ratio_p75": float(np.percentile(ratios, 75)),
        "reproj_err_mean_inliers": float(errs.mean()) if errs.size else None,
        "ransac_ok_frac": float(np.mean([r["ransac_ok"] for r in rows])),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dino-cache", type=Path,
                        default=ROOT / "outputs/dynamic_fusion/v3_direction_a/features_vitb14_s0_k1/anomalydino_visual")
    parser.add_argument("--out-dir", type=Path,
                        default=ROOT / "experiments/dynamic_fusion/innovation_v10_portfolio/capm")
    parser.add_argument("--max-images", type=int, default=MAX_IMAGES)
    args = parser.parse_args()

    out_dir = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    protocol = out_dir / "R0_PROTOCOL.json"
    if not protocol.is_file():
        raise SystemExit(f"missing pre-registered protocol: {protocol}")

    t0 = time.time()
    rows = []
    for cat in CATEGORIES:
        p = args.dino_cache / f"{cat}.npz"
        if not p.is_file():
            print(f"skip {cat}: missing {p}")
            continue
        r = run_category(p, args.max_images)
        rows.append(r)
        print(f"[CAPM {cat}] inlier median={r['inlier_ratio_median']:.3f} "
              f"mean={r['inlier_ratio_mean']:.3f} frac≥0.3={r['frac_images_inlier_ge_0.3']:.2%} "
              f"reproj={r['reproj_err_mean_inliers']}", flush=True)

    medians = np.array([r["inlier_ratio_median"] for r in rows])
    fracs = np.array([r["frac_images_inlier_ge_0.3"] for r in rows])
    report = {
        "route": "C_CAPM",
        "pipeline": "v10_portfolio_r0_feasibility",
        "seed": 0, "shot": 1, "data_role": "development",
        "ransac_reproj_thresh_patch": RANSAC_REPROJ_THRESH,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "protocol_sha256": sha256_file(protocol),
        "code_sha256": {"runner": sha256_file(Path(__file__))},
        "per_category": rows,
        "full6_summary": {
            "median_of_class_medians": float(np.median(medians)),
            "mean_of_class_medians": float(medians.mean()),
            "mean_frac_aligned_per_class": float(fracs.mean()),
        },
        "early_fail_check": {
            "majority_classes_median_lt_0.3": bool((medians < 0.3).sum() > len(rows) / 2),
            "frac_aligned_lt_0.4": bool(fracs.mean() < 0.4),
        },
        "decision": "PENDING",
    }
    (out_dir / "R0_RESULT.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\nfull-6: median-of-medians {report['full6_summary']['median_of_class_medians']:.3f} "
          f"| aligned-frac {report['full6_summary']['mean_frac_aligned_per_class']:.2%} | "
          f"elapsed {time.time()-t0:.0f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
