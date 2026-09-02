"""P5-D probe - same-backbone normal-geometry heterogeneity (pure CPU).

Question: A1 (exemplar-memory KNN) vs SubspaceAD-giant (subspace reconstruction)
showed fixed-combinable complementarity (Wave1 mean-combo +0.0164). Is that due
to *geometry type* (memory vs subspace) or merely to the bigger backbone (giant)?

Design: build a weak subspace-reconstruction expert on the SAME frozen vitb14
patch features that A1's DINO half already uses (outputs/.../features_vitb14_*).
Per (cat, k): L2-normalize ref+test patch tokens (same normalisation as A1's KNN
memory), fit PCA (ev=0.99) on normal refs only, score each test patch by
reconstruction residual ||x - P x||^2 -> 32x32 grid -> dists2map(448).
Compare vs frozen A1 concat maps with the Wave1/P5C diagnostic skeleton.

Interpretation:
  + fixed combo gain (concat+subspace > best single) on the SAME backbone =>
    geometry heterogeneity (KNN-memory vs linear subspace) carries the fusion
    value -> re-frames the S0 story away from 'two backbones / two branches'.
  no gain => A1-vs-SUB complementarity is backbone-scale driven; the 'two visual
  branches' novelty critique stands and future work must bring a different axis.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
for p in (str(ROOT), str(ROOT / "src"), str(ROOT / "scripts")):
    if p not in sys.path:
        sys.path.insert(0, p)

from industrial_ad.innovation_v6_dgsafe import maps  # noqa: E402

OUT = maps.EXPERIMENT_ROOT / "p5d_samebackbone_subspace"
A1_MAPS = maps.A1_MAPS_ROOT
FEAT_ROOT = (ROOT / "outputs" / "dynamic_fusion" / "v3_direction_a")
GRID = 32
PCA_EV = 0.99


def minmax_pooled(x):
    lo, hi = float(np.min(x)), float(np.max(x))
    if hi <= lo:
        return np.zeros_like(x)
    return (x - lo) / (hi - lo)


def per_image_ap(mask_img, map_img):
    from sklearn.metrics import average_precision_score
    y = mask_img.ravel()[::maps.STRIDE]
    s = map_img.ravel()[::maps.STRIDE]
    if y.sum() == 0 or y.sum() == len(y):
        return float("nan")
    return float(average_precision_score(y, s))


def subspace_maps(cat, seed, shot):
    """Return (N,448,448) subspace-residual maps built from cached vitb14 feats."""
    from sklearn.decomposition import PCA
    d = np.load(FEAT_ROOT / f"features_vitb14_s{seed}_k{shot}/anomalydino_visual/"
               f"{cat}.npz", allow_pickle=False)
    refs = np.asarray(d["ref_patch_features"], dtype=np.float64)   # (K,32,32,768)
    tests = np.asarray(d["patch_features"], dtype=np.float64)      # (N,32,32,768)
    K = refs.shape[0]
    Xr = refs.reshape(K * GRID * GRID, -1)
    Xt = tests.reshape(-1, tests.shape[-1])
    Xr = Xr / (np.linalg.norm(Xr, axis=1, keepdims=True) + 1e-12)
    Xt = Xt / (np.linalg.norm(Xt, axis=1, keepdims=True) + 1e-12)
    pca = PCA(n_components=PCA_EV, svd_solver="full", random_state=0)
    pca.fit(Xr)
    P = pca.components_.T @ pca.components_          # (d,d) projector
    resid = np.einsum("ij,ij->i", Xt - Xt @ P, Xt - Xt @ P)   # squared residual
    maps32 = resid.reshape(tests.shape[0], GRID, GRID).astype(np.float32)
    return maps.a1_maps448(maps32)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--shots", nargs="+", type=int, default=[1, 2, 4])
    args = ap.parse_args()
    maps.assert_development_only()
    OUT.mkdir(parents=True, exist_ok=True)
    cats = sorted(p.stem for p in A1_MAPS.glob(f"s{args.seed}_k1/*.npz"))
    rows, t0 = [], time.time()
    for cat in cats:
        for shot in args.shots:
            npz = np.load(A1_MAPS / f"s{args.seed}_k{shot}" / f"{cat}.npz",
                          allow_pickle=False)
            sample_ids = np.asarray(npz["sample_ids"])
            concat = np.asarray(npz["concat_patch_map"], dtype=np.float32)
            gtm = maps.gt_masks_for(sample_ids)
            cm = maps.a1_maps448(concat)
            sm = subspace_maps(cat, args.seed, shot)
            bad = np.array(["/good/" not in str(s) for s in sample_ids])
            ap_c = maps.pixel_metrics_448(cm, gtm)["pixel_ap"]
            ap_s = maps.pixel_metrics_448(sm, gtm)["pixel_ap"]
            cn, sn = minmax_pooled(cm), minmax_pooled(sm)
            ap_mean = maps.pixel_metrics_448((cn + sn) / 2, gtm)["pixel_ap"]
            ap_max = maps.pixel_metrics_448(np.maximum(cn, sn), gtm)["pixel_ap"]
            chosen, n_s_wins = [], 0
            for i in range(len(cm)):
                if not bad[i]:
                    chosen.append(cm[i])
                    continue
                a_c = per_image_ap(gtm[i], cm[i])
                a_s = per_image_ap(gtm[i], sm[i])
                use_s = (not np.isnan(a_c) and not np.isnan(a_s) and a_s >= a_c)
                n_s_wins += int(use_s)
                chosen.append(sm[i] if use_s else cm[i])
            ap_oracle = maps.pixel_metrics_448(np.stack(chosen), gtm)["pixel_ap"]
            rows.append({"category": cat, "shot": shot, "n_bad": int(bad.sum()),
                         "concat_ap": round(ap_c, 4), "subspace_ap": round(ap_s, 4),
                         "mean_ap": round(ap_mean, 4), "max_ap": round(ap_max, 4),
                         "oracle_ap": round(ap_oracle, 4),
                         "subspace_wins_bad_frac": round(n_s_wins / max(1, int(bad.sum())), 3)})
            print(f"[{cat}] k{shot} concat={ap_c:.4f} subspace={ap_s:.4f} "
                  f"mean={ap_mean:.4f} oracle={ap_oracle:.4f}", flush=True)
    per_cat = {}
    for cat in cats:
        r = [x for x in rows if x["category"] == cat]
        mm = lambda f: float(np.mean([x[f] for x in r]))
        best = max(mm("concat_ap"), mm("subspace_ap"))
        per_cat[cat] = {"concat_mean": round(mm("concat_ap"), 4),
                        "subspace_mean": round(mm("subspace_ap"), 4),
                        "mean_combo_mean": round(mm("mean_ap"), 4),
                        "oracle_headroom_vs_best": round(mm("oracle_ap") - best, 4),
                        "subspace_wins_bad_frac": round(mm("subspace_wins_bad_frac"), 3)}
    pool = lambda f: float(np.mean([per_cat[c][f] for c in cats]))
    verdict = {
        "pooled_concat": round(pool("concat_mean"), 4),
        "pooled_subspace": round(pool("subspace_mean"), 4),
        "pooled_best_single": round(max(pool("concat_mean"), pool("subspace_mean")), 4),
        "pooled_mean_combo_delta_vs_best": round(
            pool("mean_combo_mean") - max(pool("concat_mean"), pool("subspace_mean")), 4),
        "pooled_oracle_headroom_vs_best": round(
            np.mean([per_cat[c]["oracle_headroom_vs_best"] for c in cats]), 4),
        "positive_categories_oracle": int(sum(1 for c in cats
                                              if per_cat[c]["oracle_headroom_vs_best"] > 0)),
        "design": "vitb14 cached features; subspace PCA(ev .99) on L2-normalised "
                  "normal refs; reconstruction residual maps; SAME backbone as A1's DINO half",
    }
    report = {"program": "innovation_v6_dgsafe", "phase": "p5d_samebackbone_subspace",
              "dataset": "mpdd", "role": "development", "seed": args.seed,
              "rows": rows, "per_category": per_cat, "verdict": verdict,
              "created_at_utc": datetime.now(timezone.utc).isoformat(),
              "elapsed_s": round(time.time() - t0, 1)}
    (OUT / "P5D_SAMEBACKBONE.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=1), encoding="utf-8")
    print(json.dumps(verdict, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
