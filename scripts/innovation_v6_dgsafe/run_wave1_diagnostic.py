"""S0 Wave 1 — complementarity headroom diagnostic (task book 16 2.7).

Uses MPDD development GT (evaluator side only) to decide whether pixel-level
fusion of A1 and official SubspaceAD is worth pursuing.

Per (seed 0, shot {1,2,4}, category) at 448 resolution:
  - pooled Pixel-AP of A1, SUB, empirical mean, empirical max;
  - per-bad-image winner counts (SUB better ?) and a practical oracle
    (per bad image pick the better expert, goods -> A1).

Pass if (a) some fixed non-oracle combo beats the better single method by
>= +0.005 pooled, OR (b) oracle headroom >= +0.020 with >= 4/6 categories
positive (pooled over the 3 shots).
"""

from __future__ import annotations

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

SHOTS = (1, 2, 4)
OUT_ROOT = maps.EXPERIMENT_ROOT / "Wave1_complementarity"


def minmax_pooled(maps448: np.ndarray) -> np.ndarray:
    lo, hi = float(np.min(maps448)), float(np.max(maps448))
    if hi <= lo:
        return np.zeros_like(maps448)
    return (maps448 - lo) / (hi - lo)


def per_image_ap(mask_img: np.ndarray, map_img: np.ndarray) -> float:
    from sklearn.metrics import average_precision_score
    y = mask_img.ravel()[::8]
    s = map_img.ravel()[::8]
    if y.sum() == 0 or y.sum() == len(y):
        return float("nan")
    return float(average_precision_score(y, s))


def main() -> int:
    import argparse
    maps.assert_development_only()
    ap = argparse.ArgumentParser()
    ap.add_argument("--export-dir", type=Path, default=maps.EXPERIMENT_ROOT / "sub_maps_s0")
    args = ap.parse_args()
    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    cats = sorted({str(p.name).split("_s0_")[0]
                   for p in Path(args.export_dir).glob("*_s0_k*.npz")})
    t0 = time.time()
    rows = []
    for cat in cats:
        gt_cache = None
        for shot in SHOTS:
            a1 = maps.load_a1_patch_map(cat, 0, shot)
            sub = maps.load_sub_raw(args.export_dir, 0, shot, cat)
            perm = maps.align_perm(sub["sample_ids"], a1["sample_ids"])
            assert np.array_equal(sub["sample_ids"][perm], a1["sample_ids"])
            if gt_cache is None:
                gt_cache = maps.gt_masks_for(a1["sample_ids"])
            gtm = gt_cache
            a1m = maps.a1_maps448(a1["patch_map"])
            subm = maps.sub_maps448(sub["amap_raw"][perm])
            bad = np.array(["/good/" not in str(s) for s in a1["sample_ids"]])
            good = ~bad

            m_a1 = maps.pixel_metrics_448(a1m, gtm)["pixel_ap"]
            m_sub = maps.pixel_metrics_448(subm, gtm)["pixel_ap"]
            a1n, subn = minmax_pooled(a1m), minmax_pooled(subm)
            m_mean = maps.pixel_metrics_448((a1n + subn) / 2, gtm)["pixel_ap"]
            m_max = maps.pixel_metrics_448(np.maximum(a1n, subn), gtm)["pixel_ap"]

            # per-bad-image winner + practical oracle
            chosen = []
            n_sub_wins = 0
            for i in range(len(a1m)):
                if not bad[i]:
                    chosen.append(a1m[i])
                    continue
                ap_a = per_image_ap(gtm[i], a1m[i])
                ap_s = per_image_ap(gtm[i], subm[i])
                use_sub = (not np.isnan(ap_a) and not np.isnan(ap_s) and ap_s >= ap_a)
                n_sub_wins += int(use_sub)
                chosen.append(subm[i] if use_sub else a1m[i])
            m_oracle = maps.pixel_metrics_448(np.stack(chosen), gtm)["pixel_ap"]
            rows.append({"category": cat, "shot": shot, "n_bad": int(bad.sum()),
                         "a1_pixel_ap": m_a1, "sub_pixel_ap": m_sub,
                         "mean_pixel_ap": m_mean, "max_pixel_ap": m_max,
                         "oracle_pixel_ap": m_oracle,
                         "sub_wins_bad_images": int(n_sub_wins)})

    # pooled over shots per category
    per_cat, cells = {}, []
    for cat in cats:
        r = [x for x in rows if x["category"] == cat]
        def mm(f): return float(np.mean([x[f] for x in r]))
        a1m = mm("a1_pixel_ap"); subm = mm("sub_pixel_ap")
        oracle = mm("oracle_pixel_ap")
        best_single = max(a1m, subm)
        per_cat[cat] = {
            "a1_mean": round(a1m, 4), "sub_mean": round(subm, 4),
            "mean_combo_mean": round(mm("mean_pixel_ap"), 4),
            "max_combo_mean": round(mm("max_pixel_ap"), 4),
            "oracle_mean": round(oracle, 4),
            "oracle_headroom_vs_best": round(oracle - best_single, 4),
            "sub_wins_bad_frac": round(int(sum(x["sub_wins_bad_images"] for x in r)) /
                                       max(1, sum(x["n_bad"] for x in r)), 3),
        }
        cells.append(per_cat[cat])

    def pool(f): return float(np.mean([c[f] for c in cells]))
    a1_pool, sub_pool = pool("a1_mean"), pool("sub_mean")
    best_pool = max(a1_pool, sub_pool)
    best_fixed = max(pool("mean_combo_mean"), pool("max_combo_mean"))
    oracle_pool = pool("oracle_mean")
    pos_cats_oracle = sum(1 for c in cells if c["oracle_headroom_vs_best"] > 0)
    cond_fixed = best_fixed - best_pool >= 0.005
    cond_oracle = (oracle_pool - best_pool >= 0.020) and (pos_cats_oracle >= 4)
    verdict = {
        "cond_a_fixed_combo_delta_ge_005": bool(cond_fixed),
        "cond_b_oracle_headroom_ge_002_and_4of6_positive": bool(cond_oracle),
        "wave1_passed": bool(cond_fixed or cond_oracle),
        "best_fixed_combo_pooled_delta_vs_best_single": round(best_fixed - best_pool, 4),
        "oracle_headroom_pooled_vs_best_single": round(oracle_pool - best_pool, 4),
        "positive_categories_oracle": pos_cats_oracle,
    }
    report = {
        "program": "innovation_v6_dgsafe", "phase": "Wave1_complementarity",
        "dataset": "mpdd", "role": "development", "seed": 0,
        "protocol": "448 maps (A1 compact dists2map; SUB export dists2map), pooled "
                    "Pixel-AP flatten[::8]; combos on pooled min-max z; oracle per bad image",
        "pooled": {"a1": round(a1_pool, 4), "sub": round(sub_pool, 4),
                   "best_single": round(best_pool, 4),
                   "best_fixed_combo": round(best_fixed, 4),
                   "oracle": round(oracle_pool, 4)},
        "verdict": verdict,
        "per_category": per_cat,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "elapsed_s": round(time.time() - t0, 1),
    }
    (OUT_ROOT / "WAVE1_DIAGNOSTIC.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=1), encoding="utf-8")
    (OUT_ROOT / "WAVE1_DECISION.md").write_text(
        "# WAVE1 DECISION (machine-drafted, needs user/oversight sign-off)\n\n"
        + f"- **passed**: {verdict['wave1_passed']}\n"
        + f"- cond A (fixed combo >= +0.005): {verdict['cond_a_fixed_combo_delta_ge_005']}\n"
        + f"- cond B (oracle >= +0.020 & >=4/6 cats): {verdict['cond_b_oracle_headroom_ge_002_and_4of6_positive']}\n"
        + f"- best fixed combo delta vs best single: {verdict['best_fixed_combo_pooled_delta_vs_best_single']}\n"
        + f"- oracle headroom vs best single: {verdict['oracle_headroom_pooled_vs_best_single']}\n"
        + f"- positive categories (oracle): {verdict['positive_categories_oracle']}/6\n\n"
        + "Details: WAVE1_DIAGNOSTIC.json\n", encoding="utf-8")
    print(json.dumps(report, indent=1), flush=True)
    return 0 if verdict["wave1_passed"] else 2


if __name__ == "__main__":
    sys.exit(main())
