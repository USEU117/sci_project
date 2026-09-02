"""P5-C probe - expert heterogeneity WITHIN the frozen A1 system (pure CPU).

Motivation (user concern): S0 only fused two visual branches (A1 concat of DINO
+CLIP vs SubspaceAD giant). Question: is 'expert heterogeneity' tied to different
backbones, or can complementary normal-geometry experts already arise within one
frozen feature system (concat vs DINO-only)?  If intra-system complementarity is
comparable to the A1-vs-SUB case, it re-frames the fusion contribution as
'normal-geometry heterogeneity + reliability gating', not 'two backbones'.

Data: frozen compact maps carry BOTH concat_patch_map (A1) and dino_patch_map.
We replicate the Wave1 diagnostic structure between expert E1=concat(A1) and
expert E2=dino-only, at 448 with frozen evaluator GT. All CPU, MPDD dev only.
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

OUT = maps.EXPERIMENT_ROOT / "p5c_intrasystem"
A1_MAPS = maps.A1_MAPS_ROOT
SHOTS = (1, 2, 4)


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


def main() -> int:
    maps.assert_development_only()
    OUT.mkdir(parents=True, exist_ok=True)
    cats = sorted(p.stem for p in A1_MAPS.glob("s0_k1/*.npz"))
    rows, t0 = [], time.time()
    for cat in cats:
        for shot in SHOTS:
            npz = np.load(A1_MAPS / f"s0_k{shot}" / f"{cat}.npz", allow_pickle=False)
            sample_ids = np.asarray(npz["sample_ids"])
            concat = np.asarray(npz["concat_patch_map"], dtype=np.float32)
            dino = np.asarray(npz["dino_patch_map"], dtype=np.float32)
            gtm = maps.gt_masks_for(sample_ids)
            cm = maps.a1_maps448(concat)      # (N,448,448)
            dm = maps.a1_maps448(dino)
            bad = np.array(["/good/" not in str(s) for s in sample_ids])
            ap_c = maps.pixel_metrics_448(cm, gtm)["pixel_ap"]
            ap_d = maps.pixel_metrics_448(dm, gtm)["pixel_ap"]
            cn, dn = minmax_pooled(cm), minmax_pooled(dm)
            ap_mean = maps.pixel_metrics_448((cn + dn) / 2, gtm)["pixel_ap"]
            ap_max = maps.pixel_metrics_448(np.maximum(cn, dn), gtm)["pixel_ap"]
            chosen, n_dino_wins = [], 0
            for i in range(len(cm)):
                if not bad[i]:
                    chosen.append(cm[i])
                    continue
                a_c = per_image_ap(gtm[i], cm[i])
                a_d = per_image_ap(gtm[i], dm[i])
                use_d = (not np.isnan(a_c) and not np.isnan(a_d) and a_d >= a_c)
                n_dino_wins += int(use_d)
                chosen.append(dm[i] if use_d else cm[i])
            ap_oracle = maps.pixel_metrics_448(np.stack(chosen), gtm)["pixel_ap"]
            rows.append({"category": cat, "shot": shot, "n_bad": int(bad.sum()),
                         "concat_ap": round(ap_c, 4), "dino_ap": round(ap_d, 4),
                         "mean_ap": round(ap_mean, 4), "max_ap": round(ap_max, 4),
                         "oracle_ap": round(ap_oracle, 4),
                         "dino_wins_bad_frac": round(n_dino_wins / max(1, int(bad.sum())), 3)})
    per_cat = {}
    for cat in cats:
        r = [x for x in rows if x["category"] == cat]
        mm = lambda f: float(np.mean([x[f] for x in r]))
        best = max(mm("concat_ap"), mm("dino_ap"))
        per_cat[cat] = {"concat_mean": round(mm("concat_ap"), 4),
                        "dino_mean": round(mm("dino_ap"), 4),
                        "mean_combo_mean": round(mm("mean_ap"), 4),
                        "oracle_headroom": round(mm("oracle_ap") - best, 4),
                        "dino_wins_bad_frac": round(mm("dino_wins_bad_frac"), 3)}
    pool = lambda f: float(np.mean([per_cat[c][f] for c in cats]))
    verdict = {
        "pooled_concat": round(pool("concat_mean"), 4),
        "pooled_dino": round(pool("dino_mean"), 4),
        "pooled_best_single": round(max(pool("concat_mean"), pool("dino_mean")), 4),
        "pooled_mean_combo": round(pool("mean_combo_mean"), 4),
        "pooled_mean_combo_delta_vs_best": round(
            pool("mean_combo_mean") - max(pool("concat_mean"), pool("dino_mean")), 4),
        "pooled_oracle_headroom_vs_best": round(
            np.mean([per_cat[c]["oracle_headroom"] for c in cats]), 4),
        "positive_categories_oracle": int(sum(1 for c in cats
                                              if per_cat[c]["oracle_headroom"] > 0)),
        "note": "dino-only is the A1 subsystem (no CLIP/text); concat is A1 itself",
    }
    report = {"program": "innovation_v6_dgsafe", "phase": "p5c_intrasystem",
              "dataset": "mpdd", "role": "development", "seed": 0,
              "rows": rows, "per_category": per_cat, "verdict": verdict,
              "created_at_utc": datetime.now(timezone.utc).isoformat(),
              "elapsed_s": round(time.time() - t0, 1)}
    (OUT / "P5C_INTRASYSTEM.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=1), encoding="utf-8")
    print(json.dumps(verdict, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
