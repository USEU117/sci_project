"""D1 — defect-scale & frequency diagnostic (task book 14 section 10 D1).

Runs on MPDD development (seed0 x shot {1,2,4}, all categories):
1. GT defect-area tiering (small/mid/large) — evaluator-only, after scoring;
2. frozen A1 pixel maps per tier (pooled AP / AUROC);
3. a parameter-free two-scale stationary-wavelet spectral score (Sf-NM style);
4. oracle complementarity headroom (per-bad-image instance-level choice between
   A1 and spectral maps) and A1<->spectral rank correlation per tier.

Decision rule (document): prefer SF-NM / DC-SZoom when the SMALL tier headroom
>= +0.03 Pixel-AP AND rank correlation < 0.90.
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
for p in (str(ROOT / "src"), str(ROOT / "scripts")):
    if p not in sys.path:
        sys.path.insert(0, p)

from industrial_ad.innovation_v4_diagnostics import common, spectral  # noqa: E402
from industrial_ad.innovation_v4_diagnostics.common import (  # noqa: E402
    aligned_category, evaluator_gt, image_path_for, manifest_for,
    oracle_pooled_map, reference_ids_for, tier_pooled_map_scores,
)
from industrial_ad.innovation_v2.common import a1_grid, grids_to_maps  # noqa: E402

SHOTS = (1, 2, 4)
SEED = 0


def load_image_desc(relative: str, cache: dict, size: int = 448) -> np.ndarray:
    import cv2

    if relative not in cache:
        img = cv2.imread(str(image_path_for("mpdd", relative)))
        if img is None:
            raise RuntimeError(f"cannot read {image_path_for('mpdd', relative)}")
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        cache[relative] = spectral.spectral_descriptor_image(img)
    return cache[relative]


def category_d1(seed: int, shot: int, category: str, img_cache: dict,
                map_size=(448, 448)) -> dict:
    aligned = aligned_category("mpdd", seed, shot, category)
    gt = evaluator_gt("mpdd", seed, shot, category)
    bad_idx = np.nonzero(gt["gt_sp"] == 1)[0]
    good_idx = np.nonzero(gt["gt_sp"] == 0)[0]

    s_a1 = a1_grid(aligned)
    maps_a1 = grids_to_maps(s_a1, map_size)

    # ---- spectral memory score (label-free; refs then query test images)
    ref_ids = reference_ids_for(manifest_for("mpdd"), category, seed, shot)
    ref_desc = np.stack([load_image_desc(r, img_cache) for r in ref_ids])
    q_desc = np.stack([load_image_desc(sid, img_cache) for sid in aligned.sample_ids])
    s_freq = spectral.spectral_scores(ref_desc, q_desc)
    maps_freq = grids_to_maps(s_freq, map_size)

    gtm = gt["imgs_masks"].astype(np.uint8)
    maps_freq = maps_freq.astype(np.float32)
    maps_a1 = maps_a1.astype(np.float32)
    if maps_freq.shape != gtm.shape or maps_a1.shape != gtm.shape:
        raise RuntimeError("shape mismatch after upsampling")

    tiers = ("small", "mid", "large")
    rows = {}
    oracle = oracle_pooled_map(maps_a1, maps_freq, gtm, bad_idx)
    from scipy.stats import spearmanr
    for t in tiers:
        r_a1 = tier_pooled_map_scores(maps_a1, gtm, bad_idx, good_idx, t)
        r_fr = tier_pooled_map_scores(maps_freq, gtm, bad_idx, good_idx, t)
        r_or = tier_pooled_map_scores(oracle, gtm, bad_idx, good_idx, t)
        # rank correlation between A1 and spectral scores inside the tier
        if r_a1["n_bad_images"]:
            pos_a = np.asarray(r_a1["pos_scores"]); neg_a = np.asarray(r_a1["neg_scores"])
            pos_f = np.asarray(r_fr["pos_scores"]); neg_f = np.asarray(r_fr["neg_scores"])
            s_all = np.concatenate([pos_a, neg_a]); f_all = np.concatenate([pos_f, neg_f])
            corr = float(spearmanr(s_all, f_all).correlation) if s_all.size > 1 else None
        else:
            corr = None
        rows[t] = {
            "tier": t,
            "n_bad_images": int(r_a1["n_bad_images"]),
            "n_pos_px": int(r_a1["n_pos_px"]),
            "a1_ap": r_a1["ap"], "a1_auroc": r_a1["auroc"],
            "freq_ap": r_fr["ap"], "freq_auroc": r_fr["auroc"],
            "oracle_ap": r_or["ap"], "oracle_auroc": r_or["auroc"],
            "oracle_headroom_ap": (None if (r_or["ap"] is None or r_a1["ap"] is None)
                                   else round(float(r_or["ap"] - r_a1["ap"]), 6)),
            "a1_freq_rank_corr": (round(corr, 6) if corr is not None else None),
        }
    return {
        "dataset": "mpdd", "role": "development", "seed": seed, "shot": shot,
        "category": category,
        "n_test_images": aligned.n_images,
        "n_good": int(len(good_idx)), "n_bad": int(len(bad_idx)),
        "tiers": rows,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-root", type=Path,
                        default=common.EXPERIMENT_ROOT / "D1_scale_frequency")
    parser.add_argument("--shots", nargs="*", type=int, default=list(SHOTS))
    parser.add_argument("--category", default=None)
    args = parser.parse_args()

    common.assert_development_only()
    manifest = manifest_for("mpdd")
    cats = [args.category] if args.category else sorted(manifest["categories"])
    out_root = args.out_root
    out_root.mkdir(parents=True, exist_ok=True)

    img_cache: dict = {}
    t0 = time.time()
    reports = []
    for shot in args.shots:
        for cat in cats:
            print(f"[D1] {cat} s{SEED}/k{shot}", flush=True)
            rep = category_d1(SEED, shot, cat, img_cache)
            reports.append(rep)
            (out_root / f"{cat}_s{SEED}_k{shot}.json").write_text(
                json.dumps(rep, ensure_ascii=False, indent=1), encoding="utf-8")
            small = rep["tiers"]["small"]
            print(f"      small: n_bad={small['n_bad_images']} "
                  f"a1_ap={small['a1_ap']} oracle_ap={small['oracle_ap']} "
                  f"headroom={small['oracle_headroom_ap']} corr={small['a1_freq_rank_corr']}",
                  flush=True)
    # aggregate across categories per (shot, tier)
    agg = {}
    for shot in args.shots:
        agg[str(shot)] = {}
        for t in ("small", "mid", "large"):
            cell = []
            for r in reports:
                if r["shot"] != shot:
                    continue
                row = r["tiers"][t]
                cell.append({"category": r["category"], "n_bad": row["n_bad_images"],
                             "headroom": row["oracle_headroom_ap"],
                             "corr": row["a1_freq_rank_corr"],
                             "a1_ap": row["a1_ap"], "freq_ap": row["freq_ap"]})
            agg[str(shot)][t] = cell
    summary = {
        "schema_version": 1, "program": "innovation_v4_diagnostics",
        "diagnostic": "D1_defect_scale_frequency", "dataset": "mpdd",
        "role": "development", "seeds": [SEED], "shots": args.shots,
        "tier_thresholds": {"small_max": 0.005, "mid_max": 0.05},
        "rule": "prefer SF-NM/DC-SZoom if small-tier oracle headroom >= +0.03 "
                "Pixel-AP and rank corr < 0.90",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "elapsed_s": round(time.time() - t0, 1),
        "by_shot_tier": agg,
    }
    (out_root / "D1_SUMMARY.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=1), encoding="utf-8")
    print(json.dumps({k: {t: {"small_headroom_mean": np.nanmean(
        [c["headroom"] for c in v[t] if c["headroom"] is not None]) if any(
            c["headroom"] is not None for c in v[t]) else None}
        for t in ("small",)} for k, v in agg.items()}, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
