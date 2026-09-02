"""S0 premise check (CPU, no GPU): reproduce task-book-16 table of
official SubspaceAD - A1 per-category mean Pixel-AP on MPDD development.

SUB pixel metrics come from the frozen audit (06_v2_g2_audit/per_config.jsonl,
pca_ev 0.99). A1 pixel metrics are recomputed here from the frozen compact maps
(submission_repro_20260827/predictions_compact) with the identical 448/stride-8
protocol. GT is loaded by the evaluator side only.
"""

from __future__ import annotations

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

from industrial_ad.innovation_v6_dgsafe import maps  # noqa: E402

AUDIT_PER_CONFIG = (ROOT / "experiments/dynamic_fusion/v4_vision_text_20260819"
                    / "06_v2_g2_audit/per_config.jsonl")
OUT_ROOT = maps.EXPERIMENT_ROOT / "premise"
SEEDS = [0, 1, 2]
SHOTS = [1, 2, 4]


def sub_pixel_ap_table() -> dict:
    sub = {}
    for line in AUDIT_PER_CONFIG.read_text(encoding="utf-8").splitlines():
        r = json.loads(line)
        sub[(r["category"], int(r["seed"]), int(r["shot"]))] = r["pixel_ap"]
    return sub


def main() -> int:
    maps.assert_development_only()
    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    sub = sub_pixel_ap_table()
    cats = sorted({k[0] for k in sub})

    gt_cache: dict[str, np.ndarray] = {}
    t0 = time.time()
    cells, a1_overall, n = [], 0.0, 0
    for cat in cats:
        for seed in SEEDS:
            for shot in SHOTS:
                a = maps.load_a1_patch_map(cat, seed, shot)
                sid = a["sample_ids"]
                key = cat
                if key not in gt_cache:
                    gt_cache[key] = maps.gt_masks_for(sid)
                maps448 = maps.a1_maps448(a["patch_map"])
                a1m = maps.pixel_metrics_448(maps448, gt_cache[key])["pixel_ap"]
                sub_ap = sub[(cat, seed, shot)]
                d = (sub_ap - a1m) if (a1m is not None and sub_ap is not None) else None
                cells.append({"category": cat, "seed": seed, "shot": shot,
                              "sub_pixel_ap": sub_ap, "a1_pixel_ap": a1m,
                              "delta_ap": d})
                if d is not None:
                    a1_overall += a1m
                    n += 1

    per_cat = {}
    pos_cells = 0
    for cat in cats:
        ds = [c["delta_ap"] for c in cells if c["category"] == cat and c["delta_ap"] is not None]
        pos = sum(1 for c in cells if c["category"] == cat and (c["delta_ap"] or 0) > 0)
        total = sum(1 for c in cells if c["category"] == cat and c["delta_ap"] is not None)
        per_cat[cat] = {"mean_delta_ap": round(float(np.mean(ds)), 4) if ds else None,
                        "positive_cells": pos, "n_cells": total,
                        "worst_delta_ap": round(float(min(ds)), 4) if ds else None}
        pos_cells += pos
    overall_delta = round(float(np.mean([c["delta_ap"] for c in cells if c["delta_ap"] is not None])), 4)
    a1_mean = round(a1_overall / n, 4) if n else None

    report = {
        "program": "innovation_v6_dgsafe", "phase": "premise_check",
        "dataset": "mpdd", "role": "development",
        "a1_protocol": "compact concat patch map -> 448 dists2map -> pooled Pixel-AP "
                       "flatten[::8] (frozen A1 convention)",
        "sub_protocol": "frozen audit per_config pixel_ap (672 map, stride-8) pca_ev 0.99",
        "expected_doc16": {"overall_mean_delta": 0.02132, "positive_cells_54": 38,
                           "bracket_black": 0.15, "bracket_brown": 0.03,
                           "bracket_white": 0.0, "connector": -0.16,
                           "metal_plate": 0.04, "tubes": 0.06},
        "computed": {
            "n_cells": n, "positive_cells": pos_cells,
            "overall_mean_delta_ap": overall_delta, "a1_overall_mean_pixel_ap": a1_mean,
            "per_category": per_cat,
        },
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "elapsed_s": round(time.time() - t0, 1),
    }
    (OUT_ROOT / "PREMISE_SUMMARY.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"overall mean delta (SUB - A1) = {overall_delta}  positive {pos_cells}/{n}  "
          f"A1 mean = {a1_mean}", flush=True)
    for cat in cats:
        c = report["computed"]["per_category"][cat]
        print(f"  {cat}: mean_delta={c['mean_delta_ap']} pos={c['positive_cells']}/{c['n_cells']} "
              f"worst={c['worst_delta_ap']}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
