"""Summarize the A1 MPDD weight-scan (9 configs x {0.3,0.4,0.5,0.6,0.7})."""

from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

SCAN_ROOT = ROOT / "experiments" / "dynamic_fusion" / "v3_direction_a" / "a1_weight_scan_20260817"
WEIGHTS = [0.3, 0.4, 0.5, 0.6, 0.7]


def main() -> int:
    seeds = [0, 1, 2]
    shots = [1, 2, 4]
    # mean delta_ap per weight
    per_weight = {w: [] for w in WEIGHTS}
    rows = []
    for seed in seeds:
        for shot in shots:
            for w in WEIGHTS:
                path = SCAN_ROOT / f"seed{seed}_k{shot}" / f"concat_pca0_whiten0_w{w:g}_report.json"
                if not path.is_file():
                    raise SystemExit(f"missing: {path}")
                r = json.loads(path.read_text(encoding="utf-8"))
                delta = r["mean_delta_ap_vs_dino"]
                per_weight[w].append(delta)
                rows.append({"seed": seed, "shot": shot, "weight": w, "delta_ap": delta, "fused_ap": r["mean_fused"]["pixel_ap"]})

    summary = {}
    for w in WEIGHTS:
        vals = per_weight[w]
        summary[w] = {
            "mean_delta_ap": float(np_mean(vals)),
            "positive_configs": int(sum(1 for v in vals if v > 0)),
            "min_delta": float(min(vals)),
            "max_delta": float(max(vals)),
        }

    best_w = max(WEIGHTS, key=lambda w: summary[w]["mean_delta_ap"])
    report = {
        "schema_version": 1,
        "run_id": "a1_mpdd_weight_scan_20260817",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "stage": "phase_5_2_weight_scan_pre_freeze",
        "dataset": "mpdd",
        "dataset_role": "development",
        "config": "concat pca_dim=0 whiten=0 (vitb14 DINO + CLIP)",
        "weights": WEIGHTS,
        "per_weight": summary,
        "best_weight_by_mean_delta": best_w,
        "frozen_w0_5_rank": sorted(
            WEIGHTS, key=lambda w: summary[w]["mean_delta_ap"], reverse=True
        ).index(0.5)
        + 1,
        "w0_5_vs_best_gap": round(summary[best_w]["mean_delta_ap"] - summary[0.5]["mean_delta_ap"], 6),
        "rows": rows,
    }
    out = SCAN_ROOT / "weight_scan_summary.json"
    out.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"per_weight": summary, "best": best_w, "w0.5_rank": report["frozen_w0_5_rank"], "gap": report["w0_5_vs_best_gap"]}, indent=2))
    return 0


def np_mean(values) -> float:
    vals = list(values)
    return float(sum(vals) / len(vals))


if __name__ == "__main__":
    raise SystemExit(main())
