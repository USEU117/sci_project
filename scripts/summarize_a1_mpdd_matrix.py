"""Summarize the A1 MPDD 3x3 development matrix into one report (docs 阶段五 5.2)."""

from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

MATRIX_ROOT = ROOT / "experiments" / "dynamic_fusion" / "v3_direction_a" / "a1_matrix_20260817"


def main() -> int:
    seeds = [0, 1, 2]
    shots = [1, 2, 4]
    rows = []
    for seed in seeds:
        for shot in shots:
            path = MATRIX_ROOT / f"seed{seed}_k{shot}" / "concat_pca0_whiten0_w0.5_report.json"
            if not path.is_file():
                raise SystemExit(f"missing report: {path}")
            r = json.loads(path.read_text(encoding="utf-8"))
            rows.append(
                {
                    "seed": seed,
                    "shot": shot,
                    "mean_fused_pixel_ap": r["mean_fused"]["pixel_ap"],
                    "mean_fused_pixel_auroc": r["mean_fused"]["pixel_auroc"],
                    "mean_fused_pixel_aupro": r["mean_fused"]["pixel_aupro"],
                    "mean_dino_baseline_ap": r["mean_dino_baseline_ap"],
                    "mean_delta_ap_vs_dino": r["mean_delta_ap_vs_dino"],
                    "positive_categories": int(
                        sum(1 for c in r["per_category"] if c["delta_ap"] > 0)
                    ),
                    "max_regression": round(
                        float(min(c["delta_ap"] for c in r["per_category"])), 6
                    ),
                    "per_category": {
                        c["category"]: {
                            "fused_ap": c["fused"]["pixel_ap"],
                            "dino_ap": c["baselines"]["anomalydino_visual"]["pixel_ap"],
                            "clip_ap": c["baselines"]["anomalyclip_text"]["pixel_ap"],
                            "delta_ap": c["delta_ap"],
                        }
                        for c in r["per_category"]
                    },
                }
            )

    # by-shot aggregates across seeds
    by_shot = {}
    for shot in shots:
        subset = [r for r in rows if r["shot"] == shot]
        by_shot[shot] = {
            "mean_delta_ap": float(np_mean([r["mean_delta_ap_vs_dino"] for r in subset])),
            "positive_seeds": int(sum(1 for r in subset if r["mean_delta_ap_vs_dino"] > 0)),
        }
    # by-seed aggregates
    by_seed = {}
    for seed in seeds:
        subset = [r for r in rows if r["seed"] == seed]
        by_seed[seed] = {
            "mean_delta_ap": float(np_mean([r["mean_delta_ap_vs_dino"] for r in subset])),
        }
    overall_delta = float(np_mean([r["mean_delta_ap_vs_dino"] for r in rows]))
    overall_positive = int(sum(1 for r in rows if r["mean_delta_ap_vs_dino"] > 0))

    report = {
        "schema_version": 1,
        "run_id": "a1_mpdd_matrix_20260817",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "stage": "phase_5_2_a1_full_development_matrix",
        "dataset": "mpdd",
        "dataset_role": "development",
        "config": "concat pca_dim=0 whiten=0 dino_weight=0.5 (vitb14 DINO + CLIP)",
        "gpu_used": "reference-features-only export only (12 small runs); evaluation CPU/faiss",
        "n_configs": len(rows),
        "overall": {
            "mean_delta_ap_vs_dino": overall_delta,
            "positive_configs": overall_positive,
            "all_positive": overall_positive == len(rows),
        },
        "by_seed": by_seed,
        "by_shot": by_shot,
        "rows": rows,
    }
    out = MATRIX_ROOT / "matrix_summary.json"
    out.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(
        {
            "overall": report["overall"],
            "by_seed": by_seed,
            "by_shot": by_shot,
            "n_rows": len(rows),
        },
        indent=2,
    ))
    return 0


def np_mean(values) -> float:
    vals = list(values)
    return float(sum(vals) / len(vals))


if __name__ == "__main__":
    raise SystemExit(main())
