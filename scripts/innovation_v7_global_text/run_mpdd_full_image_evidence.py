"""Task book 17 - Phase 1 Full-MPDD image-level text evidence.

Runs TEXT p_abn (from the frozen s1 caches; never re-exported per seed) against
A1-max / A1-top1% across MPDD 3 seeds x 3 shots, with:
  - primary macro (mean over categories of per-cat mean over configs)
  - micro pooled (secondary; per-image rows over all config units)
  - discovery (seed0) / confirmation (seed1+seed2) splits reported separately
  - per-config and per-category deltas (TEXT - A1-max)

G1 numeric checks are computed here except the bootstrap CI (run_paired_bootstrap
merges its bootstrap.json into summary.json afterwards).

Run env: .venv-patchcore (CPU).
"""

from __future__ import annotations

import csv
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

from industrial_ad.innovation_v7_global_text import (  # noqa: E402
    EXPERIMENT_ROOT, SEEDS, SHOTS,
)
from industrial_ad.innovation_v7_global_text import assert_development_only  # noqa: E402
from industrial_ad.innovation_v7_global_text.evaluator import image_metrics  # noqa: E402
from industrial_ad.innovation_v7_global_text.scoring import per_config_scores  # noqa: E402

CATS = ["bracket_black", "bracket_brown", "bracket_white",
        "connector", "metal_plate", "tubes"]
OUT = EXPERIMENT_ROOT / "01_mpdd_full"
SIGNALS = ("a1_max", "a1_top1", "text")


def main() -> int:
    assert_development_only()
    OUT.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    rows = []
    cfg_store = {}

    for cat in CATS:
        for seed in SEEDS:
            for shot in SHOTS:
                cfg = per_config_scores(cat, seed, shot)
                cfg_store[(cat, seed, shot)] = cfg
                rec = {"category": cat, "seed": seed, "shot": shot,
                       "n_normal": cfg["n_normal"], "n_anomaly": cfg["n_anomaly"],
                       "n_total": len(cfg["labels"])}
                for sig in SIGNALS:
                    m = image_metrics(cfg[sig], cfg["labels"])
                    rec[f"ap_{sig}"] = m["image_ap"]
                    rec[f"auroc_{sig}"] = m["image_auroc"]
                rec["dap_text"] = rec["ap_text"] - rec["ap_a1_max"]
                rec["dauroc_text"] = rec["auroc_text"] - rec["auroc_a1_max"]
                rows.append(rec)

    # ---------- macro helpers ----------
    def col_mean(rows_, key):
        return float(np.mean([r[key] for r in rows_]))

    per_cat = {}
    for cat in CATS:
        rr = [r for r in rows if r["category"] == cat]
        per_cat[cat] = {
            "mean_dap": round(col_mean(rr, "dap_text"), 4),
            "mean_dauroc": round(col_mean(rr, "dauroc_text"), 4),
            "n_pos_of_9": int(sum(1 for r in rr if r["dap_text"] > 0)),
            "worst_dap": round(min(r["dap_text"] for r in rr), 4),
            "best_dap": round(max(r["dap_text"] for r in rr), 4),
            "mean_ap_a1": round(col_mean(rr, "ap_a1_max"), 4),
            "mean_ap_text": round(col_mean(rr, "ap_text"), 4),
        }

    # per-(seed,shot) macro delta over categories
    config_macro = []
    for seed in SEEDS:
        for shot in SHOTS:
            rr = [r for r in rows if r["seed"] == seed and r["shot"] == shot]
            config_macro.append({
                "seed": seed, "shot": shot,
                "macro_ap_a1": round(float(np.mean([r["ap_a1_max"] for r in rr])), 4),
                "macro_ap_text": round(float(np.mean([r["ap_text"] for r in rr])), 4),
                "macro_dap": round(float(np.mean([r["dap_text"] for r in rr])), 4),
                "macro_dauroc": round(float(np.mean([r["dauroc_text"] for r in rr])), 4),
                "n_pos_of_6": int(sum(1 for r in rr if r["dap_text"] > 0)),
            })

    def mean_over(field):
        return float(np.mean([c[field] for c in config_macro]))

    pooled_macro = {
        "9cfg_macro_dap": round(mean_over("macro_dap"), 4),
        "9cfg_macro_dauroc": round(mean_over("macro_dauroc"), 4),
        "9cfg_n_positive": int(sum(1 for c in config_macro if c["macro_dap"] > 0)),
    }

    # discovery (seed0) vs confirmation (seed1+seed2)
    def subset_delta(seed_list):
        cm = [c for c in config_macro if c["seed"] in seed_list]
        return {"mean_dap": round(float(np.mean([c["macro_dap"] for c in cm])), 4),
                "mean_dauroc": round(float(np.mean([c["macro_dauroc"] for c in cm])), 4),
                "n_pos_of_%d" % len(cm): int(sum(1 for c in cm if c["macro_dap"] > 0)),
                "n_configs": len(cm)}
    disc = subset_delta([0])
    conf = subset_delta([1, 2])

    # micro pooled (secondary): concatenate per-image rows over configs
    micro = {}
    for sig in SIGNALS:
        scores, labels = [], []
        for cat in CATS:
            for seed in SEEDS:
                for shot in SHOTS:
                    cfg = cfg_store[(cat, seed, shot)]
                    scores.append(cfg[sig]); labels.append(cfg["labels"])
        micro[f"ap_{sig}"] = image_metrics(np.concatenate(scores),
                                           np.concatenate(labels))["image_ap"]
    micro["dap_text"] = round(micro["ap_text"] - micro["ap_a1_max"], 4)

    # ---------- G1 pre-checks (bootstrap CI appended later) ----------
    cat_daps = [per_cat[c]["mean_dap"] for c in CATS]
    g1 = {
        "g1a_9cfg_macro_dap_ge_0015": bool(pooled_macro["9cfg_macro_dap"] >= 0.015),
        "g1b_ge7of9_configs_positive": bool(pooled_macro["9cfg_n_positive"] >= 7),
        "g1c_conf6_mean_dap_ge_0010": bool(conf["mean_dap"] >= 0.010),
        "g1c_conf_at_least_4of6_positive": bool(conf["n_pos_of_6"] >= 4),
        "g1e_macro_dauroc_ge_-0.02": bool(pooled_macro["9cfg_macro_dauroc"] >= -0.020),
        "g1f_ge3of6_cats_positive": bool(sum(1 for d in cat_daps if d > 0) >= 3),
        "g1f_worst_cat_ge_-0.100": bool(min(cat_daps) >= -0.100),
        # g1d (bootstrap 95% CI lower > 0) filled by run_paired_bootstrap.py
    }

    report = {
        "program": "innovation_v7_global_text", "phase": "phase1_mpdd_full_image_evidence",
        "dataset": "mpdd", "role": "development",
        "task_book": "17 s.3", "signal": "TEXT p_abn vs A1-max (paired per image)",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "elapsed_total_s": round(time.time() - t0, 1),
        "pooled_macro": pooled_macro,
        "discovery_seed0": disc,
        "confirmation_seed1_2": conf,
        "micro_pooled_secondary": micro,
        "g1_prechecks": g1,
        "note_micro": "micro pools per-image rows over the 9 configs (images recur "
                      "across configs/shot; TEXT per image is config-independent)",
    }

    # csvs
    with open(OUT / "per_config.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader(); w.writerows(rows)
    with open(OUT / "per_category.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["category", *[k for k in per_cat[CATS[0]]]])
        for c in CATS:
            w.writerow([c, *[per_cat[c][k] for k in per_cat[c]]])
    (OUT / "summary.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=1), encoding="utf-8")
    print(json.dumps({"pooled_macro": pooled_macro, "discovery": disc,
                      "confirmation": conf, "micro": micro,
                      "g1_prechecks": g1}, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
