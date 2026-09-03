"""Task book 17 - Phase 1 paired stratified bootstrap for TEXT vs A1-max.

Primary statistic: macro mean Delta Image-AP over the 9 (seed,shot) configs
(mean over 6 categories of per-category AP(TEXT) - AP(A1-max)); also the same
for Delta Image-AUROC.

Resampling (s.3.4): per category, resample normal images and anomaly images
WITH replacement (stratified, same sizes); the SAME image indices are used for
TEXT and A1 (paired). 10,000 iterations; percentile 95% CI + median + P(d>0).
Sensitivity: category-cluster bootstrap (resample the 6 categories, 2,000 iters).
Auxiliary (not independent): sign test & Wilcoxon over the 9 config macro deltas.

Deterministic: fixed rng seed -> fully reproducible.
Run env: .venv-patchcore (CPU). Output 01_mpdd_full/bootstrap.json.
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

from industrial_ad.innovation_v7_global_text import (  # noqa: E402
    EXPERIMENT_ROOT, SEEDS, SHOTS,
)
from industrial_ad.innovation_v7_global_text import assert_development_only  # noqa: E402
from industrial_ad.innovation_v7_global_text.scoring import per_config_scores  # noqa: E402

CATS = ["bracket_black", "bracket_brown", "bracket_white",
        "connector", "metal_plate", "tubes"]
OUT = EXPERIMENT_ROOT / "01_mpdd_full"
B = 10000
B_CLUSTER = 2000
RNG_SEED = 20260903


def ap_binary(y: np.ndarray, s: np.ndarray) -> float:
    """Average precision for a single config image subset (fast, no sklearn)."""
    y = np.asarray(y, dtype=np.int64)
    s = np.asarray(s, dtype=np.float64)
    n_pos = int(y.sum())
    if n_pos == 0 or n_pos == len(y):
        return float("nan")
    order = np.argsort(-s, kind="mergesort")
    ys = y[order]
    tp = np.cumsum(ys)
    prec = tp / np.arange(1, len(ys) + 1)
    return float((prec[ys == 1]).sum() / n_pos)


def main() -> int:
    assert_development_only()
    OUT.mkdir(parents=True, exist_ok=True)
    t0 = time.time()

    # Precompute per-image score arrays: cfg_scores[(cat,seed,shot)] aligned
    cells = {}
    for cat in CATS:
        for seed in SEEDS:
            for shot in SHOTS:
                cfg = per_config_scores(cat, seed, shot)
                cells[(cat, seed, shot)] = {
                    "y": cfg["labels"],
                    "s_text": cfg["text"],
                    "s_a1": cfg["a1_max"],
                }
    configs = [(seed, shot) for seed in SEEDS for shot in SHOTS]

    def macro_delta_for(idx_by_cat: dict) -> tuple:
        """Given per-category selected image indices, return 9-config macro
        mean DeltaAP and DeltaAUROC (lists of per-config macro deltas)."""
        per_cfg_dap, per_cfg_dauroc = [], []
        for seed, shot in configs:
            daps, daurocs = [], []
            for cat in CATS:
                c = cells[(cat, seed, shot)]
                idx = idx_by_cat[cat]
                dap = ap_binary(c["y"][idx], c["s_text"][idx]) - \
                      ap_binary(c["y"][idx], c["s_a1"][idx])
                daps.append(dap)
                # AUROC delta via rank-based trapezoid
                y = c["y"][idx]
                if (y == 1).sum() and (y == 0).sum():
                    r1 = _auc(c["s_text"][idx], y)
                    r0 = _auc(c["s_a1"][idx], y)
                    daurocs.append(r1 - r0)
            per_cfg_dap.append(float(np.nanmean(daps)))
            per_cfg_dauroc.append(float(np.nanmean(daurocs)))
        return float(np.mean(per_cfg_dap)), float(np.mean(per_cfg_dauroc)), \
            per_cfg_dap, per_cfg_dauroc

    def _auc(s, y):
        order = np.argsort(-s, kind="mergesort")
        ys = y[order]
        n0 = int((ys == 0).sum())
        n1 = int((ys == 1).sum())
        if n0 == 0 or n1 == 0:
            return float("nan")
        ranks = np.arange(1, len(ys) + 1)
        return (ranks[ys == 1].sum() - n1 * (n1 + 1) / 2.0) / (n1 * n0)

    # observed per-cat normal/anomaly counts
    counts = {}
    for cat in CATS:
        y = cells[(cat, 0, 1)]["y"]
        counts[cat] = {"n0": int((y == 0).sum()), "n1": int((y == 1).sum())}

    rng = np.random.default_rng(RNG_SEED)
    dap_iters = np.empty(B)
    dauroc_iters = np.empty(B)
    cfg_sign_iters = np.empty((B, len(configs)))
    t_start = time.time()
    for b in range(B):
        idx_by_cat = {}
        for cat in CATS:
            n0, n1 = counts[cat]["n0"], counts[cat]["n1"]
            y = cells[(cat, 0, 1)]["y"]
            norm_pos = np.flatnonzero(y == 0)
            anom_pos = np.flatnonzero(y == 1)
            i_n = rng.integers(0, n0, n0)
            i_a = rng.integers(0, n1, n1)
            idx_by_cat[cat] = np.r_[norm_pos[i_n], anom_pos[i_a]]
        dap, dauroc, cfg_dap, _ = macro_delta_for(idx_by_cat)
        dap_iters[b] = dap
        dauroc_iters[b] = dauroc
        cfg_sign_iters[b] = np.asarray(cfg_dap) > 0
        if b % 2000 == 0:
            print(f"iter {b}/{B}  mean_dap={dap_iters[:b+1].mean():.4f} "
                  f"({time.time()-t_start:.0f}s)", flush=True)

    def ci(x):
        return tuple(float(v) for v in np.percentile(x, [2.5, 97.5]))

    primary = {
        "b": B, "seed": RNG_SEED,
        "mean_dap": round(float(dap_iters.mean()), 4),
        "median_dap": round(float(np.median(dap_iters)), 4),
        "ci95_dap": [round(v, 4) for v in ci(dap_iters)],
        "p_dap_gt0": round(float((dap_iters > 0).mean()), 4),
        "mean_dauroc": round(float(dauroc_iters.mean()), 4),
        "ci95_dauroc": [round(v, 4) for v in ci(dauroc_iters)],
        "p_dauroc_gt0": round(float((dauroc_iters > 0).mean()), 4),
        "g1d_ci95_lower_gt_0": bool(ci(dap_iters)[0] > 0),
    }

    # per-config positive-rate over bootstrap iters
    per_cfg = {}
    for j, (seed, shot) in enumerate(configs):
        per_cfg[f"s{seed}_k{shot}"] = {
            "p_dap_gt0_bootstrap": round(float(cfg_sign_iters[:, j].mean()), 4)}

    # category-cluster bootstrap (sensitivity)
    rng2 = np.random.default_rng(RNG_SEED + 1)
    cl_dap = np.empty(B_CLUSTER)
    for b in range(B_CLUSTER):
        cats_s = [CATS[i] for i in rng2.integers(0, len(CATS), len(CATS))]
        idx_by_cat = {}
        for cat in cats_s:
            n0, n1 = counts[cat]["n0"], counts[cat]["n1"]
            y = cells[(cat, 0, 1)]["y"]
            i_n = rng2.integers(0, n0, n0)
            i_a = rng2.integers(0, n1, n1)
            idx_by_cat[cat] = np.r_[np.flatnonzero(y == 0)[i_n],
                                    np.flatnonzero(y == 1)[i_a]]
        dap, _, _, _ = macro_delta_for(idx_by_cat)
        cl_dap[b] = dap
    cluster = {"b": B_CLUSTER,
               "mean_dap": round(float(cl_dap.mean()), 4),
               "ci95_dap": [round(v, 4) for v in ci(cl_dap)],
               "p_dap_gt0": round(float((cl_dap > 0).mean()), 4)}

    # aux: observed 9-config macro deltas (computed from full data)
    full_idx = {cat: np.arange(counts[cat]["n0"] + counts[cat]["n1"])
                for cat in CATS}
    obs_dap, obs_dauroc, cfg_dap_obs, _ = macro_delta_for(full_idx)
    cfg_macro_obs = {f"s{s}_k{k}": round(float(d), 4)
                     for (s, k), d in zip(configs, cfg_dap_obs)}

    aux = {}
    try:
        from scipy.stats import wilcoxon, binomtest
        w = wilcoxon(cfg_dap_obs, alternative="greater")
        n_pos = int(sum(1 for d in cfg_dap_obs if d > 0))
        bt = binomtest(n_pos, len(cfg_dap_obs), 0.5, alternative="greater")
        aux = {"wilcoxon_p_one_sided": round(float(w.pvalue), 5),
               "sign_test_n_pos_of_9": n_pos,
               "sign_test_p": round(float(bt.pvalue), 5),
               "note": "auxiliary only; configs share test images (not independent)"}
    except Exception as e:  # noqa: BLE001
        aux = {"error": str(e)}

    report = {
        "program": "innovation_v7_global_text",
        "phase": "phase1_paired_bootstrap",
        "task_book": "17 s.3.4",
        "primary_statistic": "macro mean over 9 configs of per-cat mean "
                             "DeltaImageAP / DeltaImageAUROC (paired stratified)",
        "observed_9cfg_macro_dap": round(float(obs_dap), 4),
        "observed_9cfg_macro_dauroc": round(float(obs_dauroc), 4),
        "per_config_observed_macro_dap": cfg_macro_obs,
        "primary": primary,
        "per_config_bootstrap_p_positive": per_cfg,
        "category_cluster_bootstrap": cluster,
        "auxiliary_tests": aux,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "elapsed_total_s": round(time.time() - t0, 1),
        "hist_dap_bins": [round(float(v), 4) for v in
                          np.histogram(dap_iters, bins=50,
                                       range=(-0.10, 0.20))[0].tolist()],
    }
    (OUT / "bootstrap.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=1), encoding="utf-8")
    print(json.dumps({k: report[k] for k in (
        "observed_9cfg_macro_dap", "observed_9cfg_macro_dauroc", "primary",
        "category_cluster_bootstrap", "auxiliary_tests")}, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
