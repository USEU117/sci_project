"""PRS R0 first mechanism gate (doc 22 s5.3 g1 / doc 26 s4.2) - CPU.

Uses the ALREADY-EXPORTED NTOF intervention ladder on NORMAL support images
(outputs/dynamic_fusion/ntof_features_{dino,clip}_s0_k1): for each ref (normal)
image the 15 fit variants = 5 families x 3 strengths. g1 asks: on normal support,
is the per-patch response to a MONOTONE intensity ladder itself monotone
(rank-correlated with the known strength order)? If not, the 'response spectrum'
has no stable axis to deviate from, and PRS is archived before any GPU export.

Families here: exposure (strengths .70,1.15,1.40) and gamma (.80,1.20,1.50), both
monotone-increasing image transforms. Response per patch: L2 feature displacement
||f(T_a x) - f(x)||, evaluated per encoder grid (dino 32x32, clip 37x37).

Gate (doc 22 s5.3 g1): >=80% of support patches must have Spearman(known strength
order, response) >= 0.6. Computed per category per branch per family; reported
median across 6 categories (as NTOF gates used median across cats).

Diagnostics also report the fraction with |corr|>=0.6 (monotone in EITHER
direction) to distinguish "no axis" from "inverted axis".
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "outputs/dynamic_fusion"
CATEGORIES = ["bracket_black", "bracket_brown", "bracket_white", "connector", "metal_plate", "tubes"]
FAM_KEYS = {"exposure": [0, 1, 2], "gamma": [3, 4, 5]}   # index into the 15 fit variants
STRENGTHS = {"exposure": [0.70, 1.15, 1.40], "gamma": [0.80, 1.20, 1.50]}
GATE_FRAC = 0.80
GATE_CORR = 0.60


def run_category(cat: str) -> dict:
    out = {"category": cat}
    for branch in ("dino", "clip"):
        z = np.load(OUT / f"ntof_features_{branch}_s0_k1" / f"{cat}.npz", allow_pickle=False)
        orig = np.asarray(z["ref_orig_feat"], dtype=np.float32)   # [K,H,W,768]
        var = np.asarray(z["ref_var_feat"], dtype=np.float32)     # [K,15,H,W,768]
        K = orig.shape[0]
        per_cat = {}
        for fam, idxs in FAM_KEYS.items():
            fracs = []
            abs_fracs = []
            mean_resp = []
            for k in range(K):
                f0 = orig[k]
                resp = []
                for j in idxs:
                    dv = var[k, j] - f0
                    resp.append(np.sqrt(np.maximum((dv ** 2).sum(-1), 0.0)))  # [H,W]
                # per-patch Spearman between strength order and response (3 points)
                R = np.stack(resp)                                # [3,H,W]
                order = np.arange(3, dtype=np.float64)
                flat = R.reshape(3, -1)
                corr = np.empty(flat.shape[1])
                for p in range(flat.shape[1]):
                    res = spearmanr(order, flat[:, p])
                    r = getattr(res, "statistic", None)
                    if r is None:
                        r = getattr(res, "correlation", np.nan)
                    corr[p] = r if r == r else 0.0
                frac = float((corr >= GATE_CORR).mean())
                fracs.append(frac)
                abs_fracs.append(float((np.abs(corr) >= GATE_CORR).mean()))
                mean_resp.append(flat.mean(1))                    # [3]
            per_cat[fam] = float(np.mean(fracs)) if fracs else float("nan")
            per_cat[f"{fam}_abs"] = float(np.mean(abs_fracs)) if abs_fracs else float("nan")
            per_cat[f"{fam}_mean_resp"] = (
                np.mean(np.stack(mean_resp), axis=0).round(4).tolist() if mean_resp else [])
        out[branch] = per_cat
        del z
    return out


def main() -> int:
    rows = [run_category(c) for c in CATEGORIES]
    for r in rows:
        print(json.dumps(r), flush=True)
    summary = {"note": "PRS g1 (doc22 s5.3): frac of normal-support patches with "
                       "Spearman(strength-order, response)>=0.6; gate >=0.80 median-across-cats"}
    for branch in ("dino", "clip"):
        for fam in FAM_KEYS:
            vals = [r[branch][fam] for r in rows]
            med = float(np.median(vals))
            summary[f"{branch}_{fam}_median_frac"] = round(med, 4)
            summary[f"{branch}_{fam}_gate_pass"] = bool(med >= GATE_FRAC)
            abs_vals = [r[branch][f"{fam}_abs"] for r in rows]
            summary[f"{branch}_{fam}_median_abs_frac"] = round(float(np.median(abs_vals)), 4)
    print("SUMMARY " + json.dumps(summary), flush=True)
    out_root = ROOT / "experiments/dynamic_fusion/innovation_v12_new_observables/prs"
    out_root.mkdir(parents=True, exist_ok=True)
    (out_root / "PRS_G1.json").write_text(
        json.dumps({"per_category": rows, "summary": summary}, indent=1), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
