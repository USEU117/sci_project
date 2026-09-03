"""S1-HGLC (task book 16 s.3.3 items 3-5) - global-gate calibration on the A1
pixel maps, using the single global signal that passed the image-level gate
(AnomalyCLIP abnormal text probability, pooled Image-AP delta +0.0249).

Frozen formula (doc 16 s.3.2), same global gate for every pixel of the image:
    z_final(x,p) = z_A1(x,p) + beta * ReLU(z_global(x) - tau_g) * h(z_A1(x,p))
  z_A1   = raw frozen A1 concat 448 map (higher = more anomalous), used as-is
  z_glob = text abnormal probability p_abn (zero-shot, config-independent)
  tau_g  = 0.5 fixed a priori (probability tie point; not fit on any data)
  h      = z/(1+z)  or  top-q mask (q = 0.10), both fixed forms (doc)
  beta   = {0.1, 0.25, 0.5} fixed grid, not per-category

Evaluator side (development only): GT masks at 448 (maps.gt_masks_for) and the
frozen stride-8 flattened Pixel-AP protocol (maps.pixel_metrics_448).

Acceptance (doc item 4): pooled Pixel-AP gain over A1 >= +0.005 AND worst
category (per-cat mean over shots) >= -0.02, else archive S1-HGLC.

Controls (doc item 5): A1 alone; shuffled global gate (fixed seed 0) at the
winning beta; direct multiplicative gate; h = top-q mask variant.
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

OUT = maps.EXPERIMENT_ROOT / "s1_hglc"
CACHE = OUT / "cache"
BETAS = (0.1, 0.25, 0.5)
TAU_G = 0.5
TOP_Q = 0.10
SHOTS = (1, 2, 4)
SEED = 0


def h_z1pz(z: np.ndarray) -> np.ndarray:
    return z / (1.0 + np.maximum(z, 0.0))


def h_topq(z: np.ndarray) -> np.ndarray:
    """Per-map top-q mask (q=0.10 of pixels)."""
    k = max(1, int(round(TOP_Q * z.shape[1])))
    thr = np.partition(z, z.shape[1] - k, axis=1)[:, z.shape[1] - k][:, None]
    return (z >= thr).astype(np.float64)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=SEED)
    ap.add_argument("--shots", nargs="+", type=int, default=list(SHOTS))
    ap.add_argument("--betas", nargs="+", type=float, default=list(BETAS))
    ap.add_argument("--shuffle-seed", type=int, default=0)
    args = ap.parse_args()
    maps.assert_development_only()
    OUT.mkdir(parents=True, exist_ok=True)

    exp = json.loads((CACHE / "s1_hglc_export_report.json").read_text(encoding="utf-8"))
    cats = sorted(c["category"] for c in exp["categories"])

    # ---- gather per (cat, shot) z_A1 maps, GT masks and text gates ----
    cells = []  # dicts: cat, shot, zA1 (N,448,448), gtm (N,448,448), gate (N,)
    for cat in cats:
        cache = np.load(CACHE / f"{cat}.npz", allow_pickle=False)
        cache_ids = [str(s) for s in cache["sample_ids"]]
        text_p = np.asarray(cache["text_prob_test"], dtype=np.float64)
        for shot in args.shots:
            a1 = maps.load_a1_patch_map(cat, args.seed, shot)
            ids = [str(s) for s in a1["sample_ids"]]
            perm = maps.align_perm(np.asarray(cache_ids), np.asarray(ids))
            zA1 = maps.a1_maps448(a1["patch_map"])
            gtm = maps.gt_masks_for(np.asarray(ids))
            cells.append({"cat": cat, "shot": shot,
                          "zA1": zA1, "gtm": gtm,
                          "gate": text_p[perm]})
    t0 = time.time()

    def pixel_ap(z: np.ndarray, gtm: np.ndarray) -> float | None:
        return maps.pixel_metrics_448(z, gtm)["pixel_ap"]

    # per-image gate values with ReLU(z_glob - tau)
    for c in cells:
        c["g"] = np.maximum(c["gate"] - TAU_G, 0.0)

    # ---- baseline + variants ----
    def delta_for(variant_fn) -> dict:
        ap_b = np.asarray([pixel_ap(c["zA1"], c["gtm"]) for c in cells])
        ap_v = np.asarray([pixel_ap(v, c["gtm"]) for v, c in zip(variant_fn(), cells)])
        # per-cell delta, then per-cat mean over shots, then pooled mean
        per_cat = {}
        for ci, c in enumerate(cells):
            per_cat.setdefault(c["cat"], []).append(float(ap_v[ci] - ap_b[ci]))
        per_cat_m = {k: float(np.mean(v)) for k, v in per_cat.items()}
        return {"pooled_delta": round(float(np.mean(list(per_cat_m.values()))), 4),
                "worst_category": round(float(min(per_cat_m.values())), 4),
                "worst_category_name": min(per_cat_m, key=per_cat_m.get),
                "per_category_delta": {k: round(v, 4) for k, v in per_cat_m.items()},
                "n_cells": len(cells)}

    base_ap = {}
    for ci, c in enumerate(cells):
        base_ap[f"{c['cat']}|{c['shot']}"] = pixel_ap(c["zA1"], c["gtm"])
    report_base_ap = base_ap

    results = {}
    hforms = {"z1pz": h_z1pz, "topq": h_topq}
    winners = {}
    for hname, hfn in hforms.items():
        best = None
        for beta in args.betas:
            def var(hfn=hfn, beta=beta):
                return [c["zA1"] + beta * c["g"][:, None, None] * hfn(c["zA1"])
                        for c in cells]
            r = delta_for(var)
            results[f"{hname}_b{beta}"] = r
            if best is None or r["pooled_delta"] > best["pooled_delta"]:
                best = r
                winners[hname] = beta
        results[f"{hname}_best"] = best

    # ---- controls at each h-form winning beta ----
    controls = {}
    for hname, hfn in hforms.items():
        beta = winners[hname]
        # (a) shuffled image-level gate (fixed seed)
        rng = np.random.default_rng(args.shuffle_seed)
        shuf_g = {c["cat"]: rng.permutation(c["g"]) for c in cells}
        def var_shuf():
            return [c["zA1"] + beta * shuf_g[c["cat"]][:, None, None] * hfn(c["zA1"])
                    for c in cells]
        # (b) direct multiplicative: zA1 * (1 + beta*g)
        def var_mult():
            return [c["zA1"] * (1.0 + beta * c["g"][:, None, None])
                    for c in cells]
        controls[f"{hname}_shuffled_gate_b{beta}"] = delta_for(var_shuf)
        controls[f"{hname}_multiplicative_b{beta}"] = delta_for(var_mult)

    verdict = {}
    for hname in hforms:
        b = winners[hname]
        rb = results[f"{hname}_best"]
        verdict[hname] = {
            "winning_beta": b,
            "pooled_delta_pixel_ap": rb["pooled_delta"],
            "worst_category_delta": rb["worst_category"],
            "accept": bool(rb["pooled_delta"] >= 0.005 and rb["worst_category"] >= -0.02),
        }

    report = {
        "program": "innovation_v6_dgsafe", "phase": "s1_hglc_calibration",
        "dataset": "mpdd", "role": "development", "seed": args.seed,
        "task_book_section": "16 s.3.3 items 3-5",
        "formula": "z_final = z_A1 + beta * ReLU(p_abn - 0.5) * h(z_A1)",
        "tau_g": TAU_G, "top_q": TOP_Q, "betas": list(args.betas),
        "shuffle_seed": args.shuffle_seed,
        "pixel_ap_protocol": "448 stride-8 pooled per config (maps.pixel_metrics_448)",
        "pooled_convention": "mean over categories of per-cat mean over shots",
        "a1_baseline_pixel_ap_per_config": {k: (None if v is None else round(v, 4))
                                            for k, v in report_base_ap.items()},
        "results_by_variant": results,
        "controls": controls,
        "verdict": verdict,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "elapsed_total_s": round(time.time() - t0, 1),
    }
    (OUT / "S1_CALIB.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=1), encoding="utf-8")

    lines = ["# S1-HGLC calibration on A1 maps (doc 16 s.3.3 items 3-5)", "",
             "z_final = z_A1 + beta * ReLU(p_abn - 0.5) * h(z_A1);  "
             "beta grid {0.1,0.25,0.5}; h in {z/(1+z), top-q(0.10)}", ""]
    for hname in hforms:
        b = winners[hname]
        rb = results[f"{hname}_best"]
        ctrl = controls[f"{hname}_shuffled_gate_b{b}"]
        ctrl_m = controls[f"{hname}_multiplicative_b{b}"]
        lines += [
            f"- h={hname}: best beta={b}  pooled dPixel-AP={rb['pooled_delta']:+.4f}  "
            f"worst cat={rb['worst_category']:+.4f} ({rb['worst_category_name']})  "
            f"accept={verdict[hname]['accept']}",
            f"    control shuffled-gate: {ctrl['pooled_delta']:+.4f}  |  "
            f"control multiplicative: {ctrl_m['pooled_delta']:+.4f}",
        ]
    lines += ["", "Details: S1_CALIB.json", ""]
    (OUT / "S1_CALIB_DECISION.md").write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines))
    return 0 if any(v["accept"] for v in verdict.values()) else 2


if __name__ == "__main__":
    raise SystemExit(main())
