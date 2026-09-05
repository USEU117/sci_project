"""W0.1 PRS response-axis definition audit (doc 27 s4.1; CPU, <=20 min).

Question: r(a)=||f(T_a x)-f(x)|| is an unsigned displacement from the IDENTITY
input, while the NTOF ladder exposure [0.70,1.15,1.40] / gamma [0.80,1.20,1.50]
was ordered by raw parameter value. exposure is multiplicative (x*s), gamma is
x**(1/g), identity == 1.0 for both. So brightness-monotone != distance-from-
identity monotone: the V-shaped response cannot by itself prove "no intensity
axis". This AUDITS the axis definition; it does not convert the archived PRS
G1 FAIL into PASS, and re-ordering the 3 points is not anomaly separability.

Output columns per ladder point: family | a | |log(a)| | pixel RMS(T_a x, x)
| feature response (dino/clip mean |r| over support patches, from PRS_G1.json)
| saturated-pixel fraction (0/255 clip). 3 points only -> descriptive, no
reliable curve claim.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts" / "innovation_v12_new_observables"))
from ntof_render import apply_exposure, apply_gamma  # noqa: E402

FAMILIES = {"exposure": [0.70, 1.15, 1.40], "gamma": [0.80, 1.20, 1.50]}
IDENTITY = 1.0


def main() -> int:
    # representative normal image (bracket_black shot-k1 ref) for pixel RMS
    z = np.load(ROOT / "outputs/dynamic_fusion/v12_early_fusion/ml_dino_s0_k1/bracket_black.npz",
                allow_pickle=False)
    # masks/sample ids only; need an actual RGB image -> use manifest refs
    del z
    manifest = json.loads((ROOT / "data/splits/mpdd/manifest.json").read_text(encoding="utf-8"))
    rel = manifest["categories"]["bracket_black"]["0"]["1"][0]
    img = cv2.cvtColor(cv2.imread(str(ROOT / "data/mpdd_raw/MPDD" / rel)), cv2.COLOR_BGR2RGB)
    img = img.astype(np.float32)

    prs = json.loads((ROOT / "experiments/dynamic_fusion/innovation_v12_new_observables/prs"
                      / "PRS_G1.json").read_text(encoding="utf-8"))
    # feature response per family: median over the 6 cats of the 3-strength mean resp
    def med_resp(branch, fam):
        rows = prs["per_category"]
        key = f"{fam}_mean_resp"
        vals = [r[branch][key] for r in rows if key in r[branch]]
        return [float(np.median([v[k] for v in vals])) for k in range(3)] if vals else None

    rows = []
    for fam, params in FAMILIES.items():
        fn = apply_exposure if fam == "exposure" else apply_gamma
        for k, a in enumerate(params):
            out = fn(img.astype(np.uint8), a).astype(np.float32)
            rms = float(np.sqrt(np.mean((out - img) ** 2)))
            sat = float(np.mean((out <= 1.0) | (out >= 254.0)))   # 0/255 clip fraction
            dr = med_resp("dino", fam)[k] if med_resp("dino", fam) else float("nan")
            cr = med_resp("clip", fam)[k] if med_resp("clip", fam) else float("nan")
            rows.append({"family": fam, "a": a,
                         "log_abs_ratio": round(float(np.log(a / IDENTITY)), 4),
                         "pixel_rms": round(rms, 2),
                         "dino_mean_resp": round(dr, 3),
                         "clip_mean_resp": round(cr, 3),
                         "clip_frac_01_254": round(sat, 4)})
            print(json.dumps(rows[-1]), flush=True)

    out_root = ROOT / "experiments/dynamic_fusion/innovation_v13_overnight_20260904"
    out_root.mkdir(parents=True, exist_ok=True)
    md = ["# W0.1 PRS response-axis definition audit (2026-09-04)",
          "",
          "r(a)=||f(T_a x)-f(x)|| is displacement from the IDENTITY transform; the fit ladder",
          "orders by raw parameter a, but identity==1 for both exposure (x*s) and gamma (x**(1/g)).",
          "|log(a/1)| = distance-from-identity in transform-parameter space.",
          "",
          "| family | a | |log(a/1)| | pixel RMS | dino mean resp | clip mean resp | clip(0/255) frac |",
          "|---|---|---:|---:|---:|---:|---:|"]
    for r in rows:
        md.append(f"| {r['family']} | {r['a']} | {r['log_abs_ratio']:.4f} | {r['pixel_rms']} | "
                  f"{r['dino_mean_resp']} | {r['clip_mean_resp']} | {r['clip_frac_01_254']} |")
    md += ["", "Notes: pixel RMS (on one bracket_black k1 ref image at 1024) is NOT monotone in the",
           "raw ladder order either: the near-identity middle point is smallest. Feature responses",
           "(median over 6 cats, PRS_G1.json) show the same V shape. The archived G1 FAIL (rank corr",
           "vs raw order) stays FAIL; an axis re-parameterised as distance-from-identity is only",
           "descriptive with 3 points and is not anomaly separability. 3 points -> no curve claim.",
           "Saturation: clip frac is the fraction of pixels hitting 0/255 after the transform."]
    (out_root / "W0_AUDITS.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    (out_root / "W0_prs_axis_audit.json").write_text(
        json.dumps({"note": "axis audit; PRS G1 FAIL not re-opened", "rows": rows},
                   ensure_ascii=False, indent=1), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
