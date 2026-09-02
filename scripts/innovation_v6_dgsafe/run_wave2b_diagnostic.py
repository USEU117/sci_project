"""S0 Wave 2b - GT-based check whether normal-only reliability explains real risk
(task book 16 ss.2.4/2.7 Wave 2).

Inputs (evaluator-side, GT allowed here - this is the diagnostic stage):
  reliability/reliability_raw.json   (Wave 2a output: per-config r_sub, risk ranks)
  Wave1_complementarity/WAVE1_DIAGNOSTIC.json  (per-config SUB/A1 448 Pixel-AP)
Optional:
  --rerun-same-seed <json>  Wave 2a rerun file to verify max err < 1e-7
  --rerun-aug-seed  <json>  Wave 2a rerun with different aug seed for Kendall tau

Pass conditions (all must hold):
  1. Spearman rho(r_sub, SUB-A1 delta_PixelAP) >= +0.40  (18 configs, seed 0)
  2. connector r_sub lies in the bottom quartile of the 18 configs (no cat name
     used in the reliability computation itself)
  3. same-seed rerun max error < 1e-7  (checked only if rerun file provided)
  4. category-level order Kendall tau >= 0.60 vs aug-seed rerun (checked only if
     aug-seed rerun file provided)

Writes Wave2_complementarity/WAVE2_DECISION.json + WAVE2_DECISION.md.
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
for p in (str(ROOT), str(ROOT / "src"), str(ROOT / "scripts")):
    if p not in sys.path:
        sys.path.insert(0, p)

from industrial_ad.innovation_v6_dgsafe import maps  # noqa: E402

WAVE2_OUT = maps.EXPERIMENT_ROOT / "Wave2_complementarity"


def _stat(res):
    return float(res.statistic if hasattr(res, "statistic") else res.correlation)


def spearman(x, y):
    from scipy.stats import spearmanr
    return _stat(spearmanr(x, y)) if len(x) > 2 else float("nan")


def kendall(x, y):
    from scipy.stats import kendalltau
    return _stat(kendalltau(x, y))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--reliability-json", type=Path,
                    default=maps.EXPERIMENT_ROOT / "reliability" / "reliability_raw.json")
    ap.add_argument("--wave1-json", type=Path,
                    default=maps.EXPERIMENT_ROOT / "Wave1_complementarity"
                    / "WAVE1_DIAGNOSTIC.json")
    ap.add_argument("--rerun-same-seed", type=Path, default=None)
    ap.add_argument("--rerun-aug-seed", type=Path, default=None)
    args = ap.parse_args()

    maps.assert_development_only()
    WAVE2_OUT.mkdir(parents=True, exist_ok=True)

    reliab = json.loads(args.reliability_json.read_text(encoding="utf-8"))
    wave1 = json.loads(args.wave1_json.read_text(encoding="utf-8"))
    rel = {f"{r['category']}_{r['shot']}": r for r in reliab["reliability"]}
    # delta = SUB - A1 per (cat, shot) at 448 (Wave 1 rows)
    delta = {}
    for r in wave1["rows"]:
        delta[f"{r['category']}_{r['shot']}"] = r["sub_pixel_ap"] - r["a1_pixel_ap"]

    cats = sorted({k.split("_")[0] for k in rel})
    shots = sorted({int(k.split("_")[1]) for k in rel})
    keys = [f"{c}_{k}" for c in cats for k in shots]
    r_sub = np.asarray([rel[kk]["r_sub"] for kk in keys], dtype=np.float64)
    d = np.asarray([delta[kk] for kk in keys], dtype=np.float64)

    rho = spearman(r_sub.tolist(), d.tolist())
    cond1 = rho >= 0.40
    # connector bottom-quartile (r_sub at most the 25th percentile of the 18)
    q25 = float(np.quantile(r_sub, 0.25))
    con_keys = [kk for kk in keys if kk.startswith("connector_")]
    con_rsub = [rel[kk]["r_sub"] for kk in con_keys]
    cond2 = bool(con_rsub) and all(x <= q25 for x in con_rsub)

    checks = {
        "n_configs": len(keys),
        "spearman_r_sub_vs_delta": round(rho, 4),
        "cond1_rho_ge_040": bool(cond1),
        "connector_r_sub": {kk.split("_")[1]: rel[kk]["r_sub"] for kk in con_keys},
        "connector_bottom_quartile_threshold_q25": round(q25, 4),
        "cond2_connector_bottom_quartile": bool(cond2),
    }

    # optional determinism / ordering checks
    if args.rerun_same_seed is not None:
        r2 = json.loads(args.rerun_same_seed.read_text(encoding="utf-8"))
        r2map = {f"{r['category']}_{r['shot']}": r for r in r2["reliability"]}
        errs = {kk: abs(rel[kk]["r_sub"] - r2map[kk]["r_sub"]) for kk in keys}
        errs_scalar = {kk: max(abs(rel[kk][s] - r2map[kk][s])
                               for s in ("u_aug", "u_layer", "b_tail"))
                       for kk in keys}
        max_err = max(max(errs.values()), max(errs_scalar.values()))
        checks["rerun_same_seed_max_err"] = round(max_err, 10)
        checks["cond3_same_seed_rerun_lt_1e-7"] = bool(max_err < 1e-7)
    if args.rerun_aug_seed is not None:
        r3 = json.loads(args.rerun_aug_seed.read_text(encoding="utf-8"))
        r3map = {f"{r['category']}_{r['shot']}": r for r in r3["reliability"]}
        # category-level ordering: mean r_sub per category from both runs
        def cat_order(m):
            out = {}
            for c in cats:
                out[c] = float(np.mean([m[f'{c}_{k}']['r_sub'] for k in shots]))
            return out
        o1, o3 = cat_order(rel), cat_order(r3map)
        c1 = np.asarray([o1[c] for c in cats]); c3 = np.asarray([o3[c] for c in cats])
        checks["rerun_aug_seed_category_kendall_tau"] = round(kendall(c1, c3), 4)
        checks["cond4_aug_seed_kendall_tau_ge_060"] = bool(checks[
            "rerun_aug_seed_category_kendall_tau"] >= 0.60)

    conds = ["cond1_rho_ge_040", "cond2_connector_bottom_quartile"]
    if args.rerun_same_seed is not None:
        conds.append("cond3_same_seed_rerun_lt_1e-7")
    if args.rerun_aug_seed is not None:
        conds.append("cond4_aug_seed_kendall_tau_ge_060")
    passed = all(checks[c] for c in conds)
    checks["wave2_passed"] = bool(passed)

    report = {
        "program": "innovation_v6_dgsafe", "phase": "Wave2_normal_only_reliability",
        "dataset": "mpdd", "role": "development", "seed": 0,
        "protocol": "r_sub (Wave2a, normal-only, frozen formula) vs SUB-A1 "
                    "448 Pixel-AP delta (Wave1 rows); connector bottom-quartile check",
        "per_config": [{"config": kk, "r_sub": rel[kk]["r_sub"],
                        "delta_sub_a1": round(float(delta[kk]), 5)}
                       for kk in keys],
        "checks": checks,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    (WAVE2_OUT / "WAVE2_DECISION.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=1), encoding="utf-8")
    lines = [
        "# WAVE2 DECISION (machine-drafted; S0 route gating)",
        "",
        f"- **passed**: {passed}",
        f"- cond1 Spearman rho(r_sub, delta) >= 0.40: {checks.get('cond1_rho_ge_040')} "
        f"(rho={checks.get('spearman_r_sub_vs_delta')})",
        f"- cond2 connector r_sub bottom quartile: "
        f"{checks.get('cond2_connector_bottom_quartile')} "
        f"(connector={checks.get('connector_r_sub')}, q25={checks.get('connector_bottom_quartile_threshold_q25')})",
    ]
    if "cond3_same_seed_rerun_lt_1e-7" in checks:
        lines.append(f"- cond3 same-seed rerun max err < 1e-7: "
                     f"{checks.get('cond3_same_seed_rerun_lt_1e-7')} "
                     f"(max={checks.get('rerun_same_seed_max_err')})")
    if "cond4_aug_seed_kendall_tau_ge_060" in checks:
        lines.append(f"- cond4 aug-seed category Kendall tau >= 0.60: "
                     f"{checks.get('cond4_aug_seed_kendall_tau_ge_060')} "
                     f"(tau={checks.get('rerun_aug_seed_category_kendall_tau')})")
    lines += ["", "Details: WAVE2_DECISION.json", ""]
    (WAVE2_OUT / "WAVE2_DECISION.md").write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines))
    return 0 if passed else 2


if __name__ == "__main__":
    sys.exit(main())
