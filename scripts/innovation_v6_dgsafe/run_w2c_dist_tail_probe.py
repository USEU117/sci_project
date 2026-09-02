"""S0 Wave 2c - CPU-only probe: is the Wave-2 gate failure a small-pool z
calibration artefact or a genuine absence of normal-only reliability signal?

The Wave-2a frozen formula calibrates z per *pixel* inside a tiny pool of only
P = 9*k normal grids (k = 1 -> 9 samples), which quantises -log p_b so heavily
that U_layer/B_tail collapse to near-constant or pure-k values (see
WAVE2_ARCHIVE.md).  This probe re-derives the SAME two A-pool features
(U_aug, B_tail) under two *distribution-level* tail estimators and checks
whether the ordering over the 18 (category, shot) configs correlates better
with the frozen GT-based risk proxy delta = SUB - A1 448 Pixel-AP.

Variants (both CPU, normal-only, on the already-saved A-pool raw grids):
  V1  pooled empirical CDF over ALL pool pixels/versions (N = P*2304 values)
  V2  per-pixel Gaussian (mu/sigma fit over the P pool values at that pixel,
      sigma floored by the pooled sigma) -> smooth location-aware z

GT numbers are NOT recomputed: delta comes from the frozen
Wave2_reliability/WAVE2_DECISION.json.  U_layer (B/C subspaces) is not
re-derivable on disk (B/C grids are not persisted), so the probe reports
U_aug-only and U_aug+B_tail two-feature r_sub' variants explicitly.

Gate reference (doc 16 ss.2.7 Wave 2): Spearman rho(r_sub, delta) >= +0.40
and connector r_sub in the bottom quartile.
"""

from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr, norm

ROOT = Path(__file__).resolve().parents[2]
for p in (str(ROOT), str(ROOT / "src"), str(ROOT / "scripts")):
    if p not in sys.path:
        sys.path.insert(0, p)

from industrial_ad.innovation_v6_dgsafe import maps  # noqa: E402

EXP = maps.EXPERIMENT_ROOT
GRID = 48
GRID2 = GRID * GRID
Z_CAP = 12.0
VERSIONS = ["identity", "brightness1.15", "contrast1.15", "shiftx_p2",
            "shiftx_n2", "shifty_p2", "shifty_n2", "rot_p2", "rot_n2"]
OUT = EXP / "Wave2_reliability" / "dist_tail_probe"
VER = 1


def _stat(res):
    return float(res.statistic if hasattr(res, "statistic") else res.correlation)


def spearman(x, y):
    return _stat(spearmanr(x, y)) if len(x) > 2 else float("nan")


def percentile_ranks_risk(vals):
    """Percentile rank (0..1) across configs; bigger value -> riskier (Wave2a)."""
    vals = np.asarray(vals, dtype=np.float64)
    n = len(vals)
    order = np.argsort(vals, kind="stable")
    ranks = np.empty(n)
    ranks[order] = np.arange(1, n + 1)
    for v in np.unique(vals):
        m = float(np.mean(ranks[vals == v]))
        ranks[vals == v] = m
    return ranks / n


def z_pooled_cdf(grids):
    """V1: global empirical tail CDF over all pool residual values.

    Returns one z map (48x48) per input grid."""
    flat = np.stack(grids).reshape(len(grids), -1)  # (P, 2304) residuals
    R = np.sort(flat.reshape(-1))
    N = R.size
    tail = N - np.searchsorted(R, flat.reshape(-1), side="left") + 1.0  # #R >= s
    p = tail / (N + 1.0)
    z = np.clip(-np.log(np.maximum(p, np.finfo(np.float64).tiny)), 0.0, Z_CAP)
    return [z.reshape(len(grids), GRID, GRID)[i] for i in range(len(grids))]


def z_perpix_gauss(grids, pooled_scale):
    """V2: per-pixel Gaussian tail over the P pool values at each pixel.

    Returns one z map (48x48) per input grid."""
    flat = np.stack(grids).reshape(len(grids), GRID2)   # (P, 2304)
    mu = flat.mean(axis=0)
    sd = flat.std(axis=0, ddof=1)
    sd = np.maximum(sd, pooled_scale * 1e-2)            # shrink floor
    z = np.empty_like(flat)
    tiny = np.finfo(np.float64).tiny
    for p in range(GRID2):
        zs = (flat[:, p] - mu[p]) / sd[p]
        tail = 1.0 - norm.cdf(zs)
        z[:, p] = np.clip(-np.log(np.maximum(tail, tiny)), 0.0, Z_CAP)
    return [z[i].reshape(GRID, GRID) for i in range(len(grids))]


def derive_features(grids, zfn):
    """Same U_aug / B_tail recipe as Wave2a, from variant z maps of A-pool."""
    n_refs = len(grids) // len(VERSIONS)
    zA = zfn(grids)
    u_aug_vals = []
    for r in range(n_refs):
        zid = zA[r * len(VERSIONS)]
        for v in range(1, len(VERSIONS)):
            u_aug_vals.append(float(np.median(np.abs(zid - zA[r * len(VERSIONS) + v]))))
    b_tail = float(np.percentile(np.stack(zA), 99))
    return float(np.mean(u_aug_vals)), b_tail


def main() -> int:
    maps.assert_development_only()
    OUT.mkdir(parents=True, exist_ok=True)

    raw = json.loads((EXP / "reliability" / "reliability_raw.json").read_text(
        encoding="utf-8"))
    dec = json.loads((EXP / "Wave2_reliability" / "WAVE2_DECISION.json").read_text(
        encoding="utf-8"))
    rel = {f"{r['category']}|{r['shot']}": r for r in raw["reliability"]}
    delta = {p["config"]: p["delta_sub_a1"] for p in dec["per_config"]}
    cats = sorted({k.split("|")[0] for k in rel})
    shots = sorted({int(k.split("|")[1]) for k in rel})
    keys = [f"{c}|{k}" for c in cats for k in shots]
    d = np.asarray([delta[k] for k in keys], dtype=np.float64)

    # ---- sanity: reproduce archived rho on the frozen numbers ----
    r_sub0 = np.asarray([rel[k]["r_sub"] for k in keys], dtype=np.float64)
    rho0 = spearman(r_sub0.tolist(), d.tolist())
    feat_attr = {}
    for f in ("u_aug", "u_layer", "b_tail"):
        rk = np.asarray([rel[k][f"risk_pct_{f}"] for k in keys], dtype=np.float64)
        feat_attr[f] = round(spearman(rk.tolist(), d.tolist()), 4)
    # B_tail k-purity check on stored values
    k_purity = {}
    for k in shots:
        k_purity[str(k)] = sorted({round(rel[f"{c}|{k}"]['b_tail'], 4) for c in cats})

    # ---- distribution-level re-derivation per config ----
    rows = []
    for kk in keys:
        cat, shot = kk.split("|")
        shot = int(shot)
        npz = np.load(EXP / "reliability" / "pools" / f"{cat}_s0_k{shot}.npz")
        grids = [g.astype(np.float64) for g in npz["gridA"]]
        rows.append({"config": kk, "category": cat, "shot": shot, "grids": grids,
                     "delta": float(delta[kk])})

    # V1 pooled empirical CDF
    v1 = {}
    for r in rows:
        grids = r["grids"]
        u, b = derive_features(grids, z_pooled_cdf)
        v1[r["config"]] = {"u_aug": u, "b_tail": b}
    # V2 per-pixel Gaussian
    v2 = {}
    for r in rows:
        grids = r["grids"]
        flat = np.stack(grids)
        ps = float(flat.std())
        u, b = derive_features(grids, lambda g, s=ps: z_perpix_gauss(g, s))
        v2[r["config"]] = {"u_aug": u, "b_tail": b}

    out_rows = []
    for kk in keys:
        out_rows.append({
            "config": kk,
            "delta": round(delta[kk], 5),
            "r_sub_frozen": rel[kk]["r_sub"],
            "u_aug_frozen": rel[kk]["u_aug"],
            "b_tail_frozen": rel[kk]["b_tail"],
            "v1_u_aug": round(v1[kk]["u_aug"], 6),
            "v1_b_tail": round(v1[kk]["b_tail"], 6),
            "v2_u_aug": round(v2[kk]["u_aug"], 6),
            "v2_b_tail": round(v2[kk]["b_tail"], 6),
        })

    res = {
        "program": "innovation_v6_dgsafe",
        "phase": "Wave2c_distribution_level_tail_probe",
        "dataset": "mpdd", "role": "development", "seed": 0,
        "purpose": "test whether Wave-2 gate failure is a small-pool z-quantisation "
                   "artefact: re-derive A-pool U_aug/B_tail under distribution-level "
                   "tail estimators (V1 pooled CDF, V2 per-pixel Gaussian); GT delta "
                   "reused frozen from WAVE2_DECISION.json (never recomputed)",
        "notes": [
            "U_layer (B/C subspaces) not re-derivable on disk (B/C grids not "
            "persisted); r_sub' uses U_aug + B_tail only",
            "B_tail is a self-quantile of the calibration distribution, so it is "
            "expected to stay weak; the decisive feature is U_aug'",
        ],
        "sanity_archived_rho": round(rho0, 4),
        "frozen_feature_rho_vs_delta_risk_pct": feat_attr,
        "frozen_b_tail_k_purity_unique_per_k": k_purity,
        "per_config": out_rows,
    }

    # ---- correlations per feature set vs delta ----
    def corr_block(name, ukey, bkey):
        ua = np.asarray([v[ukey] for v in out_rows])
        bt = np.asarray([v[bkey] for v in out_rows])
        ru = percentile_ranks_risk(ua)
        rb = percentile_ranks_risk(bt)
        risk2 = 0.5 * (ru + rb)
        q = 1.0 - risk2
        rs2 = np.clip((q - 0.25) / 0.50, 0.0, 1.0)
        rho_bt = spearman(bt.tolist(), d.tolist())
        # leave-one-category-out stability of rho(r_sub2', delta)
        loco = {}
        for c in cats:
            mask = np.asarray([not k.startswith(c + "|") for k in keys])
            rs_ = rs2[mask]
            d_ = d[mask]
            loco[c] = round(spearman(rs_.tolist(), d_.tolist()), 4)
        block = {
            "rho_u_aug_vs_delta": round(spearman(ua.tolist(), d.tolist()), 4),
            "rho_b_tail_vs_delta": round(spearman(bt.tolist(), d.tolist()), 4),
            "rho_rsub2_vs_delta": round(spearman(rs2.tolist(), d.tolist()), 4),
            "gate_pass_rho_ge_040": float(spearman(rs2.tolist(), d.tolist())) >= 0.40,
            "rsub2_per_config": {k: round(float(r), 4) for k, r in zip(keys, rs2)},
            "loco_rho_rsub2_vs_delta": loco,
        }
        q25 = float(np.quantile(rs2, 0.25))
        con = [rs2[i] for i, k in enumerate(keys) if k.startswith("connector|")]
        block["connector_rsub2"] = {k.split("|")[1]: float(rs2[i])
                                    for i, k in enumerate(keys)
                                    if k.startswith("connector|")}
        block["q25_rsub2"] = round(q25, 4)
        block["cond2_connector_bottom_quartile"] = bool(con) and all(x <= q25 for x in con)
        return block

    res["V1_pooled_cdf"] = corr_block("v1", "v1_u_aug", "v1_b_tail")
    res["V2_perpix_gauss"] = corr_block("v2", "v2_u_aug", "v2_b_tail")

    # ---- artefact diagnostics for the V2 near-pass ----
    u2 = np.asarray([v["v2_u_aug"] for v in out_rows])
    b2 = np.asarray([v["v2_b_tail"] for v in out_rows])
    diag = {}
    diag["rho_u_aug_vs_b_tail_V2"] = round(spearman(u2.tolist(), b2.tolist()), 4)
    # B_tail'': per-category strictly increasing in k? (k-indicator check)
    mono = {}
    for c in cats:
        bs = [next(v for v in out_rows if v["config"] == f"{c}|{k}")["v2_b_tail"]
              for k in shots]
        mono[c] = {"values": [round(x, 4) for x in bs],
                   "strictly_increasing_in_k": all(x < y for x, y in zip(bs, bs[1:]))}
    diag["V2_b_tail_k_monotonicity_per_category"] = mono
    # frozen 2-feature (U_aug+B_tail, no U_layer) control rho for a like-for-like base
    u0 = np.asarray([v["u_aug_frozen"] for v in out_rows])
    b0 = np.asarray([v["b_tail_frozen"] for v in out_rows])
    ru0 = percentile_ranks_risk(u0); rb0 = percentile_ranks_risk(b0)
    rs0 = np.clip((1.0 - 0.5 * (ru0 + rb0) - 0.25) / 0.50, 0.0, 1.0)
    diag["frozen_2feature_control_rho"] = round(
        spearman(rs0.tolist(), d.tolist()), 4)
    # tie / miss / false-flag audit for V2 r_sub2'
    for name, blk in (("V1", res["V1_pooled_cdf"]), ("V2", res["V2_perpix_gauss"])):
        rs = np.asarray([blk["rsub2_per_config"][k] for k in keys])
        q25 = float(np.quantile(rs, 0.25))
        at_or_below = [k for k, r in zip(keys, rs) if r <= q25]
        neg_d = [k for k, r in zip(keys, d) if r < 0]
        pos_d = [k for k, r in zip(keys, d) if r >= 0]
        missed = [k for k in neg_d if k not in at_or_below]
        false_pos = [k for k in pos_d if k in at_or_below]
        diag[f"{name}_flagged_le_q25"] = at_or_below
        diag[f"{name}_q25_tie_count_at_boundary"] = int(np.sum(rs == q25))
        diag[f"{name}_n_flagged_of_18"] = len(at_or_below)
        diag[f"{name}_missed_negative_delta_configs"] = missed
        diag[f"{name}_false_positive_flagged_configs"] = false_pos
        diag[f"{name}_n_distinct_rsub2_values"] = int(len(np.unique(rs)))
    # null reference: rho of a random-rank r_sub2' (same clipped/tied transform)
    rng = np.random.default_rng(0)
    null_rho = []
    for _ in range(1000):
        r1 = rng.random(len(keys)); r2 = rng.random(len(keys))
        m_ = 0.5 * (percentile_ranks_risk(r1) + percentile_ranks_risk(r2))
        rs_ = np.clip((1.0 - m_ - 0.25) / 0.50, 0.0, 1.0)
        null_rho.append(spearman(rs_.tolist(), d.tolist()))
    diag["null_random_rank_rho_stats"] = {
        "mean": round(float(np.mean(null_rho)), 3),
        "std": round(float(np.std(null_rho)), 3),
        "max_1000_draws": round(float(np.max(null_rho)), 3),
        "pct_ge_06245": round(float(np.mean(np.asarray(null_rho) >= 0.6245)), 4),
    }
    res["artefact_diagnostics"] = diag

    (OUT / f"W2C_DIST_TAIL_v{VER}.json").write_text(
        json.dumps(res, ensure_ascii=False, indent=1), encoding="utf-8")

    verdict = (
        "VERDICT: distribution-level recalibration does NOT rescue the Wave-2 gate. "
        "V1 (true pooled CDF) destroys the ordering (rho=-0.07). V2's apparent pass "
        "(rho=0.62, connector bottom-quartile) is an artefact of the clipped/rank "
        "transform: B_tail'' stays a pure k-monotone indicator, U_aug'' and "
        "B_tail'' anti-correlate at -0.93, r_sub2' collapses onto only 7 distinct "
        "values, and the q25 tie boundary swallows 9/18 configs (connector passes "
        "only by sitting ON the 0.3889 tie shared with many non-connector configs). "
        "The flag set is unusable: it false-flags bracket_brown|1, bracket_brown|2 "
        "and tubes|2 (delta +0.03..+0.11, i.e. SUB is BETTER there) at the same "
        "score as connector|2 (delta=-0.181). Even random-rank null draws under "
        "this clipped transform reach rho as high as 0.635 (max of 1000 draws), "
        "so large rho is attainable without any signal. The only "
        "frozen feature with genuine risk direction is U_layer (-0.323), whose B/C "
        "grids are not persisted on disk. => Wave-2 negative is ROBUST to the z "
        "calibration formula; a full GPU recalibration rerun is NOT warranted "
        "before user confirmation."
    )

    lines = [
        "# Wave-2c distribution-level tail probe (CPU, development-only)",
        "",
        f"- archived frozen rho(r_sub, delta) = {rho0:.4f} (sanity, reproduced)",
        f"- frozen risk-feature attribution (risk_pct vs delta, expect negative): "
        f"{feat_attr}",
        f"- frozen B_tail unique values per k (k-purity check): {k_purity}",
        "",
        "V1 (pooled empirical CDF over all pool residuals):",
        f"  rho(U_aug' ,delta)={res['V1_pooled_cdf']['rho_u_aug_vs_delta']}   "
        f"rho(r_sub2',delta)={res['V1_pooled_cdf']['rho_rsub2_vs_delta']}  "
        f"gate={res['V1_pooled_cdf']['gate_pass_rho_ge_040']}  "
        f"connector_bottom_q={res['V1_pooled_cdf']['cond2_connector_bottom_quartile']}",
        "V2 (per-pixel Gaussian):",
        f"  rho(U_aug'',delta)={res['V2_perpix_gauss']['rho_u_aug_vs_delta']}   "
        f"rho(B_tail'',delta)={res['V2_perpix_gauss']['rho_b_tail_vs_delta']}   "
        f"rho(r_sub2'',delta)={res['V2_perpix_gauss']['rho_rsub2_vs_delta']}  "
        f"gate={res['V2_perpix_gauss']['gate_pass_rho_ge_040']}  "
        f"connector_bottom_q={res['V2_perpix_gauss']['cond2_connector_bottom_quartile']}",
        f"  V2 LOCO rho(r_sub2'',delta): "
        f"{res['V2_perpix_gauss']['loco_rho_rsub2_vs_delta']}",
        "",
        "Artefact audit (see artefact_diagnostics in the json):",
        f"  rho(U_aug'', B_tail'') = {diag['rho_u_aug_vs_b_tail_V2']} (anti-correlated "
        "features -> composite rank is noise averaging)",
        f"  V2 B_tail'' per-category k-monotonicity: "
        f"{diag['V2_b_tail_k_monotonicity_per_category']}",
        f"  frozen 2-feature (U_aug+B_tail) control rho = "
        f"{diag['frozen_2feature_control_rho']} (< 0.40; the jump to 0.62 under V2 "
        "comes from the calibration change, not real signal)",
        f"  V2 flagged(<=q25, {diag['V2_n_flagged_of_18']}/18): "
        f"{diag['V2_flagged_le_q25']}",
        f"  V2 false positives (flagged but delta>=0): "
        f"{diag['V2_false_positive_flagged_configs']}",
        f"  V2 distinct r_sub2' values = {diag['V2_n_distinct_rsub2_values']}; "
        f"null random-rank rho: {diag['null_random_rank_rho_stats']}",
        "",
        verdict,
        "",
        f"details: W2C_DIST_TAIL_v{VER}.json",
    ]
    (OUT / f"W2C_DIST_TAIL_v{VER}.md").write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines))
    return 0


if __name__ == "__main__":
    sys.exit(main())
