"""P2-A image-domain capacity transfer gate (doc28 s6.1), 16-grid, CPU.

Uses the P1 support caches (variants were rendered on SUPPORT images at 1024 and
re-encoded through the frozen extractors => real image-domain interventions with
boundary/resampling/context effects, unlike the V13 feature-token copies).

Per (cat, shot, held-out support h): anchors = pooled-16 cells of the OTHER K-1
support images' CLEAN features; queries = image h's clean / syn (cutpaste=0,
erasure=1, scratch=2) / photometric-nuisance episodes. Methods: free NN baseline,
per-row entropy soft (free), DINO soft-capacity, CLIP soft-capacity, concat
soft-capacity. Score = capacity premium (semi-OT expected row cost minus free
soft row cost). Report inside/ring/far premium per mask episode, nuisance p95/max,
mismatch (dino-cost vs other-seed clip features).

Frozen: eps=0.05, tau=4.0, rho uniform over anchors, grid16 (mean-pool of 32).
Gates (doc28 s6.1/s6.3, frozen, macro over cats x h):
  A1 capacity transfer: mean concat inside premium over syn (copy/erase/scratch)
     minus mean clean-image premium p95 >= +0.02 (anomaly region pays capacity cost
     above normal background 95th percentile);
  A2 spillover: syn far (and ring) mean premium minus clean p95 <= +0.02 (no
     far-background lift accompanying the inside signal);
  A3 nuisance: photometric nuisance prem p95 minus clean prem p95 <= +0.02 and
     image-level max within +0.02 of clean max;
  A4 monotonicity: Spearman(mask_area, concat inside premium) over syn episodes > 0
     (premium rises with anomaly size);
  A5 pairing: matched concat inside premium >= mismatch-paired concat inside premium
     (real cross-branch pair wins over a mismatched one);
  per-branch dino/clip rows recorded (single-branch capacity path CAP-D) but the
  decision uses concat for the cross-branch claim.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
for p in (str(ROOT), str(ROOT / "src"), str(ROOT / "scripts")):
    sys.path.insert(0, p)
sys.path.insert(0, str(ROOT / "scripts" / "innovation_v14_decisive_validation_20260905"))

import cv2  # noqa: E402

from semi_ot import capacity_premium, spillover  # noqa: E402
from industrial_ad.innovation_v10_portfolio.common import resize_patches  # noqa: E402

CATEGORIES = ["bracket_black", "bracket_brown", "bracket_white", "connector", "metal_plate", "tubes"]
CACHE = ROOT / "outputs/dynamic_fusion/v14_p1_support"
OUT = ROOT / "experiments/dynamic_fusion/innovation_v14_decisive_validation_20260905/P2_soft_capacity"
EPS = 0.05
TAU = 4.0
N_RING = 2


def _load(cat, shot, branch):
    z = np.load(CACHE / f"v14_p1_support_{branch}_s0_k{shot}" / f"{cat}.npz", allow_pickle=False)
    return (np.asarray(z["clean_feat"]), np.asarray(z["syn_feat"]),
            np.asarray(z["syn_masks"]), np.asarray(z["nui_feat"]))


def _pool16(x):
    """[..., 32, 32, d] -> [..., 16, 16, d] via 2x2 mean over the two '2' axes."""
    shp = x.shape
    return x.reshape(*shp[:-3], 16, 2, 16, 2, -1).mean(axis=(-4, -2))


def _mask16(m1024):
    m = cv2.resize(m1024.astype(np.uint8), (16, 16), interpolation=cv2.INTER_AREA)
    return (m > 127).astype(bool).ravel()


def _row_norm(x):
    x = np.asarray(x, dtype=np.float64)
    f = x.reshape(-1, x.shape[-1])
    return f / np.maximum(np.linalg.norm(f, axis=1, keepdims=True), 1e-8)


def _cos(q, a):
    return 1.0 - q @ a.T


def run_cat_shot(cat, shot):
    cd, sd, md, nd = _load(cat, shot, "dino")
    cc, sc, _mc, nc = _load(cat, shot, "clip")
    cc = resize_patches(cc.reshape(-1, *cc.shape[1:]), (32, 32)).reshape(cc.shape[0], 32, 32, -1)
    sc = resize_patches(sc.reshape(-1, *sc.shape[2:]), (32, 32)).reshape(*sc.shape[:2], 32, 32, -1)
    nc = resize_patches(nc.reshape(-1, *nc.shape[2:]), (32, 32)).reshape(nc.shape[0], 15, 32, 32, -1)
    cd, sd, nd = _pool16(cd), _pool16(sd), _pool16(nd)
    cc, sc, nc = _pool16(cc), _pool16(sc), _pool16(nc)
    K = cd.shape[0]
    rows = []
    for h in range(K):
        oth = [k for k in range(K) if k != h]
        aD = _row_norm(np.concatenate([cd[k] for k in oth]))          # [(K-1)*256, 768]
        aC = _row_norm(np.concatenate([cc[k] for k in oth]))
        aF = _row_norm(np.concatenate([0.5 * aD, 0.5 * aC], axis=-1))
        A = aD.shape[0]
        rho = np.full(A, 1.0 / A)
        a = np.full(256, 1.0 / 256)
        lb_cache = {}   # warm-start per method (same A/rho for all probes of this h)

        def probe(dq, cq, kind, seed_e, mask=None):
            qD = _row_norm(dq.reshape(-1, 768))
            qC = _row_norm(cq.reshape(-1, 768))
            qF = _row_norm(np.concatenate([0.5 * qD, 0.5 * qC], axis=-1))
            CD, CC, CF = _cos(qD, aD), _cos(qC, aC), _cos(qF, aF)
            res = {"kind": kind, "e": seed_e}
            for name, C in (("dino", CD), ("clip", CC), ("concat", CF)):
                prem, _, free, st = capacity_premium(C, a, rho, eps=EPS, tau=TAU,
                                                     lb0=lb_cache.get(name))
                lb_cache[name] = st["lb"]
                sp = spillover(mask, prem, grid=(16, 16)) if mask is not None else None
                res[f"{name}_inside"] = sp["inside"] if sp else float("nan")
                res[f"{name}_ring"] = sp["ring"] if sp else float("nan")
                res[f"{name}_far"] = sp["far"] if sp else float("nan")
                res[f"{name}_free_p95"] = float(np.percentile(free, 95))
                res[f"{name}_prem_p95"] = float(np.percentile(prem, 95))
                res[f"{name}_prem_max"] = float(prem.max())
            return res

        # clean control
        rows.append({"cat": cat, "shot": shot, "h": h, **probe(cd[h], cc[h], "clean", -1)})
        # syn episodes (kind-major: 0 cutpaste, 1 erasure, 2 scratch)
        for e in range(sd.shape[1]):
            m = _mask16(md[h, e])
            kind = ["cutpaste", "local_erasure", "thin_scratch"][e // 3]
            rows.append({"cat": cat, "shot": shot, "h": h,
                         "mask_area": float(m.mean()),
                         **probe(sd[h, e], sc[h, e], f"syn_{kind}", e, mask=m)})
        # nuisance episodes (no mask)
        for e in range(nd.shape[1]):
            rows.append({"cat": cat, "shot": shot, "h": h,
                         **probe(nd[h, e], nc[h, e], "nuisance", e)})
        # mismatch control: dino episode feature vs clip feature from a DIFFERENT episode
        for e in range(sd.shape[1]):
            e2 = (e + 3) % sd.shape[1]
            m = _mask16(md[h, e])
            r = probe(sd[h, e], sc[h, e2], "mismatch", e, mask=m)
            rows.append({"cat": cat, "shot": shot, "h": h, "mask_area": float(m.mean()), **r})
    return rows


def _mean_nan(xs):
    xs = [x for x in xs if x is not None and not (isinstance(x, float) and np.isnan(x))]
    return float(np.mean(xs)) if xs else float("nan")


def aggregate(rows, out_json):
    syn = [r for r in rows if r["kind"].startswith("syn_")]
    mismatch = [r for r in rows if r["kind"] == "mismatch"]
    clean = [r for r in rows if r["kind"] == "clean"]
    nui = [r for r in rows if r["kind"] == "nuisance"]
    g = {}
    for m in ("dino", "clip", "concat"):
        clean_p95 = _mean_nan([r[f"{m}_prem_p95"] for r in clean])
        clean_max = _mean_nan([r[f"{m}_prem_max"] for r in clean])
        nui_p95 = _mean_nan([r[f"{m}_prem_p95"] for r in nui])
        nui_max = _mean_nan([r[f"{m}_prem_max"] for r in nui])
        syn_in = _mean_nan([r[f"{m}_inside"] for r in syn])
        syn_ring = _mean_nan([r[f"{m}_ring"] for r in syn])
        syn_far = _mean_nan([r[f"{m}_far"] for r in syn])
        mis_in = _mean_nan([r[f"{m}_inside"] for r in mismatch])
        areas = np.asarray([r["mask_area"] for r in syn])
        ins = np.asarray([r[f"{m}_inside"] for r in syn])
        ok = np.isfinite(ins)
        corr = float(np.corrcoef(areas[ok], ins[ok])[0, 1]) if ok.sum() > 2 and ins[ok].std() > 0 else float("nan")
        g[m] = {"clean_prem_p95": clean_p95, "clean_prem_max": clean_max,
                "syn_inside": syn_in, "syn_ring": syn_ring, "syn_far": syn_far,
                "nui_prem_p95": nui_p95, "nui_prem_max": nui_max,
                "mismatch_inside": mis_in,
                "inside_vs_clean_p95": (syn_in - clean_p95) if np.isfinite([syn_in, clean_p95]).all() else float("nan"),
                "far_vs_clean_p95": (syn_far - clean_p95) if np.isfinite([syn_far, clean_p95]).all() else float("nan"),
                "ring_vs_clean_p95": (syn_ring - clean_p95) if np.isfinite([syn_ring, clean_p95]).all() else float("nan"),
                "nui_p95_vs_clean_p95": (nui_p95 - clean_p95) if np.isfinite([nui_p95, clean_p95]).all() else float("nan"),
                "nui_max_vs_clean_max": (nui_max - clean_max) if np.isfinite([nui_max, clean_max]).all() else float("nan"),
                "area_corr": corr}
    c = g["concat"]
    a1 = c["inside_vs_clean_p95"] >= 0.02
    a2 = c["far_vs_clean_p95"] <= 0.02 and c["ring_vs_clean_p95"] <= 0.02
    a3 = c["nui_p95_vs_clean_p95"] <= 0.02 and c["nui_max_vs_clean_max"] <= 0.02
    a4 = c["area_corr"] > 0.0
    a5 = (c["syn_inside"] - c["mismatch_inside"]) >= 0.0
    single_branch = any(
        (g[m]["inside_vs_clean_p95"] >= 0.02 and g[m]["far_vs_clean_p95"] <= 0.02)
        for m in ("dino", "clip"))
    decision = "P2A_PASS" if (a1 and a2 and a3 and a4 and a5) else "P2A_FAIL_IMAGE_DOMAIN"
    out = {"per_method": g, "A1_capacity_transfer": a1, "A2_spillover": a2,
           "A3_nuisance": a3, "A4_monotonic": a4, "A5_matched_gt_mismatch": a5,
           "single_branch_cap_d_ok": single_branch, "decision": decision}
    Path(out_json).write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cats", default=None)
    ap.add_argument("--shots", type=int, nargs="+", default=[2])
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    cats = [c.strip() for c in args.cats.split(",")] if args.cats else CATEGORIES
    all_rows = []
    for shot in args.shots:
        for cat in cats:
            all_rows += run_cat_shot(cat, shot)
            print(f"  done {cat} k{shot}", flush=True)
    (OUT / "IMAGE_PROBE.json").write_text(
        json.dumps({"rows": all_rows, "config": {"eps": EPS, "tau": TAU}},
                   ensure_ascii=False, indent=1), encoding="utf-8")
    gates = aggregate(all_rows, OUT / "P2A_GATES.json")
    print("rows:", len(all_rows), flush=True)
    print("\n==== P2-A GATES (concat) ====")
    c = gates["per_method"]["concat"]
    print(f"A1 transfer {gates['A1_capacity_transfer']}  inside-clean_p95={c['inside_vs_clean_p95']:.4f}")
    print(f"A2 spillover {gates['A2_spillover']}  far={c['far_vs_clean_p95']:.4f} ring={c['ring_vs_clean_p95']:.4f}")
    print(f"A3 nuisance  {gates['A3_nuisance']}  p95={c['nui_p95_vs_clean_p95']:.4f} max={c['nui_max_vs_clean_max']:.4f}")
    print(f"A4 monotonic {gates['A4_monotonic']}  corr={c['area_corr']:.3f}")
    print(f"A5 pairing   {gates['A5_matched_gt_mismatch']}  diff={c['syn_inside'] - c['mismatch_inside']:.4f}")
    print("single-branch CAP-D ok:", gates["single_branch_cap_d_ok"])
    print("DECISION:", gates["decision"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
