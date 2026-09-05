"""Track-3 Probe-E1: efficiency-preservation of A1 under light compression (doc32).

Pure evaluation, no fitting (PCA fit uses only support clean cells), k2.
Reuses v14 support caches only (no t2 needed). 16-grid A1 cell z =
rowL2([0.5*dino_final, 0.5*clip_final]) (1536-D), score s = 1 - max_cos(cell, memory),
memory = other (K-1) clean support images. Pixel-AP per family; scale-free
nuisance AUC (clean h cells vs its 15 photometric variants).

Candidates (all deterministic):
  T0 = A1 full memory (baseline, 256 cells/img x 1536-D)
  T1 = chessboard 50% coreset memory  (keep (i+j)%2==0 -> 128 cells/img)
  T2 = 25% coreset memory             (keep i%2==0 & j%2==0 -> 64 cells/img)
  T3 = support-PCA 1536 -> 384-D (memory & query projected, re-unit)
  T4 = support-PCA 1536 -> 192-D
PCA fit: on all K clean support cells of the cat (support-only, allowed).
Gates: G-T1 (some T: cutpaste >= T0-0.03 and erasure >= T0-0.01);
G-T2 (same T cost <= 0.5x); G-T3 (same T AUC <= 0.60 and -T0 <= +0.05).
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
import torch  # noqa: E402
from sklearn.decomposition import PCA  # noqa: E402
from sklearn.metrics import average_precision_score, roc_auc_score  # noqa: E402

from industrial_ad.innovation_v10_portfolio.common import resize_patches  # noqa: E402

CATEGORIES = ["bracket_black", "bracket_brown", "bracket_white", "connector", "metal_plate", "tubes"]
CACHE = ROOT / "outputs/dynamic_fusion/v14_p1_support"
OUT = ROOT / "experiments/dynamic_fusion/innovation_t3_efficiency_20260905"
SYN = ["cutpaste", "local_erasure"]
T = ("T0", "T1", "T2", "T3", "T4")
G = 16
COST = {"T0": 1.0, "T1": 0.5, "T2": 0.25, "T3": 0.25, "T4": 0.125}
torch.set_num_threads(max(1, torch.get_num_threads()))


def _load_v14(cat, branch):
    z = np.load(CACHE / f"v14_p1_support_{branch}_s0_k2" / f"{cat}.npz", allow_pickle=False)
    return (np.asarray(z["clean_feat"]), np.asarray(z["syn_feat"]),
            np.asarray(z["syn_masks"]), np.asarray(z["nui_feat"]))


def _row_norm(x: torch.Tensor):
    return x / torch.clamp(x.norm(dim=-1, keepdim=True), min=1e-12)


def _pool16_unit(z32):
    """[n,32,32,d] -> [n,16,16,d] unit cells (numpy in/out)."""
    t = torch.as_tensor(z32, dtype=torch.float32)
    p = t.reshape(-1, 16, 2, 16, 2, t.shape[-1]).mean(dim=(2, 4))
    return _row_norm(p.reshape(p.shape[0], G * G, t.shape[-1])).reshape(p.shape[0], G, G, -1).numpy()


def _mask16(m1024):
    m = cv2.resize(m1024.astype(np.uint8), (G, G), interpolation=cv2.INTER_AREA)
    return (m > 127).astype(np.float32)


def load_cells(cat):
    """Return d/c unit 16-grid cells: clean [K,G,G,768], syn [K,9,...], nui [K,15,...], masks."""
    cd, sd, md, nd = _load_v14(cat, "dino")
    cc, sc, _mc, nc = _load_v14(cat, "clip")
    cc = resize_patches(cc.reshape(-1, *cc.shape[1:]), (32, 32)).reshape(cc.shape[0], 32, 32, -1)
    sc = resize_patches(sc.reshape(-1, *sc.shape[2:]), (32, 32)).reshape(*sc.shape[:2], 32, 32, -1)
    nc = resize_patches(nc.reshape(-1, *nc.shape[2:]), (32, 32)).reshape(*nc.shape[:2], 32, 32, -1)
    K = cd.shape[0]

    def pk(x, extra):
        p = _pool16_unit(x)
        return p.reshape(K, extra, G, G, -1)

    masks = np.asarray([[_mask16(md[h, e]) for e in range(9)] for h in range(K)])
    return {"K": K, "masks": masks,
            "d": _pool16_unit(cd), "ds": pk(sd, 9), "dn": pk(nd, 15),
            "c": _pool16_unit(cc), "cs": pk(sc, 9), "cn": pk(nc, 15)}


def a1(zd, zc):
    """A1 fused cell rows [n,1536] unit from 16-grid d/c arrays of same leading shape."""
    n = zd.shape[0]
    d = torch.as_tensor(zd, dtype=torch.float32).reshape(n, -1, 768)
    c = torch.as_tensor(zc, dtype=torch.float32).reshape(n, -1, 768)
    return _row_norm(torch.cat([0.5 * d, 0.5 * c], dim=-1)).numpy().astype(np.float32)


def build_ops(o):
    """Precompute per-candidate query/memory transform helpers once per cat."""
    K = o["K"]
    # clean fused [K,256,1536], syn [K,9,256,1536], nui [K,15,256,1536]
    f_clean = a1(o["d"], o["c"])                      # [K,256,1536]
    f_syn = a1(o["ds"].reshape(-1, G, G, 768), o["cs"].reshape(-1, G, G, 768)).reshape(K, 9, 256, 1536)
    f_nui = a1(o["dn"].reshape(-1, G, G, 768), o["cn"].reshape(-1, G, G, 768)).reshape(K, 15, 256, 1536)
    idx = np.arange(256).reshape(G, G)
    keep_half = ((idx // G + idx % G) % 2 == 0).ravel()      # 128
    keep_q = (((idx // G) % 2 == 0) & ((idx % G) % 2 == 0)).ravel()     # 64
    pca384 = PCA(n_components=384, random_state=0, svd_solver="auto").fit(f_clean.reshape(-1, 1536))
    pca192 = PCA(n_components=192, random_state=0, svd_solver="auto").fit(f_clean.reshape(-1, 1536))
    return {"K": K, "f_clean": f_clean, "f_syn": f_syn, "f_nui": f_nui,
            "masks": o["masks"], "keep_half": keep_half, "keep_q": keep_q,
            "pca384": pca384, "pca192": pca192}


def _unit_t(x):
    return _row_norm(torch.as_tensor(x, dtype=torch.float32))


def apply_T(tname, cells, op):
    """Bank transform [n,256,1536] -> unit rows [n, n_cell, D]. Coreset shrinks bank."""
    if tname == "T1":
        return cells[:, op["keep_half"], :]
    if tname == "T2":
        return cells[:, op["keep_q"], :]
    if tname == "T3":
        return op["pca384"].transform(cells.reshape(-1, 1536)).reshape(*cells.shape[:2], -1).astype(np.float32)
    if tname == "T4":
        return op["pca192"].transform(cells.reshape(-1, 1536)).reshape(*cells.shape[:2], -1).astype(np.float32)
    return cells


def apply_Q(tname, cells, op):
    """Query transform: never coreset (all 256 cells scored), only PCA-projects."""
    if tname in ("T3", "T4"):
        return apply_T(tname, cells, op)
    return cells


def _scores(q, b):
    return 1.0 - (q @ b.T).max(dim=1).values


def _ap(y, s):
    if y.sum() == 0 or y.sum() == y.size:
        return float("nan")
    return float(average_precision_score(y.astype(int), s))


def _mnan(xs):
    xs = [x for x in xs if x is not None and not (isinstance(x, float) and np.isnan(x))]
    return float(np.mean(xs)) if xs else float("nan")


def run_cat(cat):
    op = build_ops(load_cells(cat))
    K = op["K"]
    rows = []
    for h in range(K):
        bank = [k for k in range(K) if k != h]
        for fam in (0, 1):
            for e in range(fam * 3, fam * 3 + 3):
                y = op["masks"][h, e].ravel() > 0
                if y.sum() == 0:
                    continue
                for tname in T:
                    q = apply_Q(tname, op["f_syn"][h, e:e + 1], op)[0]
                    b = apply_T(tname, op["f_clean"][bank], op).reshape(-1, q.shape[-1])
                    b = _unit_t(b)
                    rows.append({"cat": cat, "h": h, "family": SYN[fam], "t": tname,
                                 "ap": _ap(y, _scores(_unit_t(q), b).numpy())})
        vc = op["f_clean"][h]
        for tname in T:
            q0 = apply_Q(tname, vc[None], op)[0]
            b = apply_T(tname, op["f_clean"][bank], op).reshape(-1, q0.shape[-1])
            b = _unit_t(b)
            s_c = _scores(_unit_t(q0), b).numpy()
            qn = apply_Q(tname, op["f_nui"][h], op)      # [15,256,D] (query never coreset)
            s_n = np.concatenate([_scores(_unit_t(qn[e:e + 1])[0], b).numpy() for e in range(15)])
            y = np.concatenate([np.zeros(256), np.ones_like(s_n)])
            s = np.concatenate([s_c, s_n])
            rows.append({"cat": cat, "h": h, "family": "normal", "t": tname,
                         "auc": float(roc_auc_score(y.astype(int), s))})
    return rows


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cats", default=None)
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    cats = [c.strip() for c in args.cats.split(",")] if args.cats else CATEGORIES
    rows = []
    for cat in cats:
        rows += run_cat(cat)
        print(f"  done {cat} k2", flush=True)
    agg = {}
    for tname in T:
        for fam in SYN:
            agg[f"{fam}_{tname}"] = _mnan([r["ap"] for r in rows if r["t"] == tname and r["family"] == fam])
        agg[f"auc_{tname}"] = _mnan([r["auc"] for r in rows if r["t"] == tname and r["family"] == "normal"])
    base = {fam: agg[f"{fam}_T0"] for fam in SYN}
    cands = [t for t in ("T1", "T2", "T3", "T4")]
    ok = {t: (agg[f"cutpaste_{t}"] >= base["cutpaste"] - 0.03 and
              agg[f"local_erasure_{t}"] >= base["local_erasure"] - 0.01) for t in cands}
    cost_ok = {t: (ok[t] and COST[t] <= 0.5) for t in cands}
    auc_ok = {t: (ok[t] and agg[f"auc_{t}"] <= 0.60 and agg[f"auc_{t}"] - agg["auc_T0"] <= 0.05) for t in cands}
    pass_T = [t for t in cands if ok[t] and cost_ok[t] and auc_ok[t]]
    out = {"agg": agg, "cost": COST, "pass_candidates": pass_T,
           "G_T1_hold": ok, "G_T2_cost": cost_ok, "G_T3_auc": auc_ok,
           "decision": "TRACK3_PROBE_PASS" if pass_T else "TRACK3_PROBE_FAIL_ARCHIVE"}
    (OUT / "PROBE_E1_RESULTS.json").write_text(json.dumps(out, ensure_ascii=False, indent=1),
                                               encoding="utf-8")
    print("macro per family/T:", {k: round(v, 4) for k, v in agg.items()})
    print("pass candidates:", pass_T)
    print("DECISION:", out["decision"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
