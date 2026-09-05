"""Track-2 Probe-M1: do mid layers add usable complementarity over final A1?

doc31 s3/s4. Pure evaluation, no fitting, k2 (declared first shot).
For each cat & held-out image h: memory = other (K-1) clean support images at a
16-grid; held-out episodes scored per candidate by 1-max_cos on the fused
descriptor; Pixel-AP by family; scale-free nuisance AUC (clean h cells vs its 15
photometric variants) per candidate.

Candidates (each: row-L2 unit layers, 0.5-weighted concat per A1 style, L2):
  M0 = A1: dino_final + clip_final                (v14 cache)
  M1 = M0 + dino_mid(block5)                      (t2 dm_*)
  M2 = M0 + clip_mid(resblock6)                   (t2 c6_*)
  M3 = M0 + dino_mid + clip_mid(resblock12)       (t2 dm_* + c12_*)
Gates: G-M1 cutpaste macro best - M0 >= +0.05; G-M2 erasure macro >= -0.01;
G-M3 nuisance-AUC best - M0 <= +0.05 and <= 0.60.
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
from sklearn.metrics import average_precision_score, roc_auc_score  # noqa: E402

from industrial_ad.innovation_v10_portfolio.common import resize_patches  # noqa: E402

CATEGORIES = ["bracket_black", "bracket_brown", "bracket_white", "connector", "metal_plate", "tubes"]
CACHE = ROOT / "outputs/dynamic_fusion/v14_p1_support"
T2 = ROOT / "outputs/dynamic_fusion/t2_multilayer_support"
OUT = ROOT / "experiments/dynamic_fusion/innovation_t2_multilayer_20260905"
SYN = ["cutpaste", "local_erasure", "thin_scratch"]
CAND = ("M0", "M1", "M2", "M3")
LAYERS = ("d", "c", "dm", "c6", "c12")   # clean suffix "", syn "s", nui "n"
G = 16
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


def load_cell(cat):
    cd, sd, md, nd = _load_v14(cat, "dino")
    cc, sc, _mc, nc = _load_v14(cat, "clip")
    cc = resize_patches(cc.reshape(-1, *cc.shape[1:]), (32, 32)).reshape(cc.shape[0], 32, 32, -1)
    sc = resize_patches(sc.reshape(-1, *sc.shape[2:]), (32, 32)).reshape(*sc.shape[:2], 32, 32, -1)
    nc = resize_patches(nc.reshape(-1, *nc.shape[2:]), (32, 32))
    K = cd.shape[0]
    nc = nc.reshape(K, 15, 32, 32, -1)
    t2 = np.load(T2 / "k2" / f"{cat}.npz", allow_pickle=False)
    dm_c, dm_s, dm_n = (np.asarray(t2[k]) for k in ("dm_clean", "dm_syn", "dm_nui"))
    c6_c, c6_s, c6_n = (np.asarray(t2[k]) for k in ("c6_clean", "c6_syn", "c6_nui"))
    c12_c, c12_s, c12_n = (np.asarray(t2[k]) for k in ("c12_clean", "c12_syn", "c12_nui"))

    def to32(f):
        if f.ndim == 4:  # clean [K, 37, 37, D]
            return resize_patches(f.reshape(-1, *f.shape[1:]), (32, 32)).reshape(f.shape[0], 32, 32, -1)
        return resize_patches(f.reshape(-1, *f.shape[2:]), (32, 32)).reshape(f.shape[0], f.shape[1], 32, 32, -1)  # [K,15,...]

    c6_c, c6_s, c6_n = to32(c6_c), to32(c6_s), to32(c6_n)
    c12_c, c12_s, c12_n = to32(c12_c), to32(c12_s), to32(c12_n)
    masks = np.asarray([[_mask16(md[h, e]) for e in range(9)] for h in range(K)])
    o = {"K": K, "masks": masks}

    def pk(x, extra):
        # _pool16_unit flattens leading axes (K, extra) -> (K*extra); restore (K, extra, G, G, D)
        p = _pool16_unit(x)
        return p.reshape(K, extra, G, G, -1)

    # clean (suffix ""), syn ("s"), nui ("n") per layer
    o["d"], o["ds"], o["dn"] = _pool16_unit(cd), pk(sd, 9), pk(nd, 15)
    o["c"], o["cs"], o["cn"] = _pool16_unit(cc), pk(sc, 9), pk(nc, 15)
    o["dm"], o["dms"], o["dmn"] = _pool16_unit(dm_c), pk(dm_s, 9), pk(dm_n, 15)
    o["c6"], o["c6s"], o["c6n"] = _pool16_unit(c6_c), pk(c6_s, 9), pk(c6_n, 15)
    o["c12"], o["c12s"], o["c12n"] = _pool16_unit(c12_c), pk(c12_s, 9), pk(c12_n, 15)
    return o


def _unit(img):
    return _row_norm(torch.as_tensor(img, dtype=torch.float32).reshape(G * G, -1))


def cand_vectors(img):
    """img: {layer: [G,G,768] clean-grid} -> {cand: [G*G,D] unit}."""
    d, c = _unit(img["d"]), _unit(img["c"])
    m0 = _row_norm(torch.cat([0.5 * d, 0.5 * c], dim=-1))
    dm, c6, c12 = _unit(img["dm"]), _unit(img["c6"]), _unit(img["c12"])
    return {"M0": m0,
            "M1": _row_norm(torch.cat([m0, dm], dim=-1)),
            "M2": _row_norm(torch.cat([m0, c6], dim=-1)),
            "M3": _row_norm(torch.cat([m0, dm, c12], dim=-1))}


def _scores(qd, bd):
    return 1.0 - (qd @ bd.T).max(dim=1).values


def _ap(y, s):
    if y.sum() == 0 or y.sum() == y.size:
        return float("nan")
    return float(average_precision_score(y.astype(int), s))


def _mnan(xs):
    xs = [x for x in xs if x is not None and not (isinstance(x, float) and np.isnan(x))]
    return float(np.mean(xs)) if xs else float("nan")


def _im(o, h, suffix, e=None):
    return {lk: (o[lk + suffix][h] if e is None else o[lk + suffix][h, e]) for lk in LAYERS}


def run_cat(cat):
    o = load_cell(cat)
    K = o["K"]
    rows = []
    for h in range(K):
        bank = [k for k in range(K) if k != h]
        mem = {c: torch.cat([cand_vectors(_im(o, k, ""))[c] for k in bank], dim=0) for c in CAND}
        for fam in (0, 1):
            for e in range(fam * 3, fam * 3 + 3):
                y = o["masks"][h, e].ravel() > 0
                if y.sum() == 0:
                    continue
                v = cand_vectors(_im(o, h, "s", e))
                rows.append({"cat": cat, "h": h, "held_family": SYN[fam], "e": e,
                             **{c: _ap(y, _scores(v[c], mem[c]).numpy()) for c in CAND}})
        vc = cand_vectors(_im(o, h, ""))
        s_clean = {c: _scores(vc[c], mem[c]).numpy() for c in CAND}
        for c in CAND:
            s_nui = np.concatenate([_scores(cand_vectors(_im(o, h, "n", e))[c], mem[c]).numpy()
                                    for e in range(15)])
            y = np.concatenate([np.zeros(G * G), np.ones_like(s_nui)])
            s = np.concatenate([s_clean[c], s_nui])
            rows.append({"cat": cat, "h": h, "held_family": "normal", "variant": c,
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
    syn = [r for r in rows if r.get("held_family") in ("cutpaste", "local_erasure")]
    agg = {f"{fam}_{c}": _mnan([r[c] for r in syn if r["held_family"] == fam])
           for fam in ("cutpaste", "local_erasure") for c in CAND}
    norm = [r for r in rows if r.get("held_family") == "normal"]
    auc = {c: _mnan([r["auc"] for r in norm if r["variant"] == c]) for c in CAND}
    cut0 = agg["cutpaste_M0"]
    best = max(("M1", "M2", "M3"), key=lambda c: agg[f"cutpaste_{c}"])
    d_cut = agg[f"cutpaste_{best}"] - cut0
    d_era = agg[f"local_erasure_{best}"] - agg["local_erasure_M0"]
    g_m1 = bool(d_cut >= 0.05)
    g_m2 = bool(d_era >= -0.01)
    g_m3 = bool((auc[best] - auc["M0"] <= 0.05) and auc[best] <= 0.60)
    out = {"agg": agg, "auc": auc, "best": best, "delta_cutpaste": d_cut,
           "erasure_delta": d_era, "G_M1": g_m1, "G_M2": g_m2, "G_M3": g_m3,
           "decision": "TRACK2_PROBE_PASS" if (g_m1 and g_m2 and g_m3)
           else "TRACK2_PROBE_FAIL_ARCHIVE"}
    (OUT / "PROBE_M1_RESULTS.json").write_text(json.dumps(out, ensure_ascii=False, indent=1),
                                               encoding="utf-8")
    print("per-family macro AP:", {k: round(v, 4) for k, v in agg.items()})
    print("auc:", {k: round(v, 4) for k, v in auc.items()})
    print(f"best={best} delta_cutpaste={d_cut:+.4f} erasure_delta={d_era:+.4f}")
    print("G:", g_m1, g_m2, g_m3)
    print("DECISION:", out["decision"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
