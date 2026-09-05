"""Track-1 Probe-C1: is there usable RELATIONAL info in frozen patch features?

doc30 s4 pre-registered feasibility probe. Pure evaluation (no training), on the
legal v14 support-only synthetic caches (no /test/ touched). For each
(cat, shot in {2,4}) x held family {cutpaste (context anomaly, primary),
local_erasure (structural control)}: memory = other (K-1) clean support images at
a 16x16 grid; score = 1 - max_cos of a DESCRIPTOR to that memory; Pixel-AP inside
the held-out image's held-family mask.

Descriptors (same 16-grid, all L2-normalised):
  C0 = A1 fused cell vector z            (baseline: patch similarity only)
  C1 = concat(z, 3x3-neighbour mean z)   (local-context deviation)
  C2 = concat(z, up/down/left/right z)   (directional neighbourhood)
z = L2-normalised 2x2 mean-pool of the A1 concat (1536-D) at the 32-grid.

Pre-registered gates (doc30 s4; NOT tuned on results):
  G-C1  cutpaste held-out macro AP: best context variant - C0 >= +0.05 (k2 or k4);
  G-C2  erasure held-out macro AP: same variant - C0 >= -0.01 (no structural loss);
  G-C3  normal path (scale-free): nuisance-sensitivity AUC(clean h cells vs its 15
        photometric variants) of the best context variants - C0 <= +0.05 and <= 0.60.
thin_scratch is not evaluable at 32/16 grid -> excluded (recorded).
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
OUT = ROOT / "experiments/dynamic_fusion/innovation_t1_context_defect_20260905"
SYN = ["cutpaste", "local_erasure", "thin_scratch"]
VARIANT_NAMES = ("C0", "C1", "C2")
G = 16          # probe grid (2x2 mean pool of the 32-grid A1 fused cells)
torch.set_num_threads(max(1, torch.get_num_threads()))


def _load(cat, shot, branch):
    z = np.load(CACHE / f"v14_p1_support_{branch}_s0_k{shot}" / f"{cat}.npz", allow_pickle=False)
    return (np.asarray(z["clean_feat"]), np.asarray(z["syn_feat"]),
            np.asarray(z["syn_masks"]), np.asarray(z["nui_feat"]))


def _mask16(m1024, g=G):
    m = cv2.resize(m1024.astype(np.uint8), (g, g), interpolation=cv2.INTER_AREA)
    return (m > 127).astype(np.float32)


def _row_norm(x: torch.Tensor):
    return x / torch.clamp(x.norm(dim=-1, keepdim=True), min=1e-12)


def _fused_cells(d, c):
    """A1 concat fusion of [N,768] branch cells -> [N,1536] unit rows."""
    dd = _row_norm(d)
    cc = _row_norm(c)
    z = torch.cat([0.5 * dd, 0.5 * cc], dim=-1)
    return _row_norm(z)


def _load_cell_z16(cat, shot):
    """Return fused 16-grid unit cells for clean/syn(9)/nui(15) + masks16."""
    cd, sd, md, nd = _load(cat, shot, "dino")
    cc, sc, _mc, nc = _load(cat, shot, "clip")
    cc = resize_patches(cc.reshape(-1, *cc.shape[1:]), (32, 32)).reshape(cc.shape[0], 32, 32, -1)
    sc = resize_patches(sc.reshape(-1, *sc.shape[2:]), (32, 32)).reshape(*sc.shape[:2], 32, 32, -1)
    nc = resize_patches(nc.reshape(-1, *nc.shape[2:]), (32, 32))
    K = cd.shape[0]
    nc = nc.reshape(K, 15, 32, 32, -1)

    def fused_pair(dA, cA, n):
        """A1 concat of [n,1024,768] dino/clip cells -> [n,32,32,1536] unit."""
        z32 = _fused_cells(torch.tensor(dA.reshape(n, 32 * 32, 768).astype(np.float32)),
                           torch.tensor(cA.reshape(n, 32 * 32, 768).astype(np.float32)))
        return z32.reshape(n, 32, 32, 1536)

    def pool16_grid(z32):  # z32 [n,32,32,1536] torch
        p = z32.reshape(-1, 16, 2, 16, 2, 1536).mean(dim=(2, 4))   # [n,16,16,1536]
        return _row_norm(p.reshape(p.shape[0], G * G, 1536)).reshape(p.shape[0], G, G, 1536)

    n_syn = sd.shape[1]
    zc16 = pool16_grid(fused_pair(cd, cc, K))
    zs16 = pool16_grid(fused_pair(sd.reshape(-1, 32 * 32, 768), sc.reshape(-1, 32 * 32, 768),
                                  K * n_syn)).reshape(K, n_syn, G, G, 1536)
    zn16 = pool16_grid(fused_pair(nd.reshape(-1, 32 * 32, 768), nc.reshape(-1, 32 * 32, 768),
                                  K * 15)).reshape(K, 15, G, G, 1536)
    masks = np.asarray([[_mask16(md[h, e], G) for e in range(9)] for h in range(K)])
    return zc16, zs16, zn16, masks, K


def _desc(xg, variant):
    """xg [G,G,1536] unit cells -> descriptor cells [G*G,dv] L2 rows."""
    cells = xg.reshape(G * G, 1536)
    if variant == "C0":
        return cells
    p = torch.nn.functional.pad(xg.permute(2, 0, 1)[None], (1, 1, 1, 1), mode="reflect")[0]
    p = p.permute(1, 2, 0)  # [G+2,G+2,1536]
    nb = (p[0:G, 0:G] + p[0:G, 1:G + 1] + p[0:G, 2:G + 2] +
          p[1:G + 1, 0:G] + p[1:G + 1, 2:G + 2] +
          p[2:G + 2, 0:G] + p[2:G + 2, 1:G + 1] + p[2:G + 2, 2:G + 2]) / 8.0
    nb = nb.reshape(G * G, 1536)
    if variant == "C1":
        return _row_norm(torch.cat([cells, nb], dim=-1))
    # C2 directional neighbours
    up = p[0:G, 1:G + 1].reshape(G * G, 1536)
    dn = p[2:G + 2, 1:G + 1].reshape(G * G, 1536)
    lf = p[1:G + 1, 0:G].reshape(G * G, 1536)
    rt = p[1:G + 1, 2:G + 2].reshape(G * G, 1536)
    return _row_norm(torch.cat([cells, up, dn, lf, rt], dim=-1))


def _scores(qdesc, bdesc):
    return 1.0 - (qdesc @ bdesc.T).max(dim=1).values


def _ap(y, s):
    if y.sum() == 0 or y.sum() == y.size:
        return float("nan")
    return float(average_precision_score(y.astype(int), s))


def _run_cell(cat, shot):
    zc16, zs16, zn16, masks, K = _load_cell_z16(cat, shot)
    rows = []
    for h in range(K):
        bank = [k for k in range(K) if k != h]
        # descriptors are per-image (neighbour context inside each image), then cells concat
        bdesc = {v: torch.cat([_desc(zc16[k], v) for k in bank], dim=0) for v in VARIANT_NAMES}
        # held-family AP for cutpaste (0) and erasure (1)
        for fam in (0, 1):
            for e in range(fam * 3, fam * 3 + 3):
                y = masks[h, e].ravel() > 0
                if y.sum() == 0:
                    continue
                aps = {}
                for v in VARIANT_NAMES:
                    qd = _desc(zs16[h, e], v)
                    s = _scores(qd, bdesc[v])
                    aps[v] = _ap(y, s.numpy())
                rows.append({"cat": cat, "shot": shot, "h": h,
                             "held_family": SYN[fam], "e": e, **aps})
        # normal path per variant (scale-free): cell-level AUC clean(h) vs the 15
        # photometric nuisance variants of h, within the SAME descriptor space.
        for v in VARIANT_NAMES:
            b = bdesc[v]
            s_clean = _scores(_desc(zc16[h], v), b).numpy()          # [256]
            s_nui = np.concatenate([_scores(_desc(zn16[h, e], v), b).numpy() for e in range(15)])
            y = np.concatenate([np.zeros_like(s_clean), np.ones_like(s_nui)])
            s_all = np.concatenate([s_clean, s_nui])
            rows.append({"cat": cat, "shot": shot, "h": h, "held_family": "normal",
                         "variant": v,
                         "auc_clean_nui": float(roc_auc_score(y.astype(int), s_all))})
    return rows


def _mnan(xs):
    xs = [x for x in xs if x is not None and not (isinstance(x, float) and np.isnan(x))]
    return float(np.mean(xs)) if xs else float("nan")


def aggregate(rows, out_json):
    syn = [r for r in rows if "held_family" in r and r["held_family"] in ("cutpaste", "local_erasure")]
    out = {}
    for s in (2, 4):
        for fam in ("cutpaste", "local_erasure"):
            for v in VARIANT_NAMES:
                aps = [r[v] for r in syn if r["held_family"] == fam and r["shot"] == s and r[v] == r[v]]
                out.setdefault((s, fam), {})[v] = _mnan(aps) if aps else float("nan")
    norm = [r for r in rows if r.get("held_family") == "normal"]
    gates = {}
    for v in VARIANT_NAMES:
        a = [r["auc_clean_nui"] for r in norm if r["variant"] == v]
        gates[v] = {"auc_clean_nui_macro": _mnan(a)}
    # G-C1/G-C2 per shot using best context variant
    verdict = {}
    for s in (2, 4):
        c0 = out[(s, "cutpaste")]["C0"]
        best = max(("C1", "C2"), key=lambda v: out[(s, "cutpaste")][v])
        d_cut = out[(s, "cutpaste")][best] - c0
        d_era = out[(s, "local_erasure")][best] - out[(s, "local_erasure")]["C0"]
        verdict[s] = {"best_variant": best, "cutpaste_C0": c0, "cutpaste_best": out[(s, "cutpaste")][best],
                      "delta_cutpaste": d_cut, "erasure_delta": d_era}
    any_shot = verdict[2]["delta_cutpaste"] >= 0.05 or verdict[4]["delta_cutpaste"] >= 0.05
    era_ok = all(verdict[s]["erasure_delta"] >= -0.01 for s in (2, 4))
    # G-C3 (scale-free): nuisance-sensitivity AUC of best context variants ~ C0
    auc_c0 = gates["C0"]["auc_clean_nui_macro"]
    auc_ok = {v: bool(gates[v]["auc_clean_nui_macro"] - auc_c0 <= 0.05
                      and gates[v]["auc_clean_nui_macro"] <= 0.60) for v in ("C1", "C2")}
    norm_ok = all(auc_ok.values())
    g = {"G_C1_relational_info": bool(any_shot), "G_C2_no_structural_loss": bool(era_ok),
         "G_C3_normal_path_stable": bool(norm_ok),
         "per_shot": verdict,
         "auc_clean_nui": {v: gates[v]["auc_clean_nui_macro"] for v in VARIANT_NAMES},
         "auc_ok": auc_ok,
         "decision": "TRACK1_PROBE_PASS" if (any_shot and era_ok and norm_ok)
         else "TRACK1_PROBE_FAIL_ARCHIVE"}
    Path(out_json).write_text(json.dumps({"per_shot_family_ap": {f"{k[0]}_{k[1]}": v for k, v in out.items()},
                                          "normal": gates, "gates": g},
                                         ensure_ascii=False, indent=1), encoding="utf-8")
    return g


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cats", default=None)
    ap.add_argument("--shots", type=int, nargs="+", default=[2, 4])
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    cats = [c.strip() for c in args.cats.split(",")] if args.cats else CATEGORIES
    all_rows = []
    for shot in args.shots:
        for cat in cats:
            all_rows += _run_cell(cat, shot)
            print(f"  done {cat} k{shot}", flush=True)
    gates = aggregate(all_rows, OUT / "PROBE_C1_RESULTS.json")
    print("rows:", len(all_rows))
    for s in (2, 4):
        v = gates["per_shot"][s]
        print(f"k{s}: best={v['best_variant']} cutpaste C0={v['cutpaste_C0']:.3f} -> "
              f"{v['cutpaste_best']:.3f} (d={v['delta_cutpaste']:+.3f}) erasure d={v['erasure_delta']:+.3f}")
    print("auc_clean_nui C0/C1/C2:", {k: round(val, 3) for k, val in gates["auc_clean_nui"].items()})
    print("G:", gates["G_C1_relational_info"], gates["G_C2_no_structural_loss"],
          gates["G_C3_normal_path_stable"])
    print("DECISION:", gates["decision"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
