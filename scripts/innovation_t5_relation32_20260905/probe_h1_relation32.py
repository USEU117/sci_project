"""Direction-5 Probe-H1: do relational descriptors work at the real 32-grid?

doc35 s2/s3 pre-registered. Pure evaluation on v14 support-only synthetic caches
(no /test/, no real defects, no fitting). For each (cat, shot in {2,4}) x held
family {cutpaste (primary context anomaly), local_erasure (structural control)}:
memory = other (K-1) clean support images at the 32x32 patch grid; score =
1 - max_cos of a DESCRIPTOR to that memory; Pixel-AP over the held-out image's
held-family mask (downsampled to 32).

Descriptors (32-grid fused A1 patch z = rowL2([0.5 dino, 0.5 clip]), 1536-D):
  C0 = z                                   (patch similarity baseline)
  C1 = concat(z, 3x3-neighbour mean z)     (local context, 32-grid)
  C2 = concat(z, up/down/left/right z)     (directional neighbourhood)
  C3 = concat(z, 16-grid 3x3-neighbour mean, nearest-upsampled to 32) (cross-scale)
Gates (doc35 s3, not tuned): G-H1 cutpaste macro best-C0 >= +0.05 (k2 or k4);
G-H2 erasure best-C0 >= -0.01; G-H3 nuisance-AUC best-C0 <= +0.05 & <= 0.60.
thin_scratch not evaluable -> excluded (same as Track-1).
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
OUT = ROOT / "experiments/dynamic_fusion/innovation_t5_relation32_20260905"
SYN = ["cutpaste", "local_erasure", "thin_scratch"]
VARIANT_NAMES = ("C0", "C1", "C2", "C3")
G = 32          # probe grid = real A1 patch grid
torch.set_num_threads(max(1, torch.get_num_threads()))


def _load(cat, shot, branch):
    z = np.load(CACHE / f"v14_p1_support_{branch}_s0_k{shot}" / f"{cat}.npz", allow_pickle=False)
    return (np.asarray(z["clean_feat"]), np.asarray(z["syn_feat"]),
            np.asarray(z["syn_masks"]), np.asarray(z["nui_feat"]))


def _mask32(m1024):
    m = cv2.resize(m1024.astype(np.uint8), (G, G), interpolation=cv2.INTER_AREA)
    return (m > 127).astype(np.float32)


def _row_norm(x: torch.Tensor):
    return x / torch.clamp(x.norm(dim=-1, keepdim=True), min=1e-12)


def _fused_patch(d, c):
    """A1 concat of [N,1024,768] -> [N,1024,1536] unit rows (32-grid)."""
    dd = _row_norm(d)
    cc = _row_norm(c)
    return _row_norm(torch.cat([0.5 * dd, 0.5 * cc], dim=-1))


def _load_patch_z32(cat, shot):
    """Fused 32-grid unit patches for clean/syn(9)/nui(15) + masks32."""
    cd, sd, md, nd = _load(cat, shot, "dino")
    cc, sc, _mc, nc = _load(cat, shot, "clip")
    cc = resize_patches(cc.reshape(-1, *cc.shape[1:]), (G, G)).reshape(cc.shape[0], G, G, -1)
    sc = resize_patches(sc.reshape(-1, *sc.shape[2:]), (G, G)).reshape(*sc.shape[:2], G, G, -1)
    nc = resize_patches(nc.reshape(-1, *nc.shape[2:]), (G, G)).reshape(*nc.shape[:2], G, G, -1)
    K = cd.shape[0]
    n_syn = sd.shape[1]

    def fused_pair(dA, cA, n):
        return _fused_patch(torch.tensor(dA.reshape(n, G * G, 768).astype(np.float32)),
                            torch.tensor(cA.reshape(n, G * G, 768).astype(np.float32))
                            ).reshape(n, G, G, 1536)

    zc = fused_pair(cd, cc, K)
    zs = fused_pair(sd.reshape(-1, G * G, 768), sc.reshape(-1, G * G, 768), K * n_syn)
    zs = zs.reshape(K, n_syn, G, G, 1536)
    zn = fused_pair(nd.reshape(-1, G * G, 768), nc.reshape(-1, G * G, 768), K * 15)
    zn = zn.reshape(K, 15, G, G, 1536)
    masks = np.asarray([[_mask32(md[h, e]) for e in range(9)] for h in range(K)])
    return zc, zs, zn, masks, K


def _neighbour_mean(xg):
    """xg [gg,gg,D] -> 3x3 reflect-padded mean [gg,gg,D]."""
    gg = xg.shape[0]
    p = torch.nn.functional.pad(xg.permute(2, 0, 1)[None], (1, 1, 1, 1), mode="reflect")[0]
    p = p.permute(1, 2, 0)  # [gg+2,gg+2,D]
    nb = (p[0:gg, 0:gg] + p[0:gg, 1:gg + 1] + p[0:gg, 2:gg + 2] +
          p[1:gg + 1, 0:gg] + p[1:gg + 1, 2:gg + 2] +
          p[2:gg + 2, 0:gg] + p[2:gg + 2, 1:gg + 1] + p[2:gg + 2, 2:gg + 2]) / 8.0
    return nb


def _coarse_context(xg):
    """16-grid 3x3-neighbour mean of xg, nearest-upsampled back to 32-grid [G,G,D]."""
    p16 = xg.reshape(16, 2, 16, 2, -1).mean(dim=(1, 3))          # [16,16,D]
    nb16 = _neighbour_mean(p16)
    # nearest upsample: repeat each coarse cell to its 2x2 footprint
    return torch.repeat_interleave(torch.repeat_interleave(nb16, 2, dim=0), 2, dim=1)


def _desc(xg, variant):
    """xg [G,G,1536] unit patches -> descriptor rows [G*G,dv] L2 (torch)."""
    cells = xg.reshape(G * G, 1536)
    if variant == "C0":
        return cells
    if variant == "C1":
        nb = _neighbour_mean(xg).reshape(G * G, 1536)
        return _row_norm(torch.cat([cells, nb], dim=-1))
    if variant == "C3":
        cc3 = _coarse_context(xg).reshape(G * G, 1536)
        return _row_norm(torch.cat([cells, cc3], dim=-1))
    # C2 directional neighbours
    p = torch.nn.functional.pad(xg.permute(2, 0, 1)[None], (1, 1, 1, 1), mode="reflect")[0]
    p = p.permute(1, 2, 0)  # [G+2,G+2,1536]
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


def _run_patch(cat, shot):
    zc, zs, zn, masks, K = _load_patch_z32(cat, shot)
    rows = []
    for h in range(K):
        bank = [k for k in range(K) if k != h]
        bdesc = {v: torch.cat([_desc(zc[k], v) for k in bank], dim=0) for v in VARIANT_NAMES}
        for fam in (0, 1):
            for e in range(fam * 3, fam * 3 + 3):
                y = masks[h, e].ravel() > 0
                if y.sum() == 0:
                    continue
                aps = {}
                for v in VARIANT_NAMES:
                    qd = _desc(zs[h, e], v)
                    aps[v] = _ap(y, _scores(qd, bdesc[v]).numpy())
                rows.append({"cat": cat, "shot": shot, "h": h,
                             "held_family": SYN[fam], "e": e, **aps})
        for v in VARIANT_NAMES:
            b = bdesc[v]
            s_clean = _scores(_desc(zc[h], v), b).numpy()
            s_nui = np.concatenate([_scores(_desc(zn[h, e], v), b).numpy() for e in range(15)])
            y = np.concatenate([np.zeros_like(s_clean), np.ones_like(s_nui)])
            s_all = np.concatenate([s_clean, s_nui])
            rows.append({"cat": cat, "shot": shot, "h": h, "held_family": "normal",
                         "variant": v, "auc_clean_nui": float(roc_auc_score(y.astype(int), s_all))})
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
    gates = {v: {"auc_clean_nui_macro": _mnan([r["auc_clean_nui"] for r in norm if r["variant"] == v])}
             for v in VARIANT_NAMES}
    verdict = {}
    for s in (2, 4):
        c0 = out[(s, "cutpaste")]["C0"]
        best = max(("C1", "C2", "C3"), key=lambda v: out[(s, "cutpaste")][v])
        verdict[s] = {"best_variant": best, "cutpaste_C0": c0, "cutpaste_best": out[(s, "cutpaste")][best],
                      "delta_cutpaste": out[(s, "cutpaste")][best] - c0,
                      "erasure_delta": out[(s, "local_erasure")][best] - out[(s, "local_erasure")]["C0"]}
    any_shot = verdict[2]["delta_cutpaste"] >= 0.05 or verdict[4]["delta_cutpaste"] >= 0.05
    era_ok = all(verdict[s]["erasure_delta"] >= -0.01 for s in (2, 4))
    auc_c0 = gates["C0"]["auc_clean_nui_macro"]
    auc_ok = {v: bool(gates[v]["auc_clean_nui_macro"] - auc_c0 <= 0.05
                      and gates[v]["auc_clean_nui_macro"] <= 0.60) for v in ("C1", "C2", "C3")}
    norm_ok = all(auc_ok.values())
    g = {"G_H1_relational_info_32": bool(any_shot), "G_H2_no_structural_loss": bool(era_ok),
         "G_H3_normal_path_stable": bool(norm_ok), "per_shot": verdict,
         "auc_clean_nui": {v: gates[v]["auc_clean_nui_macro"] for v in VARIANT_NAMES},
         "auc_ok": auc_ok,
         "decision": "D5_PROBE_PASS" if (any_shot and era_ok and norm_ok)
         else "D5_PROBE_FAIL_ARCHIVE"}
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
            all_rows += _run_patch(cat, shot)
            print(f"  done {cat} k{shot}", flush=True)
    gates = aggregate(all_rows, OUT / "PROBE_H1_RESULTS.json")
    print("rows:", len(all_rows))
    for s in (2, 4):
        v = gates["per_shot"][s]
        print(f"k{s}: best={v['best_variant']} cutpaste C0={v['cutpaste_C0']:.3f} -> "
              f"{v['cutpaste_best']:.3f} (d={v['delta_cutpaste']:+.3f}) erasure d={v['erasure_delta']:+.3f}")
    print("auc_clean_nui:", {k: round(val, 3) for k, val in gates["auc_clean_nui"].items()})
    print("G:", gates["G_H1_relational_info_32"], gates["G_H2_no_structural_loss"],
          gates["G_H3_normal_path_stable"])
    print("DECISION:", gates["decision"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
