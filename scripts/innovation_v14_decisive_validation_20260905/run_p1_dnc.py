"""P1-B legal DNC mechanism gate (doc28 s5.3), support-only.

Reads P1-A caches: outputs/dynamic_fusion/v14_p1_support_{branch}_s0_k{shot}/{cat}.npz
  clean_feat [K, g, g, 768], syn_feat [K, 9, g, g, 768] (kind-major, 3 kinds x 3 seeds),
  syn_masks [K, 9, 1024, 1024], nui_feat [K, 15, g, g, 768] (5 fams x 3 strengths),
  nui_keys, syn_kinds, grid_size, ref_rel.
Structure: per (cat, shot): LOO over held-out support h AND held-out syn family f:
  fit q on (K-1) images x (2 families x 3 seeds) + nui; select methods;
  eval reduced-fused KNN AP on image h's family-f episodes (mask from syn_masks).
Gates (mechanism, frozen; macro over cats then 2/3 families):
  G1 DNC-I - max(random-mean, low_nui) >= +0.02 on >=2/3 held-out families;
  G2 nuis-FP (photometric, p99-of-bank threshold) reduced vs full <= +10% relative
     (robust: absolute slack +0.01 rate);
  G3 DNC-C vs DNC-I combined-set Jaccard < 0.95 AND chosen cross-branch mean |corr|
     drops >= 10%;
  G4 DNC-C - DNC-I synthetic macro AP >= +0.003 and (true gain - shuffled-corr gain)
     >= 0.003.
Decision (doc28 s5.3): if G1&G2 fail -> no real-mask run (archive). If G1&G2 pass but
G3/G4 fail -> only DNC-I may run ONE frozen real diagnostic ("channel adaptation").
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

import faiss  # noqa: E402
import cv2  # noqa: E402
from sklearn.metrics import average_precision_score  # noqa: E402

from dnc_selector import select_dnc_i, select_dnc_c, redundancy_stats  # noqa: E402

CATEGORIES = ["bracket_black", "bracket_brown", "bracket_white", "connector", "metal_plate", "tubes"]
CACHE = ROOT / "outputs/dynamic_fusion/v14_p1_support"
OUT = ROOT / "experiments/dynamic_fusion/innovation_v14_decisive_validation_20260905/P1_dnc_fixed"
KEEP = 256
LAM = 0.3
EPS_Q = 0.05
N_RANDOM = 10
SEED0 = 12345
SYN = ["cutpaste", "local_erasure", "thin_scratch"]


def _load(cat, shot, branch):
    z = np.load(CACHE / f"v14_p1_support_{branch}_s0_k{shot}" / f"{cat}.npz", allow_pickle=False)
    return (np.asarray(z["clean_feat"]), np.asarray(z["syn_feat"]),
            np.asarray(z["syn_masks"]), np.asarray(z["nui_feat"]))


def _mask_cells(m1024, g):
    m = cv2.resize(m1024.astype(np.uint8), (g, g), interpolation=cv2.INTER_AREA)
    return (m > 127).astype(np.uint8)


def _robust_std(x):
    med = np.median(x)
    return 1.4826 * np.median(np.abs(x - med))


def _resp(clean, syn, masks, nui, g):
    """Channel response stats from support image arrays (single image)."""
    d = clean.shape[-1]
    cells_c = clean.reshape(-1, d)
    s = np.maximum(_robust_std(cells_c), 1e-8)
    med_c = np.median(cells_c, axis=0)
    # defect responses over episodes (kind-major flattened); self-centred on CLEAN
    defs = []
    S = syn.shape[0]
    for e in range(S):
        m = (_mask_cells(masks[e], g) > 0).ravel()
        fe = syn[e].reshape(g * g, d)
        if m.sum() == 0:
            continue
        r = np.median(fe[m], axis=0) - med_c
        defs.append(r)
    defs = np.stack(defs) if defs else np.zeros((1, d))
    # nuisance responses
    ns = []
    for e in range(nui.shape[0]):
        ns.append(np.median(np.abs(nui[e] - clean).reshape(-1, d), axis=0))
    ns = np.stack(ns)
    q = np.median(np.abs(defs) / s, axis=0) / (np.median(ns, axis=0) / s + EPS_Q)
    return q, s, med_c, ns


def _fused_reduced(feat_cells, sel):
    """Row-L2-normalise selected branch channels; empty sel -> zeros."""
    if sel.size == 0:
        return np.zeros((feat_cells.shape[0], 0))
    x = feat_cells[:, sel].astype(np.float64)
    return x / np.maximum(np.linalg.norm(x, axis=1, keepdims=True), 1e-8)


def _fused_bank(dino_cells, clip_cells, selD, selC):
    """concat fused bank (A1 protocol on reduced channels), cells-first arrays."""
    d = _fused_reduced(dino_cells, selD)
    c = _fused_reduced(clip_cells, selC)
    if d.shape[1] and c.shape[1]:
        f = np.concatenate([0.5 * d, 0.5 * c], axis=1)
    else:
        f = d if d.shape[1] else c
    f = f / np.maximum(np.linalg.norm(f, axis=1, keepdims=True), 1e-8)
    return np.ascontiguousarray(f, dtype=np.float32)


def _ap_scored(bank, q_feat_cells, mask32):
    idx = faiss.IndexFlatL2(bank.shape[1])
    idx.add(bank)
    d2, _ = idx.search(np.ascontiguousarray(q_feat_cells, dtype=np.float32), 1)
    sc = d2[:, 0] / 2.0
    y = mask32.ravel() > 0
    if y.sum() == 0 or y.sum() == y.size:
        return float("nan")
    return float(average_precision_score(y.astype(int), sc))


def _nuis_fp(bank, query_cells_list):
    """Fraction of query images whose max cell d2 exceeds bank LOO p99 (A1 normal
    reference). bank: cells-first float32. query_cells_list: list of cells-first."""
    idx = faiss.IndexFlatL2(bank.shape[1])
    idx.add(bank)
    d2n, _ = idx.search(bank, 2)
    thr = float(np.percentile(d2n[:, 1], 99))
    flags = []
    for q in query_cells_list:
        d2, _ = idx.search(np.ascontiguousarray(q, dtype=np.float32), 1)
        flags.append(bool((d2[:, 0] > thr).any()))
    return thr, float(np.mean(flags)) if flags else float("nan")


def _cell_sel_q(qD, qC, corrDC, lam):
    """DNC selection helpers returning (selD, selC, red_stats)."""
    sI = select_dnc_i(qD, qC, KEEP)
    sC = select_dnc_c(qD, qC, corrDC, lam, KEEP)
    redI = redundancy_stats(*sI, corrDC)
    redC = redundancy_stats(*sC, corrDC)
    return sI, sC, redI, redC


def _set_jaccard(sel_i, sel_c):
    """Combined-set Jaccard over branches (C offset by 768 to avoid index clash)."""
    a = set(np.asarray(sel_i[0]).tolist()) | set((np.asarray(sel_i[1]) + 768).tolist())
    b = set(np.asarray(sel_c[0]).tolist()) | set((np.asarray(sel_c[1]) + 768).tolist())
    u = a | b
    return len(a & b) / len(u) if u else 1.0


def run_cat_shot(cat, shot, gD, gC):
    cd, sd, md, nd = _load(cat, shot, "dino")      # [K,32,32,768] / [K,9,32,32,768] / masks / nui
    cc, sc, _mc, nc = _load(cat, shot, "clip")     # [K,37,37,768] clip grids -> 32
    from industrial_ad.innovation_v10_portfolio.common import resize_patches
    cc = resize_patches(cc.reshape(-1, *cc.shape[1:]), (32, 32)).reshape(cc.shape[0], 32, 32, -1)
    sc = resize_patches(sc.reshape(-1, *sc.shape[2:]), (32, 32)).reshape(*sc.shape[:2], 32, 32, -1)
    K = cd.shape[0]
    nc = resize_patches(nc.reshape(-1, *nc.shape[2:]), (32, 32))
    nc = nc.reshape(K, 15, 32, 32, -1)
    rows, diags, g2rows = [], [], []
    for h in range(K):
        sel_imgs = [k for k in range(K) if k != h]
        for fi, held_f in enumerate(SYN):
            # fit on (K-1) imgs x non-held families (kind-major index over 9)
            keep_idx = [kind * 3 + s for kind in range(3) if kind != fi for s in range(3)]
            qD_acc, qC_acc = [], []
            for k in sel_imgs:
                qD, sD, _, _ = _resp(cd[k], sd[k, keep_idx], md[k, keep_idx], nd[k], 32)
                qC, sC, _, _ = _resp(cc[k], sc[k, keep_idx], md[k, keep_idx], nc[k], 32)
                qD_acc.append(qD)
                qC_acc.append(qC)
            qD = np.mean(qD_acc, axis=0)
            qC = np.mean(qC_acc, axis=0)
            cleanD = np.concatenate([cd[k].reshape(-1, 768) for k in sel_imgs])
            cleanC = np.concatenate([cc[k].reshape(-1, 768) for k in sel_imgs])
            varD = np.var(cleanD, axis=0)
            varC = np.var(cleanC, axis=0)
            nuiD = np.median([np.median(np.abs(nd[k] - cd[k]).reshape(-1, 768), axis=0)
                              for k in sel_imgs], axis=0)
            nuiC = np.median([np.median(np.abs(nc[k] - cc[k]).reshape(-1, 768), axis=0)
                              for k in sel_imgs], axis=0)
            lowD = np.argsort(nuiD)[:KEEP]
            lowC = np.argsort(nuiC)[:KEEP]
            # dino/clip response vectors over common episodes for DNC-C redundancy
            epD, epC = [], []
            for k in sel_imgs:
                for e in keep_idx:
                    mk = (_mask_cells(md[k, e], 32) > 0).ravel()
                    epD.append(np.median(sd[k, e].reshape(-1, 768)[mk], axis=0)
                               - np.median(cd[k].reshape(-1, 768), axis=0))
                    epC.append(np.median(sc[k, e].reshape(-1, 768)[mk], axis=0)
                               - np.median(cc[k].reshape(-1, 768), axis=0))
            rD = np.stack(epD)
            rC = np.stack(epC)
            corrDC = np.nan_to_num(np.corrcoef(np.hstack([rD, rC]).T)[:768, 768:])
            selI, selC, redI, redC = _cell_sel_q(qD, qC, corrDC, LAM)
            methods = {
                "full": (np.arange(768), np.arange(768)),
                "dnc_i": selI,
                "dnc_c": selC,
                "low_nui": (lowD, lowC),
                "highvar": (np.argsort(-varD)[:KEEP], np.argsort(-varC)[:KEEP]),
                "dino_only": (np.argsort(-qD)[:2 * KEEP], np.asarray([], dtype=int)),
                "clip_only": (np.asarray([], dtype=int), np.argsort(-qC)[:2 * KEEP]),
            }
            # shuffled cross-branch correlation control (G4): fixed per (h, fi)
            rng = np.random.default_rng(SEED0 + h * 97 + fi)
            corr_shuf = corrDC[:, rng.permutation(768)]
            methods["dnc_c_shuf"] = select_dnc_c(qD, qC, corr_shuf, LAM, KEEP)
            diags.append({"cat": cat, "shot": shot, "h": h, "held_family": held_f,
                          "jac": _set_jaccard(selI, selC),
                          "mean_corr_i": redI["mean_corr"], "mean_corr_c": redC["mean_corr"]})
            rng10 = np.random.default_rng(SEED0)
            rsel = [(rng10.permutation(768)[:KEEP], rng10.permutation(768)[:KEEP])
                    for _ in range(N_RANDOM)]
            eval_eps = [e for e in range(9) if e // 3 == fi]      # held family episodes on h
            all_sel = list(methods.items()) + [(f"random{ri}", s) for ri, s in enumerate(rsel)]
            for name, (selD, selC_) in all_sel:
                bank = _fused_bank(cleanD, cleanC, selD, selC_)
                aps = []
                for e in eval_eps:
                    mask32 = _mask_cells(md[h, e], 32) > 0
                    if mask32.sum() == 0:
                        continue
                    qb = _fused_bank(sd[h, e].reshape(-1, 768), sc[h, e].reshape(-1, 768), selD, selC_)
                    aps.append(_ap_scored(bank, qb, mask32))
                rows.append({"cat": cat, "shot": shot, "h": h, "held_family": held_f,
                             "method": name, "ap": float(np.mean(aps)) if aps else float("nan")})
        # G2 nuisance-FP block per held-out h (selection fit on all 9 episodes of
        # the (K-1) images; bank excludes h; nuis queries are photometric variants of h)
        qD_acc, qC_acc = [], []
        for k in sel_imgs:
            qD, _, _, _ = _resp(cd[k], sd[k], md[k], nd[k], 32)
            qC, _, _, _ = _resp(cc[k], sc[k], md[k], nc[k], 32)
            qD_acc.append(qD)
            qC_acc.append(qC)
        qD = np.mean(qD_acc, axis=0)
        qC = np.mean(qC_acc, axis=0)
        sI_g2 = select_dnc_i(qD, qC, KEEP)
        cleanD = np.concatenate([cd[k].reshape(-1, 768) for k in sel_imgs])
        cleanC = np.concatenate([cc[k].reshape(-1, 768) for k in sel_imgs])
        bank_full = _fused_bank(cleanD, cleanC, np.arange(768), np.arange(768))
        bank_dnc = _fused_bank(cleanD, cleanC, *sI_g2)
        qF = [_fused_bank(nd[h][e].reshape(-1, 768), nc[h][e].reshape(-1, 768),
                          np.arange(768), np.arange(768)) for e in range(15)]
        qI = [_fused_bank(nd[h][e].reshape(-1, 768), nc[h][e].reshape(-1, 768),
                          *sI_g2) for e in range(15)]
        _, fp_full = _nuis_fp(bank_full, qF)
        _, fp_dnc = _nuis_fp(bank_dnc, qI)
        g2rows.append({"cat": cat, "shot": shot, "h": h,
                       "fp_full": fp_full, "fp_dnci": fp_dnc})
    return rows, diags, g2rows


# ---------------- gate aggregation (frozen, doc28 s5.3) ----------------

def _macro(rows, method, fam=None):
    vals = [r["ap"] for r in rows
            if r["method"] == method and (fam is None or r["held_family"] == fam)
            and not (isinstance(r["ap"], float) and np.isnan(r["ap"]))]
    return float(np.mean(vals)) if vals else float("nan")


def aggregate(rows, diags, g2rows, out_json):
    # G1: per held-out family, DNC-I minus max(random-mean, low_nui); macro over cats/shots/h
    fam = {}
    for f in SYN:
        dncI = _macro(rows, "dnc_i", f)
        rnd = np.mean([_macro(rows, f"random{r}", f) for r in range(N_RANDOM)])
        lowN = _macro(rows, "low_nui", f)
        fam[f] = {"dnc_i": dncI, "random_mean": float(rnd), "low_nui": lowN,
                  "delta": dncI - max(rnd, lowN)}
    n_pass = sum(1 for v in fam.values() if v["delta"] >= 0.02)
    g1 = {"pass": n_pass >= 2, "pass_families": n_pass, "per_family": fam}

    # G2: nuisance FP macro rate, reduced vs full
    fp_full = float(np.mean([r["fp_full"] for r in g2rows]))
    fp_dnc = float(np.mean([r["fp_dnci"] for r in g2rows]))
    rel = (fp_dnc - fp_full) / max(fp_full, 1e-4)
    g2 = {"pass": (fp_dnc - fp_full) <= max(0.10 * fp_full, 0.01),
          "fp_full_macro": fp_full, "fp_dnci_macro": fp_dnc,
          "rel_increase": float(rel), "n_fits": len(g2rows)}

    # G3: combined-set Jaccard and cross-branch mean-|corr| drop (macro over fits)
    jac = float(np.mean([d["jac"] for d in diags]))
    mc_i = float(np.mean([d["mean_corr_i"] for d in diags]))
    mc_c = float(np.mean([d["mean_corr_c"] for d in diags]))
    drop = 1.0 - mc_c / mc_i if mc_i > 0 else 0.0
    g3 = {"pass": jac < 0.95 and drop >= 0.10, "jac_mean": jac,
          "mean_corr_i": mc_i, "mean_corr_c": mc_c, "drop": float(drop), "n_fits": len(diags)}

    # G4: DNC-C vs DNC-I macro AP gain; shuffled-corr control must be weaker by >=0.003
    ap_i = _macro(rows, "dnc_i")
    ap_c = _macro(rows, "dnc_c")
    ap_s = _macro(rows, "dnc_c_shuf")
    gain = ap_c - ap_i
    shuf_gain = ap_s - ap_i
    g4 = {"pass": gain >= 0.003 and (gain - shuf_gain) >= 0.003,
          "ap_dnci_macro": ap_i, "ap_dncc_macro": ap_c, "ap_dncc_shuf_macro": ap_s,
          "gain": float(gain), "shuf_gain": float(shuf_gain)}

    g1_ok, g2_ok = g1["pass"], g2["pass"]
    if g1_ok and g2_ok and g3["pass"] and g4["pass"]:
        decision = "P1_PASS_FULL"      # DNC-C and DNC-I may enter ONE frozen real diagnostic
        claim = "dnc_c_and_dnc_i"
    elif g1_ok and g2_ok:
        decision = "P1_PASS_DNC_I_ONLY"  # G3/G4 failed -> channel adaptation claim only
        claim = "dnc_i_only"
    else:
        decision = "P1_FAIL_ARCHIVE"     # doc28 s5.3/s8: no real-mask run
        claim = "none"
    out = {"g1": g1, "g2": g2, "g3": g3, "g4": g4, "decision": decision, "claim": claim}
    Path(out_json).write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cats", default=None)
    ap.add_argument("--shots", type=int, nargs="+", default=[2, 4])
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    cats = [c.strip() for c in args.cats.split(",")] if args.cats else CATEGORIES
    rows, diags, g2rows = [], [], []
    for shot in args.shots:
        for cat in cats:
            r, d, g2 = run_cat_shot(cat, shot, 32, 37)
            rows += r
            diags += d
            g2rows += g2
            print(f"  done {cat} k{shot}: rows={len(r)} fits={len(d)}", flush=True)
    (OUT / "SYNTH_RESULTS.json").write_text(
        json.dumps({"rows": rows, "diags": diags, "g2": g2rows},
                   ensure_ascii=False, indent=1), encoding="utf-8")
    gates = aggregate(rows, diags, g2rows, OUT / "GATES.json")
    print("\n==== P1-B GATES ====")
    print(f"G1 {gates['g1']['pass']}  pass_families={gates['g1']['pass_families']}/3")
    print(f"G2 {gates['g2']['pass']}  fp_full={gates['g2']['fp_full_macro']:.4f} "
          f"fp_dnci={gates['g2']['fp_dnci_macro']:.4f} rel={gates['g2']['rel_increase']:.3f}")
    print(f"G3 {gates['g3']['pass']}  jac={gates['g3']['jac_mean']:.4f} "
          f"drop={gates['g3']['drop']:.3f}")
    print(f"G4 {gates['g4']['pass']}  gain={gates['g4']['gain']:.4f} "
          f"shuf_gain={gates['g4']['shuf_gain']:.4f}")
    print("DECISION:", gates["decision"], "| claim:", gates["claim"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
