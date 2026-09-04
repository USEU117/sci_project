"""V12-EARLY-FUSION Stage 0 oracle NULL-CONTROL audit (doc 25 s5) - CPU.

Reproduces the archived per-GT-component layer-oracle on the cached layer maps and
runs the three zero-information controls required by doc 25 s5 to test whether the
reported headroom (+0.39 macro, k1/2/4) can be produced WITHOUT new information:

  real     : per GT-connected-component best single-branch layer expert (as archived)
  a1copy   : expert pool = {A1} only -> replacement is identity (procedure sanity)
  scale    : expert pool = {c*A1, c in scales} -> measures pure scale/GT-privilege headroom
  shuffle  : expert pool = real 7 layers but expert assigned at random per component
             (3 seeds) -> measures how much gain survives when the D<->CLIP/layer
             correspondence is destroyed

The archived oracle's per-component selection rule is reused verbatim
(bce + (1-ap) + mean(normal)) so "all experts under the same normal-calibration rule".

Run (.venv-patchcore, CPU):
  .venv-patchcore\\Scripts\\python.exe scripts\\innovation_v12_new_observables\\run_r3_ef_stage0_null_audit.py --shot 1
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2  # noqa: E402
import numpy as np
from sklearn.metrics import average_precision_score  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "scripts" / "innovation_v12_new_observables"))

# Reuse the probe's map pipeline so the audit operates on bit-identical maps.
from run_r3_ef_stage0_probe import (CATEGORIES, ML_ROOT, fused_rows, knn_map,  # noqa: E402
                                    l2, pooled_ap, resize32, to56)

SCALES = [0.25, 0.5, 1.0, 2.0, 4.0, 8.0]
SHUFFLE_SEEDS = [0, 1, 2]
D_LAY = [6, 9, 11]
C_LAY = [6, 12, 18, 24]


def _load(cat: str, shot: int):
    z = np.load(ML_ROOT / f"ml_dino_s0_k{shot}/{cat}.npz", allow_pickle=False)
    d_feat = np.asarray(z["patch_features"])       # [N,3,32,32,768]
    d_ref = np.asarray(z["ref_patch_features"])     # [3,K,32,32,768]
    masks = np.asarray(z["imgs_masks"], dtype=np.uint8)
    del z
    zc = np.load(ML_ROOT / f"ml_clip_s0_k{shot}/{cat}.npz", allow_pickle=False)
    c_feat = np.asarray(zc["patch_features"])       # [N,4,37,37,768]
    c_ref = np.asarray(zc["ref_patch_features"])    # [4,K,37,37,768]
    del zc
    return d_feat, d_ref, c_feat, c_ref, masks


def build_maps(cat: str, shot: int) -> dict:
    """Same maps as the archived probe: 7 single-branch experts + A1 concat (D11C24).

    Peak host RAM is the constraint on this box (16 GB); big arrays are freed as soon
    as each map family is done so the audit can run after the GPU training job.
    """
    d_feat, d_ref, c_feat, c_ref, masks = _load(cat, shot)
    m56 = (masks[:, ::8, ::8] > 0.5).astype(np.uint8)
    n = d_feat.shape[0]
    del masks

    maps = {}
    for li, lay in enumerate(D_LAY):
        bank = l2(d_ref[li].reshape(-1, 768))
        maps[f"dino_L{lay}"] = np.stack([to56(knn_map(l2(d_feat[i, li]), bank)) for i in range(n)])
        del bank
    for li, lay in enumerate(C_LAY):   # clip experts on the RAW 37-grid, as the archived probe
        bank = l2(c_ref[li].reshape(-1, 768))
        maps[f"clip_L{lay}"] = np.stack([to56(knn_map(l2(c_feat[i, li]), bank)) for i in range(n)])
        del bank
    # A1 == concat D11 + C24 (needs clip at 32-grid; resize only now to cap peak RAM)
    c32 = resize32(c_feat)      # [N,4,32,32,768]
    c32_ref = resize32(c_ref)   # [4,K,32,32,768]
    del c_feat, c_ref
    bank = fused_rows(d_ref[2].reshape(-1, 32, 32, 768), c32_ref[3].reshape(-1, 32, 32, 768))
    rows = []
    for i in range(n):
        q = fused_rows(d_feat[i, 2].reshape(1, 32, 32, 768),
                       c32[i, 3].reshape(1, 32, 32, 768)).reshape(32, 32, 1536)
        rows.append(to56(knn_map(q, bank)))
    maps["concat_D11C24"] = np.stack(rows)
    del bank, d_feat, d_ref, c32, c32_ref
    return maps, m56


def _comp_loss(expert_map_i: np.ndarray, cm: np.ndarray, outside: np.ndarray):
    pc = expert_map_i[cm]
    po = expert_map_i[outside]
    if pc.size == 0 or po.size == 0:
        return None
    bce = -float(np.mean(np.log(np.clip(pc, 1e-7, 1.0))))
    sc = np.r_[pc, po]
    y = np.r_[np.ones(len(pc)), np.zeros(len(po))]
    try:
        ap = float(average_precision_score(y, sc))
    except ValueError:
        ap = 0.5
    return bce + (1.0 - ap) + float(po.mean())


def oracle_paste(maps: dict, m56: np.ndarray, mode: str, seed: int = 0) -> tuple:
    """Return (oracle_maps, n_components_replaced) under the chosen expert pool."""
    if mode == "real":
        pool = [maps[f"dino_L{l}"] for l in D_LAY] + [maps[f"clip_L{l}"] for l in C_LAY]
    elif mode == "a1copy":
        pool = [maps["concat_D11C24"]]
    elif mode == "scale":
        pool = [c * maps["concat_D11C24"] for c in SCALES]
    elif mode == "shuffle":
        pool = [maps[f"dino_L{l}"] for l in D_LAY] + [maps[f"clip_L{l}"] for l in C_LAY]
    else:
        raise ValueError(mode)
    rng = np.random.default_rng(seed)
    oracle = maps["concat_D11C24"].copy()
    n = m56.shape[0]
    comps = 0
    for i in range(n):
        lbl = m56[i]
        if lbl.sum() == 0:
            continue
        num, comp = cv2.connectedComponents(lbl, connectivity=8)
        outside = lbl == 0
        if outside.sum() < 2:
            continue
        for cid in range(1, num):
            cm = comp == cid
            if mode == "shuffle":
                best = pool[rng.integers(len(pool))]
                oracle[i][cm] = best[i][cm]
                comps += 1
                continue
            bl = float("inf")
            chosen = None
            for em in pool:
                loss = _comp_loss(em[i], cm, outside)
                if loss is not None and loss < bl:
                    bl = loss
                    chosen = em
            if chosen is not None:
                oracle[i][cm] = chosen[i][cm]
                comps += 1
    return oracle, comps


def run_category(cat: str, shot: int) -> dict:
    maps, m56 = build_maps(cat, shot)
    a1_ap = pooled_ap(maps["concat_D11C24"], m56)
    out = {"category": cat, "shot": shot, "n": m56.shape[0],
           "a1_ap": round(a1_ap, 6) if a1_ap == a1_ap else None}
    for mode in ("real", "a1copy", "scale"):
        oracle, comps = oracle_paste(maps, m56, mode)
        ap = pooled_ap(oracle, m56)
        out[f"{mode}_ap"] = round(ap, 6) if ap == ap else None
        out[f"{mode}_delta"] = round(ap - a1_ap, 6) if (ap == ap and a1_ap == a1_ap) else None
        out[f"{mode}_comps"] = comps
    shuf = []
    for s in SHUFFLE_SEEDS:
        oracle, comps = oracle_paste(maps, m56, "shuffle", seed=s)
        ap = pooled_ap(oracle, m56)
        shuf.append(round(ap - a1_ap, 6) if (ap == ap and a1_ap == a1_ap) else None)
    out["shuffle_mean_delta"] = round(float(np.nanmean(shuf)), 6) if shuf else None
    out["shuffle_deltas"] = shuf
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--shot", type=int, choices=[1, 2, 4], required=True)
    ap.add_argument("--category", default=None)
    args = ap.parse_args()
    p2 = ROOT / "experiments/dynamic_fusion/innovation_v12_early_fusion/02_stage0_probe"
    p2.mkdir(parents=True, exist_ok=True)
    cats = [args.category] if args.category else CATEGORIES
    rows = [run_category(c, args.shot) for c in cats]
    for r in rows:
        print(f"  {r['category']} a1={r['a1_ap']} "
              f"realΔ={r['real_delta']} a1copyΔ={r['a1copy_delta']} "
              f"scaleΔ={r['scale_delta']} shuffleΔ={r['shuffle_mean_delta']}", flush=True)

    def macro(key):
        vals = [r[key] for r in rows if r.get(key) is not None]
        return round(float(np.mean(vals)), 6) if vals else None

    summary = {
        "shot": args.shot,
        "macro_a1_ap": macro("a1_ap"),
        "macro_real_delta": macro("real_delta"),
        "macro_a1copy_delta": macro("a1copy_delta"),
        "macro_scale_delta": macro("scale_delta"),
        "macro_shuffle_mean_delta": macro("shuffle_mean_delta"),
        "scales_tested": SCALES,
        "shuffle_seeds": SHUFFLE_SEEDS,
        "per_category": [{k: r[k] for k in ("category", "a1_ap", "real_delta", "a1copy_delta",
                                             "scale_delta", "shuffle_mean_delta",
                                             "shuffle_deltas")} for r in rows],
    }
    (p2 / f"ORACLE_NULL_AUDIT_k{args.shot}.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=1), encoding="utf-8")
    print("SUMMARY " + json.dumps(summary), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
