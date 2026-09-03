"""V12 NTOF (doc 22 s3) R0 gate runner (CPU, .venv-patchcore, after exports).

Normal-only mechanism gate (doc 22 s3.4 + pre-registered R0_PROTOCOL).
All maps are scored on the frozen 56x56 (STRIDE=8) protocol grid.

Run:
  .venv-patchcore\\Scripts\\python.exe scripts\\innovation_v12_new_observables\\run_r2_ntof_gates.py
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

import cv2  # noqa: E402
import faiss  # noqa: E402
from sklearn.preprocessing import normalize as sk_norm  # noqa: E402

from industrial_ad.innovation_v10_portfolio.common import (  # noqa: E402
    MAP_SIZE, STRIDE, build_fused_blocks, load_features,
)
from src.utils import dists2map  # noqa: E402

CATEGORIES = ["bracket_black", "bracket_brown", "bracket_white", "connector", "metal_plate", "tubes"]
BASE = ROOT / "outputs/dynamic_fusion/v3_direction_a"
VAR_ROOT = ROOT / "outputs/dynamic_fusion"
TAN_DIR = VAR_ROOT / "ntof_tangents_s0_k1"
RANKS = [2, 3, 4]
RNG_SEED = 0
PCT = 95.0
HELD_KEYS = ["held_exposure", "held_gamma", "held_white_balance",
             "held_lr_brightness_gradient", "held_specular_blob"]
SYN_KEYS = ["cutpaste", "local_erasure", "thin_scratch"]


def to56(g: np.ndarray) -> np.ndarray:
    return dists2map(g, MAP_SIZE)[::STRIDE, ::STRIDE].astype(np.float32)


def l2(x: np.ndarray) -> np.ndarray:
    x = x.astype(np.float32)
    return sk_norm(x.reshape(-1, x.shape[-1])).reshape(x.shape)


def resize32(x: np.ndarray) -> np.ndarray:
    """[...,37,37,d] -> [...,32,32,d] bilinear via torch (A1 alignment parity)."""
    import torch
    from torch.nn import functional as F

    x = x.astype(np.float32)
    pre = x.shape[:-3]
    h, w, d = x.shape[-3], x.shape[-2], x.shape[-1]
    t = torch.from_numpy(x.reshape(-1, h, w, d)).permute(0, 3, 1, 2)  # [n,d,h,w]
    t = F.interpolate(t, size=(32, 32), mode="bilinear", align_corners=False)
    return t.permute(0, 2, 3, 1).numpy().reshape(*pre, 32, 32, d)


def fused_rows(d32: np.ndarray, c32: np.ndarray) -> np.ndarray:
    """A1 fused rows [n,1536] unit (per-branch L2 + w=0.5 concat + L2)."""
    d = l2(d32.reshape(-1, d32.shape[-1]))
    c = l2(c32.reshape(-1, c32.shape[-1]))
    f = np.concatenate([0.5 * d, 0.5 * c], axis=-1)
    return l2(f)


def nearest_dev(q: np.ndarray, bank: np.ndarray) -> np.ndarray:
    """q [P,d] unit; returns d = q - nearest bank row (unit rows)."""
    index = faiss.IndexFlatL2(q.shape[1])
    index.add(bank.astype(np.float32))
    _, idx = index.search(q.astype(np.float32), k=1)
    return q - bank[idx[:, 0]]


def residual_norm(diff: np.ndarray, u: np.ndarray) -> np.ndarray:
    if u is None or u.shape[1] == 0:
        return np.linalg.norm(diff, axis=1)
    return np.linalg.norm(diff - (diff @ u) @ u.T, axis=1)


def basis(v: np.ndarray, r: int) -> np.ndarray:
    _, _, vh = np.linalg.svd(v.astype(np.float64), full_matrices=False)
    return vh[:r].T.astype(np.float32)


def random_basis(d: int, r: int, seed: int) -> np.ndarray:
    q, _ = np.linalg.qr(np.random.RandomState(seed).standard_normal((d, r)))
    return q.astype(np.float32)


# ------------------------------------------------------------ tangent fit pass

def fit_tangent(cat: str) -> dict:
    """Fit per-branch tangents U (ranks 2/3/4) + fused U_f(r=3) + returns info."""
    with np.load(VAR_ROOT / f"ntof_features_dino_s0_k1/{cat}.npz", allow_pickle=False) as vd, \
         np.load(VAR_ROOT / f"ntof_features_clip_s0_k1/{cat}.npz", allow_pickle=False) as vc:
        d_o = np.asarray(vd["ref_orig_feat"], dtype=np.float32)   # [K,32,32,768]
        d_v = np.asarray(vd["ref_var_feat"], dtype=np.float32)    # [K,15,32,32,768]
        c_o = np.asarray(vc["ref_orig_feat"], dtype=np.float32)   # [K,37,37,768]
        c_v = np.asarray(vc["ref_var_feat"], dtype=np.float32)    # [K,15,37,37,768]
    k, h, w, d = d_o.shape
    p_d = h * w
    d_o_flat = l2(d_o.reshape(-1, d)).reshape(k, p_d, d)
    d_v_flat = l2(d_v.reshape(-1, d)).reshape(k, 15, p_d, d)
    vd_pool = (d_v_flat - d_o_flat[:, None, :, :]).reshape(-1, d)
    h2, w2, _ = c_o.shape[1:4]
    p_c = h2 * w2
    c_o_flat = l2(c_o.reshape(-1, 768)).reshape(k, p_c, 768)
    c_v_flat = l2(c_v.reshape(-1, 768)).reshape(k, 15, p_c, 768)
    vc_pool = (c_v_flat - c_o_flat[:, None, :, :]).reshape(-1, 768)
    # fused tangent (r=3) at 32 grid
    c_o32 = resize32(c_o.reshape(-1, h2, w2, 768)).reshape(k, 32, 32, 768)
    c_v32 = resize32(c_v.reshape(-1, h2, w2, 768)).reshape(k, 15, 32, 32, 768)
    f_o_rows = fused_rows(d_o, c_o32)                          # [k*1024, 1536]
    f_v_rows = fused_rows(d_v.reshape(-1, 32, 32, 768),
                          c_v32.reshape(-1, 32, 32, 768))      # [k*15*1024, 1536]
    vf_pool = (f_v_rows.reshape(k, 15, 1024, 1536)
               - f_o_rows.reshape(k, 1024, 1536)[:, None, :, :]).reshape(-1, 1536)
    tan = {"d": {r: basis(vd_pool, r) for r in RANKS},
           "c": {r: basis(vc_pool, r) for r in RANKS},
           "f": basis(vf_pool, 3)}
    # shuffled-pairing control: break PATCH correspondence between variant and base
    # (variant of patch p is paired with the base feature of a random other patch)
    def shuffled(v_flat: np.ndarray, o_flat: np.ndarray, seed: int) -> np.ndarray:
        rng = np.random.RandomState(seed)
        kk, nv, pp, dd = v_flat.shape
        out = []
        for r in range(kk):
            for j in range(nv):
                perm = rng.permutation(pp)
                out.append(v_flat[r, j] - o_flat[r, perm])
        return np.concatenate(out, axis=0)

    tan["shuf_d"] = basis(shuffled(d_v_flat, d_o_flat, 7), 3)
    tan["shuf_c"] = basis(shuffled(c_v_flat, c_o_flat, 8), 3)
    return tan


def tangent_cache():
    TAN_DIR.mkdir(parents=True, exist_ok=True)
    for cat in CATEGORIES:
        out = TAN_DIR / f"{cat}.npz"
        if out.is_file():
            continue
        t = fit_tangent(cat)
        np.savez(out, **{f"d_{r}": t["d"][r] for r in RANKS},
                 **{f"c_{r}": t["c"][r] for r in RANKS},
                 f_3=t["f"], shuf_d=t["shuf_d"], shuf_c=t["shuf_c"])
        print(f"[tangent] {cat} done", flush=True)


def load_tangent(cat: str) -> dict:
    z = np.load(TAN_DIR / f"{cat}.npz", allow_pickle=False)
    return {"d": {r: z[f"d_{r}"] for r in RANKS},
            "c": {r: z[f"c_{r}"] for r in RANKS},
            "f": z["f_3"], "shuf_d": z["shuf_d"], "shuf_c": z["shuf_c"]}


# ------------------------------------------------------------- score pipeline

def score_queries(qd_list, qc_list, tan, need_ranks):
    """Return dict of 56-maps for every query: a1, ntof_r, dino_r, clip_r, concat_r3.

    qd_list/qc_list: list of raw branch grids (native sizes). Uses banks from cache
    built per-category (passed via module-level _BANKS set by run_category).
    """
    b = _BANKS  # {d_bank,c_bank,f_bank}
    out = {m: [] for m in _MAPKEYS(need_ranks)}
    for i in range(len(qd_list)):
        qd = qd_list[i]
        qc = qc_list[i]
        hd, wd, dd = qd.shape
        hc, wc, dc = qc.shape
        qdn = l2(qd.reshape(1, hd, wd, dd)).reshape(-1, dd)
        qcn = l2(qc.reshape(1, hc, wc, dc)).reshape(-1, dc)
        ddq = nearest_dev(qdn, b["d_bank"])
        dcq = nearest_dev(qcn, b["c_bank"])
        # A1 fused distance (raw branch grids; fused_rows normalizes internally)
        qf = fused_rows(qd.reshape(1, hd, wd, dd),
                        resize32(qc.reshape(1, hc, wc, dc).astype(np.float32)).reshape(hd, wd, dc))
        out["a1"].append(to56(nearest_map(qf, b["f_bank"], hd, wd)))
        for r in need_ranks:
            rd = residual_norm(ddq, tan["d"][r]).reshape(hd, wd)
            rc = residual_norm(dcq, tan["c"][r]).reshape(hc, wc)
            rc32 = cv2.resize(rc, (32, 32), interpolation=cv2.INTER_LINEAR)
            out[f"ntof_{r}"].append(to56(0.5 * rd + 0.5 * rc32))
            out[f"dino_{r}"].append(to56(rd))
            out[f"clip_{r}"].append(to56(rc))
        if 3 in need_ranks:
            qf32 = fused_rows(qd, resize32(qc.reshape(1, hc, wc, dc)).reshape(hd, wd, dc))
            dqf = nearest_dev(qf32, b["f_bank"])
            rf = residual_norm(dqf, tan["f"]).reshape(hd, wd)
            out[f"concat_3"].append(to56(rf))
            for tag, ud, uc in (("rand", tan.get("rand_d"), tan.get("rand_c")),
                                ("wrong", tan.get("wrong_d"), tan.get("wrong_c")),
                                ("shuf", tan.get("shuf_d"), tan.get("shuf_c"))):
                if ud is not None and uc is not None:
                    rd = residual_norm(ddq, ud).reshape(hd, wd)
                    rc = residual_norm(dcq, uc).reshape(hc, wc)
                    rc32 = cv2.resize(rc, (32, 32), interpolation=cv2.INTER_LINEAR)
                    out[f"ctrl_{tag}"].append(to56(0.5 * rd + 0.5 * rc32))
    return {m: np.stack(v) for m, v in out.items()}


def nearest_map(qf: np.ndarray, bank: np.ndarray, h: int, w: int) -> np.ndarray:
    index = faiss.IndexFlatL2(qf.shape[1])
    index.add(bank.astype(np.float32))
    dists, _ = index.search(qf.astype(np.float32), k=1)
    return (dists[:, 0] / 2.0).reshape(h, w)


def _MAPKEYS(need_ranks):
    keys = ["a1"]
    for r in need_ranks:
        keys += [f"ntof_{r}", f"dino_{r}", f"clip_{r}"]
    if 3 in need_ranks:
        keys += ["concat_3", "ctrl_rand", "ctrl_wrong", "ctrl_shuf"]
    return keys


_BANKS: dict = {}


def run_category(cat: str, tan: dict, wrong_tan: dict) -> dict:
    dino = load_features(BASE / f"features_vitb14_s0_k1/anomalydino_visual/{cat}.npz")
    clip = load_features(BASE / f"features_s0_k1/anomalyclip_text/{cat}.npz")
    with np.load(VAR_ROOT / f"ntof_features_dino_s0_k1/{cat}.npz", allow_pickle=False) as vd, \
         np.load(VAR_ROOT / f"ntof_features_clip_s0_k1/{cat}.npz", allow_pickle=False) as vc, \
         np.load(VAR_ROOT / f"ntof_syn_masks_s0_k1/{cat}.npz", allow_pickle=False) as vm:
        good_rel = [str(x) for x in vd["good_rel"]]
        d_good_h = np.asarray(vd["good_held_feat"], dtype=np.float32)   # [G,5,32,32,768]
        d_good_s = np.asarray(vd["good_syn_feat"], dtype=np.float32)    # [G,3,32,32,768]
        c_good_h = np.asarray(vc["good_held_feat"], dtype=np.float32)   # [G,5,37,37,768]
        c_good_s = np.asarray(vc["good_syn_feat"], dtype=np.float32)
        syn_masks = np.asarray(vm["syn_masks"], dtype=np.uint8)         # [G,3,1024,1024]

    good_ids_cache = [str(s) for s in dino["sample_ids"]]
    pos = {sid: i for i, sid in enumerate(good_ids_cache)}
    d_orig = np.asarray([dino["patch_features"][pos[rel]] for rel in good_rel], dtype=np.float32)
    c_orig = np.asarray([clip["patch_features"][pos[rel]] for rel in good_rel], dtype=np.float32)
    n_g = len(good_rel)

    # fused A1 bank from the cache refs (guaranteed parity with the frozen A1)
    fref = build_fused_blocks(dino, clip, dino_weight=0.5)[1]            # [K,32,32,1536]
    _BANKS["d_bank"] = l2(np.asarray(dino["ref_patch_features"]).reshape(-1, 768))
    _BANKS["c_bank"] = l2(np.asarray(clip["ref_patch_features"]).reshape(-1, 768))
    _BANKS["f_bank"] = fref.reshape(-1, fref.shape[-1]).astype(np.float32)
    _BANKS["f_bank"] = l2(_BANKS["f_bank"])

    tan = dict(tan)
    tan["rand_d"] = random_basis(768, 3, RNG_SEED)
    tan["rand_c"] = random_basis(768, 3, RNG_SEED + 1)
    tan["wrong_d"] = wrong_tan["d"][3]
    tan["wrong_c"] = wrong_tan["c"][3]

    # ---- original good maps (thresholds) - all methods
    maps_orig = score_queries(list(d_orig), list(c_orig), tan, RANKS)
    # ---- held-out illumination maps
    maps_held = score_queries(
        [d_good_h[i, f] for i in range(n_g) for f in range(5)],
        [c_good_h[i, f] for i in range(n_g) for f in range(5)], tan, RANKS)
    # ---- synthetic defect maps
    maps_syn = score_queries(
        [d_good_s[i, f] for i in range(n_g) for f in range(3)],
        [c_good_s[i, f] for i in range(n_g) for f in range(3)], tan, RANKS)

    # thresholds from original-good pooled maps per method
    th = {m: float(np.percentile(maps_orig[m].ravel(), PCT)) for m in maps_orig}

    def fp_ratio_stats(mapm, refmap, thr_m, thr_ref):
        n = mapm.shape[0]
        fpm = (mapm > thr_m).sum(axis=(1, 2))
        fpr = (refmap > thr_ref).sum(axis=(1, 2))
        fpr = np.maximum(fpr, 1)
        return (fpm / fpr).astype(np.float64)

    stats = {"n_good": n_g}
    for m in maps_held:
        if m == "a1":
            continue
        ratio = fp_ratio_stats(maps_held[m], maps_held["a1"], th[m], th["a1"])
        ratio = ratio.reshape(n_g, 5)
        stats[f"fp_ratio_median_{m}"] = round(float(np.median(ratio)), 4)
        stats[f"fp_ratio_mean_{m}"] = round(float(np.mean(ratio)), 4)
        # normalized effect
        fpm = (maps_held[m] > th[m]).sum(axis=(1, 2)).astype(np.float64)
        fpa = np.maximum((maps_held["a1"] > th["a1"]).sum(axis=(1, 2)), 1)
        stats[f"effect_{m}"] = round(float(np.mean(1.0 - fpm / fpa)), 4)
    # per-family median ratios for ntof_3
    ratio3 = fp_ratio_stats(maps_held["ntof_3"], maps_held["a1"], th["ntof_3"], th["a1"]).reshape(n_g, 5)
    stats["fp_ratio_median_ntof_3_per_family"] = {
        HELD_KEYS[f]: round(float(np.median(ratio3[:, f])), 4) for f in range(5)}

    # g2: synthetic preservation medians (ntof_3 & a1 in-cell means)
    for kind_i, kind in enumerate(SYN_KEYS):
        vals = []
        for i in range(n_g):
            m56 = cv2.resize(syn_masks[i, kind_i], (56, 56), interpolation=cv2.INTER_NEAREST) > 0
            if m56.sum() == 0:
                continue
            idx = i * 3 + kind_i
            a = maps_syn["a1"][idx][m56]
            nt = maps_syn["ntof_3"][idx][m56]
            if a.mean() > 0:
                vals.append(float(nt.mean() / a.mean()))
        stats[f"syn_preserve_median_{kind}"] = round(float(np.median(vals)), 4) if vals else None
        stats[f"syn_preserve_n_{kind}"] = len(vals)

    # g5 encoder controls (rank 3): effect of dino/clip/concat vs ntof
    stats["g5_best_single"] = max(stats.get("effect_dino_3", -9), stats.get("effect_clip_3", -9))
    return {"category": cat, **stats}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", type=Path, default=ROOT /
                        "experiments/dynamic_fusion/innovation_v12_new_observables/ntof")
    parser.add_argument("--category", default=None)
    parser.add_argument("--fit-tangents-only", action="store_true")
    args = parser.parse_args()

    protocol = args.out_dir / "R0_PROTOCOL.json"
    if not protocol.is_file():
        raise SystemExit(f"missing pre-registered protocol: {protocol}")

    if args.fit_tangents_only:
        tangent_cache()
        return 0

    tangent_cache()
    cats = [args.category] if args.category else CATEGORIES
    tangents = {c: load_tangent(c) for c in cats}
    rows = []
    for i, cat in enumerate(cats):
        wrong = tangents[cats[(i + 1) % len(cats)]]
        print(f"[NTOF gates] {cat}", flush=True)
        rows.append(run_category(cat, tangents[cat], wrong))
        print("   " + json.dumps({k: v for k, v in rows[-1].items()
                                  if v is not None}), flush=True)

    def med(key):
        vals = [r[key] for r in rows if r.get(key) is not None]
        return round(float(np.median(vals)), 4) if vals else None

    fp_ratio_r = {r: med(f"fp_ratio_median_ntof_{r}") for r in RANKS}
    fp_ratio_r_mean = {r: med(f"fp_ratio_mean_ntof_{r}") for r in RANKS}
    syn_pres = {k: med(f"syn_preserve_median_{k}") for k in SYN_KEYS}
    eff = {m: med(f"effect_{m}") for m in
           ["ntof_3", "dino_3", "clip_3", "concat_3", "ctrl_rand", "ctrl_wrong", "ctrl_shuf"]}
    g2_pass = all(v is not None and v >= 0.90 for v in syn_pres.values())
    g1_pass = fp_ratio_r[3] is not None and fp_ratio_r[3] <= 0.75
    g3_pass = all(eff["ntof_3"] is not None and eff.get(m) is not None
                  and eff["ntof_3"] - eff[m] >= 0.05 for m in ("ctrl_rand", "ctrl_wrong", "ctrl_shuf"))
    best_single = max([eff.get("dino_3", -9), eff.get("clip_3", -9)])
    g5_pass = eff["ntof_3"] is not None and eff["ntof_3"] >= best_single - 0.01
    rank_consistent = all(fp_ratio_r[r] is not None for r in RANKS)

    gates = {
        "g1_fp_reduction_ratio_le_075": g1_pass,
        "g2_synthetic_preserve_ge_090": g2_pass,
        "g3_controls_diff_ge_005": g3_pass,
        "g4_rank_stable_234": rank_consistent,
        "g5_encoder_control": g5_pass,
    }
    if not g2_pass:
        decision = "STOP (doc 22 s3.6: synthetic preservation < 0.90)"
    elif g1_pass and g3_pass and g5_pass:
        decision = "PASS (mechanism gates met -> R1 3-seed gates would follow)"
    else:
        decision = "FAIL -> ARCHIVE NTOF, move to PRS (doc 22 s10/s13)"
    summary = {
        "seed": 0, "shot": 1,
        "fp_ratio_median_by_rank": fp_ratio_r,
        "fp_ratio_mean_by_rank": fp_ratio_r_mean,
        "syn_preserve_median": syn_pres,
        "effect": eff,
        "best_single_encoder": round(best_single, 4) if best_single > -9 else None,
        "gates": gates,
        "decision": decision,
    }
    report = {
        "route": "V12-NTOF", "seed": 0, "shot": 1,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "per_category": rows,
        "summary": summary,
    }
    (args.out_dir / "R0_RESULT.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=1), encoding="utf-8")
    (args.out_dir / "controls.json").write_text(
        json.dumps({"effect": eff, "fp_ratio_by_rank": fp_ratio_r,
                    "syn_preserve": syn_pres}, ensure_ascii=False, indent=1), encoding="utf-8")
    with open(args.out_dir / "per_category.csv", "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        header = ["category"] + [k for k in rows[0] if k != "category"]
        w.writerow(header)
        for r in rows:
            w.writerow([r["category"]] + [r.get(k) for k in header[1:]])
    print("SUMMARY " + json.dumps(summary), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
