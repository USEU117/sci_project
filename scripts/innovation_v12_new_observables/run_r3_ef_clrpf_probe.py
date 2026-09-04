"""V12-EARLY-FUSION CL-RPF cache probe (doc 26 §4.1 / §6.3) - CPU.

Question: do per-patch CROSS-DEPTH trajectories of normal-standardized residual
distances carry ORDERING information that fixed per-layer equal-weight mean/max
(static multi-layer integration) does not?

Normalization is normal-only: for each layer l the calibration (mean_l, std_l) comes
from residual distances of the SUPPORT (normal) patches against the support bank with
leave-one-patch-out. K=1 here -> neighbouring normal patches are spatially correlated;
that is a stated limitation (doc 26 §4.1), not treated as independent samples.

Pre-registered (doc 26 §4.1; fixed, no effect-selected variants):
  shot 1 (K=1 normal image per category); all 6 MPDD dev cats; D=[6,9,11], C=[6,12,18,24]
  -> all on the 32-grid (clip bilinear-resized as in A1).
  residual r_l(p) = L2 distance to the nearest layer-l support patch.
  z_l(p) = (r_l(p) - mean_l)/std_l, mean_l/std_l normal-only (LOO support stats).

Per-patch candidate score maps -> dists2map 56 grid -> pooled Pixel-AP @56:
  controls      : a1 & static2 (archived REFERENCES), mean_std, max_std
  trajectory    : dino_slope, clip_slope, mean_slope, |slope| variants, late_persist,
                  second_diff, d_minus_c
  order controls: reversed_slope, shuffled_slope (3 seeds), final_repeat_slope
Gates (macro over 6 cats at k1):
  G1 best trajectory variant - strongest static control  >= +0.003
  G2 best trajectory variant - best order control        >= +0.003
If G1/G2 fail the trajectory has no independent ordering signal on MPDD s0 k1
(= only a multi-layer ensemble), archive as negative for the trajectory claim.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "scripts" / "innovation_v12_new_observables"))

import faiss  # noqa: E402
from sklearn.metrics import average_precision_score  # noqa: E402

from run_r3_ef_stage0_null_audit import CATEGORIES, ML_ROOT  # noqa: E402
from src.utils import dists2map  # noqa: E402

D_LAY = [6, 9, 11]
C_LAY = [6, 12, 18, 24]
SHUFFLE_SEEDS = [0, 1, 2]
Z_HIGH = 1.0
TRAJ = ["dino_slope", "clip_slope", "mean_slope", "dino_aslope", "clip_aslope",
        "mean_aslope", "late_persist", "second_diff", "d_minus_c"]
ORDER_CTL = ["reversed_slope", "shuffled_slope", "final_repeat_slope"]
STATIC_CTL = ["mean_std", "max_std", "static2"]


def _l2(x: np.ndarray) -> np.ndarray:
    x = np.ascontiguousarray(x, dtype=np.float32)
    shp = x.shape
    flat = x.reshape(-1, shp[-1])
    n = np.linalg.norm(flat, axis=1, keepdims=True)
    return (flat / np.maximum(n, 1e-8)).reshape(shp)


def _resize32(x: np.ndarray) -> np.ndarray:
    import torch
    from torch.nn import functional as F

    x = np.ascontiguousarray(x, dtype=np.float32)
    pre = x.shape[:-3]
    h, w, d = x.shape[-3], x.shape[-2], x.shape[-1]
    t = torch.from_numpy(x.reshape(-1, h, w, d)).permute(0, 3, 1, 2)
    t = F.interpolate(t, size=(32, 32), mode="bilinear", align_corners=False)
    return t.permute(0, 2, 3, 1).numpy().reshape(*pre, 32, 32, d)


def _nearest_dist(q_flat, bank):
    idx = faiss.IndexFlatL2(bank.shape[1])
    idx.add(bank.astype(np.float32))
    d2, _ = idx.search(q_flat.astype(np.float32), 1)
    return np.sqrt(np.maximum(d2[:, 0], 0.0).astype(np.float64)).astype(np.float32)


def _loo_support_stats(ref_layer, excl_radius=1):
    """Normal-only (mean,std) of residual distances of support patches.

    K>1: leave-one-IMAGE-out (bank = all patches of the other normal images).
    K==1: leave-one-image-out is impossible -> per-patch residual against the same
    image's other patches with a spatial exclusion radius (Chebyshev <= excl_radius)
    so neighbouring normal patches are not treated as independent (doc 26 §4.1
    stated limitation, not an independent-sample claim).
    """
    K = ref_layer.shape[0]
    H = ref_layer.shape[1]
    r = []
    if K > 1:
        for k in range(K):
            others = np.concatenate([ref_layer[j].reshape(-1, 768) for j in range(K) if j != k], 0)
            bank = _l2(others)
            r.append(_nearest_dist(_l2(ref_layer[k]).reshape(-1, 768), bank))
    else:
        feats = _l2(ref_layer[0]).reshape(H * H, 768)
        idx = faiss.IndexFlatL2(feats.shape[1])
        idx.add(feats.astype(np.float32))
        d2, nbi = idx.search(feats.astype(np.float32), H * H)
        for i in range(H):
            for j in range(H):
                row = i * H + j
                dmin = None
                for d, nb in zip(d2[row], nbi[row]):
                    ni, nj = int(nb) // H, int(nb) % H
                    if max(abs(ni - i), abs(nj - j)) > excl_radius:
                        dmin = float(d)
                        break
                r.append(dmin if dmin is not None else float(d2[row, -1]))
    rr = np.concatenate([np.asarray(r, dtype=np.float64)])
    return float(rr.mean()), float(rr.std() + 1e-8)


def _slope_map(zmaps):
    """Per-pixel least-squares slope over depth. zmaps: list of [n,32,32] in depth order."""
    A = np.stack(zmaps)                       # [L,n,32,32]
    L = A.shape[0]
    t = np.arange(L, dtype=np.float64)
    t = (t - t.mean()) / (t.std() + 1e-9)
    s = np.sum(t[:, None, None, None] * A, axis=0) / float(np.dot(t, t))
    return s


def _late_map(zmaps):
    A = np.stack(zmaps)
    half = A[len(A) // 2:]
    return (half > Z_HIGH).mean(axis=0)


def _second_diff_map(zmaps):
    A = np.stack(zmaps)
    if A.shape[0] < 3:
        return np.zeros_like(A[0])
    return np.abs(A[2:] - 2.0 * A[1:-1] + A[:-2]).mean(axis=0)


def run_category(cat: str, shot: int) -> dict:
    z = np.load(ML_ROOT / f"ml_dino_s0_k{shot}" / f"{cat}.npz", allow_pickle=False)
    d_feat = np.asarray(z["patch_features"])     # [N,3,32,32,768]
    d_ref = np.asarray(z["ref_patch_features"])  # [3,K,32,32,768]
    masks = np.asarray(z["imgs_masks"], dtype=np.uint8)
    del z
    zc = np.load(ML_ROOT / f"ml_clip_s0_k{shot}" / f"{cat}.npz", allow_pickle=False)
    c_feat = _resize32(np.asarray(zc["patch_features"]))     # [N,4,32,32,768]
    c_ref = _resize32(np.asarray(zc["ref_patch_features"]))  # [4,K,32,32,768]
    del zc
    n = d_feat.shape[0]
    m56 = (masks[:, ::8, ::8] > 0.5).astype(np.uint8)
    del masks

    zmap = {}
    for li, lay in enumerate(D_LAY):
        bank = _l2(d_ref[li].reshape(-1, 768))
        mean_l, std_l = _loo_support_stats(d_ref[li])
        rows = np.stack([_nearest_dist(_l2(d_feat[i, li]).reshape(-1, 768), bank)
                         for i in range(n)]).reshape(n, 32, 32)
        zmap[f"dino_L{lay}"] = (rows - mean_l) / std_l
    for li, lay in enumerate(C_LAY):
        bank = _l2(c_ref[li].reshape(-1, 768))
        mean_l, std_l = _loo_support_stats(c_ref[li])
        rows = np.stack([_nearest_dist(_l2(c_feat[i, li]).reshape(-1, 768), bank)
                         for i in range(n)]).reshape(n, 32, 32)
        zmap[f"clip_L{lay}"] = (rows - mean_l) / std_l

    dZ = [zmap[f"dino_L{l}"] for l in D_LAY]
    cZ = [zmap[f"clip_L{l}"] for l in C_LAY]

    cand = {
        "mean_std": np.mean(list(zmap.values()), axis=0),
        "max_std": np.max(list(zmap.values()), axis=0),
        "dino_slope": _slope_map(dZ),
        "clip_slope": _slope_map(cZ),
        "mean_slope": 0.5 * (_slope_map(dZ) + _slope_map(cZ)),
        "dino_aslope": np.abs(_slope_map(dZ)),
        "clip_aslope": np.abs(_slope_map(cZ)),
        "mean_aslope": np.abs(0.5 * (_slope_map(dZ) + _slope_map(cZ))),
        "late_persist": 0.5 * (_late_map(dZ) + _late_map(cZ)),
        "second_diff": 0.5 * (_second_diff_map(dZ) + _second_diff_map(cZ)),
        "d_minus_c": np.mean(dZ, axis=0) - np.mean(cZ, axis=0),
        # order controls
        "reversed_slope": 0.5 * (_slope_map(dZ[::-1]) + _slope_map(cZ[::-1])),
        "final_repeat_slope": 0.5 * (_slope_map([dZ[-1]] * len(dZ))
                                     + _slope_map([cZ[-1]] * len(cZ))),
    }
    shuf = []
    for seed in SHUFFLE_SEEDS:
        rng = np.random.default_rng(seed)
        dP = rng.permutation(len(dZ))
        cP = rng.permutation(len(cZ))
        shuf.append(0.5 * (_slope_map([dZ[i] for i in dP]) + _slope_map([cZ[i] for i in cP])))
    cand["shuffled_slope"] = np.mean(shuf, axis=0)

    aps = {}
    for name, m32 in cand.items():
        m56m = np.stack([dists2map(m32[i], (448, 448))[::8, ::8] for i in range(n)])
        aps[name] = round(_pooled_ap(m56m, m56), 6)
    return {"category": cat, "aps": aps, "n": n}


def _pooled_ap(maps56, m56):
    y = (m56.ravel() > 0.5).astype(np.int32)
    s = maps56.ravel()
    if y.sum() == 0 or y.sum() == y.size:
        return float("nan")
    return float(average_precision_score(y, s))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--category", default=None)
    ap.add_argument("--shot", type=int, default=1, choices=[1, 2, 4])
    args = ap.parse_args()
    out_root = ROOT / "experiments/dynamic_fusion/innovation_v12_early_fusion/04_clrpf_probe"
    out_root.mkdir(parents=True, exist_ok=True)
    cats = [args.category] if args.category else CATEGORIES
    rows = [run_category(c, args.shot) for c in cats]
    for r in rows:
        print(" ", r["category"], json.dumps(r["aps"]), flush=True)

    names = list(rows[0]["aps"].keys())

    def macro(name):
        v = [r["aps"][name] for r in rows]
        v = [x for x in v if x == x]
        return round(float(np.mean(v)), 6) if v else None

    macro_aps = {k: macro(k) for k in names}
    refs = json.loads((ROOT / "experiments/dynamic_fusion/innovation_v12_early_fusion/03_scaif_small_gate/REFERENCES.json").read_text(encoding="utf-8"))
    macro_aps["a1"] = round(float(np.mean([r["pixel_ap_56"] for r in refs
                                           if r["kind"] == "a1" and r["shot"] == args.shot])), 6)
    macro_aps["static2"] = round(float(np.mean([r["pixel_ap_56"] for r in refs
                                                if r["kind"] == "static2" and r["shot"] == args.shot])), 6)
    # in-protocol static map controls: archived per-layer single-branch maps (same
    # 56-grid map protocol as the candidates), equal-weight mean of the 7 maps.
    lw_path = ROOT / "experiments/dynamic_fusion/innovation_v12_early_fusion/02_stage0_probe" / f"LAYERWISE_RESULTS_k{args.shot}.csv"
    if lw_path.exists():
        import csv as _csv

        lw = {}
        with open(lw_path, newline="", encoding="utf-8") as fh:
            for row in _csv.DictReader(fh):
                for k in ("dino_L6", "dino_L9", "dino_L11", "clip_L6", "clip_L12", "clip_L18", "clip_L24"):
                    lw.setdefault(k, []).append(float(row[k]))
        best_single = max(float(np.mean(lw[k])) for k in lw)
        macro_aps["best_single_layer_map"] = round(best_single, 6)
    print("MACRO", json.dumps(macro_aps), flush=True)

    static_set = STATIC_CTL + (["best_single_layer_map"] if "best_single_layer_map" in macro_aps else [])
    strong_static = max(macro_aps[k] for k in static_set)
    best_traj = max(macro_aps[k] for k in TRAJ)
    best_traj_name = max(TRAJ, key=lambda k: macro_aps[k])
    strong_order = max(macro_aps[k] for k in ORDER_CTL)
    g1 = (best_traj - strong_static) >= 0.003
    g2 = (best_traj - strong_order) >= 0.003
    summary = {"shot": args.shot, "macro_aps": macro_aps,
               "best_trajectory": {"name": best_traj_name, "ap": best_traj},
               "strongest_static_control": strong_static,
               "best_order_control": strong_order,
               "g1_delta_vs_static": round(best_traj - strong_static, 6),
               "g2_delta_vs_order": round(best_traj - strong_order, 6),
               "g1_traj_vs_static_ge_0p003": bool(g1),
               "g2_traj_vs_shuffled_order_ge_0p003": bool(g2),
               "per_category": rows}
    (out_root / f"CLRPF_PROBE_k{args.shot}.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=1), encoding="utf-8")
    print("SUMMARY", json.dumps({k: summary[k] for k in
          ("best_trajectory", "strongest_static_control", "best_order_control",
           "g1_delta_vs_static", "g2_delta_vs_order",
           "g1_traj_vs_static_ge_0p003", "g2_traj_vs_shuffled_order_ge_0p003")}),
          flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
