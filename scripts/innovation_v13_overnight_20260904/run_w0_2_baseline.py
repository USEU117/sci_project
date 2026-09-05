"""W0.2 Baseline identity + cheap bilinear signal on FROZEN A1 (doc 27 s4.2).

Deliverables:
  (i)   naming reconciliation: A1_FROZEN (REFERENCES kind=a1 / A1_REFERENCE_MAPS)
        vs STATIC2 vs MEAN_STD7 per-cat k1 numbers from archived JSONs.
  (ii)  frozen-A1 small-sample MAP parity: ml-cache final-layer fused maps vs
        v3-cache canonical fused maps (1 cat, k1), atol/rtol 1e-6 reported.
  (iii) six-class k1: frozen A1 "32-grid match then dists2map-interpolate" vs
        "same final-layer features bilinear to 56 then match" (cheap engineering
        line; NOT an A1 claim, NOT cross-branch innovation).

All CPU (faiss). Deterministic.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
for p in (str(ROOT), str(ROOT / "src"), str(ROOT / "scripts")):
    sys.path.insert(0, p)

import faiss  # noqa: E402
from sklearn.metrics import average_precision_score  # noqa: E402
from src.utils import dists2map  # noqa: E402

from industrial_ad.innovation_v10_portfolio.common import (  # noqa: E402
    MAP_SIZE, STRIDE, build_fused_blocks, load_features, resize_patches,
)

CATEGORIES = ["bracket_black", "bracket_brown", "bracket_white", "connector", "metal_plate", "tubes"]
V3 = ROOT / "outputs/dynamic_fusion/v3_direction_a"
ML = ROOT / "outputs/dynamic_fusion/v12_early_fusion"


def _pooled_ap(maps56, m56):
    y = (m56.ravel() > 0.5).astype(np.int32)
    s = maps56.ravel()
    if y.sum() == 0 or y.sum() == y.size:
        return float("nan")
    return float(average_precision_score(y, s))


def fused_maps(feat, ref, masks, grid):
    """KNN at `grid` on L2-normalised fused rows -> dists2map 448 [::8] maps56."""
    n = feat.shape[0]
    h, w = grid
    d = feat.shape[-1]
    bank = ref.reshape(-1, d).astype(np.float32)
    idx = faiss.IndexFlatL2(d)
    idx.add(bank)
    qf = feat.reshape(-1, d).astype(np.float32)
    dists, _ = idx.search(qf, k=1)
    maps56 = np.stack([
        dists2map((dists[:, 0].reshape(n, h, w)[i] / 2.0), MAP_SIZE)[::STRIDE, ::STRIDE]
        for i in range(n)]).astype(np.float32)
    m56 = (masks[:, ::STRIDE, ::STRIDE] > 0.5).astype(np.uint8)
    return maps56, m56


def v3_frozen(cat, shot=1):
    dino = load_features(V3 / f"features_vitb14_s0_k{shot}/anomalydino_visual/{cat}.npz")
    clip = load_features(V3 / f"features_s0_k{shot}/anomalyclip_text/{cat}.npz")
    feat, ref, _, masks, grid = build_fused_blocks(dino, clip)
    return feat, ref, masks, grid


def _ml_final_dict(branch, cat, shot):
    z = np.load(ML / f"ml_{branch}_s0_k{shot}" / f"{cat}.npz", allow_pickle=False)
    feat = np.asarray(z["patch_features"])
    ref = np.asarray(z["ref_patch_features"])
    li = 2 if branch == "dino" else 3          # L11 (dino) / L24 (clip)
    grid = tuple(int(v) for v in z["grid_size"])
    return {
        "patch_features": feat[:, li],
        "ref_patch_features": ref[li],
        "sample_ids": np.asarray(z["sample_ids"]),
        "imgs_masks": np.asarray(z["imgs_masks"]),
        "grid_size": grid,
    }


def ml_frozen(cat, shot=1):
    dd = _ml_final_dict("dino", cat, shot)
    cc = _ml_final_dict("clip", cat, shot)
    feat, ref, _, masks, grid = build_fused_blocks(dd, cc)
    return feat, ref, masks, grid


def frozen_bilinear56(cat, shot=1):
    """Same final-layer features, both branches bilinear-resized to 56 BEFORE the
    fused KNN (everything else identical to build_fused_blocks)."""
    dino = load_features(V3 / f"features_vitb14_s0_k{shot}/anomalydino_visual/{cat}.npz")
    clip = load_features(V3 / f"features_s0_k{shot}/anomalyclip_text/{cat}.npz")
    g56 = (56, 56)
    fd = dict(dino)
    fd["patch_features"] = resize_patches(dino["patch_features"], g56)
    fd["ref_patch_features"] = resize_patches(dino["ref_patch_features"], g56)
    fd["grid_size"] = g56
    fc = dict(clip)
    fc["patch_features"] = resize_patches(clip["patch_features"], g56)
    fc["ref_patch_features"] = resize_patches(clip["ref_patch_features"], g56)
    feat, ref, _, masks, _ = build_fused_blocks(fd, fc)
    return feat, ref, masks, g56


def main() -> int:
    out_root = ROOT / "experiments/dynamic_fusion/innovation_v13_overnight_20260904"
    out_root.mkdir(parents=True, exist_ok=True)
    rows, parity = [], {}
    for cat in CATEGORIES:
        # canonical frozen A1 at 32
        f32, r32, m32, g32 = v3_frozen(cat, 1)
        m56_32, mm56 = fused_maps(f32, r32, m32, g32)
        ap32 = _pooled_ap(m56_32, mm56)
        # frozen bilinear-56
        f56, r56, m56_, g56 = frozen_bilinear56(cat, 1)
        m56b, mm56b = fused_maps(f56, r56, m56_, g56)
        ap56b = _pooled_ap(m56b, mm56b)
        rows.append({"category": cat, "shot": 1, "A1_frozen_32": round(ap32, 6),
                     "A1_bilinear56": round(ap56b, 6),
                     "delta_bl56_minus_32": round(ap56b - ap32, 6)})
        print(" ", json.dumps(rows[-1]), flush=True)
        del f32, r32, m32, f56, r56, m56_

    # (ii) small-sample map parity: ml-cache vs v3-cache fused maps, bracket_brown k1
    cat = "bracket_brown"
    f_a, r_a, m_a, g_a = v3_frozen(cat, 1)
    ma_a, mm = fused_maps(f_a, r_a, m_a, g_a)
    f_b, r_b, m_b, g_b = ml_frozen(cat, 1)
    ma_b, _ = fused_maps(f_b, r_b, m_b, g_b)
    diff = np.abs(ma_a.astype(np.float64) - ma_b.astype(np.float64))
    parity = {"category": cat, "n_maps": int(ma_a.shape[0]),
              "max_abs_diff": float(diff.max()),
              "mean_abs_diff": float(diff.mean()),
              "identical_atol1e-6": bool(np.allclose(ma_a, ma_b, atol=1e-6, rtol=0)),
              "identical_rtol1e-6": bool(np.allclose(ma_a, ma_b, atol=0, rtol=1e-6))}
    print(" PARITY", json.dumps(parity), flush=True)

    macro32 = round(float(np.mean([r["A1_frozen_32"] for r in rows])), 6)
    macro56 = round(float(np.mean([r["A1_bilinear56"] for r in rows])), 6)
    payload = {"rows": rows, "macro_k1": {"A1_frozen_32": macro32, "A1_bilinear56": macro56,
                                          "delta": round(macro56 - macro32, 6)},
               "parity_ml_vs_v3": parity,
               "note": "A1_frozen_32 = v3-cache canonical concat (REFERENCES a1); naming audit table in W0_AUDITS.md"}
    (out_root / "W0_baseline.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=1), encoding="utf-8")
    print("MACRO", macro32, macro56, round(macro56 - macro32, 6), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
