"""PSMF R0 CPU pre-check (doc 22 s7.1/s7.3): defect-component size stratification.

Uses the ALREADY-EXPORTED 448x448 GT masks (ml_dino npz imgs_masks) to measure,
per MPDD development category (seed0 k1): number of defect images, defect
component area distribution, and micro-defect share. This selects the 2-3 cats
for the PSMF phase R0 and pre-registers the "smallest-25%-by-area" bin used by
the doc 22 s7.3 gate (+0.015 Pixel-AP over A1 on that bin).

Micro-definition used for the pre-check only (reported, not gating): component
area <= 14x14 px (one dino patch cell at the 448 grid). The gating bin is the
within-category smallest-25%-by-area component set.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts" / "innovation_v12_new_observables"))
from run_r3_ef_stage0_null_audit import CATEGORIES, ML_ROOT  # noqa: E402

PATCH_PX = 14 * 14          # one dino patch cell at the 448-grid (196 px)


def _components(mask: np.ndarray):
    """8-connectivity connected components -> list of areas (px)."""
    from scipy.ndimage import label

    lbl, n = label(mask > 0, structure=np.ones((3, 3), dtype=int))
    if n == 0:
        return np.array([], dtype=np.int64)
    areas = np.bincount(lbl.ravel())[1:]
    return areas


def run_cat(cat: str) -> dict:
    z = np.load(ML_ROOT / f"ml_dino_s0_k1" / f"{cat}.npz", allow_pickle=False)
    masks = np.asarray(z["imgs_masks"], dtype=np.uint8)     # [N,448,448]
    del z
    n_def = int((masks.reshape(masks.shape[0], -1).sum(1) > 0).sum())
    areas = []
    for i in range(masks.shape[0]):
        areas.extend(_components(masks[i]).tolist())
    areas = np.asarray(areas, dtype=np.float64)
    if areas.size == 0:
        return {"category": cat, "n_images": masks.shape[0], "n_defect_images": n_def,
                "n_components": 0, "note": "no defect components in this split"}
    pcts = {f"p{q}": float(np.percentile(areas, q)) for q in (25, 50, 75, 90)}
    micro_share = float((areas <= PATCH_PX).mean())         # <=1 dino patch cell
    small25_area_max = float(np.quantile(areas, 0.25))
    return {"category": cat, "n_images": masks.shape[0], "n_defect_images": n_def,
            "n_components": int(areas.size), "area_pcts": {k: round(v, 1) for k, v in pcts.items()},
            "micro_share_le1patch": round(micro_share, 4),
            "smallest25_area_max": round(small25_area_max, 1),
            "area_min_max": [float(areas.min()), float(areas.max())]}


def main() -> int:
    rows = [run_cat(c) for c in CATEGORIES]
    for r in rows:
        print(json.dumps(r), flush=True)
    out_root = ROOT / "experiments/dynamic_fusion/innovation_v12_new_observables/psmf"
    out_root.mkdir(parents=True, exist_ok=True)
    (out_root / "PSMF_PRE_CHECK.json").write_text(
        json.dumps(rows, ensure_ascii=False, indent=1), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
