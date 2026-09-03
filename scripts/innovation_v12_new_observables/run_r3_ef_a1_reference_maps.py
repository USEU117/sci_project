"""V12-EARLY-FUSION: frozen-A1 reference Pixel-AP maps from the v3 A1 caches.

Purpose (doc 23 Stage 0 / g4 leg-2 + reference table for g1):
  Build the A1 concat map (deepest fused: DINO L11 grid32 + CLIP L24 resized32,
  w=0.5, per-branch L2 -> concat -> L2, faiss IndexFlatL2 k=1, dist/2,
  dists2map 448 sigma=4 -> [::8,::8] = 56x56) DIRECTLY from the frozen A1
  caches (v3_direction_a features_vitb14_s0_k{shot} + features_s0_k{shot}),
  using common.build_fused_blocks (exact A1 parity).  Writes per-category
  pooled Pixel-AP so DEEPEST_PARITY_REPORT can diff against the probe's
  internal A1-equivalent map (concat_D11C24 built from ml caches).

Run (.venv-patchcore, CPU; after stage0 probe to avoid CPU contention):
  .venv-patchcore\\Scripts\\python.exe scripts\\innovation_v12_new_observables\\run_r3_ef_a1_reference_maps.py
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

import faiss  # noqa: E402

from industrial_ad.innovation_v10_portfolio.common import (  # noqa: E402
    MAP_SIZE, STRIDE, build_fused_blocks, load_features,
)

from src.utils import dists2map  # noqa: E402

CATEGORIES = ["bracket_black", "bracket_brown", "bracket_white", "connector", "metal_plate", "tubes"]
BASE = ROOT / "outputs/dynamic_fusion/v3_direction_a"


def pooled_ap(maps56: np.ndarray, m56: np.ndarray) -> float:
    from sklearn.metrics import average_precision_score

    y = (m56.ravel() > 0.5).astype(np.int32)
    s = maps56.ravel()
    if y.sum() == 0 or y.sum() == y.size:
        return float("nan")
    return float(average_precision_score(y, s))


def run_category(cat: str, shot: int) -> dict:
    dino = load_features(BASE / f"features_vitb14_s0_k{shot}/anomalydino_visual/{cat}.npz")
    clip = load_features(BASE / f"features_s0_k{shot}/anomalyclip_text/{cat}.npz")
    feat, ref, sample_ids, masks, grid = build_fused_blocks(dino, clip)
    del dino, clip

    n = feat.shape[0]
    h, w, d = ref.shape[1:4]
    bank = ref.reshape(-1, d).astype(np.float32)
    idx = faiss.IndexFlatL2(d)
    idx.add(bank)
    n_ref = ref.shape[0]

    grid_dist = np.empty((n, h * w), dtype=np.float32)
    qf = feat.reshape(-1, d).astype(np.float32)
    dists, _ = idx.search(qf, k=1)
    grid_dist[:] = dists[:, 0].reshape(n, h * w)
    del feat, ref, qf, dists

    map56 = np.stack([
        dists2map((grid_dist[i] / 2.0).reshape(h, w), MAP_SIZE)[::STRIDE, ::STRIDE]
        for i in range(n)
    ]).astype(np.float32)
    del grid_dist

    m56 = (masks[:, ::STRIDE, ::STRIDE] > 0.5).astype(np.uint8)
    return {
        "category": cat,
        "shot": shot,
        "n": n,
        "n_ref": int(n_ref),
        "grid": list(grid),
        "pixel_ap_56": pooled_ap(map56, m56),
        "alignment": {
            "sample_ids_ok": int(sample_ids.shape[0]),
            "masks_448": list(masks.shape),
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--shots", type=int, nargs="+", default=[1, 2, 4])
    parser.add_argument("--category", default=None)
    args = parser.parse_args()
    out_dir = ROOT / "experiments/dynamic_fusion/innovation_v12_early_fusion/02_stage0_probe"
    out_dir.mkdir(parents=True, exist_ok=True)

    cats = [args.category] if args.category else CATEGORIES
    rows = []
    for shot in args.shots:
        for cat in cats:
            r = run_category(cat, shot)
            rows.append(r)
            print(f"  A1ref {cat} k{shot} ap56={r['pixel_ap_56']} grid={r['grid']}", flush=True)

    per_shot = {}
    for shot in args.shots:
        rr = [r for r in rows if r["shot"] == shot]
        per_shot[str(shot)] = {
            "mean_pixel_ap_56": round(float(np.mean([r["pixel_ap_56"] for r in rr])), 6),
            "categories": {r["category"]: round(r["pixel_ap_56"], 6) for r in rr},
        }
    payload = {"rows": rows, "per_shot": per_shot}
    (out_dir / "A1_REFERENCE_MAPS.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=1), encoding="utf-8")
    print("A1REF " + json.dumps(per_shot), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
