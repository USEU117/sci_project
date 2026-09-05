"""Track-4 real gate: preserve-ranking memory coresets (doc34).

Fix direction for doc33 REAL_GATE_FAIL: geometric chessboard coreset (P1) drops
high-leverage neighbors; C1 keeps the top-50% highest-leverage memory units (2nd
nearest-neighbour citation counts over ref clean patches), C2 keeps a
farthest-point 50% cover. Full A1 frozen protocol (pca0/whiten0/w0.5, 32-grid,
faiss k=1 -> 448 map, Pixel-AP stride 8); only the ref (memory) rows change.
Deterministic, no fitting, no /test/ into memory. Gates G-F1/G-F2/G-F3 per doc34.
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import faiss  # noqa: E402
import numpy as np  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
for p in (str(ROOT / "src"), str(ROOT / "scripts"),
          str(ROOT / "methods" / "anomalydino")):
    sys.path.insert(0, p)

import evaluate_a1_feature_fusion as A1  # noqa: E402
from industrial_ad.fusion.alignment import build_alignment_plan  # noqa: E402
from sklearn.preprocessing import normalize  # noqa: E402
from src.utils import dists2map  # noqa: E402

MANIFEST = ROOT / "data/splits/mpdd/manifest.json"
DINO_DIR = ROOT / "outputs/dynamic_fusion/v3_direction_a"
STRIDE = A1.STRIDE
CANDS = ("full", "P1", "C1", "C2")


def chess_mask(grid):
    gi, gj = np.meshgrid(np.arange(grid[0]), np.arange(grid[1]), indexing="ij")
    return ((gi + gj) % 2 == 0).ravel()


def fused_concat(dino, clip):
    grid = dino["grid_size"]
    alignment = build_alignment_plan(dino["sample_ids"], clip["sample_ids"])
    clip_feat = clip["patch_features"][alignment.candidate_order]
    clip_ref = clip["ref_patch_features"]
    clip_feat = A1.resize_patches(clip_feat, grid)
    clip_ref = A1.resize_patches(clip_ref, grid)
    dino_feat = dino["patch_features"]
    dino_ref = dino["ref_patch_features"]
    dino_feat = normalize(dino_feat.reshape(-1, dino_feat.shape[-1])).reshape(dino_feat.shape)
    dino_ref = normalize(dino_ref.reshape(-1, dino_ref.shape[-1])).reshape(dino_ref.shape)
    clip_feat = normalize(clip_feat.reshape(-1, clip_feat.shape[-1])).reshape(clip_feat.shape)
    clip_ref = normalize(clip_ref.reshape(-1, clip_ref.shape[-1])).reshape(clip_ref.shape)
    w = 0.5
    feat = np.concatenate([w * dino_feat, (1.0 - w) * clip_feat], axis=-1)
    ref = np.concatenate([w * dino_ref, (1.0 - w) * clip_ref], axis=-1)
    return feat, ref, grid


def score_maps(feat_flat, ref_flat, grid, n_images, map_size):
    faiss.normalize_L2(ref_flat)
    faiss.normalize_L2(feat_flat)
    index = faiss.IndexFlatL2(ref_flat.shape[1])
    index.add(ref_flat)
    distances, _ = index.search(feat_flat, k=1)
    dists = (distances[:, 0] / 2.0).reshape(n_images, *grid)
    maps = np.stack([dists2map(d, map_size) for d in dists]).astype(np.float32)
    return maps


def _unit_rows(x):
    faiss.normalize_L2(x)
    return x


def keep_indices(method: str, ref_flat: np.ndarray, grid):
    """Return deterministic 50% row indices of ref_flat for P1/C1/C2."""
    M = ref_flat.shape[0]
    K = ref_flat.shape[0] // (grid[0] * grid[1])
    half = M // 2
    if method == "P1":
        return np.nonzero(np.tile(chess_mask(grid), K))[0]
    if method == "C1":
        # high-leverage: 2nd-NN citation counts among ref clean patches (unit rows)
        X = _unit_rows(ref_flat.copy())
        idx = faiss.IndexFlatL2(X.shape[1])
        idx.add(X)
        _, nn = idx.search(X, k=2)          # nn[:,0] == self
        cites = np.bincount(nn[:, 1], minlength=M).astype(np.float64)
        return np.argsort(-cites)[:half]
    if method == "C2":
        # farthest-point 50% cover on unit rows (fixed seed 0 -> order independent)
        X = _unit_rows(ref_flat.copy()).astype(np.float32)
        sel = [0]
        dmin = np.full(M, np.inf, dtype=np.float32)
        while len(sel) < half:
            d = np.sum((X - X[sel[-1]]) ** 2, axis=1)
            dmin = np.minimum(dmin, d)
            sel.append(int(np.argmax(dmin)))
        return np.asarray(sel, dtype=np.int64)
    raise ValueError(method)


def run_cat(cat, dino_dir, clip_dir, map_size):
    dino = A1.load_features(dino_dir / f"{cat}.npz")
    clip = A1.load_features(clip_dir / f"{cat}.npz")
    feat, ref, grid = fused_concat(dino, clip)
    n = feat.shape[0]
    d = feat.shape[-1]
    feat_flat = feat.reshape(-1, d)
    ref_flat = ref.reshape(-1, d)
    row = {"category": cat, "ref_patches_full": int(ref_flat.shape[0])}
    full_maps = score_maps(feat_flat, ref_flat, grid, n, map_size)
    row["full"] = A1.compute_metrics(full_maps.astype(np.float64), dino["imgs_masks"])
    for method in ("P1", "C1", "C2"):
        keep = keep_indices(method, ref_flat, grid)
        assert keep.shape[0] == ref_flat.shape[0] // 2
        maps = score_maps(feat_flat, ref_flat[keep], grid, n, map_size)
        row[method] = A1.compute_metrics(maps.astype(np.float64), dino["imgs_masks"])
        row[f"d_ap_{method}"] = row[method]["pixel_ap"] - row["full"]["pixel_ap"]
    return row


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--shot", type=int, default=2, choices=[2, 4])
    ap.add_argument("--cats", default=None)
    ap.add_argument("--map-size", type=int, default=448)
    args = ap.parse_args()
    m = json.loads(MANIFEST.read_text(encoding="utf-8"))
    cats = [c.strip() for c in args.cats.split(",")] if args.cats else sorted(m["categories"])
    dino_dir = DINO_DIR / f"features_vitb14_s{args.seed}_k{args.shot}" / "anomalydino_visual"
    clip_dir = DINO_DIR / f"features_s{args.seed}_k{args.shot}" / "anomalyclip_text"
    map_size = (args.map_size, args.map_size)
    rows = [run_cat(cat, dino_dir, clip_dir, map_size) for cat in cats]
    for r in rows:
        print(f"  {r['category']}: full AP={r['full']['pixel_ap']:.4f} | "
              + " ".join(f"{method} dAP={r[f'd_ap_{method}']:+.4f}" for method in ("P1", "C1", "C2")), flush=True)

    def mean(fn):
        return float(np.mean([fn(r) for r in rows]))

    agg = {}
    for method in ("P1", "C1", "C2"):
        agg[method] = {"mean_d_ap": round(mean(lambda r, mm=method: r[f"d_ap_{mm}"]), 6),
                       "worst_d_ap": round(min(r[f"d_ap_{method}"] for r in rows), 6),
                       "mean_d_auroc": round(mean(lambda r, mm=method: r[mm]["pixel_auroc"] - r["full"]["pixel_auroc"]), 6),
                       "mean_d_aupro": round(mean(lambda r, mm=method: r[mm]["pixel_aupro"] - r["full"]["pixel_aupro"]), 6)}
    p1_violate = agg["P1"]["mean_d_ap"] < -0.01 or agg["P1"]["worst_d_ap"] < -0.03   # expected fail (calibration)
    wins = {}
    for method in ("C1", "C2"):
        wins[method] = (agg[method]["mean_d_ap"] >= -0.01 and agg[method]["worst_d_ap"] >= -0.03
                        and agg[method]["mean_d_auroc"] >= -0.005 and agg[method]["mean_d_aupro"] >= -0.005)
    # note: this file is per-shot; two-shot consistency is resolved at decision time.
    out = {"seed": args.seed, "shot": args.shot, "created_utc": datetime.now(timezone.utc).isoformat(),
           "mean_full_ap": round(mean(lambda r: r["full"]["pixel_ap"]), 6),
           "per_method": agg, "p1_calibration_violated": p1_violate,
           "per_shot_wins": wins, "per_category": rows}
    OUT = ROOT / "experiments/dynamic_fusion/innovation_t3_efficiency_20260905"
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / f"TRACK4_REAL_s{args.seed}_k{args.shot}.json").write_text(
        json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"[k{args.shot}] P1 calibration violated={p1_violate} | C1 {agg['C1']} C2 {agg['C2']}")
    print(f"[k{args.shot}] per-shot wins C1={wins['C1']} C2={wins['C2']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
