"""Track-3 MAIN real-MPDD gate: A1 full memory vs A1 + 50% chessboard coreset (doc33).

Reuses the A1 frozen concat protocol (pca0/whiten0/w0.5, 32-grid, faiss k=1,
dists2map->448, Pixel-AP stride 8). Only the ref (memory) side is coreset via a
deterministic chessboard mask (i+j)%2==0 per ref image -> 50% of patches; query
patches are untouched. No fitting, no /test/ labels into memory, read-only eval
on the frozen feature cache. Output per-cat full/coreset Pixel-AP/AUROC + delta.
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import cv2  # noqa: E402
import faiss  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
for p in (str(ROOT / "src"), str(ROOT / "scripts"),
          str(ROOT / "methods" / "anomalydino")):
    sys.path.insert(0, p)

import evaluate_a1_feature_fusion as A1  # noqa: E402  (load_features, resize_patches, compute_metrics, STRIDE)
from industrial_ad.fusion.alignment import build_alignment_plan  # noqa: E402
from sklearn.preprocessing import normalize  # noqa: E402
from src.utils import dists2map  # noqa: E402

MANIFEST = ROOT / "data/splits/mpdd/manifest.json"
DINO_DIR = ROOT / "outputs/dynamic_fusion/v3_direction_a"
STRIDE = A1.STRIDE


def chess_mask(grid: tuple[int, int]) -> np.ndarray:
    """[G,G] bool keep (i+j)%2==0 -> ~50% of patch positions (deterministic)."""
    gi, gj = np.meshgrid(np.arange(grid[0]), np.arange(grid[1]), indexing="ij")
    return ((gi + gj) % 2 == 0).ravel()


def fused_concat(dino: dict, clip: dict):
    """Return (feat_flat, ref_flat, grid) exactly as A1 frozen concat (no PCA)."""
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
    feat = np.concatenate([w * dino_feat, (1.0 - w) * clip_feat], axis=-1)  # [N,G,G,1536]
    ref = np.concatenate([w * dino_ref, (1.0 - w) * clip_ref], axis=-1)     # [K,G,G,1536]
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


def run_cat(cat, dino_dir, clip_dir, map_size):
    dino = A1.load_features(dino_dir / f"{cat}.npz")
    clip = A1.load_features(clip_dir / f"{cat}.npz")
    feat, ref, grid = fused_concat(dino, clip)          # ref [K,G,G,1536]
    n = feat.shape[0]
    d = feat.shape[-1]
    feat_flat = feat.reshape(-1, d)
    ref_flat = ref.reshape(-1, d)
    K = ref.shape[0]
    keep = chess_mask(grid)                              # per-image 50%
    ref_cs = ref.reshape(K, -1, d)[:, keep, :].reshape(-1, d)   # K*512
    maps_full = score_maps(feat_flat, ref_flat, grid, n, map_size)
    maps_cs = score_maps(feat_flat, ref_cs, grid, n, map_size)
    gt = dino["imgs_masks"]
    mf = A1.compute_metrics(maps_full.astype(np.float64), gt)
    mc = A1.compute_metrics(maps_cs.astype(np.float64), gt)
    return {"category": cat, "ref_patches_full": int(ref_flat.shape[0]),
            "ref_patches_coreset": int(ref_cs.shape[0]),
            "full": mf, "coreset50": mc,
            "delta_ap": mc["pixel_ap"] - mf["pixel_ap"],
            "delta_auroc": mc["pixel_auroc"] - mf["pixel_auroc"],
            "delta_aupro": mc["pixel_aupro"] - mf["pixel_aupro"]}


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
    rows = []
    for cat in cats:
        r = run_cat(cat, dino_dir, clip_dir, map_size)
        rows.append(r)
        print(f"  {cat}: full AP={r['full']['pixel_ap']:.4f} cs50 AP={r['coreset50']['pixel_ap']:.4f} "
              f"dAP={r['delta_ap']:+.4f} dAUROC={r['delta_auroc']:+.4f} ref {r['ref_patches_full']}->{r['ref_patches_coreset']}",
              flush=True)
    def mean(fn):
        return float(np.mean([fn(r) for r in rows]))

    dmin = min(r["delta_ap"] for r in rows)
    g_r1 = mean(lambda r: r["delta_ap"]) >= -0.01
    g_r2 = dmin >= -0.03
    out = {"seed": args.seed, "shot": args.shot, "created_utc": datetime.now(timezone.utc).isoformat(),
           "mean_delta_ap": round(mean(lambda r: r["delta_ap"]), 6),
           "mean_full_ap": round(mean(lambda r: r["full"]["pixel_ap"]), 6),
           "mean_cs_ap": round(mean(lambda r: r["coreset50"]["pixel_ap"]), 6),
           "mean_delta_auroc": round(mean(lambda r: r["delta_auroc"]), 6),
           "mean_delta_aupro": round(mean(lambda r: r["delta_aupro"]), 6),
           "worst_delta_ap": round(dmin, 6),
           "memory_ratio": float(rows[0]["ref_patches_coreset"] / rows[0]["ref_patches_full"]),
           "G_R1": g_r1, "G_R2": g_r2,
           "decision": "REAL_GATE_PASS" if (g_r1 and g_r2) else "REAL_GATE_FAIL",
           "per_category": rows}
    OUT = ROOT / "experiments/dynamic_fusion/innovation_t3_efficiency_20260905"
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / f"REAL_GATE_s{args.seed}_k{args.shot}.json").write_text(
        json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"mean dAP={out['mean_delta_ap']:+.4f} worst={out['worst_delta_ap']:+.4f} "
          f"G_R1={g_r1} G_R2={g_r2} DECISION={out['decision']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
