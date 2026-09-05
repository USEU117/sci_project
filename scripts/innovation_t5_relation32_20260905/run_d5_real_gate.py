"""Direction-5 MAIN real gate: 32-grid relational descriptors on real MPDD (doc36).

A1 frozen concat protocol (pca0/whiten0/w0.5, 32-grid, faiss k=1 -> 448 map,
Pixel-AP stride 8). Candidates computed per image (neighbour context inside each
image, memory and query under the same rule):
  C0 = z (A1 baseline, unchanged frozen path -> must reproduce frozen numbers)
  C1 = concat(z, 3x3-neighbour mean z)   (3072-D)
  C2 = concat(z, up/down/left/right z)   (5x1536-D)
Overall 6-cat macro and the parts_mismatch subgroup (sample_ids containing
'/parts_mismatch/'), A1 Pixel-AP/AUROC on the full test split. No fitting.
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import faiss  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402

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
VARS = ("C0", "C1", "C2")


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


def _row_unit_flat(x):
    x = np.ascontiguousarray(x, dtype=np.float32)
    faiss.normalize_L2(x)
    return x


def _neigh_mean(xg):
    gg = xg.shape[0]
    p = torch.nn.functional.pad(xg.permute(2, 0, 1)[None], (1, 1, 1, 1), mode="reflect")[0]
    p = p.permute(1, 2, 0)
    return (p[0:gg, 0:gg] + p[0:gg, 1:gg + 1] + p[0:gg, 2:gg + 2] +
            p[1:gg + 1, 0:gg] + p[1:gg + 1, 2:gg + 2] +
            p[2:gg + 2, 0:gg] + p[2:gg + 2, 1:gg + 1] + p[2:gg + 2, 2:gg + 2]) / 8.0


def _desc_flat(x_img, variant):
    """x_img [G,G,D] unit rows -> [G*G,dv] unit rows (torch)."""
    Gx = x_img.shape[0]
    cells = x_img.reshape(Gx * Gx, -1)
    if variant == "C0":
        return cells
    if variant == "C1":
        nb = _neigh_mean(x_img).reshape(Gx * Gx, -1)
        return torch.nn.functional.normalize(torch.cat([cells, nb], dim=-1), dim=-1)
    p = torch.nn.functional.pad(x_img.permute(2, 0, 1)[None], (1, 1, 1, 1), mode="reflect")[0]
    p = p.permute(1, 2, 0)
    up = p[0:Gx, 1:Gx + 1].reshape(Gx * Gx, -1)
    dn = p[2:Gx + 2, 1:Gx + 1].reshape(Gx * Gx, -1)
    lf = p[1:Gx + 1, 0:Gx].reshape(Gx * Gx, -1)
    rt = p[1:Gx + 1, 2:Gx + 2].reshape(Gx * Gx, -1)
    return torch.nn.functional.normalize(torch.cat([cells, up, dn, lf, rt], dim=-1), dim=-1)


def score_maps_from_desc(desc_feat, desc_ref, grid, n_images, map_size):
    ref = np.ascontiguousarray(desc_ref.numpy(), dtype=np.float32)
    q = np.ascontiguousarray(desc_feat.numpy(), dtype=np.float32)
    faiss.normalize_L2(ref)
    faiss.normalize_L2(q)
    index = faiss.IndexFlatL2(ref.shape[1])
    index.add(ref)
    distances, _ = index.search(q, k=1)
    dists = (distances[:, 0] / 2.0).reshape(n_images, *grid)
    maps = np.stack([dists2map(d, map_size) for d in dists]).astype(np.float32)
    return maps


def build_bank_per_image(x, variant):
    """x [n,G,G,D] unit rows -> per-image desc [n, G*G, dv]."""
    out = []
    for i in range(x.shape[0]):
        t = torch.as_tensor(x[i], dtype=torch.float32)
        out.append(_desc_flat(t, variant))
    return torch.cat(out, dim=0) if False else out


def metrics_subset(maps, masks, idx):
    return A1.compute_metrics(maps[idx].astype(np.float64), masks[idx])


def run_cat(cat, dino_dir, clip_dir, map_size):
    dino = A1.load_features(dino_dir / f"{cat}.npz")
    clip = A1.load_features(clip_dir / f"{cat}.npz")
    feat, ref, grid = fused_concat(dino, clip)
    n = feat.shape[0]
    d = feat.shape[-1]
    Gx = grid[0]
    # unit rows (A1 score step does the same); keep C0 on the frozen path exactly:
    feat_u = _row_unit_flat(feat.reshape(-1, d)).reshape(n, Gx, Gx, d)
    ref_u = _row_unit_flat(ref.reshape(-1, d)).reshape(ref.shape[0], Gx, Gx, d)
    masks = dino["imgs_masks"]
    ids = dino["sample_ids"]
    pm_idx = np.array([i for i, s in enumerate(ids) if "/parts_mismatch/" in str(s)])
    row = {"category": cat, "n_test": n, "n_pm": int(pm_idx.size)}
    for var in VARS:
        qs = []
        for i in range(n):
            qs.append(_desc_flat(torch.as_tensor(feat_u[i], dtype=torch.float32), var))
        q = torch.cat(qs, dim=0)
        bs = [None] * ref.shape[0]
        for i in range(ref.shape[0]):
            bs[i] = _desc_flat(torch.as_tensor(ref_u[i], dtype=torch.float32), var)
        b = torch.cat(bs, dim=0)
        maps = score_maps_from_desc(q, b, grid, n, map_size)
        row[var] = A1.compute_metrics(maps.astype(np.float64), masks)
        row[f"d_ap_{var}"] = 0.0
        if var == "C0":
            row["full_ap_ref"] = row["C0"]["pixel_ap"]
        if pm_idx.size:
            row[f"pm_{var}"] = metrics_subset(maps, masks, pm_idx)
        else:
            row[f"pm_{var}"] = None
    row["d_ap_C1"] = row["C1"]["pixel_ap"] - row["C0"]["pixel_ap"]
    row["d_ap_C2"] = row["C2"]["pixel_ap"] - row["C0"]["pixel_ap"]
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
    rows = []
    for cat in cats:
        r = run_cat(cat, dino_dir, clip_dir, map_size)
        rows.append(r)
        print(f"  {cat}: A1 AP={r['C0']['pixel_ap']:.4f} C1 dAP={r['d_ap_C1']:+.4f} "
              f"C2 dAP={r['d_ap_C2']:+.4f} | pm C1 dAP="
              f"{(r['pm_C1']['pixel_ap'] - r['pm_C0']['pixel_ap']) if r['pm_C0'] is not None else float('nan'):.4f}",
              flush=True)

    def mean(fn):
        return float(np.mean([fn(r) for r in rows]))

    best_var = max(("C1", "C2"), key=lambda v: mean(lambda r, vv=v: r[f"d_ap_{vv}"]))
    d_ap = {v: round(mean(lambda r, vv=v: r[f"d_ap_{vv}"]), 6) for v in ("C1", "C2")}
    worst = {v: round(min(r[f"d_ap_{v}"] for r in rows), 6) for v in ("C1", "C2")}
    d_auroc = {v: round(mean(lambda r, vv=v: r[vv]["pixel_auroc"] - r["C0"]["pixel_auroc"]), 6) for v in ("C1", "C2")}
    pm_cats = [r for r in rows if r["pm_C0"] is not None]

    def pm_mean(fn):
        return float(np.mean([fn(r) for r in pm_cats])) if pm_cats else None

    pm_d = {v: (round(pm_mean(lambda r, vv=v: r[f"pm_{vv}"]["pixel_ap"] - r["pm_C0"]["pixel_ap"]), 6)
                if pm_cats else float("nan")) for v in ("C1", "C2")}
    g_r1 = d_ap[best_var] >= 0.01
    g_r2 = worst[best_var] >= -0.03
    g_r3 = bool(pm_cats) and pm_d[best_var] >= 0.0
    g_r4 = d_auroc[best_var] >= -0.005
    out = {"seed": args.seed, "shot": args.shot, "created_utc": datetime.now(timezone.utc).isoformat(),
           "best_var": best_var, "mean_A1_ap": round(mean(lambda r: r["C0"]["pixel_ap"]), 6),
           "delta_ap": d_ap, "worst_delta_ap": worst, "delta_auroc": d_auroc,
           "pm_mean_A1_ap": round(pm_mean(lambda r: r["pm_C0"]["pixel_ap"]), 6) if pm_cats else None,
           "pm_delta_ap": pm_d,
           "G_R1": g_r1, "G_R2": g_r2, "G_R3": g_r3, "G_R4": g_r4,
           "per_category": rows}
    OUT = ROOT / "experiments/dynamic_fusion/innovation_t5_relation32_20260905"
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / f"REAL_D5_s0_k{args.shot}.json").write_text(json.dumps(out, ensure_ascii=False, indent=1),
                                                       encoding="utf-8")
    print(f"[k{args.shot}] best={best_var} mean dAP={d_ap[best_var]:+.4f} worst={worst[best_var]:+.4f} "
          f"dAUROC={d_auroc[best_var]:+.4f} pm dAP={pm_d[best_var]:+.4f}")
    print(f"[k{args.shot}] G_R1={g_r1} G_R2={g_r2} G_R3={g_r3} G_R4={g_r4}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
