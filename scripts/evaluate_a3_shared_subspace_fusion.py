"""Direction A — A3: shared-subspace cross-modal fusion (NO centering).

A2 failed because a *random* orthogonal projection cannot align the heterogeneous
DINO(384) / CLIP(768) spaces. A2b (CCA) failed because it centers on the ref mean,
which breaks the origin semantics that L2-normalize + KNN rely on.

A3 fixes both: it learns the cross-modal alignment from the *normal references only*
via the SVD of the cross-covariance  M = D_ref^T C_ref  (refs L2-normalized first,
but NOT centered). The leading left/right singular vectors give orthonormal
projections A (DINO -> r) and B (CLIP -> r) onto a shared r-dim subspace. This is
exactly CCA *without* the centering step (since on L2-normalized refs D^T D ~ I,
the CCA whitening collapses to identity), and the rank-truncation to r regularizes
the few-shot fit (prevents overfitting, unlike full Procrustes).

Fusion happens in the shared r-dim subspace, then the fused features go through the
same normalize + KNN memory bank (distance = anomaly) as A1.

Fuse modes:
  align  -> same-position concat [d', c']
  cross  -> full cross-attention context concat [d', ctx]   (DINO queries, CLIP keys/values)
  gate   -> diagonal (same-position) cosine gate, concat [d', g*c']

Leakage discipline: A/B are fit on normal refs only; test patches use the same A/B.
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from scipy.special import softmax
from sklearn.preprocessing import normalize

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "methods" / "anomalydino"))

from evaluate_a1_feature_fusion import (
    load_features,
    resize_patches,
    score_via_memory,
    compute_metrics,
)
from industrial_ad.fusion.alignment import build_alignment_plan

STRIDE: int = 8


def _shared_subspace(d_ref_norm: np.ndarray, c_ref_norm: np.ndarray, r: int, skip_first: bool):
    """SVD of cross-covariance (no centering) -> orthonormal projections A, B.

    d_ref_norm: [P, dino_dim] L2-normalized ref patches
    c_ref_norm: [P, clip_dim] L2-normalized ref patches
    Returns A [dino_dim, r], B [clip_dim, r], singular values s[:r].

    The leading singular vector of M = D^T C is the global mean direction (rank-1
    term d_mean^T c_mean), which dominates the cross-covariance. `skip_first` drops
    it, which is the no-centering analogue of CCA's centering: it aligns on the
    *variation* directions without ever translating the origin.
    """
    M = d_ref_norm.T @ c_ref_norm  # [dino_dim, clip_dim]
    U, s, Vt = np.linalg.svd(M, full_matrices=False)
    if skip_first:
        U = U[:, 1:]
        Vt = Vt[1:, :]
        s = s[1:]
    r = min(r, len(s))
    A = U[:, :r].astype(np.float32)   # [dino_dim, r]
    B = Vt[:r, :].T.astype(np.float32)  # [clip_dim, r]
    return A, B, s[:r]


def fuse_category(dino: dict, clip: dict, r: int, fuse: str, tau: float, skip_first: bool, map_size) -> np.ndarray:
    grid = dino["grid_size"]
    P = grid[0] * grid[1]

    alignment = build_alignment_plan(dino["sample_ids"], clip["sample_ids"])
    clip_feat = clip["patch_features"][alignment.candidate_order]
    clip_ref = clip["ref_patch_features"]

    dino_feat = dino["patch_features"]        # [N, P, dino_dim]
    dino_ref = dino["ref_patch_features"]     # [R, P, dino_dim]

    clip_feat = resize_patches(clip_feat, grid)   # [N, P, clip_dim]
    clip_ref = resize_patches(clip_ref, grid)     # [R, P, clip_dim]

    Dd = dino_feat.shape[-1]
    Dc = clip_feat.shape[-1]

    # Fit A/B on L2-normalized refs (no centering).
    d_ref_norm = normalize(dino_ref.reshape(-1, Dd))
    c_ref_norm = normalize(clip_ref.reshape(-1, Dc))
    A, B, sv = _shared_subspace(d_ref_norm, c_ref_norm, r, skip_first)
    r_eff = A.shape[1]

    def project(d, c):
        M = d.shape[0]
        dn = normalize(d.reshape(-1, Dd)).reshape(M, P, Dd)
        cn = normalize(c.reshape(-1, Dc)).reshape(M, P, Dc)
        d_proj = normalize((dn.reshape(-1, Dd) @ A).reshape(-1, r_eff)).reshape(M, P, r_eff)
        c_proj = normalize((cn.reshape(-1, Dc) @ B).reshape(-1, r_eff)).reshape(M, P, r_eff)
        out = []
        for i in range(M):
            q = d_proj[i]  # [P, r_eff]
            v = c_proj[i]  # [P, r_eff]
            if fuse == "align":
                out.append(np.concatenate([q, v], axis=1))
            elif fuse == "cross":
                attn = softmax((q @ v.T) / tau, axis=1)  # [P, P]
                ctx = attn @ v                            # [P, r_eff]
                out.append(np.concatenate([q, ctx], axis=1))
            elif fuse == "gate":
                g = np.clip(np.sum(q * v, axis=1), 0.0, None)[:, None]  # cosine, both unit-norm
                out.append(np.concatenate([q, g * v], axis=1))
            else:
                raise ValueError(f"unknown fuse: {fuse}")
        return np.stack(out, axis=0)

    ref_mod = project(dino_ref, clip_ref)
    feat_mod = project(dino_feat, clip_feat)

    D = feat_mod.shape[-1]
    feat_flat = feat_mod.reshape(-1, D).astype(np.float32)
    ref_flat = ref_mod.reshape(-1, D).astype(np.float32)
    n = feat_mod.shape[0]
    return score_via_memory(feat_flat, ref_flat, 0, False, n, grid, map_size), sv


def main() -> int:
    parser = argparse.ArgumentParser(description="A3 shared-subspace cross-modal fusion evaluation")
    parser.add_argument("--dino-features", type=Path, required=True)
    parser.add_argument("--clip-features", type=Path, required=True)
    parser.add_argument("--baseline-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=0, choices=[0, 1, 2])
    parser.add_argument("--r", type=int, default=128, help="shared subspace dim (rank truncation)")
    parser.add_argument("--fuse", choices=("align", "cross", "gate"), default="cross")
    parser.add_argument("--tau", type=float, default=0.3, help="cross-attention temperature")
    parser.add_argument("--skip-first", action="store_true",
                        help="drop the leading (global-mean) singular direction")
    parser.add_argument("--map-size", type=int, default=448)
    parser.add_argument("--output-dir", type=Path, default=None)
    args = parser.parse_args()

    map_size = (args.map_size, args.map_size)
    manifest = json.loads(
        (ROOT / "data" / "splits" / "mpdd" / "manifest.json").read_text(encoding="utf-8"))
    categories = sorted(manifest["categories"])

    out_dir = args.output_dir or (
        ROOT / "experiments" / "dynamic_fusion" / "v3_direction_a" / "a3_shared_subspace" / f"seed{args.seed}")
    out_dir.mkdir(parents=True, exist_ok=True)

    results = []
    print(f"r={args.r} fuse={args.fuse} tau={args.tau} skip_first={args.skip_first}", flush=True)
    for cat in categories:
        dino_path = args.dino_features / f"{cat}.npz"
        clip_path = args.clip_features / f"{cat}.npz"
        if not dino_path.exists() or not clip_path.exists():
            print(f"  [SKIP] missing features for {cat}", flush=True)
            continue
        dino = load_features(dino_path)
        clip = load_features(clip_path)
        fused_maps, sv = fuse_category(dino, clip, args.r, args.fuse, args.tau, args.skip_first, map_size)
        fused_metrics = compute_metrics(fused_maps.astype(np.float64), dino["imgs_masks"])

        baselines = {}
        for bname in ("anomalydino_visual", "anomalyclip_text"):
            p = args.baseline_dir / bname / f"{cat}.npz"
            if p.exists():
                with np.load(p, allow_pickle=False) as data:
                    bmaps = np.asarray(data["anomaly_maps"], dtype=np.float32)
                    bmasks = np.asarray(data["imgs_masks"], dtype=np.uint8)
                baselines[bname] = compute_metrics(bmaps.astype(np.float64), bmasks)

        dino_ap = baselines.get("anomalydino_visual", {}).get("pixel_ap", float("nan"))
        row = {
            "category": cat,
            "r": args.r,
            "fuse": args.fuse,
            "tau": args.tau,
            "skip_first": args.skip_first,
            "top_singular_values": [round(float(x), 4) for x in sv[:8]],
            "fused": fused_metrics,
            "baselines": baselines,
            "delta_ap": round(fused_metrics["pixel_ap"] - dino_ap, 6),
        }
        results.append(row)
        print(f"  {cat}: fused AP={fused_metrics['pixel_ap']:.4f} AUROC={fused_metrics['pixel_auroc']:.4f} "
              f"AUPRO={fused_metrics['pixel_aupro']:.4f} | DINO AP={dino_ap:.4f} "
              f"(ΔAP={row['delta_ap']:+.4f}) sv={row['top_singular_values']}", flush=True)

    mean = {
        key: round(float(np.mean([r["fused"][key] for r in results])), 6)
        for key in ("pixel_auroc", "pixel_ap", "pixel_aupro")
    }
    mean_delta_ap = round(float(np.mean([r["delta_ap"] for r in results])), 6)
    mean_dino_ap = round(float(np.mean([r["baselines"]["anomalydino_visual"]["pixel_ap"] for r in results])), 6)

    report = {
        "pipeline": "v3_direction_a_a3_shared_subspace",
        "direction": "A_feature_level_fusion",
        "r": args.r,
        "fuse": args.fuse,
        "tau": args.tau,
        "skip_first": args.skip_first,
        "seed": args.seed,
        "stride": STRIDE,
        "dataset": "mpdd",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "mean_fused": mean,
        "mean_dino_baseline_ap": mean_dino_ap,
        "mean_delta_ap_vs_dino": mean_delta_ap,
        "per_category": results,
    }
    report_path = out_dir / f"{args.fuse}_r{args.r}_tau{args.tau:g}{'_skipfirst' if args.skip_first else ''}_report.json"
    report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nReport: {report_path}")
    print(f"Mean fused: {mean}")
    print(f"Mean DINO baseline AP: {mean_dino_ap}, mean ΔAP = {mean_delta_ap:+.6f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
