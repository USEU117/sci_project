"""Direction A — A2b: Canonical Correlation Analysis (CCA) fusion.

CCA is the canonical unsupervised multi-view fusion: fitted on the normal
references only, it learns linear projections A (DINO -> shared) and B
(CLIP -> shared) that maximize cross-modal correlation. The projected views are
fused and scored with the same KNN memory bank (distance = anomaly) as A1.

Regularized to handle few-shot (n=1024 patches, d=384/768). Fitted on refs only;
test patches are projected with the same A/B (no test labels used for fitting).
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

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


def _inv_sqrt_sym(M: np.ndarray) -> np.ndarray:
    """Symmetric inverse square root via eigen-decomposition (M = V Λ Vᵀ)."""
    w, V = np.linalg.eigh(M)
    w = np.clip(w, 1e-12, None)
    return (V / np.sqrt(w)) @ V.T


def cca_fuse_category(dino: dict, clip: dict, rho: float, n_components: int,
                      fuse: str, map_size) -> np.ndarray:
    grid = dino["grid_size"]
    P = grid[0] * grid[1]

    alignment = build_alignment_plan(dino["sample_ids"], clip["sample_ids"])
    clip_feat = clip["patch_features"][alignment.candidate_order]
    clip_ref = clip["ref_patch_features"]

    D_ref = dino["ref_patch_features"].reshape(-1, dino["ref_patch_features"].shape[-1])
    C_ref = resize_patches(clip_ref, grid).reshape(-1, clip_ref.shape[-1])
    D = dino["patch_features"].reshape(-1, dino["patch_features"].shape[-1])
    C = resize_patches(clip_feat, grid).reshape(-1, clip_feat.shape[-1])

    # center on normal ref mean
    d_mean = D_ref.mean(0)
    c_mean = C_ref.mean(0)
    Dr = D_ref - d_mean
    Cr = C_ref - c_mean

    dd = D_ref.shape[-1]
    cd = C_ref.shape[-1]
    Sdd = (Dr.T @ Dr) / P + rho * np.eye(dd)
    Scc = (Cr.T @ Cr) / P + rho * np.eye(cd)
    Sdc = (Dr.T @ Cr) / P

    Sdd_is = _inv_sqrt_sym(Sdd)
    Scc_is = _inv_sqrt_sym(Scc)
    T = Sdd_is @ Sdc @ Scc_is
    U, s, Vt = np.linalg.svd(T, full_matrices=False)
    r = min(n_components, len(s))

    A = Sdd_is @ U[:, :r]        # [384, r]
    B = Scc_is @ Vt[:r].T        # [768, r]

    Dp = (D - d_mean) @ A
    Cp = (C - c_mean) @ B
    Drp = Dr @ A
    Crp = Cr @ B

    if fuse == "concat":
        F = np.concatenate([Dp, Cp], axis=1)
        Fr = np.concatenate([Drp, Crp], axis=1)
    elif fuse == "sum":
        F = Dp + Cp
        Fr = Drp + Crp
    else:
        raise ValueError(f"unknown fuse: {fuse}")

    F = F.astype(np.float32)
    Fr = Fr.astype(np.float32)
    n = dino["patch_features"].shape[0]
    return score_via_memory(F, Fr, 0, False, n, grid, map_size)


def main() -> int:
    parser = argparse.ArgumentParser(description="A2b CCA fusion evaluation")
    parser.add_argument("--dino-features", type=Path, required=True)
    parser.add_argument("--clip-features", type=Path, required=True)
    parser.add_argument("--baseline-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=0, choices=[0, 1, 2])
    parser.add_argument("--rho", type=float, default=1e-2, help="regularization")
    parser.add_argument("--n-components", type=int, default=128)
    parser.add_argument("--fuse", choices=("concat", "sum"), default="concat")
    parser.add_argument("--map-size", type=int, default=448)
    parser.add_argument("--output-dir", type=Path, default=None)
    args = parser.parse_args()

    map_size = (args.map_size, args.map_size)
    manifest = json.loads(
        (ROOT / "data" / "splits" / "mpdd" / "manifest.json").read_text(encoding="utf-8"))
    categories = sorted(manifest["categories"])

    out_dir = args.output_dir or (
        ROOT / "experiments" / "dynamic_fusion" / "v3_direction_a" / "a2b_cca" / f"seed{args.seed}")
    out_dir.mkdir(parents=True, exist_ok=True)

    results = []
    print(f"rho={args.rho} n_components={args.n_components} fuse={args.fuse}", flush=True)
    for cat in categories:
        dino_path = args.dino_features / f"{cat}.npz"
        clip_path = args.clip_features / f"{cat}.npz"
        if not dino_path.exists() or not clip_path.exists():
            print(f"  [SKIP] missing features for {cat}", flush=True)
            continue
        dino = load_features(dino_path)
        clip = load_features(clip_path)
        fused_maps = cca_fuse_category(dino, clip, args.rho, args.n_components, args.fuse, map_size)
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
            "rho": args.rho,
            "n_components": args.n_components,
            "fuse": args.fuse,
            "fused": fused_metrics,
            "baselines": baselines,
            "delta_ap": round(fused_metrics["pixel_ap"] - dino_ap, 6),
        }
        results.append(row)
        print(f"  {cat}: fused AP={fused_metrics['pixel_ap']:.4f} AUROC={fused_metrics['pixel_auroc']:.4f} "
              f"AUPRO={fused_metrics['pixel_aupro']:.4f} | DINO AP={dino_ap:.4f} "
              f"(ΔAP={row['delta_ap']:+.4f})", flush=True)

    mean = {
        key: round(float(np.mean([r["fused"][key] for r in results])), 6)
        for key in ("pixel_auroc", "pixel_ap", "pixel_aupro")
    }
    mean_delta_ap = round(float(np.mean([r["delta_ap"] for r in results])), 6)
    mean_dino_ap = round(float(np.mean([r["baselines"]["anomalydino_visual"]["pixel_ap"] for r in results])), 6)

    report = {
        "pipeline": "v3_direction_a_a2b_cca",
        "direction": "A_feature_level_fusion",
        "rho": args.rho,
        "n_components": args.n_components,
        "fuse": args.fuse,
        "seed": args.seed,
        "stride": STRIDE,
        "dataset": "mpdd",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "mean_fused": mean,
        "mean_dino_baseline_ap": mean_dino_ap,
        "mean_delta_ap_vs_dino": mean_delta_ap,
        "per_category": results,
    }
    report_path = out_dir / f"{args.fuse}_rho{args.rho:g}_r{args.n_components}_report.json"
    report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nReport: {report_path}")
    print(f"Mean fused: {mean}")
    print(f"Mean DINO baseline AP: {mean_dino_ap}, mean ΔAP = {mean_delta_ap:+.6f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
