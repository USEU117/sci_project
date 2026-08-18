"""Direction A — A2: cross-modal attention modulation of DINO (no learned params).

For each image, DINO patches act as queries and CLIP patches act as keys/values.
A cosine cross-attention (temperature-scaled softmax) aggregates a CLIP context
per DINO patch, which is then fused with the DINO patch. Fused features build a
shared KNN memory bank (distance = anomaly), exactly as in A1.

This is "attention modulates DINO": the CLIP context is sample-adaptive (dynamic),
unlike A1's static same-location concatenation.

Leakage discipline: the memory bank is built from normal references only.
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


def _fixed_clip_projection(dino_dim: int, clip_dim: int) -> np.ndarray:
    """Fixed (seeded) random projection CLIP -> DINO dim, used for cross-modal attention.

    Columns are unit-normed so projected CLIP patches keep a comparable scale.
    Reproducible and label-free (no training on labels; the same matrix is reused
    for refs and tests, keeping the memory-bank distance well defined).
    """
    rng = np.random.default_rng(0)
    W = rng.normal(size=(dino_dim, clip_dim)).astype(np.float32)
    W /= np.linalg.norm(W, axis=0, keepdims=True)
    return W


def modulate_category(dino: dict, clip: dict, tau: float, fuse_mode: str, map_size) -> np.ndarray:
    grid = dino["grid_size"]
    P = grid[0] * grid[1]

    alignment = build_alignment_plan(dino["sample_ids"], clip["sample_ids"])
    clip_feat = clip["patch_features"][alignment.candidate_order]
    clip_ref = clip["ref_patch_features"]

    dino_feat = dino["patch_features"]
    dino_ref = dino["ref_patch_features"]

    clip_feat = resize_patches(clip_feat, grid)
    clip_ref = resize_patches(clip_ref, grid)

    Dd = dino_feat.shape[-1]
    Dc = clip_feat.shape[-1]
    W = _fixed_clip_projection(Dd, Dc)

    def modulate(d, c):
        M = d.shape[0]
        d = normalize(d.reshape(-1, Dd)).reshape(M, P, Dd)
        c = normalize(c.reshape(-1, Dc)).reshape(M, P, Dc)
        c = (c @ W.T).astype(np.float32)  # [M, P, Dd]
        c = normalize(c.reshape(-1, Dd)).reshape(M, P, Dd)
        out = []
        for i in range(M):
            q = d[i]  # [P, Dd]
            v = c[i]  # [P, Dd]
            attn = softmax((q @ v.T) / tau, axis=1)  # [P, P]
            ctx = attn @ v  # [P, Dd]
            if fuse_mode == "concat":
                out.append(np.concatenate([q, ctx], axis=1))
            elif fuse_mode == "gate":
                # scalar relevance gate per patch, modulated DINO only (residual gate)
                gate = np.clip((q @ ctx.T).diagonal() / tau, 0.0, None)[:, None]
                out.append(q * (1.0 + gate))
            else:
                raise ValueError(f"unknown fuse_mode: {fuse_mode}")
        return np.stack(out, axis=0)

    ref_mod = modulate(dino_ref, clip_ref)
    feat_mod = modulate(dino_feat, clip_feat)

    D = feat_mod.shape[-1]
    feat_flat = feat_mod.reshape(-1, D).astype(np.float32)
    ref_flat = ref_mod.reshape(-1, D).astype(np.float32)
    n = feat_mod.shape[0]
    return score_via_memory(feat_flat, ref_flat, 0, False, n, grid, map_size)


def main() -> int:
    parser = argparse.ArgumentParser(description="A2 cross-modal attention fusion evaluation")
    parser.add_argument("--dino-features", type=Path, required=True)
    parser.add_argument("--clip-features", type=Path, required=True)
    parser.add_argument("--baseline-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=0, choices=[0, 1, 2])
    parser.add_argument("--tau", type=float, default=0.3, help="attention temperature")
    parser.add_argument("--fuse-mode", choices=("concat", "gate"), default="concat")
    parser.add_argument("--map-size", type=int, default=448)
    parser.add_argument("--output-dir", type=Path, default=None)
    args = parser.parse_args()

    map_size = (args.map_size, args.map_size)
    manifest = json.loads(
        (ROOT / "data" / "splits" / "mpdd" / "manifest.json").read_text(encoding="utf-8"))
    categories = sorted(manifest["categories"])

    out_dir = args.output_dir or (
        ROOT / "experiments" / "dynamic_fusion" / "v3_direction_a" / "a2_attention" / f"seed{args.seed}")
    out_dir.mkdir(parents=True, exist_ok=True)

    results = []
    print(f"tau={args.tau} fuse_mode={args.fuse_mode}", flush=True)
    for cat in categories:
        dino_path = args.dino_features / f"{cat}.npz"
        clip_path = args.clip_features / f"{cat}.npz"
        if not dino_path.exists() or not clip_path.exists():
            print(f"  [SKIP] missing features for {cat}", flush=True)
            continue
        dino = load_features(dino_path)
        clip = load_features(clip_path)
        fused_maps = modulate_category(dino, clip, args.tau, args.fuse_mode, map_size)
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
            "tau": args.tau,
            "fuse_mode": args.fuse_mode,
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
        "pipeline": "v3_direction_a_a2",
        "direction": "A_feature_level_fusion",
        "tau": args.tau,
        "fuse_mode": args.fuse_mode,
        "seed": args.seed,
        "stride": STRIDE,
        "dataset": "mpdd",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "mean_fused": mean,
        "mean_dino_baseline_ap": mean_dino_ap,
        "mean_delta_ap_vs_dino": mean_delta_ap,
        "per_category": results,
    }
    report_path = out_dir / f"{args.fuse_mode}_tau{args.tau:g}_report.json"
    report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nReport: {report_path}")
    print(f"Mean fused: {mean}")
    print(f"Mean DINO baseline AP: {mean_dino_ap}, mean ΔAP = {mean_delta_ap:+.6f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
