"""Route C — CAPM R0 pixel evaluation (MPDD seed0 k1, all 6 classes).

Feasibility (inlier stats) PASSED -> this pre-registered post-pass evaluation.
score = d_global + 0.25 * relu(d_pos - d_global), d_pos = fused nearest neighbour
restricted to the aligned-coordinate radius-2 neighbourhood; alignment-reliable
images only (RANSAC inlier >= 0.30); otherwise exact A1 identity.

  .venv-patchcore\\Scripts\\python.exe scripts\\innovation_v10_portfolio\\run_r0_capm_eval.py
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from industrial_ad.innovation_v10_portfolio import capm, common
from industrial_ad.innovation_v10_portfolio.common import build_fused_blocks, load_features

CATEGORIES = ["bracket_black", "bracket_brown", "bracket_white", "connector", "metal_plate", "tubes"]


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def run_category(dino_cache: Path, clip_cache: Path, cat: str) -> dict:
    dino = load_features(dino_cache / f"{cat}.npz")
    clip = load_features(clip_cache / f"{cat}.npz")
    feat, ref, _, masks, grid = build_fused_blocks(dino, clip, dino_weight=0.5)
    n, h, w, d = feat.shape
    k = ref.shape[0]
    assert k == 1, "CAPM R0 uses k1 (single reference)"

    dino_feat = dino["patch_features"]                 # [N, H, W, 768] for alignment
    dino_ref = dino["ref_patch_features"][0]           # [H, W, 768]
    rf_align = capm.normalize_rows(dino_ref.reshape(-1, dino_ref.shape[-1]))
    ref_fused = ref[0]                                  # [H, W, D] fused ref

    # A1 d_global grids [N, H, W]
    import faiss

    ff = feat.reshape(-1, d).astype(np.float32)
    faiss.normalize_L2(ff)  # already normalized by builder; idempotent
    ref_f = ref.reshape(-1, d).astype(np.float32)
    faiss.normalize_L2(ref_f)
    index = faiss.IndexFlatL2(d)
    index.add(ref_f)
    dists, _ = index.search(ff, 1)
    d_global_grid = (dists[:, 0] / 2.0).reshape(n, h, w).astype(np.float32)

    rng = np.random.default_rng(20260903)
    capm_grids = np.empty_like(d_global_grid)
    rnd_grids = np.empty_like(d_global_grid)
    align_stats = []
    n_unreliable = 0
    n_random_unreliable = 0
    for i in range(n):
        qf = capm.normalize_rows(dino_feat[i].reshape(-1, dino_feat.shape[-1]))
        est = capm.estimate_affine(qf, rf_align, grid)
        # identity fallback when unreliable
        capm_grids[i] = capm.candidate_grid(feat[i], ref_fused, est.get("M"), est["inlier_ratio"],
                                            d_global_grid[i])
        if not est["ok"] or est["inlier_ratio"] < capm.RELIABLE_INLIER:
            n_unreliable += 1
        # random-homography control (deterministic per image)
        a = float(rng.uniform(-8, 8))
        tx, ty = float(rng.uniform(-4, 4)), float(rng.uniform(-4, 4))
        s = float(rng.uniform(0.95, 1.05))
        rad = np.deg2rad(a)
        M_rand = np.array([[s * np.cos(rad), -np.sin(rad), tx],
                           [np.sin(rad), s * np.cos(rad), ty]], dtype=np.float32)
        rnd_grids[i] = capm.candidate_grid(feat[i], ref_fused, M_rand, est["inlier_ratio"],
                                           d_global_grid[i])
        if not est["ok"] or est["inlier_ratio"] < capm.RELIABLE_INLIER:
            n_random_unreliable += 1
        align_stats.append({"image": i, "ok": est["ok"], "inlier_ratio": est["inlier_ratio"],
                            "identity": not est["ok"] or est["inlier_ratio"] < capm.RELIABLE_INLIER})

    maps_a1 = common.maps_from_patches(d_global_grid)
    maps_capm = common.maps_from_patches(capm_grids)
    maps_rnd = common.maps_from_patches(rnd_grids)
    met_a1 = common.compute_pixel_metrics(maps_a1.astype(np.float64), masks)
    met_capm = common.compute_pixel_metrics(maps_capm.astype(np.float64), masks)
    met_rnd = common.compute_pixel_metrics(maps_rnd.astype(np.float64), masks)

    # identity-exactness for unreliable images: candidate grid == d_global grid
    id_maxdiff = float(np.max(np.abs(capm_grids[np.array([s["identity"] for s in align_stats])]
                                     - d_global_grid[np.array([s["identity"] for s in align_stats])]))
                       if n_unreliable else 0.0)

    row = {
        "category": cat,
        "n_images": n,
        "n_reliable": n - n_unreliable,
        "n_identity_fallback": n_unreliable,
        "identity_fallback_ratio": round(n_unreliable / n, 4),
        "identity_maxdiff_on_fallback": id_maxdiff,
        "a1_pixel_ap": round(met_a1["pixel_ap"], 6),
        "capm_pixel_ap": round(met_capm["pixel_ap"], 6),
        "random_ctrl_pixel_ap": round(met_rnd["pixel_ap"], 6),
        "delta_ap_vs_a1": round(met_capm["pixel_ap"] - met_a1["pixel_ap"], 6),
        "random_delta_ap_vs_a1": round(met_rnd["pixel_ap"] - met_a1["pixel_ap"], 6),
        "capm_pixel_auroc": round(met_capm["pixel_auroc"], 6),
        "a1_pixel_auroc": round(met_a1["pixel_auroc"], 6),
        "mean_inlier_reliable": round(float(np.mean(
            [s["inlier_ratio"] for s in align_stats if not s["identity"]])), 4),
    }
    print(f"[CAPM-eval {cat}] A1 AP={row['a1_pixel_ap']:.4f} CAPM AP={row['capm_pixel_ap']:.4f} "
          f"Δ={row['delta_ap_vs_a1']:+.4f} random Δ={row['random_delta_ap_vs_a1']:+.4f} "
          f"identity={row['n_identity_fallback']}/{n}", flush=True)
    return row


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dino-cache", type=Path,
                        default=ROOT / "outputs/dynamic_fusion/v3_direction_a/features_vitb14_s0_k1/anomalydino_visual")
    parser.add_argument("--clip-cache", type=Path,
                        default=ROOT / "outputs/dynamic_fusion/v3_direction_a/features_s0_k1/anomalyclip_text")
    parser.add_argument("--out-dir", type=Path,
                        default=ROOT / "experiments/dynamic_fusion/innovation_v10_portfolio/capm")
    args = parser.parse_args()

    out_dir = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    protocol = out_dir / "R0_PROTOCOL.json"
    if not protocol.is_file():
        raise SystemExit(f"missing pre-registered protocol: {protocol}")

    rows = []
    t0 = time.time()
    for cat in CATEGORIES:
        if not (args.dino_cache / f"{cat}.npz").is_file():
            print(f"skip {cat}: cache missing")
            continue
        rows.append(run_category(args.dino_cache, args.clip_cache, cat))

    def mean(key: str) -> float:
        return round(float(np.mean([r[key] for r in rows])), 6)

    report = {
        "route": "C_CAPM",
        "pipeline": "v10_portfolio_r0_pixel_eval",
        "seed": 0, "shot": 1, "data_role": "development",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "protocol_sha256": sha256_file(protocol),
        "code_sha256": {
            "capm": sha256_file(ROOT / "src/industrial_ad/innovation_v10_portfolio/capm.py"),
            "common": sha256_file(ROOT / "src/industrial_ad/innovation_v10_portfolio/common.py"),
            "runner": sha256_file(Path(__file__)),
        },
        "per_category": rows,
        "mean_delta_ap": mean("delta_ap_vs_a1"),
        "n_positive": sum(1 for r in rows if r["delta_ap_vs_a1"] > 0),
        "worst_category": round(min(r["delta_ap_vs_a1"] for r in rows), 6),
        "mean_random_delta_ap": mean("random_delta_ap_vs_a1"),
        "real_vs_random": round(mean("delta_ap_vs_a1") - mean("random_delta_ap_vs_a1"), 6),
        "mean_auroc_loss": round(mean("a1_pixel_auroc") - mean("capm_pixel_auroc"), 6),
        "mean_identity_ratio": mean("identity_fallback_ratio"),
        "gates": {
            "g_ap_ge_0.003": mean("delta_ap_vs_a1") >= 0.003,
            "g_4of6_positive": sum(1 for r in rows if r["delta_ap_vs_a1"] > 0) >= 4,
            "g_worst_ge_-0.015": min(r["delta_ap_vs_a1"] for r in rows) >= -0.015,
            "g_real_vs_random_ge_0.003": mean("delta_ap_vs_a1") - mean("random_delta_ap_vs_a1") >= 0.003,
        },
        "decision": "PENDING",
    }
    (out_dir / "R0_RESULT.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\nmean ΔAP = {report['mean_delta_ap']} | pos {report['n_positive']}/6 | "
          f"worst {report['worst_category']} | real-vs-random {report['real_vs_random']} | "
          f"elapsed {time.time()-t0:.0f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
