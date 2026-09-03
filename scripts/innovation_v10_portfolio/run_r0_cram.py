"""Run R0 for Route A CRAM (task book 19 §4): MPDD seed0 shots 2 & 4.

Usage (from repo root, in .venv-patchcore):
  .venv-patchcore\\Scripts\\python.exe scripts\\innovation_v10_portfolio\\run_r0_cram.py

Outputs:
  experiments/dynamic_fusion/innovation_v10_portfolio/cram/R0_RESULT.json
  experiments/dynamic_fusion/innovation_v10_portfolio/cram/R0_DECISION.md
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

from industrial_ad.innovation_v10_portfolio import common, cram
from industrial_ad.innovation_v10_portfolio.common import build_fused_blocks, compute_pixel_metrics, load_features

CATEGORIES = ["bracket_black", "bracket_brown", "bracket_white", "connector", "metal_plate", "tubes"]


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def run_category(dino_cache: Path, clip_cache: Path, cat: str, shot: int,
                 rng: np.random.Generator) -> dict:
    dino = load_features(dino_cache / f"{cat}.npz")
    clip = load_features(clip_cache / f"{cat}.npz")
    feat, ref, _, masks, grid = build_fused_blocks(dino, clip, dino_weight=0.5)
    k = ref.shape[0]
    n, d = feat.shape[0], feat.shape[-1]
    feat_flat = feat.reshape(-1, d).astype(np.float32)

    # agreement stats over per-reference distances
    dr = common.per_reference_distances(feat_flat, ref)
    stats = cram.agreement_stats(dr)

    # A2 normal calibration (reference LOO only); meaningful only for K>=4
    normal_mad = common.ref_loo_calibration(ref)
    mad95 = cram.mad95_from_normal_mad(normal_mad) if k >= 4 else float("nan")

    # identity parity: a0 (min over per-image refs) vs pooled single-index NN
    pooled_grid = cram.pooled_min_map(feat, ref)
    a0_grid = stats["d_min"].reshape(n, *grid)
    max_identity_diff = float(np.max(np.abs(pooled_grid - a0_grid)))

    grids: dict[str, np.ndarray] = {}
    for name in ("a0", "a1", "a2"):
        if name == "a0":
            grids["a0"] = a0_grid
        elif name == "a1":
            grids["a1"] = cram.score_a1(stats).reshape(n, *grid) if k >= 2 else a0_grid
        else:
            grids["a2"] = (cram.score_a2(stats, mad95).reshape(n, *grid)
                           if (k >= 4 and np.isfinite(mad95) and mad95 > 0) else a0_grid)
    grids["med_only_control"] = stats["d_med"].reshape(n, *grid)

    row: dict = {
        "category": cat,
        "shot": shot,
        "n_ref_images": int(k),
        "n_test_images": int(n),
        "mad95_normal": None if not np.isfinite(mad95) else round(float(mad95), 8),
        "identity_pooled_vs_a0_maxdiff": max_identity_diff,
        "candidates": {},
    }
    for name, g in grids.items():
        m = common.maps_from_patches(g)
        met = compute_pixel_metrics(m.astype(np.float64), masks)
        row["candidates"][name] = {k2: round(v, 6) for k2, v in met.items()}

    # ---- strong controls ----
    controls: dict = {}

    # c1 duplicate refs (pool unchanged, K inflated): no spurious gains allowed
    if k >= 2:
        ref_dup = np.repeat(ref, 2, axis=0)
        stats_dup = cram.agreement_stats(common.per_reference_distances(feat_flat, ref_dup))
        dup = {}
        for name, gf in (("a0", cram.score_a0), ("a1", cram.score_a1)):
            g = gf(stats_dup).reshape(n, *grid)
            m = common.maps_from_patches(g)
            dup[name] = compute_pixel_metrics(m.astype(np.float64), masks)["pixel_ap"]
        controls["c1_duplicate_refs"] = {
            "inflated_k": int(ref_dup.shape[0]),
            "gain_a1_over_a0": round(dup["a1"] - dup["a0"], 6),
            "a0_ap_unchanged": round(dup["a0"] - row["candidates"]["a0"]["pixel_ap"], 9),
        }

    # c2 shuffled refs (pool preserved, per-image attribution destroyed)
    if k >= 2:
        kk, hh, ww, dd = ref.shape
        flat = ref.reshape(kk * hh * ww, dd).copy()
        perm = rng.permutation(flat.shape[0])
        flat = flat[perm]
        blocks = np.stack([flat[i * (hh * ww):(i + 1) * (hh * ww)]
                           for i in range(kk)]).reshape(kk, hh, ww, dd)
        stats_sh = cram.agreement_stats(common.per_reference_distances(feat_flat, blocks))
        sh = {}
        for name, gf in (("a0", cram.score_a0), ("a1", cram.score_a1)):
            g = gf(stats_sh).reshape(n, *grid)
            m = common.maps_from_patches(g)
            sh[name] = compute_pixel_metrics(m.astype(np.float64), masks)["pixel_ap"]
        controls["c2_shuffled_refs"] = {
            "gain_a1_over_a0": round(sh["a1"] - sh["a0"], 6),
            "a0_ap_unchanged": round(sh["a0"] - row["candidates"]["a0"]["pixel_ap"], 9),
        }

    row["controls"] = controls
    return row


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--shots", type=int, nargs="+", default=[2, 4])
    ap.add_argument("--categories", nargs="+", default=CATEGORIES)
    ap.add_argument("--dino-cache-root", type=Path,
                    default=ROOT / "outputs/dynamic_fusion/v3_direction_a")
    ap.add_argument("--clip-cache-root", type=Path,
                    default=ROOT / "outputs/dynamic_fusion/v3_direction_a")
    ap.add_argument("--out-dir", type=Path,
                    default=ROOT / "experiments/dynamic_fusion/innovation_v10_portfolio/cram")
    ap.add_argument("--control-seed", type=int, default=20260903)
    args = ap.parse_args()

    out_dir = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(args.control_seed)
    t0 = time.time()

    protocol = out_dir / "R0_PROTOCOL.json"
    if not protocol.is_file():
        raise SystemExit(f"missing pre-registered protocol: {protocol}")

    per_shot: dict[str, dict] = {}
    for shot in args.shots:
        dino_root = args.dino_cache_root / f"features_vitb14_s{args.seed}_k{shot}/anomalydino_visual"
        clip_root = args.clip_cache_root / f"features_s{args.seed}_k{shot}/anomalyclip_text"
        if not dino_root.is_dir() or not clip_root.is_dir():
            raise SystemExit(f"cache missing: {dino_root} / {clip_root}")
        rows = []
        for cat in args.categories:
            row = run_category(dino_root, clip_root, cat, shot, rng)
            rows.append(row)
            print(f"[s{args.seed} k{shot}] {cat}: "
                  f"A0 AP={row['candidates']['a0']['pixel_ap']:.4f} "
                  f"A1 AP={row['candidates']['a1']['pixel_ap']:.4f} "
                  f"A2 AP={row['candidates']['a2']['pixel_ap']:.4f} "
                  f"(id={row['identity_pooled_vs_a0_maxdiff']:.2e})", flush=True)
        agg: dict = {"per_category": rows}
        for cand in ("a0", "a1", "a2", "med_only_control"):
            agg[f"mean_{cand}_ap"] = round(float(np.mean(
                [r["candidates"][cand]["pixel_ap"] for r in rows])), 6)
            agg[f"mean_{cand}_auroc"] = round(float(np.mean(
                [r["candidates"][cand]["pixel_auroc"] for r in rows])), 6)
            agg[f"mean_{cand}_aupro"] = round(float(np.mean(
                [r["candidates"][cand]["pixel_aupro"] for r in rows])), 6)
        for cand in ("a1", "a2"):
            agg[f"mean_delta_ap_{cand}"] = round(
                agg[f"mean_{cand}_ap"] - agg["mean_a0_ap"], 6)
            agg[f"n_positive_{cand}"] = int(sum(
                r["candidates"][cand]["pixel_ap"] > r["candidates"]["a0"]["pixel_ap"]
                for r in rows))
        agg["worst_category"] = {
            cand: round(min((r["candidates"][cand]["pixel_ap"] - r["candidates"]["a0"]["pixel_ap"])
                            for r in rows), 6)
            for cand in ("a1", "a2")
        }
        per_shot[f"k{shot}"] = agg
        print(f"[s{args.seed} k{shot}] mean A0 AP={agg['mean_a0_ap']:.4f} | "
              f"A1 dAP={agg['mean_delta_ap_a1']:+.4f} A2 dAP={agg['mean_delta_ap_a2']:+.4f}",
              flush=True)

    def mean_over_shots(key: str) -> float:
        return float(np.mean([per_shot[f"k{s}"][key] for s in args.shots]))

    def cat_mean_delta(cand: str) -> dict:
        out = {}
        for cat in args.categories:
            vals = []
            for s in args.shots:
                r = next(x for x in per_shot[f"k{s}"]["per_category"] if x["category"] == cat)
                vals.append(r["candidates"][cand]["pixel_ap"] - r["candidates"]["a0"]["pixel_ap"])
            out[cat] = round(float(np.mean(vals)), 6)
        return out

    gates = {}
    for cand in ("a1", "a2"):
        mean_gain = mean_over_shots(f"mean_delta_ap_{cand}")
        cat_pos = cat_mean_delta(cand)
        n_pos = sum(1 for v in cat_pos.values() if v > 0)
        worst = min(cat_pos.values())
        auroc_loss = -mean_over_shots(f"mean_{cand}_auroc") + mean_over_shots("mean_a0_auroc")
        gates[cand] = {
            "g2_mean_gain_over_two_shots": mean_gain,
            "g3_n_positive_categories": n_pos,
            "g4_worst_category": worst,
            "g5_auroc_mean_loss": auroc_loss,
        }

    report = {
        "route": "A_CRAM",
        "pipeline": "v10_portfolio_r0",
        "seed": args.seed,
        "shots": args.shots,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "elapsed_s": round(time.time() - t0, 1),
        "protocol_sha256": sha256_file(protocol),
        "code_sha256": {
            "common": sha256_file(ROOT / "src/industrial_ad/innovation_v10_portfolio/common.py"),
            "cram": sha256_file(ROOT / "src/industrial_ad/innovation_v10_portfolio/cram.py"),
            "runner": sha256_file(Path(__file__)),
        },
        "manifest_sha256": sha256_file(ROOT / "data/splits/mpdd/manifest.json"),
        "identity_gate": {
            "g1_max_diff_over_categories": max(
                r["identity_pooled_vs_a0_maxdiff"]
                for s in args.shots for r in per_shot[f"k{s}"]["per_category"]),
            "g1_pass": all(r["identity_pooled_vs_a0_maxdiff"] <= 1e-6
                           for s in args.shots for r in per_shot[f"k{s}"]["per_category"]),
        },
        "per_shot": per_shot,
        "cat_mean_delta_over_shots": {cand: cat_mean_delta(cand) for cand in ("a1", "a2")},
        "gates": gates,
        "decision": "PENDING",
    }
    out_path = out_dir / "R0_RESULT.json"
    out_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\nWrote {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
