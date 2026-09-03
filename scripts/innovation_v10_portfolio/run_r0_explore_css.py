"""X2-probe: intra-image context self-similarity (CSS) vs A1 (task-book Scenario E follow-up).

Reference-free evidence stream: score(q) = mean cosine distance from patch q to its
24 spatial neighbours (Chebyshev radius 2) within the SAME test image. Orthogonal to
all memory-distance routes. Fusion: fixed per-image z-score + sum (no tuning).

  .venv-patchcore\\Scripts\\python.exe scripts\\innovation_v10_portfolio\\run_r0_explore_css.py
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

from industrial_ad.innovation_v10_portfolio import common
from industrial_ad.innovation_v10_portfolio.common import build_fused_blocks, load_features
from src.utils import dists2map

RADIUS = 2
CATEGORIES = ["bracket_black", "bracket_brown", "bracket_white", "connector", "metal_plate", "tubes"]


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def css_grid_map(feat_img: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """feat_img [H,W,D] (L2 rows) -> (d1_grid_dummy=None, css_grid [H,W]).

    css(i,j) = mean over 24 Chebyshev-r=2 neighbours of (1 - cos) = mean sq-L2/2.
    """
    import faiss

    hh, ww, d = feat_img.shape
    q = feat_img.reshape(-1, d).astype(np.float32)
    # A1 identity not needed here (caller computes it); kept faiss import for parity style.
    del faiss
    pad = np.pad(feat_img, ((RADIUS, RADIUS), (RADIUS, RADIUS), (0, 0)), mode="edge")
    acc = np.zeros((hh, ww), dtype=np.float64)
    n_off = 0
    for dr in range(-RADIUS, RADIUS + 1):
        for dc in range(-RADIUS, RADIUS + 1):
            if dr == 0 and dc == 0:
                continue
            sl = pad[RADIUS + dr:RADIUS + dr + hh, RADIUS + dc:RADIUS + dc + ww]
            acc += (1.0 - np.einsum("hwd,hwd->hw", feat_img.astype(np.float64), sl.astype(np.float64)))
            n_off += 1
    css = (acc / n_off).astype(np.float32)
    return None, css


def a1_distance_grid(feat_img: np.ndarray, bank: np.ndarray) -> np.ndarray:
    """A1 d1 grid: sq-L2/2 to nearest pooled-bank patch (matches frozen A1)."""
    import faiss

    hh, ww, d = feat_img.shape
    q = feat_img.reshape(-1, d).astype(np.float32)
    index = faiss.IndexFlatL2(d)
    index.add(bank.astype(np.float32))
    dists, _ = index.search(q, k=1)
    return (dists[:, 0] / 2.0).reshape(hh, ww).astype(np.float32)


def _z(x: np.ndarray) -> np.ndarray:
    s = float(x.std())
    if s < 1e-6:
        return np.zeros_like(x)
    return ((x - float(x.mean())) / s).astype(np.float32)


def run_category(dino_cache: Path, clip_cache: Path, cat: str) -> dict:
    dino = load_features(dino_cache / f"{cat}.npz")
    clip = load_features(clip_cache / f"{cat}.npz")
    feat, ref, _, masks, grid = build_fused_blocks(dino, clip, dino_weight=0.5)
    n = feat.shape[0]
    assert ref.shape[0] == 1
    bank = ref[0].reshape(-1, feat.shape[-1]).astype(np.float32)

    a1_maps, css_maps, fused_maps, fused_max_maps = [], [], [], []
    for i in range(n):
        d1g = a1_distance_grid(feat[i], bank)
        _, cssg = css_grid_map(feat[i])
        a1_448 = dists2map(d1g, common.MAP_SIZE).astype(np.float32)
        css_448 = dists2map(cssg, common.MAP_SIZE).astype(np.float32)
        za, zc = _z(a1_448), _z(css_448)
        a1_maps.append(a1_448)
        css_maps.append(css_448)
        fused_maps.append((za + zc).astype(np.float32))
        fused_max_maps.append(np.maximum(za, zc).astype(np.float32))
    a1 = np.stack(a1_maps)
    css = np.stack(css_maps)
    fus = np.stack(fused_maps)
    fus_max = np.stack(fused_max_maps)

    met_a1 = common.compute_pixel_metrics(a1.astype(np.float64), masks)
    met_css = common.compute_pixel_metrics(css.astype(np.float64), masks)
    met_fus = common.compute_pixel_metrics(fus.astype(np.float64), masks)
    met_fus_max = common.compute_pixel_metrics(fus_max.astype(np.float64), masks)

    # diagnostic: correlation on defect vs normal pixels (stride-8 pooled)
    s = common.STRIDE
    la = (masks[:, ::s, ::s] > 0.5).ravel()
    av = a1[:, ::s, ::s].ravel().astype(np.float64)
    cv = css[:, ::s, ::s].ravel().astype(np.float64)
    corr_def = float(np.corrcoef(av[la], cv[la])[0, 1]) if la.sum() > 1 else float("nan")
    corr_norm = float(np.corrcoef(av[~la], cv[~la])[0, 1]) if (~la).sum() > 1 else float("nan")

    return {
        "category": cat,
        "n_images": n,
        "a1_ap": round(met_a1["pixel_ap"], 6),
        "css_ap": round(met_css["pixel_ap"], 6),
        "fused_ap": round(met_fus["pixel_ap"], 6),
        "delta_fused": round(met_fus["pixel_ap"] - met_a1["pixel_ap"], 6),
        "fused_max_ap": round(met_fus_max["pixel_ap"], 6),
        "delta_fused_max": round(met_fus_max["pixel_ap"] - met_a1["pixel_ap"], 6),
        "css_auroc": round(met_css["pixel_auroc"], 6),
        "a1_auroc": round(met_a1["pixel_auroc"], 6),
        "corr_defect": round(corr_def, 4),
        "corr_normal": round(corr_norm, 4),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dino-cache", type=Path, default=None)
    parser.add_argument("--clip-cache", type=Path, default=None)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out-dir", type=Path,
                        default=ROOT / "experiments/dynamic_fusion/innovation_v10_portfolio/explore_css")
    args = parser.parse_args()

    base = ROOT / "outputs/dynamic_fusion/v3_direction_a"
    dino_cache = args.dino_cache or base / f"features_vitb14_s{args.seed}_k1/anomalydino_visual"
    clip_cache = args.clip_cache or base / f"features_s{args.seed}_k1/anomalyclip_text"
    args.dino_cache, args.clip_cache = dino_cache, clip_cache

    out_dir = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    protocol = out_dir / "R0_PROTOCOL.json"
    if not protocol.is_file():
        raise SystemExit(f"missing pre-registered protocol: {protocol}")

    rows = []
    t0 = time.time()
    for cat in CATEGORIES:
        if not (args.dino_cache / f"{cat}.npz").is_file():
            continue
        r = run_category(args.dino_cache, args.clip_cache, cat)
        rows.append(r)
        print(f"[CSS {cat}] A1 AP={r['a1_ap']:.4f} CSS AP={r['css_ap']:.4f} "
              f"fused-sum Δ={r['delta_fused']:+.4f} fused-max Δ={r['delta_fused_max']:+.4f} "
              f"corr(defect={r['corr_defect']:.2f}, normal={r['corr_normal']:.2f})", flush=True)

    def mean(k): return float(np.mean([r[k] for r in rows]))
    report = {
        "route": "X2_EXPLORE_CSS",
        "pipeline": "v10_portfolio_explore_css",
        "seed": args.seed, "shot": 1, "radius": RADIUS,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "protocol_sha256": sha256_file(protocol),
        "code_sha256": {"runner": sha256_file(Path(__file__))},
        "per_category": rows,
        "mean_a1_ap": round(mean("a1_ap"), 6),
        "mean_css_ap": round(mean("css_ap"), 6),
        "mean_delta_fused": round(mean("delta_fused"), 6),
        "n_positive": sum(1 for r in rows if r["delta_fused"] > 0),
        "worst_category": round(min(r["delta_fused"] for r in rows), 6),
        "mean_delta_fused_max": round(mean("delta_fused_max"), 6),
        "n_positive_max": sum(1 for r in rows if r["delta_fused_max"] > 0),
        "worst_category_max": round(min(r["delta_fused_max"] for r in rows), 6),
        "decision": "PENDING",
    }
    result_name = "R0_RESULT.json" if args.seed == 0 else f"R0_CONFIRM_s{args.seed}.json"
    (out_dir / result_name).write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\nA1 mean AP = {report['mean_a1_ap']} (frozen k1 ~0.3092 sanity) | "
          f"CSS-alone mean AP = {report['mean_css_ap']} | fused-sum Δ = {report['mean_delta_fused']} "
          f"({report['n_positive']}/6 pos, worst {report['worst_category']}) | "
          f"fused-max Δ = {report['mean_delta_fused_max']} ({report['n_positive_max']}/6 pos, "
          f"worst {report['worst_category_max']}) | elapsed {time.time()-t0:.0f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
