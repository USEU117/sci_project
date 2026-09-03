"""X-probe: memory linear-reconstruction (LLSE) residual vs A1 1-NN (task book §15 spirit).

Fused A1 space, MPDD seed0 k1, all 6 classes, CPU.

  .venv-patchcore\\Scripts\\python.exe scripts\\innovation_v10_portfolio\\run_r0_explore_llse.py
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

K_NEIGH = 8
REG = 1e-3
CATEGORIES = ["bracket_black", "bracket_brown", "bracket_white", "connector", "metal_plate", "tubes"]


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _residual_from_nbr(q: np.ndarray, nbr: np.ndarray, k: int) -> np.ndarray:
    """q [Q,d] L2-normalized rows; nbr [Q,k,d] -> residual grid values (sqrt r^2)."""
    nd = nbr.astype(np.float64)
    G = nd @ nd.transpose(0, 2, 1) + REG * np.eye(k, dtype=np.float64)
    rhs = np.matmul(nd, q.astype(np.float64)[..., None])[..., 0]
    coef = np.linalg.solve(G, rhs)
    resid2 = np.maximum(1.0 - (coef * rhs).sum(axis=-1), 0.0)
    return np.sqrt(resid2).astype(np.float32)


def llse_residual_grid(qf: np.ndarray, bank: np.ndarray, mode: str | None = None,
                       rng: np.random.Generator | None = None,
                       k: int = K_NEIGH) -> tuple[np.ndarray, np.ndarray]:
    """qf [H,W,D], bank [P,D] (L2-normalized rows).

    mode None -> true top-8 nearest (also returns d1 grid); mode 'random' -> 8
    random distinct bank rows per query (locality-destroying control).
    """
    import faiss

    hh, ww, d = qf.shape
    q = qf.reshape(-1, d).astype(np.float32)
    b = bank.astype(np.float32)
    P = b.shape[0]
    if mode is None:
        index = faiss.IndexFlatL2(d)
        index.add(b)
        dists, idx = index.search(q, k=k)   # squared L2
        d1_grid = (dists[:, 0] / 2.0).reshape(hh, ww)
    else:
        idx = rng.integers(0, P, size=(q.shape[0], k))
        d1_grid = None
    nbr = b[idx]                        # [Q, k, d]
    resid = _residual_from_nbr(q, nbr, k)
    resid_grid = resid.reshape(hh, ww)
    return d1_grid, resid_grid


def run_category(dino_cache: Path, clip_cache: Path, cat: str) -> dict:
    dino = load_features(dino_cache / f"{cat}.npz")
    clip = load_features(clip_cache / f"{cat}.npz")
    feat, ref, _, masks, grid = build_fused_blocks(dino, clip, dino_weight=0.5)
    n = feat.shape[0]
    assert ref.shape[0] == 1
    bank = ref[0].reshape(-1, feat.shape[-1]).astype(np.float32)  # [P, D]
    rng = np.random.default_rng(20260903)

    d1_all, res_all, res_rnd = [], [], []
    for i in range(n):
        d1g, resg = llse_residual_grid(feat[i], bank, None)
        _, resr = llse_residual_grid(feat[i], bank, "random", rng=rng)
        d1_all.append(dists2map(d1g, common.MAP_SIZE).astype(np.float32))
        res_all.append(dists2map(resg, common.MAP_SIZE).astype(np.float32))
        res_rnd.append(dists2map(resr, common.MAP_SIZE).astype(np.float32))
    a1_map = np.stack(d1_all).astype(np.float32)
    res_map = np.stack(res_all).astype(np.float32)
    resp_map = np.stack(res_rnd).astype(np.float32)

    met_a1 = common.compute_pixel_metrics(a1_map.astype(np.float64), masks)
    met_res = common.compute_pixel_metrics(res_map.astype(np.float64), masks)
    met_resr = common.compute_pixel_metrics(resp_map.astype(np.float64), masks)
    return {
        "category": cat,
        "n_images": n,
        "a1_ap": round(met_a1["pixel_ap"], 6),
        "llse_ap": round(met_res["pixel_ap"], 6),
        "llse_random_ap": round(met_resr["pixel_ap"], 6),
        "delta_llse": round(met_res["pixel_ap"] - met_a1["pixel_ap"], 6),
        "delta_llse_random": round(met_resr["pixel_ap"] - met_a1["pixel_ap"], 6),
        "llse_auroc": round(met_res["pixel_auroc"], 6),
        "a1_auroc": round(met_a1["pixel_auroc"], 6),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dino-cache", type=Path, default=None)
    parser.add_argument("--clip-cache", type=Path, default=None)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out-dir", type=Path,
                        default=ROOT / "experiments/dynamic_fusion/innovation_v10_portfolio/explore_llse")
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
        print(f"[LLSE {cat}] A1 AP={r['a1_ap']:.4f} LLSE AP={r['llse_ap']:.4f} "
              f"Δ={r['delta_llse']:+.4f} | random-8 Δ={r['delta_llse_random']:+.4f}", flush=True)

    def mean(k): return float(np.mean([r[k] for r in rows]))
    report = {
        "route": "X_EXPLORE_LLSE",
        "pipeline": "v10_portfolio_explore_llse",
        "seed": args.seed, "shot": 1, "k_neigh": K_NEIGH, "reg": REG,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "protocol_sha256": sha256_file(protocol),
        "code_sha256": {"runner": sha256_file(Path(__file__))},
        "per_category": rows,
        "mean_a1_ap": round(mean("a1_ap"), 6),
        "mean_delta_llse": round(mean("delta_llse"), 6),
        "n_positive": sum(1 for r in rows if r["delta_llse"] > 0),
        "worst_category": round(min(r["delta_llse"] for r in rows), 6),
        "mean_delta_llse_random": round(mean("delta_llse_random"), 6),
        "control_true_vs_random": round(mean("delta_llse") - mean("delta_llse_random"), 6),
        "decision": "PENDING",
    }
    result_name = "R0_RESULT.json" if args.seed == 0 else f"R0_CONFIRM_s{args.seed}.json"
    (out_dir / result_name).write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\nA1 mean AP = {report['mean_a1_ap']} (frozen k1 ~0.3092 sanity) | "
          f"Δ LLSE = {report['mean_delta_llse']} ({report['n_positive']}/6 pos, "
          f"worst {report['worst_category']}) | random-8 Δ = {report['mean_delta_llse_random']} | "
          f"true-vs-random {report['control_true_vs_random']} | elapsed {time.time()-t0:.0f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
