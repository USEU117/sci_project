"""P5-B precheck - S2 GPMR-style responsibility entropy vs real error (task book 16 s.4.3).

Pure-CPU diagnostic on the frozen A1 feature caches (MPDD development role only).
Goal: decide whether a Gaussian-prototype / responsibility-entropy anomaly signal is
worth any further GPU investment, BEFORE building a full S2 pipeline.

Design (frozen, minimal):
  features : DINOv2 ViT-B/14 branch cache (outputs/dynamic_fusion/v3_direction_a/
             features_vitb14_s{seed}_k{shot}/anomalydino_visual/{cat}.npz) which
             stores ref_patch_features (K,32,32,384) and test patch_features
             (N,32,32,384) + sample_ids.
  bins     : coarse spatial bins of 8x8 patch cells -> 4x4 = 16 bins over 32x32.
  per bin  : shrinkage (diagonal) Gaussian Mixture with n_components=4 fitted on
             the normal (reference) patches that fall in the bin.
  per test patch : responsibility entropy H = -sum_k w_k log w_k  (higher =
             "not explained by any single normal prototype") and
             -log p(x) (mixture negative log-likelihood).
Stop gate (16 s.4.3): Spearman |rho| between per-pixel error and entropy < 0.3
            -> archive S2 direction (entropy does not carry anomaly information).

GT is evaluator-side only (development MPDD); used here to *test* the hypothesis.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
for p in (str(ROOT), str(ROOT / "src"), str(ROOT / "scripts")):
    if p not in sys.path:
        sys.path.insert(0, p)

from industrial_ad.innovation_v6_dgsafe import maps  # noqa: E402

OUT = maps.EXPERIMENT_ROOT / "p5b_gpmr_precheck"
FEAT_DIR = (ROOT / "outputs" / "dynamic_fusion" / "v3_direction_a"
            / "features_vitb14_s0_k4" / "anomalydino_visual")
SEED = 0
SHOT = 4
N_COMP = 4
BIN = 8                       # 8x8 patch cells -> 4x4 bins on 32x32
GRID = 32


def _spearman(x, y):
    from scipy.stats import spearmanr
    res = spearmanr(x, y)
    return float(res.statistic if hasattr(res, "statistic") else res.correlation)


def load_cat(cat: str) -> dict:
    p = Path(FEAT_DIR) / f"{cat}.npz"
    d = np.load(p, allow_pickle=False)
    refs = np.asarray(d["ref_patch_features"], dtype=np.float64)   # (K,32,32,384)
    test = np.asarray(d["patch_features"], dtype=np.float64)       # (N,32,32,384)
    return {"refs": refs, "test": test,
            "sample_ids": np.asarray(d["sample_ids"])}


def bin_index(grid_coords):
    return np.clip(grid_coords // BIN, 0, (GRID // BIN) - 1)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--categories", nargs="+",
                    default=["bracket_black", "bracket_brown", "bracket_white",
                             "connector", "metal_plate", "tubes"])
    args = ap.parse_args()
    maps.assert_development_only()
    OUT.mkdir(parents=True, exist_ok=True)

    from sklearn.mixture import GaussianMixture

    rows = []
    for cat in args.categories:
        t0 = time.time()
        data = load_cat(cat)
        K = data["refs"].shape[0]
        # pool normal patches per spatial bin
        norm_feats = data["refs"].reshape(K, GRID * GRID, -1)      # (K,1024,384)
        pools = {b: [] for b in range(16)}
        ref_ent = np.zeros((K, GRID, GRID))
        for k in range(K):
            for g in range(GRID * GRID):
                i, j = divmod(g, GRID)
                b = (i // BIN) * 4 + (j // BIN)
                pools[b].append(norm_feats[k, g])
        # fit per-bin diagonal GMM on refs (all k) and get ref entropy for sanity
        gmms = {}
        for b, feats in pools.items():
            X = np.stack(feats)                                    # (K*1024? no) -> per bin it is K*64
            X = X.reshape(-1, X.shape[-1])
            gm = GaussianMixture(n_components=min(N_COMP, max(1, len(X) // 10)),
                                 covariance_type="diag", reg_covar=1e-4,
                                 max_iter=100, random_state=0)
            gm.fit(X)
            gmms[b] = gm
        # per-test-patch responsibility entropy + neg loglik (grid 32x32)
        N = data["test"].shape[0]
        H = np.zeros((N, GRID, GRID))
        NLL = np.zeros((N, GRID, GRID))
        test_flat = data["test"].reshape(N, GRID * GRID, -1)
        for n in range(N):
            for g in range(GRID * GRID):
                i, j = divmod(g, GRID)
                b = (i // BIN) * 4 + (j // BIN)
                x = test_flat[n, g, None]
                gm = gmms[b]
                # responsibilities (density per component)
                resp = gm.predict_proba(x)[0]
                ent = -np.sum(resp * np.log(np.maximum(resp, 1e-12)))
                ll = gm.score(x)
                H[n, i, j] = ent
                NLL[n, i, j] = -ll
        # evaluator: GT at 448 -> 32 grid (any positive pixel => patch pos)
        gtm32 = np.zeros((N, GRID, GRID), dtype=np.int64)
        for n, sid in enumerate(data["sample_ids"]):
            rel = Path(str(sid))
            parts = rel.parts
            if len(parts) >= 4 and parts[1] == "test" and parts[2] != "good":
                cat_, defect, stem = parts[0], parts[2], Path(parts[3]).stem
                mp = Path(maps.MPDD_DATA_ROOT) / cat_ / "ground_truth" / defect \
                    / f"{stem}_mask.png"
                if mp.exists():
                    import cv2
                    m = cv2.imread(str(mp), cv2.IMREAD_GRAYSCALE)
                    m = cv2.resize(m, (GRID, GRID), interpolation=cv2.INTER_NEAREST)
                    gtm32[n] = (m > 0).astype(np.int64)
        y = gtm32.ravel()
        mask = (y == 0) | (y == 1)
        rho_h = _spearman(H.ravel()[mask], y[mask]) if mask.sum() > 10 else float("nan")
        rho_nll = _spearman(NLL.ravel()[mask], y[mask]) if mask.sum() > 10 else float("nan")
        rows.append({
            "category": cat, "seed": SEED, "shot": SHOT, "n_refs": int(K),
            "n_test": int(N), "n_bins": 16, "n_components": N_COMP,
            "spearman_entropy_vs_pixel_error": round(rho_h, 4),
            "spearman_negloglik_vs_pixel_error": round(rho_nll, 4),
            "elapsed_s": round(time.time() - t0, 1)})
        print(f"[{cat}] s{SEED}/k{SHOT} n_refs={K} "
              f"rho(entropy)={rho_h:+.3f} rho(nll)={rho_nll:+.3f}", flush=True)

    avg_rho_h = float(np.mean([r["spearman_entropy_vs_pixel_error"] for r in rows]))
    avg_rho_nll = float(np.mean([r["spearman_negloglik_vs_pixel_error"] for r in rows]))
    decision = {
        "stop_gate_abs_rho_lt_0_3": (abs(avg_rho_h) < 0.3 and abs(avg_rho_nll) < 0.3),
        "avg_abs_rho_entropy": round(abs(avg_rho_h), 4),
        "avg_abs_rho_negloglik": round(abs(avg_rho_nll), 4),
        "note": "if gate triggers -> archive S2 GPMR per task book 16 s4.3",
    }
    report = {
        "program": "innovation_v6_dgsafe", "phase": "p5b_gpmr_precheck",
        "dataset": "mpdd", "role": "development", "seed": SEED, "shot": SHOT,
        "design": "per-spatial-bin diagonal GMM (4 comps) on cached vitb14 ref "
                  "patches; per-test-patch responsibility entropy / -loglik vs GT",
        "rows": rows, "decision": decision,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    (OUT / "P5B_GPMR_PRE.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=1), encoding="utf-8")
    print(json.dumps(decision, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
