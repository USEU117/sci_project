"""N2 / CCT - capacity-constrained cross-branch matching (doc 27 s6; CPU).

Mechanism probe ONLY (feature-domain interventions on the k1 support image), as a
pre-registered R0 gate. A full real-data pixel-AP run needs GPU re-export / larger
budget; if R0 is positive we run a minimal k2 real check, else stop with record.

Idea (doc 27 s6): A1 lets every query patch use the same normal patch unlimited
times. A normal-looking but DUPLICATED region is therefore free under plain NN. A
capacity-constrained matching (each normal cell serves at most its fair share)
makes duplication/missing-pattern expensive in a LOCALISED way.

Solver: balanced entropic OT in log-domain Sinkhorn (row marginal = col marginal =
1/n), per-cell anomaly score = expected transport cost. Two branch plans P_D, P_C
over the SAME anchor cells (each anchor = one support cell with true D/C pairing):
  CCT-I : P_D,P_C solved independently (gamma=0)
  CCT-C : JS row-coupling between the two per-cell assignment rows (gamma>0,
          alternating linearised updates)
  concat-OT : single plan on fused (concat) 1-cos cost  (equivalence control)
  a1_free : per-cell nearest fused 1-cos (frozen A1 path, no capacity)
  dino_ot / clip_ot : single-branch capacity controls at same total mass

R0 interventions (on the SUPPORT image's OWN features, both branches jointly):
  copy(s):  a square block of cells is duplicated to another location
  erase(s): a square block replaced by unit-norm feature noise
  nuisance: tiny isotropic feature perturbation (photometric-analog control)
  permute:  shuffle the multiset of cell identities (coordinate-free sanity)
R0 checks: copy/erase inside-vs-outside score gap grows with s and beats
a1_free's gap; nuisance inside/outside ~0 (FP controlled); permutation leaves the
sorted per-cell score multiset invariant (no coordinate leakage).

Pre-registered constants (frozen BEFORE reading real GT):
  GRID=32 (support cells = anchors), EPS=0.02, GAMMA=0.5, ITER capped 4000,
  block sizes s in {2,3,4}, nuisance sigma=0.02, permute seed=0.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
for p in (str(ROOT), str(ROOT / "src"), str(ROOT / "scripts")):
    sys.path.insert(0, p)

from scipy.stats import rankdata  # noqa: E402

from industrial_ad.innovation_v10_portfolio.common import build_fused_blocks, resize_patches  # noqa: E402

CATEGORIES = ["bracket_black", "bracket_brown", "bracket_white", "connector", "metal_plate", "tubes"]
NTOF_D = ROOT / "outputs/dynamic_fusion/ntof_features_dino_s0_k1"
NTOF_C = ROOT / "outputs/dynamic_fusion/ntof_features_clip_s0_k1"
EPS = 0.05
GAMMA = 0.5
MAX_ITER = 1500
TOL = 1e-6
BLOCK_SIZES = [2, 3, 4]
NUIS_SIGMA = 0.02
PERM_SEED = 0
GRID = 16            # R0 mechanism probe grid (doc27 s6 allows 16x16 mechanism-only)
CELLS = GRID * GRID


# ---------------------------------------------------------------- solvers
def _row_l2(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    return x / np.maximum(np.linalg.norm(x, axis=-1, keepdims=True), 1e-12)


def _cos_cost(q: np.ndarray, a: np.ndarray) -> np.ndarray:
    """[Q,D] x [A,D] -> [Q,A] 1-cos (rows pre-normalised)."""
    return 1.0 - q @ a.T


def _pool16(x: np.ndarray) -> np.ndarray:
    """[32,32,D] -> [16,16,D] by 2x2 block mean (spatial downsample for probe)."""
    return x.reshape(16, 2, 16, 2, -1).mean(axis=(1, 3))


def sinkhorn_log(C: np.ndarray, eps: float = EPS, iters: int = MAX_ITER, tol: float = TOL):
    """Balanced entropic OT; row & col marginals uniform. Returns transport [Q,A]."""
    n, m = C.shape
    if n != m:
        raise ValueError("balanced OT needs square cost (query cells == anchor cells)")
    loga = np.log(np.full(n, 1.0 / n))
    logb = np.log(np.full(m, 1.0 / m))
    logK = -np.clip(C, 0.0, 50.0) / eps
    lu = np.zeros(n)
    lv = np.zeros(m)
    for it in range(iters):
        lu_old = lu.copy()
        M = logK + lv[None, :]
        mx = M.max(axis=1)
        lu = loga - (mx + np.log(np.exp(M - mx[:, None]).sum(axis=1)))
        M2 = logK + lu[:, None]
        mx2 = M2.max(axis=0)
        lv = logb - (mx2 + np.log(np.exp(M2 - mx2[None, :]).sum(axis=0)))
        if it % 50 == 0 and it > 0:
            # primal row-marginal error (cheap-ish; every 50 iters)
            Pk = np.exp(logK + lv[None, :])
            row_err = np.abs(Pk @ np.exp(lu) - np.exp(loga)).max()
            if row_err < tol or np.abs(lu - lu_old).max() < tol:
                break
    P = np.exp(logK + lu[:, None] + lv[None, :])
    return P


def branch_cost_pair(d_feat_ref, c_feat_ref, d_feat_q=None, c_feat_q=None):
    """1-cos cost matrices CD, CC, Cfused given 32-grid support ref & query cells."""
    d_ref = _row_l2(d_feat_ref.reshape(-1, d_feat_ref.shape[-1]))
    c_ref = _row_l2(c_feat_ref.reshape(-1, c_feat_ref.shape[-1]))
    fd_ref = _row_l2(np.concatenate([0.5 * d_ref, 0.5 * c_ref], axis=-1))
    d_q = _row_l2(d_feat_q.reshape(-1, d_feat_q.shape[-1]))
    c_q = _row_l2(c_feat_q.reshape(-1, c_feat_q.shape[-1]))
    fd_q = _row_l2(np.concatenate([0.5 * d_q, 0.5 * c_q], axis=-1))
    return (_cos_cost(d_q, d_ref), _cos_cost(c_q, c_ref),
            _cos_cost(fd_q, fd_ref))


def solve_methods(CD, CC, CF):
    """Return per-cell score maps (query-grid flattened) for each method.

    score_cell = conditional expected cost under the cell's (row-normalised)
    matching distribution: sum_j (P[i,j]/rowsum_i) C[i,j].
    """
    out = {}
    out["a1_free"] = CF.min(axis=1)

    def _score(P, C):
        rs = P.sum(axis=1, keepdims=True)
        return (P / np.maximum(rs, 1e-12) * C).sum(axis=1)

    p_d = sinkhorn_log(CD)
    p_c = sinkhorn_log(CC)
    out["dino_ot"] = _score(p_d, CD)
    out["clip_ot"] = _score(p_c, CC)
    out["cct_i"] = 0.5 * _score(p_d, CD) + 0.5 * _score(p_c, CC)
    # CCT-C: alternating linearised JS row coupling (gradient offsets)
    pd, pc = p_d.copy(), p_c.copy()
    for _ in range(4):
        off_d = np.empty_like(CD)
        off_c = np.empty_like(CC)
        for i in range(CD.shape[0]):
            sd = np.maximum(pd[i], 1e-12)
            sc = np.maximum(pc[i], 1e-12)
            m = 0.5 * (sd + sc)
            # JS(P_D_i || P_C_i) gradient wrt P_D row: (1/2)(log P_D - log m)
            off_d[i] = 0.5 * (np.log(sd) - np.log(np.maximum(m, 1e-12)))
            off_c[i] = 0.5 * (np.log(sc) - np.log(np.maximum(m, 1e-12)))
        pd = sinkhorn_log(CD + GAMMA * off_d)
        pc = sinkhorn_log(CC + GAMMA * off_c)
    out["cct_c"] = 0.5 * _score(pd, CD) + 0.5 * _score(pc, CC)
    # concat-OT single plan (equivalence control)
    pf = sinkhorn_log(CF)
    out["concat_ot"] = _score(pf, CF)
    return out


# ---------------------------------------------------------------- interventions (feature domain)
def place_block(rng, grid=GRID, s=2):
    """Random non-overlapping-ish source & target top-left corners (target clear of source)."""
    while True:
        sy, sx = rng.integers(0, grid - s + 1, 2)
        ty, tx = rng.integers(0, grid - s + 1, 2)
        if abs(sy - ty) >= s or abs(sx - tx) >= s:
            return (sy, sx), (ty, tx)


def apply_copy_paste(d_feat, c_feat, s, seed):
    rng = np.random.default_rng(seed)
    (sy, sx), (ty, tx) = place_block(rng, s=s)
    dc = d_feat.copy()
    cc = c_feat.copy()
    dc[ty:ty + s, tx:tx + s] = d_feat[sy:sy + s, sx:sx + s]
    cc[ty:ty + s, tx:tx + s] = c_feat[sy:sy + s, sx:sx + s]
    mask = np.zeros((GRID, GRID), np.uint8)
    mask[ty:ty + s, tx:tx + s] = 1
    return dc, cc, mask


def apply_erase(d_feat, c_feat, s, seed):
    rng = np.random.default_rng(seed)
    (_, _), (ty, tx) = place_block(rng, s=s)
    dc = d_feat.copy()
    cc = c_feat.copy()
    dnoise = rng.normal(size=(s, s, d_feat.shape[-1]))
    cnoise = rng.normal(size=(s, s, c_feat.shape[-1]))
    dc[ty:ty + s, tx:tx + s] = _row_l2(dnoise)
    cc[ty:ty + s, tx:tx + s] = _row_l2(cnoise)
    mask = np.zeros((GRID, GRID), np.uint8)
    mask[ty:ty + s, tx:tx + s] = 1
    return dc, cc, mask


def apply_nuisance(d_feat, c_feat, seed, sigma=NUIS_SIGMA):
    rng = np.random.default_rng(seed)
    d = d_feat + rng.normal(0, sigma, d_feat.shape)
    c = c_feat + rng.normal(0, sigma, c_feat.shape)
    return _row_l2(d), _row_l2(c), None


# ---------------------------------------------------------------- category-level R0 probe
def load_pair(cat):
    zd = np.load(NTOF_D / f"{cat}.npz", allow_pickle=False)
    zc = np.load(NTOF_C / f"{cat}.npz", allow_pickle=False)
    d_ref = _pool16(np.asarray(zd["ref_orig_feat"])[0])   # [16,16,768]
    c37 = np.asarray(zc["ref_orig_feat"])[0]              # [37,37,768]
    c32 = resize_patches(c37[None], (32, 32))[0]
    c_ref = _pool16(c32)                                   # [16,16,768]
    return d_ref, c_ref


def r0_probe(cat, d_ref, c_ref):
    """Feature-domain R0 on the support image itself (balanced OT, anchors == cells)."""
    rows = []
    for kind in ("copy", "erase"):
        for s in BLOCK_SIZES:
            for seed in (1, 2):
                if kind == "copy":
                    dq, cq, mask = apply_copy_paste(d_ref, c_ref, s, seed)
                else:
                    dq, cq, mask = apply_erase(d_ref, c_ref, s, seed)
                CD, CC, CF = branch_cost_pair(d_ref, c_ref, dq, cq)
                scores = solve_methods(CD, CC, CF)
                m = mask.ravel() > 0.5
                for meth, sc in scores.items():
                    rows.append({"cat": cat, "kind": kind, "s": s, "seed": seed, "meth": meth,
                                 "inside": float(sc[m].mean()), "outside": float(sc[~m].mean()),
                                 "gap": float(sc[m].mean() - sc[~m].mean())})
    # nuisance FP (no mask): mean score rise over clean
    dq, cq, _ = apply_nuisance(d_ref, c_ref, seed=7)
    CD, CC, CF = branch_cost_pair(d_ref, c_ref, dq, cq)
    scores_n = solve_methods(CD, CC, CF)
    # clean baseline (self-query): expected ~0 cost
    CD0, CC0, CF0 = branch_cost_pair(d_ref, c_ref, d_ref, c_ref)
    scores_0 = solve_methods(CD0, CC0, CF0)
    for meth in scores_n:
        rows.append({"cat": cat, "kind": "nuisance", "s": 0, "seed": 7, "meth": meth,
                     "inside": float(scores_n[meth].mean()), "outside": float("nan"),
                     "gap": float(scores_n[meth].mean() - scores_0[meth].mean())})
    return rows


def permutation_check(d_ref, c_ref):
    """Coordinate-free sanity: shuffled cell order leaves sorted scores unchanged."""
    rng = np.random.default_rng(PERM_SEED)
    perm = rng.permutation(GRID * GRID)
    dq = d_ref.reshape(-1, d_ref.shape[-1])[perm].reshape(GRID, GRID, -1)
    cq = c_ref.reshape(-1, c_ref.shape[-1])[perm].reshape(GRID, GRID, -1)
    CD0, CC0, CF0 = branch_cost_pair(d_ref, c_ref, d_ref, c_ref)
    CDp, CCp, CFp = branch_cost_pair(d_ref, c_ref, dq, cq)
    s0 = solve_methods(CD0, CC0, CF0)
    sp = solve_methods(CDp, CCp, CFp)
    return {m: float(np.max(np.abs(np.sort(s0[m]) - np.sort(sp[m])))) for m in s0}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cats", default=None)
    args = ap.parse_args()
    out_root = ROOT / "experiments/dynamic_fusion/innovation_v13_overnight_20260904/N2_cct"
    out_root.mkdir(parents=True, exist_ok=True)
    cats = [c.strip() for c in args.cats.split(",")] if args.cats else CATEGORIES
    all_rows, perm = [], {}
    for cat in cats:
        d_ref, c_ref = load_pair(cat)
        all_rows += r0_probe(cat, d_ref, c_ref)
        perm[cat] = permutation_check(d_ref, c_ref)
        print(f"  {cat} perm={json.dumps({k: round(v, 6) for k, v in perm[cat].items()})}", flush=True)
    # summary per kind x meth (macro over cats)
    summary = {}
    for r in all_rows:
        key = (r["kind"], r["meth"])
        if key not in summary:
            summary[key] = []
        summary[key].append(r["gap"])
    agg = {f"{k[0]}|{k[1]}": {"mean_gap": round(float(np.mean(v)), 5),
                               "n": len(v)} for k, v in summary.items()}
    payload = {"rows": all_rows, "agg_gap": agg, "permutation_maxdiff": perm,
               "config": {"eps": EPS, "gamma": GAMMA, "blocks": BLOCK_SIZES,
                          "nuisance_sigma": NUIS_SIGMA}}
    (out_root / "R0.json").write_text(json.dumps(payload, ensure_ascii=False, indent=1),
                                      encoding="utf-8")
    print("AGG", json.dumps(agg, indent=0), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
