"""P0 deterministic unit tests (doc28 s4.3/s4.4). Run: python test_p0.py

DNC-C selector: (1) lam=0 -> == DNC-I; (2) high-q/high-corr channel is replaced by
lower-q/low-corr when lam>0; (3) exactly `keep` unique per branch, deterministic.
Semi-relaxed OT: rectangular Q!=A + row err <1e-5; tau->0 == per-row softmax;
tau up -> columns closer to rho and (Q=A, rho uniform, large tau) ~ balanced OT;
simultaneous row/col permutation invariance; identical query/support premium ~0 and
duplicating an anchor's content raises premium with copy count while far rows stay
low (spillover check).
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from dnc_selector import select_dnc_i, select_dnc_c, redundancy_stats  # noqa: E402
from semi_ot import solve_semi_ot, capacity_premium, row_costs, _row_project  # noqa: E402

PASS = []


def check(name: str, cond: bool, info: str = ""):
    PASS.append((name, bool(cond), info))
    print(("PASS " if cond else "FAIL ") + name + ("  " + info if info else ""))


# --------------------------------------------------------------- DNC-C selector
def test_dnc_selector():
    rng = np.random.default_rng(7)
    qD = rng.uniform(0.5, 3.0, 60)
    qC = rng.uniform(0.5, 3.0, 60)
    corr = np.clip(rng.normal(0, 0.4, (60, 60)), -1, 1)
    dncI = select_dnc_i(qD, qC, keep=10)
    dncC0 = select_dnc_c(qD, qC, corr, lam=0.0, keep=10)
    check("dnc-c lam=0 == dnc-i (D)", set(dncI[0]) == set(dncC0[0]))
    check("dnc-c lam=0 == dnc-i (C)", set(dncI[1]) == set(dncC0[1]))

    # constructed flip: keep=1 per branch; D top high-q low-corr wins D slot; C's
    # top-q channel is highly correlated with chosen D -> replaced by lower-q.
    qD2 = np.array([10.0, 1.0, 0.0, 0.0])
    qC2 = np.array([10.0, 9.0, 1.0, 0.0])
    corr2 = np.zeros((4, 4))
    corr2[0, 0] = 0.99          # D0 (chosen) highly corr with C0 (C's argmax q)
    corr2[0, 1] = 0.05
    dI = select_dnc_i(qD2, qC2, keep=1)
    dC = select_dnc_c(qD2, qC2, corr2, lam=8.0, keep=1)
    check("dnc-c flip changes C set vs dnc-i",
          dC[1][0] != dI[1][0] and dC[1][0] == 1,
          f"dnc-i C={dI[1]} dnc-c C={dC[1]}")
    check("dnc-c flip keeps D set", dC[0][0] == dI[0][0] == 0)
    # correlation drop after fix
    st0 = redundancy_stats(dI[0], dI[1], corr2)
    st1 = redundancy_stats(dC[0], dC[1], corr2)
    check("dnc-c chosen corr drops", st1["max_corr"] < st0["max_corr"],
          f"{st0['max_corr']:.2f} -> {st1['max_corr']:.2f}")

    # uniqueness + determinism at realistic size
    qD3 = rng.uniform(0.5, 3.0, 768)
    qC3 = rng.uniform(0.5, 3.0, 768)
    corr3 = np.clip(rng.normal(0, 0.3, (768, 768)), -1, 1)
    a1 = select_dnc_c(qD3, qC3, corr3, lam=0.5, keep=256)
    a2 = select_dnc_c(qD3, qC3, corr3, lam=0.5, keep=256)
    check("dnc-c 256 unique per branch", len(set(a1[0])) == 256 and len(set(a1[1])) == 256)
    check("dnc-c deterministic", np.array_equal(a1[0], a2[0]) and np.array_equal(a1[1], a2[1]))
    ji = len(set(a1[0]) & set(select_dnc_i(qD3, qC3, 256)[0])) / 256
    print(f"  info: Jaccard(DNC-C,DNC-I) D={ji:.3f} (realistic-size, lam=0.5)")


# --------------------------------------------------------------- semi-relaxed OT
def _balanced_sinkhorn(C, a, b, eps, iters=4000):
    n, m = C.shape
    logK = -C / eps
    lu = np.zeros(n)
    lv = np.zeros(m)
    loga = np.log(a)
    logb = np.log(b)
    for it in range(iters):
        M = logK + lv[None, :]
        mx = M.max(axis=1)
        lu = loga - (mx + np.log(np.exp(M - mx[:, None]).sum(axis=1)))
        M2 = logK + lu[:, None]
        mx2 = M2.max(axis=0)
        lv = logb - (mx2 + np.log(np.exp(M2 - mx2[None, :]).sum(axis=0)))
        if it % 200 == 0 and it > 0:
            err = np.abs(np.exp(M2) @ np.ones(m) - a).max() if False else 0
    return np.exp(logK + lu[:, None] + lv[None, :])


def test_semi_ot():
    rng = np.random.default_rng(11)
    # 1. rectangular run + row err
    C = rng.uniform(0.0, 1.5, (64, 128))
    a = rng.dirichlet(np.ones(64))
    rho = rng.dirichlet(np.ones(128))
    P, st = solve_semi_ot(C, a, rho, eps=0.05, tau=2.0)
    check("semi-ot rectangular Q!=A runnable", st["Q"] == 64 and st["A"] == 128)
    check("semi-ot row marginal err < 1e-5", st["row_err"] < 1e-5, f"err={st['row_err']:.2e}")

    # 2. tau->0 == per-row softmax
    P0, _ = solve_semi_ot(C, a, rho, eps=0.05, tau=1e-9)
    ref = _row_project(C, a, 0.05)
    check("tau->0 == per-row entropy soft matching", np.abs(P0 - ref).max() < 1e-8)

    # 3. tau up: columns closer to rho; and Q=A,rho uniform,large tau -> ~balanced OT
    P0m = P0.sum(axis=0)
    Pm = P.sum(axis=0)
    check("tau=2 columns closer to rho than tau=0",
          np.abs(Pm - rho).sum() < np.abs(P0m - rho).sum(),
          f"{np.abs(P0m-rho).sum():.3f} -> {np.abs(Pm-rho).sum():.3f}")
    Cb = rng.uniform(0.0, 1.5, (48, 48))
    ab = np.full(48, 1.0 / 48)
    rhob = np.full(48, 1.0 / 48)
    Pbig, stb = solve_semi_ot(Cb, ab, rhob, eps=0.05, tau=50.0, iters=6000)
    Pbal = _balanced_sinkhorn(Cb, ab, rhob, eps=0.05)
    check("large tau Q=A -> close to balanced OT",
          np.abs(Pbig - Pbal).max() < 2e-2,
          f"max|dP|={np.abs(Pbig-Pbal).max():.3f}")

    # 4. permutation invariance (rows & cols simultaneously)
    Cq = rng.uniform(0.0, 1.5, (40, 40))
    aq = rng.dirichlet(np.ones(40))
    rhq = rng.dirichlet(np.ones(40))
    P1, _ = solve_semi_ot(Cq, aq, rhq, eps=0.05, tau=3.0)
    pr = rng.permutation(40)
    pc = rng.permutation(40)
    P2, _ = solve_semi_ot(Cq[np.ix_(pr, pc)], aq[pr], rhq[pc], eps=0.05, tau=3.0)
    back = np.zeros((40, 40))
    back[np.ix_(pr, pc)] = P2
    check("semi-ot row&col permutation invariant", np.abs(back - P1).max() < 1e-6)

    # 5. identical vs duplicated content (capacity premium)
    A = 64
    X = rng.normal(size=(A, 32))
    X = X / np.maximum(np.linalg.norm(X, axis=1, keepdims=True), 1e-8)
    Cid = 1.0 - X @ X.T                     # Q=A identical-to-anchor
    aid = np.full(A, 1.0 / A)
    rhid = np.full(A, 1.0 / A)
    prem, P, free, _ = capacity_premium(Cid, aid, rhid, eps=0.02, tau=4.0)
    check("identical query/support premium ~0", float(np.abs(prem).max()) < 1e-3,
          f"max_prem={float(np.abs(prem).max()):.2e}")
    # duplicate: rows 0..d-1 all have anchor-0 content (d copies + originals remain)
    for d in (2, 4, 8):
        Q = A + d                            # d extra duplicate rows
        Cq = np.empty((Q, A))
        Xq = np.vstack([X, np.tile(X[0], (d, 1))])
        Cq = 1.0 - Xq @ X.T
        aq = np.full(Q, 1.0 / Q)
        prem2, _, _, _ = capacity_premium(Cq, aq, rhid, eps=0.02, tau=4.0)
        dup_rows = np.arange(A, Q)           # the added copies of anchor-0 content
        check(f"duplicated rows pay more than identical baseline (d={d})",
              prem2[dup_rows].mean() > prem.max() + 1e-3,
              f"dup_mean={prem2[dup_rows].mean():.4f} base_max={prem.max():.4f}")
    # spillover-ish: far (non-dup) rows premium should stay well below dup rows
    prem2, _, _, _ = capacity_premium(
        1.0 - np.vstack([X, np.tile(X[0], (8, 1))]) @ X.T,
        np.full(A + 8, 1.0 / (A + 8)), rhid, eps=0.02, tau=4.0)
    far = prem2[:A]
    check("far rows premium < duplicated rows premium",
          far.mean() < prem2[A:].mean(),
          f"far={far.mean():.4f} dup={prem2[A:].mean():.4f}")


if __name__ == "__main__":
    test_dnc_selector()
    test_semi_ot()
    n_fail = sum(1 for _, ok, _ in PASS if not ok)
    print(f"\n{len(PASS) - n_fail}/{len(PASS)} passed")
    raise SystemExit(1 if n_fail else 0)
