"""DNC channel selectors (doc28 s4.3 fixed DNC-C).

DNC-I : per-branch top-K by defect-vs-nuisance score q.
DNC-C : per-branch greedy with cross-branch redundancy penalty applied INSIDE each
        branch: at every step, for each branch b we score every unselected channel j
        by  q_b[j] - lambda * max_{k in opposite_chosen} |corr(j,k)|  and add the
        single globally best (branch, j). This can drop a high-q/high-redundancy
        channel in favour of a lower-q/low-redundancy one (the V13 bug: penalty
        only arbitrated between the two argmax-q candidates, so sets always
        collapsed to DNC-I). Both branches keep exactly K unique indices.

Deterministic: fixed tie-break by (score desc, index asc). Pure numpy.
"""
from __future__ import annotations

import numpy as np


def select_dnc_i(qD: np.ndarray, qC: np.ndarray, keep: int = 256):
    return (np.argsort(-qD)[:keep], np.argsort(-qC)[:keep])


def select_dnc_c(qD: np.ndarray, qC: np.ndarray, corrDC: np.ndarray,
                 lam: float, keep: int = 256, rng_seed: int = 0):
    """corrDC [nD, nC] = correlation between dino channel j and clip channel k."""
    nD, nC = qD.shape[0], qC.shape[0]
    qD = np.asarray(qD, dtype=np.float64)
    qC = np.asarray(qC, dtype=np.float64)
    corr = np.abs(np.asarray(corrDC, dtype=np.float64))   # [nD, nC]
    chosenD: list[int] = []
    chosenC: list[int] = []
    freeD = np.ones(nD, dtype=bool)
    freeC = np.ones(nC, dtype=bool)
    while freeD.any() and freeC.any() and (len(chosenD) < keep or len(chosenC) < keep):
        candD = candC = None
        if len(chosenD) < keep and freeD.any():
            red = np.zeros(freeD.sum())
            if chosenC:
                red = corr[freeD][:, chosenC].max(axis=1)
            scores = qD[freeD] - lam * red
            j_rel = int(np.argmax(scores))
            j = int(np.flatnonzero(freeD)[j_rel])
            candD = (j, float(scores[j_rel]))
        if len(chosenC) < keep and freeC.any():
            red = np.zeros(freeC.sum())
            if chosenD:
                red = corr[np.ix_(chosenD, freeC)].max(axis=0)
            scores = qC[freeC] - lam * red
            k_rel = int(np.argmax(scores))
            k = int(np.flatnonzero(freeC)[k_rel])
            candC = (k, float(scores[k_rel]))
        if candD is None and candC is None:
            break
        if candC is None or (candD is not None and candD[1] >= candC[1]):
            j = candD[0]
            chosenD.append(j)
            freeD[j] = False
        else:
            k = candC[0]
            chosenC.append(k)
            freeC[k] = False
    # fill any residual slots deterministically (should not happen at keep<=n)
    if len(chosenD) < keep:
        for j in np.flatnonzero(freeD)[: keep - len(chosenD)]:
            chosenD.append(int(j))
    if len(chosenC) < keep:
        for k in np.flatnonzero(freeC)[: keep - len(chosenC)]:
            chosenC.append(int(k))
    return (np.asarray(sorted(chosenD), dtype=int), np.asarray(sorted(chosenC), dtype=int))


def redundancy_stats(chosenD, chosenC, corrDC):
    """Mean/max cross-branch |corr| among chosen sets + Jaccard vs DNC-I helper."""
    cd = np.asarray(chosenD)
    cc = np.asarray(chosenC)
    sub = np.abs(corrDC[np.ix_(cd, cc)])
    return {"nD": len(cd), "nC": len(cc),
            "max_corr": float(sub.max()) if sub.size else 0.0,
            "mean_corr": float(sub.mean()) if sub.size else 0.0}
