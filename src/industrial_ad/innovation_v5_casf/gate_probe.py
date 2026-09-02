"""Wave 0 — amplified CASF gate probe (task book 15 section 2.4).

Deterministic, pre-registered rule for the category gate set:

    c in Gset  <=>  mean(headroom_c) >= +0.02  AND  >= 2/3 seeds have headroom_c >= +0.02

where headroom_c = pseudo-Dice(asym-trained head) - pseudo-Dice(sym-trained head)
evaluated on held-out asymmetric episodes of category c.

Mechanics mirror the v4 D3 probe (diagnostics.d3_statistics / pseudo_anomaly_map /
LogisticRegression C=1.0, Dice @0.5) exactly; only the episode count and the
number of family seeds are amplified. The probe reads normal-reference features
only (aligned_category, label-free); it never loads GT or masks.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
for p in (str(ROOT / "src"), str(ROOT / "scripts")):
    if p not in sys.path:
        sys.path.insert(0, p)

from industrial_ad.innovation_v4_diagnostics import diagnostics as diag  # noqa: E402
from industrial_ad.innovation_v4_diagnostics import common  # noqa: E402

# --- pre-registered probe hyper-parameters (task book 15 section 2.4 / config) ---
N_TRAIN_EP = 24        # pseudo episodes per family (asym {dino,clip,both} 1:1:1; sym {both})
N_EVAL_EP = 24         # held-out asymmetric episodes
GAMMA = 0.35           # tangent-noise strength
LOGREG_C = 1.0         # logistic head capacity (matches v4 D3 probe)
DICE_THRESHOLD = 0.5   # fixed decision threshold for pseudo Dice
FAMILY_SEEDS = (0, 1, 2)
BASE_RNG = 20260903    # same base as the v4 D3 probe
SPLIT_SEED = 0
SHOT = 2               # two references: ref[0] perturbed, ref[1] = memory/donor

# --- deterministic gate rule (task book 15 section 2.4) ---
MIN_MEAN_HEADROOM = 0.02
MIN_SEED_VOTES = 2
N_SEEDS = len(FAMILY_SEEDS)


def rng_for(category: str, seed_ix: int) -> np.random.Generator:
    """Deterministic per-(category, family-seed) RNG."""
    return np.random.default_rng(BASE_RNG + 1000 * seed_ix + hash(category) % 4096)


def category_probe_seed(category: str, seed_ix: int) -> dict:
    """One (category, family-seed) probe run; returns asym/sym metrics + headroom.

    LOO-clean split: ref[0] is the base map that gets perturbed; ref[1] is the
    per-branch memory AND the transplant donor (identical to the v4 D3 probe).
    """
    aligned = common.aligned_category("mpdd", SPLIT_SEED, SHOT, category)
    d0 = aligned.d_ref[0].copy()          # [32,32,768]
    c0 = aligned.c_ref[0].copy()
    mem_d = aligned.d_ref[1][None]        # [1,32,32,768] per-branch memory
    mem_c = aligned.c_ref[1][None]

    rng = rng_for(category, seed_ix)

    def branch_z_episode(d_feat, c_feat):
        zd = diag._branch_z(d_feat.reshape(-1, 768), mem_d.reshape(-1, 768))
        zc = diag._branch_z(c_feat.reshape(-1, 768), mem_c.reshape(-1, 768))
        h, w = d_feat.shape[:2]
        return zd.reshape(h, w), zc.reshape(h, w)

    def make_episodes(pool: list[str], n: int) -> list[dict]:
        eps = []
        for _ in range(n):
            mode = pool[rng.integers(0, len(pool))]
            dm, cm, mask = diag.pseudo_anomaly_map(d0, c0, mem_d[0], mem_c[0],
                                                   mode, rng, gamma=GAMMA)
            zd, zc = branch_z_episode(dm, cm)
            feats = diag.d3_statistics(dm, cm, zd, zc)   # [1024,4]
            eps.append({"feats": feats, "mask": mask.ravel() > 0})
        return eps

    asym_pool = ["dino", "clip", "both"]
    sym_pool = ["both"]
    train_asym = make_episodes(asym_pool, N_TRAIN_EP)
    train_sym = make_episodes(sym_pool, N_TRAIN_EP)
    eval_asym = make_episodes(asym_pool, N_EVAL_EP)

    from sklearn.linear_model import LogisticRegression

    def train_and_eval(train_eps: list[dict]) -> dict | None:
        X = np.concatenate([e["feats"] for e in train_eps])
        y = np.concatenate([e["mask"] for e in train_eps]).astype(np.int64)
        if y.sum() == 0 or (y == 0).sum() == 0:
            return None
        clf = LogisticRegression(max_iter=500, C=LOGREG_C)
        clf.fit(X, y)
        from sklearn.metrics import average_precision_score, roc_auc_score
        Xe = np.concatenate([e["feats"] for e in eval_asym])
        ye = np.concatenate([e["mask"] for e in eval_asym])
        prob = clf.predict_proba(Xe)[:, 1]
        auroc = float(roc_auc_score(ye, prob))
        ap = float(average_precision_score(ye, prob))
        pred = prob > DICE_THRESHOLD
        inter = (pred & ye).sum()
        dice = float(2 * inter / (pred.sum() + ye.sum() + 1e-9))
        return {"auroc": round(auroc, 4), "ap": round(ap, 4),
                "dice": round(dice, 4), "n_train_px": int(y.size),
                "n_pos_train": int(y.sum())}

    res_asym = train_and_eval(train_asym)
    res_sym = train_and_eval(train_sym)
    headroom = (None if (res_asym is None or res_sym is None)
                else round(float(res_asym["dice"] - res_sym["dice"]), 4))
    return {
        "dataset": "mpdd", "role": "development", "split_seed": SPLIT_SEED,
        "shot": SHOT, "category": category, "seed_ix": int(seed_ix),
        "n_train_ep_per_family": N_TRAIN_EP, "n_eval_asym": N_EVAL_EP,
        "gamma": GAMMA, "logreg_C": LOGREG_C,
        "trained_asym_eval": res_asym, "trained_sym_eval": res_sym,
        "headroom": headroom,
    }


def derive_gset(rows: list[dict]) -> tuple[list[str], dict]:
    """Deterministic gate rule (task book 15 section 2.4).

    rows: per-(category, seed_ix) probe outputs. Returns (sorted Gset,
    rule detail keyed by category with per-seed headrooms).
    """
    by_cat: dict[str, dict] = {}
    for r in rows:
        by_cat.setdefault(r["category"], {})[r["seed_ix"]] = r["headroom"]
    detail = {}
    gset = []
    for cat, seeds in sorted(by_cat.items()):
        hrs = [seeds.get(i) for i in range(N_SEEDS)]
        vals = [h for h in hrs if h is not None]
        mean_hr = round(float(np.mean(vals)), 4) if vals else None
        votes = sum(1 for h in vals if h is not None and h >= MIN_MEAN_HEADROOM)
        active = bool(vals and mean_hr is not None and mean_hr >= MIN_MEAN_HEADROOM
                      and votes >= MIN_SEED_VOTES)
        if active:
            gset.append(cat)
        detail[cat] = {"per_seed_headroom": {str(i): hrs[i] for i in range(N_SEEDS)},
                       "mean_headroom": mean_hr, "seed_votes": votes, "active": active}
    return sorted(gset), detail


def summarize(rows: list[dict]) -> dict:
    """Aggregate probe rows into the frozen Gset report."""
    gset, detail = derive_gset(rows)
    return {
        "program": "innovation_v5_casf", "phase": "Wave0_gate_probe",
        "dataset": "mpdd", "role": "development", "split_seed": SPLIT_SEED,
        "shot": SHOT, "n_train_ep_per_family": N_TRAIN_EP,
        "n_eval_asym": N_EVAL_EP, "gamma": GAMMA, "logreg_C": LOGREG_C,
        "family_seeds": list(FAMILY_SEEDS), "base_rng": BASE_RNG,
        "gate_rule": {"min_mean_headroom": MIN_MEAN_HEADROOM,
                      "min_seed_votes": MIN_SEED_VOTES, "n_seeds": N_SEEDS},
        "Gset": gset,
        "by_category": detail,
        "probe_identity": "mirror of v4 D3 probe (diagnostics.d3_statistics/"
                          "pseudo_anomaly_map/LogisticRegression C=1.0), amplified "
                          "to 24 episodes/family x 3 family seeds",
    }
