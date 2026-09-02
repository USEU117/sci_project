"""D3 — dual-branch supervision value diagnostic (task book 14 section 10 D3).

CASF-style branch-asymmetric pseudo anomalies are generated from normal MPDD
references only (deepest layer): distant transplant / tangent noise inside a
random mask, perturbing DINO only / CLIP only / both at 1:1:1 for the
asymmetric family, and both-only for the symmetric control.

A tiny logistic head sees CASF input statistics (z_D, z_C, |z_D-z_C|,
signed disagreement) and is trained either on the asymmetric family or on the
symmetric family; both are evaluated on held-out asymmetric episodes. If the
asymmetric-trained head does not beat the symmetric-trained head by >= 0.02
pseudo Dice, CASF-style cross-branch supervision carries no learnable value on
MPDD and the route is downgraded.

Everything is synthetic (normal references only); no real test mask/label is
ever read.
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
for p in (str(ROOT / "src"), str(ROOT / "scripts")):
    if p not in sys.path:
        sys.path.insert(0, p)

from industrial_ad.innovation_v4_diagnostics import common, diagnostics as diag  # noqa: E402

SEED = 0
SHOT = 2            # two references: one is perturbed, the other is the memory
N_TRAIN_EP = 12     # pseudo episodes per family
N_EVAL_EP = 12      # held-out asymmetric episodes
GAMMA = 0.35        # tangent-noise strength (fixed, pre-registered)
BASE_RNG = 20260903


def category_d3(category: str) -> dict:
    aligned = common.aligned_category("mpdd", SEED, SHOT, category)
    # two references; perturb ref[0], memory = ref[1] (LOO clean separation)
    d0 = aligned.d_ref[0].copy()   # [32,32,768] unit rows
    c0 = aligned.c_ref[0].copy()
    d1 = aligned.d_ref[1][None]    # [1,32,32,768] memory branch
    c1 = aligned.c_ref[1][None]

    rng = np.random.default_rng(BASE_RNG + hash(category) % 4096)

    def stats_of(dm: np.ndarray, cm: np.ndarray, zd: np.ndarray, zc: np.ndarray) -> np.ndarray:
        return diag.d3_statistics(dm, cm, zd, zc)

    def branch_z_episode(d_feat, c_feat):
        zd = diag._branch_z(d_feat.reshape(-1, 768), d1.reshape(-1, 768))
        zc = diag._branch_z(c_feat.reshape(-1, 768), c1.reshape(-1, 768))
        h, w = d_feat.shape[:2]
        return zd.reshape(h, w), zc.reshape(h, w)

    # --- build episodes
    def make_episodes(pool: list[str], n: int):
        eps = []
        for _ in range(n):
            mode = pool[rng.integers(0, len(pool))]
            dm, cm, mask = diag.pseudo_anomaly_map(d0, c0, d1[0], c1[0], mode, rng, gamma=GAMMA)
            zd, zc = branch_z_episode(dm, cm)
            feats = stats_of(dm, cm, zd, zc)          # [1024,4] already unit-normalised
            eps.append({"feats": feats, "mask": mask.ravel() > 0})
        return eps

    asym_pool = ["dino", "clip", "both"]
    sym_pool = ["both"]
    train_asym = make_episodes(asym_pool, N_TRAIN_EP)
    train_sym = make_episodes(sym_pool, N_TRAIN_EP)
    eval_asym = make_episodes(asym_pool, N_EVAL_EP)

    from sklearn.linear_model import LogisticRegression

    def train_and_eval(train_eps):
        X = np.concatenate([e["feats"] for e in train_eps])
        y = np.concatenate([e["mask"] for e in train_eps]).astype(np.int64)
        if y.sum() == 0 or (y == 0).sum() == 0:
            return None
        clf = LogisticRegression(max_iter=500, C=1.0)
        clf.fit(X, y)
        from sklearn.metrics import average_precision_score, roc_auc_score
        Xe = np.concatenate([e["feats"] for e in eval_asym])
        ye = np.concatenate([e["mask"] for e in eval_asym])
        prob = clf.predict_proba(Xe)[:, 1]
        auroc = float(roc_auc_score(ye, prob))
        ap = float(average_precision_score(ye, prob))
        # Dice at the fixed 0.5 decision threshold
        pred = prob > 0.5
        inter = (pred & ye).sum()
        dice = float(2 * inter / (pred.sum() + ye.sum() + 1e-9))
        return {"auroc": round(auroc, 4), "ap": round(ap, 4), "dice": round(dice, 4),
                "n_train_px": int(y.size), "n_pos_train": int(y.sum())}

    res_asym = train_and_eval(train_asym)
    res_sym = train_and_eval(train_sym)
    headroom = (None if (res_asym is None or res_sym is None)
                else round(float(res_asym["dice"] - res_sym["dice"]), 4))
    return {
        "dataset": "mpdd", "role": "development", "seed": SEED, "shot": SHOT,
        "category": category,
        "gamma": GAMMA, "n_train_asym": N_TRAIN_EP, "n_train_sym": N_TRAIN_EP,
        "n_eval_asym": N_EVAL_EP,
        "trained_asym_eval": res_asym,
        "trained_sym_eval": res_sym,
        "asym_minus_sym_dice_headroom": headroom,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-root", type=Path,
                        default=common.EXPERIMENT_ROOT / "D3_supervision_value")
    parser.add_argument("--category", default=None)
    args = parser.parse_args()
    common.assert_development_only()
    manifest = common.manifest_for("mpdd")
    cats = [args.category] if args.category else sorted(manifest["categories"])
    out_root = args.out_root
    out_root.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    reports = []
    for cat in cats:
        print(f"[D3] {cat}", flush=True)
        rep = category_d3(cat)
        reports.append(rep)
        (out_root / f"{cat}_s{SEED}_k{SHOT}.json").write_text(
            json.dumps(rep, ensure_ascii=False, indent=1), encoding="utf-8")
        print(f"     asym={rep['trained_asym_eval']} sym={rep['trained_sym_eval']} "
              f"headroom={rep['asym_minus_sym_dice_headroom']}", flush=True)
    hr = [r["asym_minus_sym_dice_headroom"] for r in reports
          if r["asym_minus_sym_dice_headroom"] is not None]
    summary = {
        "schema_version": 1, "program": "innovation_v4_diagnostics",
        "diagnostic": "D3_supervision_value", "dataset": "mpdd", "role": "development",
        "seed": SEED, "shot": SHOT,
        "rule": "CASF downgraded if asymmetric-vs-symmetric pseudo-Dice headroom "
                "< 0.02 (task book 14 section 10 D3)",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "elapsed_s": round(time.time() - t0, 1),
        "mean_dice_headroom": (round(float(np.mean(hr)), 4) if hr else None),
        "categories_supporting_casf": sum(1 for v in hr if v is not None and v >= 0.02),
        "n_categories": len(reports),
    }
    (out_root / "D3_SUMMARY.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=1), encoding="utf-8")
    print(json.dumps(summary, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
