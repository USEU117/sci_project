"""Route-D gate D1: can unlabeled features predict WHEN A1 needs correction?

Design review D1:
  - label (development-only): per-image fraction of pixels where the GT-informed
    per-pixel best-of-branch Oracle does NOT select concat (i.e. concat is not the
    best branch). A "benefit" image is one where A1 can be improved.
  - features: UNLABELED only (cross-modal disagreement, concat score spread,
    KNN-distance statistics of the concat memory bank).
  - evaluation: leave-one-category-out (fit on 5 categories, evaluate on the 6th),
    report AUROC / AP / calibration; permutation control (shuffled labels).

Gate D1 passes only if:
  - held-out benefit-prediction AUROC clearly above the positive base rate
  - consistent direction on >= 4/6 held-out categories
  - feature permutation drops performance back to random

Outputs:
  experiments/dynamic_fusion/route_d_d0_20260818/d1_predictability_report.{json,md}
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from evaluate_a1_feature_fusion import compute_metrics, fuse_category, load_features  # noqa: E402

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score

MAP_SIZE = (448, 448)


def extract_image_features(concat_map: np.ndarray, dino_map: np.ndarray, clip_map: np.ndarray) -> dict:
    """UNLABELED per-image features. No GT anywhere."""
    concat = concat_map.astype(np.float64)
    dino = dino_map.astype(np.float64)
    clip = clip_map.astype(np.float64)
    div_dc = dino - clip
    return {
        "disagreement_abs_mean": float(np.abs(div_dc).mean()),
        "disagreement_abs_std": float(np.abs(div_dc).std()),
        "disagreement_neg_frac": float((div_dc < 0).mean()),  # CLIP above DINO share
        "concat_mean": float(concat.mean()),
        "concat_std": float(concat.std()),
        "concat_p95": float(np.percentile(concat, 95)),
        "concat_p99_minus_median": float(np.percentile(concat, 99) - np.median(concat)),
        "dino_clip_rank_spearman": float(_spearman(dino.ravel(), clip.ravel())),
    }


def _spearman(a: np.ndarray, b: np.ndarray) -> float:
    from scipy.stats import rankdata
    ra, rb = rankdata(a), rankdata(b)
    n = a.size
    d = ra - rb
    denom = n * (n * n - 1) / 6.0
    return float(1.0 - 6.0 * np.dot(d, d) / denom) if denom else 0.0


def oracle_concat_share(concat_map: np.ndarray, dino_map: np.ndarray, clip_map: np.ndarray, gt: np.ndarray) -> float:
    """Fraction of pixels where concat is chosen by the GT-informed best-of-3 Oracle."""
    stack = np.stack([concat_map, dino_map, clip_map])
    best_idx = np.where((gt > 0.5), stack.argmax(axis=0), stack.argmin(axis=0))
    return float((best_idx == 0).mean())


def run() -> None:
    manifest = json.loads((ROOT / "data" / "splits" / "mpdd" / "manifest.json").read_text(encoding="utf-8"))
    categories = sorted(manifest["categories"])

    dino_dir = ROOT / "outputs/dynamic_fusion/v3_direction_a/features_vitb14_s0_k1/anomalydino_visual"
    clip_dir = ROOT / "outputs/dynamic_fusion/v3_direction_a/features_s0_k1/anomalyclip_text"

    X, share, cat_ids = [], [], []
    for cat in categories:
        dino = load_features(dino_dir / f"{cat}.npz")
        clip = load_features(clip_dir / f"{cat}.npz")
        gt = dino["imgs_masks"]
        maps = {m: fuse_category(dino, clip, m, 0, False, MAP_SIZE, 0.5) for m in ("concat", "dino", "clip")}
        for i in range(maps["concat"].shape[0]):
            share.append(oracle_concat_share(maps["concat"][i], maps["dino"][i], maps["clip"][i], gt[i]))
            X.append(extract_image_features(maps["concat"][i], maps["dino"][i], maps["clip"][i]))
            cat_ids.append(cat)

    X = np.asarray([list(f.values()) for f in X], dtype=np.float64)
    share = np.asarray(share, dtype=np.float64)
    # label: "high correction benefit" = concat is NOT best on many pixels
    # (bottom 25% share); balanced against the majority by design (base rate 0.25)
    y = (share <= np.percentile(share, 25)).astype(np.float64)
    names = list(extract_image_features(maps["concat"][0], maps["dino"][0], maps["clip"][0]).keys())
    cat_arr = np.asarray(cat_ids)

    base_rate = float(y.mean())

    def loce_auroc(rng_seed: int, shuffle: bool = False) -> dict:
        rng = np.random.RandomState(rng_seed)
        per_cat = {}
        for held in categories:
            train_mask = cat_arr != held
            test_mask = cat_arr == held
            yt = y.copy()
            if shuffle:
                yt = yt[rng.permutation(len(y))]
            if len(np.unique(yt[train_mask])) < 2:
                per_cat[held] = {"auroc": 0.5, "ap": float(yt[test_mask].mean())}
                continue
            clf = LogisticRegression(max_iter=2000)
            clf.fit(X[train_mask], yt[train_mask])
            proba = clf.predict_proba(X[test_mask])[:, 1]
            per_cat[held] = {"auroc": float(roc_auc_score(y[test_mask], proba)), "ap": float(average_precision_score(y[test_mask], proba))}
        return per_cat

    real = loce_auroc(0, shuffle=False)
    permuted = loce_auroc(0, shuffle=True)

    mean_auroc = float(np.mean([v["auroc"] for v in real.values()]))
    mean_permuted_auroc = float(np.mean([v["auroc"] for v in permuted.values()]))
    mean_ap = float(np.mean([v["ap"] for v in real.values()]))
    cats_above_base = sum(1 for v in real.values() if v["auroc"] > base_rate)
    direction_consistent = cats_above_base >= 4

    passed = (
        mean_auroc >= 0.60
        and direction_consistent
        and (mean_auroc - mean_permuted_auroc) >= 0.10
    )

    report = {
        "schema_version": 1,
        "run_id": "route_d_d1_20260818",
        "stage": "route_D_gate_D1_predictability",
        "dataset": "mpdd", "dataset_role": "development",
        "unit": "image",
        "label": "high correction benefit: oracle picks non-concat on enough pixels (concat share in bottom 25%; label uses GT, development-only)",
        "features": names,
        "base_rate_positive": round(base_rate, 4),
        "n_images": int(len(y)),
        "loce_real": {k: {kk: round(vv, 4) for kk, vv in v.items()} for k, v in real.items()},
        "loce_permuted": {k: {kk: round(vv, 4) for kk, vv in v.items()} for k, v in permuted.items()},
        "mean_auroc_real": round(mean_auroc, 4),
        "mean_ap_real": round(mean_ap, 4),
        "mean_auroc_permuted": round(mean_permuted_auroc, 4),
        "categories_above_base_rate": cats_above_base,
        "gate_conditions": {
            "mean_auroc_ge_0_60": mean_auroc >= 0.60,
            "consistent_direction_ge_4_of_6": direction_consistent,
            "auroc_gap_vs_permutation_ge_0_10": (mean_auroc - mean_permuted_auroc) >= 0.10,
        },
        "gate_d1_passed": passed,
        "decision": (
            "STOP route D: unlabeled features cannot predict correction benefit -> archive dynamic-routing line permanently; continue route S"
            if not passed
            else "predictability holds -> proceed to D2 (single pre-registered A1-R candidate)"
        ),
    }
    out_dir = ROOT / "experiments" / "dynamic_fusion" / "route_d_d0_20260818"
    (out_dir / "d1_predictability_report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    md = [
        "# 路线 D 决策门 D1（无标签可预测性）",
        "",
        f"RunId: `route_d_d1_20260818` · MPDD s0/k1 · {len(y)} 图 · 阳性基率 {base_rate:.3f}",
        "",
        f"- LOCO mean AUROC = **{mean_auroc:.3f}**（要求 ≥ 0.60）",
        f"- LOCO mean AP = {mean_ap:.3f}",
        f"- 特征置乱后 mean AUROC = **{mean_permuted_auroc:.3f}**（差距 {mean_auroc - mean_permuted_auroc:.3f}，要求 ≥ 0.10）",
        f"- 高于基率的类别数：{cats_above_base}/6（要求 ≥ 4）",
        f"- **Gate D1：{'PASS → 进入 D2' if passed else 'FAIL → 停止路线 D，正式归档动态路线，回路线 S'}**",
        "",
        "- 特征全部无标签（跨模态分歧、concat 分布、排名一致性）。",
        "- 若 D1 失败，按设计审查第 12 节第 9 条将动态路线正式归档并停止扩展。",
        "",
    ]
    (out_dir / "d1_predictability_report.md").write_text("\n".join(md), encoding="utf-8")

    print(json.dumps(report["gate_conditions"], ensure_ascii=False, indent=2))
    print(f"mean_auroc={mean_auroc:.3f} permuted={mean_permuted_auroc:.3f} base={base_rate:.3f} gate_d1_passed={passed}")


if __name__ == "__main__":
    run()
