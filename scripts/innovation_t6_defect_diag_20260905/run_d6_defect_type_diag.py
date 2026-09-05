"""Direction-6 pure diagnostic: A1 real per-defect-type gap structure (doc37).

Read-only diagnostic over the frozen A1 concat pipeline on real MPDD seed0
(k2/k4). No candidates, no fitting, no new mechanism.

Inputs (read-only):
  - compact concat patch maps : submission_repro_20260827/predictions_compact/maps/mpdd/s0_k{shot}/{cat}.npz
                                ('concat_patch_map' float16 [n, 32, 32], replay-verified <=5e-5)
  - masks + sample_ids         : outputs/dynamic_fusion/v3_direction_a/features_vitb14_s0_k{shot}/anomalydino_visual/{cat}.npz
  - parity targets             : experiments/dynamic_fusion/innovation_t5_relation32_20260905/REAL_D5_s0_k{shot}.json

Outputs:
  experiments/dynamic_fusion/innovation_t6_defect_diag_20260905/D6_REPORT_s0_k{shot}.json
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
for p in (str(ROOT / "src"), str(ROOT / "scripts"), str(ROOT / "methods" / "anomalydino")):
    sys.path.insert(0, p)

import evaluate_a1_feature_fusion as A1  # noqa: E402
from sklearn.metrics import average_precision_score, roc_auc_score  # noqa: E402
from src.utils import dists2map  # noqa: E402

COMPACT = ROOT / "submission_repro_20260827/predictions_compact/maps/mpdd"
DINO_FEAT = ROOT / "outputs/dynamic_fusion/v3_direction_a"
D5_REAL = ROOT / "experiments/dynamic_fusion/innovation_t5_relation32_20260905"
OUT = ROOT / "experiments/dynamic_fusion/innovation_t6_defect_diag_20260905"
CONTEXT_TYPES = ("parts_mismatch", "bend_and_parts_mismatch")


def dino_dir(shot: int) -> Path:
    return DINO_FEAT / f"features_vitb14_s0_k{shot}" / "anomalydino_visual"


def defect_type_of(sample_id: str) -> str:
    parts = str(sample_id).replace("\\", "/").split("/")
    return parts[parts.index("test") + 1]


def to_maps448(patch_map: np.ndarray) -> np.ndarray:
    """float16 patch maps -> float32 448x448 maps (exact replay path of the compact cache)."""
    return np.stack([dists2map(m.astype(np.float32), (448, 448)) for m in patch_map]).astype(np.float32)


def image_metrics(maps448: np.ndarray, pos_idx: np.ndarray, neg_idx: np.ndarray) -> dict | None:
    if pos_idx.size == 0 or neg_idx.size == 0:
        return None
    scores = maps448.reshape(maps448.shape[0], -1).max(axis=1)
    y = np.zeros(maps448.shape[0], dtype=np.int32)
    y[pos_idx] = 1
    keep = np.concatenate([pos_idx, neg_idx])
    if np.unique(y[keep]).size < 2:
        return None
    return {
        "image_auroc": round(float(roc_auc_score(y[keep], scores[keep])), 6),
        "image_ap": round(float(average_precision_score(y[keep], scores[keep])), 6),
    }


def run_cat(cat: str, shot: int) -> dict:
    comp = np.load(COMPACT / f"s0_k{shot}" / f"{cat}.npz", allow_pickle=False)
    dino = np.load(dino_dir(shot) / f"{cat}.npz", allow_pickle=True)
    ids = np.asarray(dino["sample_ids"])
    masks = np.asarray(dino["imgs_masks"], dtype=np.uint8)
    assert np.array_equal(np.asarray(comp["sample_ids"]), ids), f"{cat}: sample order mismatch"
    maps448 = to_maps448(np.asarray(comp["concat_patch_map"]))
    n = ids.shape[0]

    types = [defect_type_of(s) for s in ids]
    good_idx = np.array([i for i, t in enumerate(types) if t == "good"], dtype=np.int64)
    row = {"category": cat, "n_test": int(n), "n_good": int(good_idx.size),
           "full_pixel": A1.compute_metrics(maps448.astype(np.float64), masks), "types": []}
    for t in sorted({t for t in types if t != "good"}):
        idx = np.array([i for i, x in enumerate(types) if x == t], dtype=np.int64)
        pm = A1.compute_metrics(maps448[idx].astype(np.float64), masks[idx])
        entry = {"type": t, "n": int(idx.size), **{k: round(float(v), 6) for k, v in pm.items()}}
        im = image_metrics(maps448, idx, good_idx)
        if im:
            entry.update(im)
        row["types"].append(entry)
    return row


def mean(xs: list[float]) -> float:
    return float(np.mean(xs)) if xs else float("nan")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--shot", type=int, required=True, choices=[2, 4])
    ap.add_argument("--cats", default=None, help="comma list; default all 6")
    args = ap.parse_args()

    cats = [c.strip() for c in args.cats.split(",")] if args.cats else [
        "bracket_black", "bracket_brown", "bracket_white", "connector", "metal_plate", "tubes"]
    rows = [run_cat(c, args.shot) for c in cats]

    mean_ap = mean([r["full_pixel"]["pixel_ap"] for r in rows])
    # ---- type-level aggregation (equal weight per category, A1 macro convention) ----
    type_rows: dict[str, list[dict]] = {}
    for r in rows:
        for t in r["types"]:
            type_rows.setdefault(t["type"], []).append({"cat": r["category"], **t})
    type_summary = {}
    for t, items in sorted(type_rows.items()):
        n = int(sum(it["n"] for it in items))
        type_summary[t] = {
            "cats": sorted(it["cat"] for it in items),
            "n_total": n,
            "n_cats": len(items),
            "pixel_ap_macro": round(mean([it["pixel_ap"] for it in items]), 6),
            "pixel_auroc_macro": round(mean([it["pixel_auroc"] for it in items]), 6),
            "pixel_aupro_macro": round(mean([it["pixel_aupro"] for it in items]), 6),
            "image_auroc_macro": round(mean([it["image_auroc"] for it in items if "image_auroc" in it]), 6),
            "image_ap_macro": round(mean([it["image_ap"] for it in items if "image_ap" in it]), 6),
        }
    # ---- context family: pool family types per category, then macro over cats ----
    fam = []
    for r in rows:
        dino = np.load(dino_dir(args.shot) / f"{r['category']}.npz", allow_pickle=True)
        comp = np.load(COMPACT / f"s0_k{args.shot}" / f"{r['category']}.npz", allow_pickle=False)
        ids = np.asarray(dino["sample_ids"])
        idx = np.array([i for i, s in enumerate(ids)
                        if defect_type_of(s) in CONTEXT_TYPES], dtype=np.int64)
        if idx.size:
            maps448 = to_maps448(np.asarray(comp["concat_patch_map"]))
            masks = np.asarray(dino["imgs_masks"], dtype=np.uint8)
            fam.append({"cat": r["category"], "n": int(idx.size),
                        **{k: round(float(v), 6) for k, v in
                           A1.compute_metrics(maps448[idx].astype(np.float64), masks[idx]).items()}})
    family_summary = None
    if fam:
        family_summary = {"cats": [f["cat"] for f in fam], "n_total": int(sum(f["n"] for f in fam)),
                          "n_cats": len(fam),
                          "pixel_ap_macro": round(mean([f["pixel_ap"] for f in fam]), 6),
                          "pixel_auroc_macro": round(mean([f["pixel_auroc"] for f in fam]), 6),
                          "pixel_aupro_macro": round(mean([f["pixel_aupro"] for f in fam]), 6),
                          "per_cat": [{k: f[k] for k in ("cat", "n", "pixel_ap", "pixel_auroc", "pixel_aupro")}
                                      for f in fam]}
    # ---- parity against REAL_D5 C0 (doc36) ----
    d5 = json.loads((D5_REAL / f"REAL_D5_s0_k{args.shot}.json").read_text(encoding="utf-8"))
    per_cat_ref = {r["category"]: r["C0"]["pixel_ap"] for r in d5["per_category"]}
    cat_diff = [rows_c["full_pixel"]["pixel_ap"] - per_cat_ref[rows_c["category"]] for rows_c in rows]
    pm_macro = type_summary.get("parts_mismatch", {}).get("pixel_ap_macro")
    parity = {
        "mean_ap": round(mean_ap, 6), "target_mean_ap": d5["mean_A1_ap"],
        "pm_macro_ap": pm_macro, "target_pm_macro_ap": d5.get("pm_mean_A1_ap"),
        "max_abs_cat_ap_diff": round(float(np.max(np.abs(cat_diff))), 9),
    }
    out = {"seed": 0, "shot": args.shot, "created_utc": datetime.now(timezone.utc).isoformat(),
           "parity": parity, "defect_type_summary": type_summary,
           "context_family_summary": family_summary, "per_category": rows}
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / f"D6_REPORT_s0_k{args.shot}.json").write_text(
        json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"[k{args.shot}] mean A1 AP={parity['mean_ap']:.4f} (target {parity['target_mean_ap']:.4f}, "
          f"max|cat-diff|={parity['max_abs_cat_ap_diff']:.2e})", flush=True)
    print(f"[k{args.shot}] parts_mismatch macro AP={pm_macro if pm_macro is not None else float('nan'):.4f} "
          f"(target {parity['target_pm_macro_ap']:.4f})", flush=True)
    for t, s in type_summary.items():
        print(f"  type {t:<24} n={s['n_total']:>4} AP={s['pixel_ap_macro']:.4f} "
              f"AUROC={s['pixel_auroc_macro']:.4f} imgAUROC={s['image_auroc_macro']:.4f}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

