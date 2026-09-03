"""R0: can explicit text maps re-rank A1 candidate regions?

This is an information-value diagnostic, not a deployable method. Candidate
regions and their A1/text scores are frozen before evaluator-only GT loading.
Only MPDD development seed0, shots 1/2/4 are permitted.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import cv2
import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score

ROOT = Path(__file__).resolve().parents[2]
for p in (ROOT, ROOT / "src", ROOT / "scripts"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from industrial_ad.innovation_v6_dgsafe import maps as a1io  # noqa: E402
from industrial_ad.innovation_v8_tcrr_probe import (  # noqa: E402
    component_features,
    component_masks,
    proposal_label,
    robust01,
)

PROTOCOL_PATH = ROOT / "configs" / "innovation_v8_tcrr_probe" / "r0_protocol.json"
DEFAULT_TEXT = ROOT / "outputs" / "dynamic_fusion" / "innovation_v8_tcrr_probe" / "text_maps"
DEFAULT_OUT = ROOT / "experiments" / "dynamic_fusion" / "innovation_v8_tcrr_probe" / "R0_region_value"


def stable_cat_seed(category: str, base: int) -> int:
    return base + int(hashlib.sha256(category.encode("utf-8")).hexdigest()[:8], 16)


def resize_text(maps: np.ndarray) -> np.ndarray:
    return np.stack([
        cv2.resize(m.astype(np.float32), (448, 448), interpolation=cv2.INTER_LINEAR)
        for m in maps
    ]).astype(np.float32)


def safe_metrics(labels, scores):
    y = np.asarray(labels, dtype=np.int64)
    s = np.asarray(scores, dtype=np.float64)
    if len(y) == 0 or len(np.unique(y)) < 2 or not np.isfinite(s).all():
        return {"ap": None, "auroc": None, "n": int(len(y)), "n_pos": int(y.sum())}
    return {
        "ap": float(average_precision_score(y, s)),
        "auroc": float(roc_auc_score(y, s)),
        "n": int(len(y)),
        "n_pos": int(y.sum()),
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--text-dir", type=Path, default=DEFAULT_TEXT)
    ap.add_argument("--outdir", type=Path, default=DEFAULT_OUT)
    args = ap.parse_args()

    a1io.assert_development_only()
    protocol = json.loads(PROTOCOL_PATH.read_text(encoding="utf-8"))
    args.outdir.mkdir(parents=True, exist_ok=True)
    cats = list(protocol["categories"]) if "categories" in protocol else [
        "bracket_black", "bracket_brown", "bracket_white",
        "connector", "metal_plate", "tubes",
    ]

    # Label-free stage: freeze proposal masks and both genuine/shuffled scores.
    frozen = []
    frozen_masks = []
    frozen_ids = []
    for cat in cats:
        text_path = args.text_dir / f"{cat}.npz"
        if not text_path.exists():
            raise FileNotFoundError(text_path)
        with np.load(text_path, allow_pickle=False) as z:
            text_ids = np.asarray(z["sample_ids"])
            text448 = resize_text(np.asarray(z["anomaly_maps"], dtype=np.float32))
        text56 = np.stack([robust01(m[:: protocol["stride"], :: protocol["stride"]])
                           for m in text448])

        rng = np.random.default_rng(stable_cat_seed(cat, protocol["shuffle_seed"]))
        shuffled_indices = rng.permutation(len(text_ids))

        for shot in protocol["shots"]:
            a = a1io.load_a1_patch_map(cat, 0, shot)
            perm = a1io.align_perm(text_ids, a["sample_ids"])
            a1_448 = a1io.a1_maps448(a["patch_map"])
            a1_56 = np.stack([robust01(m[:: protocol["stride"], :: protocol["stride"]])
                              for m in a1_448])
            txt = text56[perm]
            # Shuffle only after text maps have been aligned to A1 order.
            txt_shuf = txt[shuffled_indices]

            for image_index, sid in enumerate(a["sample_ids"]):
                for q in protocol["proposal_quantiles"]:
                    masks = component_masks(a1_56[image_index], q,
                                            protocol["minimum_component_cells"])
                    for component_index, mask in enumerate(masks):
                        feat = component_features(
                            mask, a1_56[image_index], txt[image_index],
                            trim_fraction=protocol["trim_fraction"],
                            consistency_threshold=protocol["text_consistency_threshold"],
                        )
                        shuf = component_features(
                            mask, a1_56[image_index], txt_shuf[image_index],
                            trim_fraction=protocol["trim_fraction"],
                            consistency_threshold=protocol["text_consistency_threshold"],
                        )
                        rec = {
                            "category": cat, "seed": 0, "shot": shot,
                            "sample_id": str(sid), "image_index": image_index,
                            "proposal_quantile": q, "component_index": component_index,
                            **feat,
                            "shuf_text_trimmed_mean": shuf["text_trimmed_mean"],
                            "shuf_text_p90": shuf["text_p90"],
                            "shuf_text_consistency": shuf["text_consistency"],
                        }
                        frozen.append(rec)
                        frozen_masks.append(mask)
                        frozen_ids.append(str(sid))

    # Evaluator-only stage: load GT after every proposal and score is frozen.
    gt_cache = {}
    for rec, mask, sid in zip(frozen, frozen_masks, frozen_ids):
        if sid not in gt_cache:
            gt448 = a1io.gt_masks_for([sid])[0]
            gt_cache[sid] = gt448[:: protocol["stride"], :: protocol["stride"]]
        rec.update(proposal_label(mask, gt_cache[sid], protocol["positive_overlap_fraction"]))

    fields = list(frozen[0]) if frozen else []
    with (args.outdir / "region_rows.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(frozen)

    poolings = protocol["text_poolings"]
    candidates = ["a1_mean", *poolings]
    per_category = {}
    for cat in cats:
        rows = [r for r in frozen if r["category"] == cat]
        y = [r["label"] for r in rows]
        per_category[cat] = {name: safe_metrics(y, [r[name] for r in rows]) for name in candidates}
        for name in poolings:
            per_category[cat][f"shuf_{name}"] = safe_metrics(
                y, [r[f"shuf_{name}"] for r in rows])

    macro = {}
    for name in candidates:
        vals = [per_category[c][name]["ap"] for c in cats]
        macro[name] = float(np.mean(vals)) if all(v is not None for v in vals) else None
    best = max(poolings, key=lambda n: macro[n] if macro[n] is not None else -np.inf)
    base = macro["a1_mean"]
    gain = macro[best] - base
    shuf_macro = float(np.mean([per_category[c][f"shuf_{best}"]["ap"] for c in cats]))
    shuffle_drop = macro[best] - shuf_macro
    cat_gains = {c: per_category[c][best]["ap"] - per_category[c]["a1_mean"]["ap"] for c in cats}
    positive = {c: v for c, v in cat_gains.items() if v > 0}
    positive_sum = sum(positive.values())
    top_share = max(positive.values()) / positive_sum if positive_sum > 0 else 1.0
    n_pos = int(sum(r["label"] for r in frozen))
    n_neg = len(frozen) - n_pos
    g = protocol["gate"]
    checks = {
        "gain_ge_005": gain >= g["macro_region_ap_gain_ge"],
        "positive_categories_ge_4": len(positive) >= g["positive_categories_ge"],
        "shuffle_drop_ge_003": shuffle_drop >= g["shuffle_drop_ge"],
        "top_share_le_050": top_share <= g["top_category_positive_gain_share_le"],
        "positive_regions_ge_100": n_pos >= g["positive_regions_ge"],
        "negative_regions_ge_100": n_neg >= g["negative_regions_ge"],
    }
    passed = all(checks.values())

    report = {
        "program": protocol["program"], "phase": protocol["phase"],
        "dataset": "mpdd", "role": "development",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "protocol": protocol,
        "counts": {"regions": len(frozen), "positive": n_pos, "negative": n_neg,
                   "unique_images": len(gt_cache)},
        "macro_region_ap": macro,
        "selected_pre_registered_text_pooling": best,
        "a1_macro_ap": base, "text_macro_ap": macro[best],
        "macro_gain": gain, "shuffled_text_macro_ap": shuf_macro,
        "shuffle_drop": shuffle_drop,
        "per_category_gain": cat_gains,
        "positive_categories": list(positive),
        "top_category_positive_gain_share": top_share,
        "per_category": per_category,
        "gate_checks": checks, "gate_passed": passed,
        "leakage": {"gt_loaded_after_proposals_and_scores_frozen": True,
                    "external_datasets_used": False,
                    "test_labels_used_to_choose_proposal_geometry": False},
    }
    (args.outdir / "R0_REGION_VALUE.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    lines = [
        "# TCRR R0 region information-value decision",
        "",
        f"- regions: {len(frozen)} (positive {n_pos}, negative {n_neg})",
        f"- A1 region AP: {base:.4f}",
        f"- best text pooling: {best}, AP {macro[best]:.4f}, gain {gain:+.4f}",
        f"- shuffled text AP: {shuf_macro:.4f}, genuine-minus-shuffle {shuffle_drop:+.4f}",
        f"- positive categories: {len(positive)}/6; top positive gain share {top_share:.3f}",
        f"- gate: {'PASS' if passed else 'FAIL / ARCHIVE'}",
        "",
        "Gate checks:",
        *[f"- {k}: {v}" for k, v in checks.items()],
        "",
        "This R0 result is diagnostic only. A PASS authorizes a minimal region reranker; "
        "it is not itself a paper contribution.",
    ]
    (args.outdir / "R0_DECISION.md").write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines), flush=True)
    return 0 if passed else 2


if __name__ == "__main__":
    raise SystemExit(main())
