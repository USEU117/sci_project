"""Frozen TCRR evaluation on one external dataset; no post-result tuning."""

from __future__ import annotations

import argparse
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

from industrial_ad.innovation_v6_dgsafe.maps import a1_maps448, align_perm  # noqa: E402
from industrial_ad.innovation_v8_tcrr_probe import region_rerank_map, robust01  # noqa: E402

PROTOCOL_PATH = ROOT / "configs/innovation_v8_tcrr_probe/r3_external_protocol.json"


def metrics(scores, labels):
    y = np.asarray(labels).reshape(-1).astype(np.int64)
    s = np.asarray(scores).reshape(-1).astype(np.float64)
    return {"pixel_ap": float(average_precision_score(y, s)),
            "pixel_auroc": float(roc_auc_score(y, s))}


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", required=True, choices=["btad", "mvtec"])
    p.add_argument("--text-dir", type=Path, required=True)
    p.add_argument("--outdir", type=Path, required=True)
    args = p.parse_args()
    protocol = json.loads(PROTOCOL_PATH.read_text(encoding="utf-8"))
    if args.dataset not in protocol["datasets"]:
        raise SystemExit("dataset not pre-registered")
    compact = ROOT / "submission_repro_20260827/predictions_compact/maps" / args.dataset
    categories = sorted(p.stem for p in (compact / "s0_k1").glob("*.npz"))
    args.outdir.mkdir(parents=True, exist_ok=True)
    rows = []

    for cat in categories:
        # Load prediction-only text artifact.  Ground truth stays unopened here.
        with np.load(args.text_dir / f"{cat}.npz", allow_pickle=False) as z:
            text_ids = np.asarray(z["sample_ids"])
            text_maps = np.asarray(z["anomaly_maps"], dtype=np.float32)
        text56_all = np.stack([robust01(cv2.resize(m, (448, 448), interpolation=cv2.INTER_LINEAR)[::8, ::8])
                               for m in text_maps])

        # Freeze all nine configurations for this category before evaluator GT.
        frozen = []
        for seed in protocol["seeds"]:
            for shot in protocol["shots"]:
                path = compact / f"s{seed}_k{shot}" / f"{cat}.npz"
                with np.load(path, allow_pickle=False) as z:
                    ids = np.asarray(z["sample_ids"])
                    patch = np.asarray(z["concat_patch_map"], dtype=np.float32)
                perm = align_perm(text_ids, ids)
                txt = text56_all[perm]
                raw = a1_maps448(patch)[:, ::8, ::8].astype(np.float32)
                prop = np.stack([robust01(m) for m in raw])
                made = {"tcrr": [], "rotate180": [], "halfroll": []}
                for i in range(len(raw)):
                    variants = {"tcrr": txt[i], "rotate180": np.rot90(txt[i], 2),
                                "halfroll": np.roll(txt[i], (28, 28), axis=(0, 1))}
                    for name, tmap in variants.items():
                        out, _ = region_rerank_map(raw[i], prop[i], tmap,
                                                   quantile=0.95, min_cells=4, max_factor=1.5)
                        made[name].append(out)
                frozen.append({"seed": seed, "shot": shot, "sample_ids": ids, "a1": raw,
                               **{k: np.stack(v) for k, v in made.items()}})

        # Evaluator-only target opened after every map for this category is frozen.
        with np.load(args.text_dir / f"{cat}_targets.npz", allow_pickle=False) as z:
            target_ids = np.asarray(z["sample_ids"])
            target_masks = np.asarray(z["imgs_masks"], dtype=np.uint8)
        for f in frozen:
            perm = align_perm(target_ids, f["sample_ids"])
            # Match the frozen evaluator exactly: resize target to 448 first,
            # then sample both spatial axes at stride 8.  Direct 518->56 is
            # close but not geometrically identical at the far boundary.
            gt56 = np.stack([cv2.resize(m, (448, 448), interpolation=cv2.INTER_NEAREST)[::8, ::8]
                             for m in target_masks[perm]]) > 0
            base = metrics(f["a1"], gt56)
            row = {"dataset": args.dataset, "category": cat, "seed": f["seed"], "shot": f["shot"],
                   **{f"a1_{k}": v for k, v in base.items()}}
            for name in ("tcrr", "rotate180", "halfroll"):
                met = metrics(f[name], gt56)
                for key, value in met.items():
                    row[f"{name}_{key}"] = value
                    row[f"{name}_delta_{key}"] = value - base[key]
            rows.append(row)
        print(f"evaluated {args.dataset}/{cat}: {len(frozen)} configs", flush=True)

    def mean(field, subset=rows):
        return float(np.mean([r[field] for r in subset]))
    gain = mean("tcrr_delta_pixel_ap")
    auc_gain = mean("tcrr_delta_pixel_auroc")
    controls = {n: mean(f"{n}_delta_pixel_ap") for n in ("rotate180", "halfroll")}
    separation = gain - max(controls.values())
    seed_shot = {f"s{s}_k{k}": mean("tcrr_delta_pixel_ap", [r for r in rows if r["seed"] == s and r["shot"] == k])
                 for s in protocol["seeds"] for k in protocol["shots"]}
    cat_gain = {c: mean("tcrr_delta_pixel_ap", [r for r in rows if r["category"] == c]) for c in categories}
    g = protocol["gate_per_dataset"]
    checks = {"macro_pixel_ap_gain_gt_0": gain > g["macro_pixel_ap_gain_gt"],
              "positive_seed_shots_ge_4": sum(v > 0 for v in seed_shot.values()) >= g["positive_seed_shot_count_ge"],
              "worst_category_gain_ge_minus003": min(cat_gain.values()) >= g["worst_category_macro_pixel_ap_gain_ge"],
              "macro_pixel_auroc_loss_ge_minus0003": auc_gain >= g["macro_pixel_auroc_loss_ge"],
              "spatial_control_separation_ge_0003": separation >= g["genuine_gain_minus_best_spatial_control_gain_ge"]}
    passed = all(checks.values())
    report = {"program": protocol["program"], "phase": protocol["phase"], "dataset": args.dataset,
              "created_at_utc": datetime.now(timezone.utc).isoformat(), "protocol": protocol,
              "method_maps_frozen_before_category_gt_load": True, "rows": rows,
              "summary": {"macro_pixel_ap_gain": gain, "macro_pixel_auroc_gain": auc_gain,
                          "seed_shot_pixel_ap_gain": seed_shot, "category_pixel_ap_gain": cat_gain,
                          "control_pixel_ap_gain": controls, "genuine_minus_best_control_gain": separation},
              "gate_checks": checks, "gate_passed": passed}
    (args.outdir / "R3_RESULT.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    lines = [f"# TCRR R3 {args.dataset.upper()} external decision", "", f"- macro Pixel-AP gain: {gain:+.6f}",
             f"- macro Pixel-AUROC gain: {auc_gain:+.6f}", f"- seed-shot gains: {seed_shot}",
             f"- category gains: {cat_gain}", f"- spatial-control separation: {separation:+.6f}",
             f"- gate: {'PASS' if passed else 'FAIL'}", "", *[f"- {k}: {v}" for k, v in checks.items()]]
    (args.outdir / "R3_DECISION.md").write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines))
    return 0 if passed else 2


if __name__ == "__main__":
    raise SystemExit(main())
