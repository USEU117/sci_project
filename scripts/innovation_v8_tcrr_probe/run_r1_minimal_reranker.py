"""R1 pre-registered minimal TCRR map reranker on MPDD development only."""

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

from industrial_ad.innovation_v6_dgsafe import maps as a1io  # noqa: E402
from industrial_ad.innovation_v8_tcrr_probe import region_rerank_map, robust01  # noqa: E402

PROTOCOL_PATH = ROOT / "configs/innovation_v8_tcrr_probe/r1_protocol.json"
DEFAULT_TEXT = ROOT / "outputs/dynamic_fusion/innovation_v8_tcrr_probe/text_maps"
DEFAULT_OUT = ROOT / "experiments/dynamic_fusion/innovation_v8_tcrr_probe/R1_minimal_reranker"
CATS = ["bracket_black", "bracket_brown", "bracket_white", "connector", "metal_plate", "tubes"]


def resize_text(arr: np.ndarray) -> np.ndarray:
    return np.stack([cv2.resize(m.astype(np.float32), (448, 448), interpolation=cv2.INTER_LINEAR)
                     for m in arr]).astype(np.float32)


def metrics(scores: np.ndarray, labels: np.ndarray) -> dict[str, float]:
    y = labels.reshape(-1).astype(np.int64)
    s = scores.reshape(-1).astype(np.float64)
    return {"pixel_ap": float(average_precision_score(y, s)),
            "pixel_auroc": float(roc_auc_score(y, s))}


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--text-dir", type=Path, default=DEFAULT_TEXT)
    p.add_argument("--outdir", type=Path, default=DEFAULT_OUT)
    p.add_argument("--seed", type=int, default=None, choices=[0, 1, 2],
                   help="Optional confirmation seed; formula/protocol remain unchanged")
    args = p.parse_args()
    protocol = json.loads(PROTOCOL_PATH.read_text(encoding="utf-8"))
    seed = protocol["seed"] if args.seed is None else args.seed
    a1io.assert_development_only()
    args.outdir.mkdir(parents=True, exist_ok=True)

    # Method stage: generate and freeze all maps without loading GT.
    frozen: dict[tuple[str, int], dict] = {}
    region_count = 0
    for cat in CATS:
        with np.load(args.text_dir / f"{cat}.npz", allow_pickle=False) as z:
            text_ids = np.asarray(z["sample_ids"])
            text448 = resize_text(np.asarray(z["anomaly_maps"], dtype=np.float32))
        text56_all = np.stack([robust01(m[::8, ::8]) for m in text448])
        for shot in protocol["shots"]:
            a = a1io.load_a1_patch_map(cat, seed, shot)
            perm = a1io.align_perm(text_ids, a["sample_ids"])
            text56 = text56_all[perm]
            raw56 = a1io.a1_maps448(a["patch_map"])[:, ::8, ::8].astype(np.float32)
            prop56 = np.stack([robust01(m) for m in raw56])
            methods = {"tcrr": [], "rotate180": [], "halfroll": []}
            audits = []
            for i in range(len(raw56)):
                variants = {
                    "tcrr": text56[i],
                    "rotate180": np.rot90(text56[i], 2),
                    "halfroll": np.roll(text56[i], shift=(28, 28), axis=(0, 1)),
                }
                for name, txt in variants.items():
                    out, audit = region_rerank_map(
                        raw56[i], prop56[i], txt,
                        quantile=protocol["proposal_quantile"],
                        min_cells=protocol["minimum_component_cells"],
                        max_factor=protocol["factor_range"][1],
                    )
                    methods[name].append(out)
                    if name == "tcrr":
                        audits.extend(audit)
            region_count += len(audits)
            frozen[(cat, shot)] = {
                "sample_ids": np.asarray(a["sample_ids"]), "a1": raw56,
                **{k: np.stack(v) for k, v in methods.items()},
                "factor_min": min((x["factor"] for x in audits), default=None),
                "factor_max": max((x["factor"] for x in audits), default=None),
                "regions": len(audits),
            }

    # Evaluator-only stage begins only after every method/control map is frozen.
    rows = []
    for cat in CATS:
        for shot in protocol["shots"]:
            f = frozen[(cat, shot)]
            gt448 = a1io.gt_masks_for(f["sample_ids"])
            gt56 = gt448[:, ::8, ::8]
            base = metrics(f["a1"], gt56)
            row = {"category": cat, "seed": seed, "shot": shot,
                   "regions": f["regions"], "factor_min": f["factor_min"],
                   "factor_max": f["factor_max"], **{f"a1_{k}": v for k, v in base.items()}}
            for name in ("tcrr", "rotate180", "halfroll"):
                met = metrics(f[name], gt56)
                for key, value in met.items():
                    row[f"{name}_{key}"] = value
                    row[f"{name}_delta_{key}"] = value - base[key]
            rows.append(row)

    def mean(field: str, subset=rows) -> float:
        return float(np.mean([r[field] for r in subset]))

    cat_gain = {c: mean("tcrr_delta_pixel_ap", [r for r in rows if r["category"] == c]) for c in CATS}
    shot_gain = {str(s): mean("tcrr_delta_pixel_ap", [r for r in rows if r["shot"] == s])
                 for s in protocol["shots"]}
    gain = mean("tcrr_delta_pixel_ap")
    auc_gain = mean("tcrr_delta_pixel_auroc")
    control_gains = {name: mean(f"{name}_delta_pixel_ap") for name in ("rotate180", "halfroll")}
    separation = gain - max(control_gains.values())
    g = protocol["gate"]
    checks = {
        "macro_pixel_ap_gain_ge_0005": gain >= g["macro_pixel_ap_gain_ge"],
        "all_3_shots_positive": sum(v > 0 for v in shot_gain.values()) >= g["positive_shot_count_ge"],
        "positive_category_shots_ge_11": sum(r["tcrr_delta_pixel_ap"] > 0 for r in rows) >= g["positive_category_shot_count_ge"],
        "worst_category_gain_ge_minus002": min(cat_gain.values()) >= g["worst_category_macro_pixel_ap_gain_ge"],
        "macro_pixel_auroc_loss_ge_minus0002": auc_gain >= g["macro_pixel_auroc_loss_ge"],
        "spatial_control_separation_ge_0003": separation >= g["genuine_gain_minus_best_control_gain_ge"],
    }
    passed = all(checks.values())
    report = {
        "program": protocol["program"], "phase": protocol["phase"],
        "created_at_utc": datetime.now(timezone.utc).isoformat(), "protocol": protocol,
        "evaluated_seed": seed,
        "method_stage_completed_before_gt_load": True, "regions": region_count,
        "rows": rows, "summary": {
            "macro_pixel_ap_gain": gain, "macro_pixel_auroc_gain": auc_gain,
            "per_shot_pixel_ap_gain": shot_gain, "per_category_pixel_ap_gain": cat_gain,
            "control_pixel_ap_gains": control_gains,
            "genuine_minus_best_control_gain": separation,
            "positive_category_shots": sum(r["tcrr_delta_pixel_ap"] > 0 for r in rows),
        }, "gate_checks": checks, "gate_passed": passed,
    }
    (args.outdir / "R1_RESULT.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    lines = ["# TCRR R1 minimal reranker decision", "",
             f"- macro Pixel-AP gain: {gain:+.6f}", f"- macro Pixel-AUROC gain: {auc_gain:+.6f}",
             f"- per-shot Pixel-AP gain: {shot_gain}", f"- per-category Pixel-AP gain: {cat_gain}",
             f"- control gains: {control_gains}; genuine separation {separation:+.6f}",
             f"- positive category-shots: {report['summary']['positive_category_shots']}/18",
             f"- gate: {'PASS — seed1/2 confirmation authorized' if passed else 'FAIL — do not scale'}", "",
             *[f"- {k}: {v}" for k, v in checks.items()]]
    (args.outdir / "R1_DECISION.md").write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines))
    return 0 if passed else 2


if __name__ == "__main__":
    raise SystemExit(main())
