"""R0b mechanism audit for the v8 TCRR region signal.

Same-image rotation and half-roll preserve each text map's histogram and global
image identity while destroying spatial correspondence. This separates genuine
region alignment from the already-known image-level AnomalyCLIP signal.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import cv2
import numpy as np
from sklearn.metrics import average_precision_score

ROOT = Path(__file__).resolve().parents[2]
for p in (ROOT, ROOT / "src", ROOT / "scripts"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from industrial_ad.innovation_v6_dgsafe import maps as a1io  # noqa: E402
from industrial_ad.innovation_v8_tcrr_probe import (  # noqa: E402
    component_features, component_masks, robust01,
)

R0_PROTOCOL = json.loads((ROOT / "configs/innovation_v8_tcrr_probe/r0_protocol.json").read_text(encoding="utf-8"))
R0B_PROTOCOL = json.loads((ROOT / "configs/innovation_v8_tcrr_probe/r0b_spatial_protocol.json").read_text(encoding="utf-8"))
DEFAULT_TEXT = ROOT / "outputs/dynamic_fusion/innovation_v8_tcrr_probe/text_maps"
DEFAULT_R0 = ROOT / "experiments/dynamic_fusion/innovation_v8_tcrr_probe/R0_region_value"
DEFAULT_OUT = ROOT / "experiments/dynamic_fusion/innovation_v8_tcrr_probe/R0b_spatial_robustness"
CATS = ["bracket_black", "bracket_brown", "bracket_white", "connector", "metal_plate", "tubes"]


def resize_text(arr):
    return np.stack([cv2.resize(m.astype(np.float32), (448, 448), interpolation=cv2.INTER_LINEAR)
                     for m in arr]).astype(np.float32)


def ap(rows, field, label_threshold=None):
    if label_threshold is None:
        y = np.asarray([int(r["label"]) for r in rows])
    else:
        y = np.asarray([float(r["overlap_fraction"]) >= label_threshold for r in rows], dtype=np.int64)
    if len(np.unique(y)) < 2:
        return None
    return float(average_precision_score(y, np.asarray([float(r[field]) for r in rows])))


def macro(rows, field, *, quantile=None, label_threshold=None, exclude=(), allow_undefined=False):
    values, per_cat = [], {}
    for cat in CATS:
        if cat in exclude:
            continue
        rr = [r for r in rows if r["category"] == cat and
              (quantile is None or abs(float(r["proposal_quantile"]) - quantile) < 1e-9)]
        value = ap(rr, field, label_threshold)
        per_cat[cat] = value
        if value is None:
            if allow_undefined:
                continue
            return None, per_cat
        values.append(value)
    return (float(np.mean(values)) if values else None), per_cat


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--text-dir", type=Path, default=DEFAULT_TEXT)
    p.add_argument("--r0-dir", type=Path, default=DEFAULT_R0)
    p.add_argument("--outdir", type=Path, default=DEFAULT_OUT)
    args = p.parse_args()
    a1io.assert_development_only()
    args.outdir.mkdir(parents=True, exist_ok=True)

    with (args.r0_dir / "region_rows.csv").open(encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f))
    keyed = {(r["category"], int(r["shot"]), r["sample_id"],
              round(float(r["proposal_quantile"]), 6), int(r["component_index"])): r for r in rows}
    if len(keyed) != len(rows):
        raise RuntimeError("R0 region keys are not unique")

    seen = set()
    for cat in CATS:
        with np.load(args.text_dir / f"{cat}.npz", allow_pickle=False) as z:
            text_ids = np.asarray(z["sample_ids"])
            text448 = resize_text(np.asarray(z["anomaly_maps"], dtype=np.float32))
        text56 = np.stack([robust01(m[::R0_PROTOCOL["stride"], ::R0_PROTOCOL["stride"]]) for m in text448])

        for shot in R0_PROTOCOL["shots"]:
            a = a1io.load_a1_patch_map(cat, 0, shot)
            perm = a1io.align_perm(text_ids, a["sample_ids"])
            txt = text56[perm]
            a1_448 = a1io.a1_maps448(a["patch_map"])
            a1_56 = np.stack([robust01(m[::R0_PROTOCOL["stride"], ::R0_PROTOCOL["stride"]]) for m in a1_448])

            for image_index, sid in enumerate(a["sample_ids"]):
                rot = np.rot90(txt[image_index], 2)
                roll = np.roll(txt[image_index], shift=(txt.shape[1] // 2, txt.shape[2] // 2), axis=(0, 1))
                for q in R0_PROTOCOL["proposal_quantiles"]:
                    comps = component_masks(a1_56[image_index], q, R0_PROTOCOL["minimum_component_cells"])
                    for ci, mask in enumerate(comps):
                        key = (cat, shot, str(sid), round(float(q), 6), ci)
                        if key not in keyed:
                            raise RuntimeError(f"cannot replay R0 proposal {key}")
                        common = dict(trim_fraction=R0_PROTOCOL["trim_fraction"],
                                      consistency_threshold=R0_PROTOCOL["text_consistency_threshold"])
                        keyed[key]["rot180_text_p90"] = component_features(mask, a1_56[image_index], rot, **common)["text_p90"]
                        keyed[key]["halfroll_text_p90"] = component_features(mask, a1_56[image_index], roll, **common)["text_p90"]
                        seen.add(key)
    if seen != set(keyed):
        raise RuntimeError(f"proposal replay incomplete: {len(seen)} vs {len(keyed)}")

    # Preserve augmented rows as an auditable artifact.
    fields = list(rows[0])
    for name in ("rot180_text_p90", "halfroll_text_p90"):
        if name not in fields:
            fields.append(name)
    with (args.outdir / "region_rows_spatial_controls.csv").open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader(); w.writerows(rows)

    original, original_cat = macro(rows, "text_p90")
    a1_all, _ = macro(rows, "a1_mean")
    rot, _ = macro(rows, "rot180_text_p90")
    roll, _ = macro(rows, "halfroll_text_p90")
    spatial_drop = original - max(rot, roll)

    q = float(R0B_PROTOCOL["single_quantile_check"])
    q_text, q_text_cat = macro(rows, "text_p90", quantile=q)
    q_a1, q_a1_cat = macro(rows, "a1_mean", quantile=q)
    q_gains = {c: q_text_cat[c] - q_a1_cat[c] for c in CATS}

    no_tubes_text, _ = macro(rows, "text_p90", exclude=("tubes",))
    no_tubes_a1, _ = macro(rows, "a1_mean", exclude=("tubes",))
    no_tubes_gain = no_tubes_text - no_tubes_a1

    sensitivity = {}
    for threshold in R0B_PROTOCOL["overlap_fraction_sensitivity"]:
        t_ap, t_cat = macro(rows, "text_p90", label_threshold=float(threshold), allow_undefined=True)
        a_ap, a_cat = macro(rows, "a1_mean", label_threshold=float(threshold), allow_undefined=True)
        evaluable = [c for c in CATS if t_cat[c] is not None and a_cat[c] is not None]
        sensitivity[str(threshold)] = {
            "text_ap": t_ap, "a1_ap": a_ap, "gain": t_ap - a_ap,
            "evaluable_categories": evaluable,
            "undefined_categories": [c for c in CATS if c not in evaluable],
        }

    g = R0B_PROTOCOL["gate"]
    checks = {
        "spatial_drop_ge_003": spatial_drop >= g["genuine_minus_best_spatial_control_ge"],
        "q095_gain_ge_005": (q_text - q_a1) >= g["q095_macro_gain_vs_a1_ge"],
        "q095_positive_categories_ge_4": sum(v > 0 for v in q_gains.values()) >= g["q095_positive_categories_ge"],
        "leave_tubes_out_gain_ge_005": no_tubes_gain >= g["leave_tubes_out_macro_gain_ge"],
        "all_overlap_sensitivity_gain_ge_003": all(
            v["gain"] >= g["each_overlap_sensitivity_gain_ge"] for v in sensitivity.values()),
        "overlap_sensitivity_coverage_ge_5": all(
            len(v["evaluable_categories"]) >= g["overlap_sensitivity_evaluable_categories_ge"]
            for v in sensitivity.values()),
    }
    passed = all(checks.values())
    report = {
        "program": R0B_PROTOCOL["program"], "phase": R0B_PROTOCOL["phase"],
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "protocol": R0B_PROTOCOL, "r0_rows_replayed": len(rows),
        "all_thresholds": {"a1_ap": a1_all, "text_ap": original,
                           "rot180_ap": rot, "halfroll_ap": roll,
                           "genuine_minus_best_spatial_control": spatial_drop},
        "q095": {"a1_ap": q_a1, "text_ap": q_text, "gain": q_text - q_a1,
                  "per_category_gain": q_gains},
        "leave_tubes_out": {"a1_ap": no_tubes_a1, "text_ap": no_tubes_text,
                            "gain": no_tubes_gain},
        "overlap_sensitivity": sensitivity,
        "gate_checks": checks, "gate_passed": passed,
        "interpretation": R0B_PROTOCOL["interpretation"],
    }
    (args.outdir / "R0B_SPATIAL_ROBUSTNESS.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    lines = [
        "# TCRR R0b spatial robustness decision", "",
        f"- genuine text P90 region AP: {original:.4f}",
        f"- rotate-180 same-image control: {rot:.4f}",
        f"- half-roll same-image control: {roll:.4f}",
        f"- genuine minus best spatial control: {spatial_drop:+.4f}",
        f"- q=0.95 only gain vs A1: {q_text-q_a1:+.4f} ({sum(v>0 for v in q_gains.values())}/6 cats positive)",
        f"- leave-tubes-out gain: {no_tubes_gain:+.4f}",
        "- overlap sensitivity gains: " + ", ".join(
            f"{k}={v['gain']:+.4f} ({len(v['evaluable_categories'])}/6 evaluable)"
            for k, v in sensitivity.items()),
        f"- gate: {'PASS — minimal R1 authorized' if passed else 'FAIL — archive spatial region route'}", "",
        *[f"- {k}: {v}" for k, v in checks.items()],
    ]
    (args.outdir / "R0B_DECISION.md").write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines))
    return 0 if passed else 2


if __name__ == "__main__":
    raise SystemExit(main())
