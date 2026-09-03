"""Aggregate frozen R1 reranker results for pre-registered seed1/2 R2."""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
PROTOCOL = ROOT / "configs/innovation_v8_tcrr_probe/r2_confirmation_protocol.json"
DEFAULT_INPUT = ROOT / "experiments/dynamic_fusion/innovation_v8_tcrr_probe"
DEFAULT_OUT = DEFAULT_INPUT / "R2_seed_confirmation"
CATS = ["bracket_black", "bracket_brown", "bracket_white", "connector", "metal_plate", "tubes"]


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--input-root", type=Path, default=DEFAULT_INPUT)
    p.add_argument("--outdir", type=Path, default=DEFAULT_OUT)
    args = p.parse_args()
    protocol = json.loads(PROTOCOL.read_text(encoding="utf-8"))
    rows = []
    source_files = []
    for seed in protocol["seeds"]:
        path = args.input_root / f"R2_seed{seed}" / "R1_RESULT.json"
        data = json.loads(path.read_text(encoding="utf-8"))
        if data.get("evaluated_seed") != seed:
            raise RuntimeError(f"seed provenance mismatch in {path}")
        rows.extend(data["rows"])
        source_files.append(str(path.relative_to(ROOT)))
    expected = len(CATS) * len(protocol["seeds"]) * len(protocol["shots"])
    if len(rows) != expected:
        raise RuntimeError(f"expected {expected} rows, got {len(rows)}")

    def mean(field: str, subset=rows) -> float:
        return float(np.mean([r[field] for r in subset]))

    gain = mean("tcrr_delta_pixel_ap")
    auc_gain = mean("tcrr_delta_pixel_auroc")
    controls = {n: mean(f"{n}_delta_pixel_ap") for n in ("rotate180", "halfroll")}
    separation = gain - max(controls.values())
    seed_shot = {f"s{s}_k{k}": mean("tcrr_delta_pixel_ap", [r for r in rows if r["seed"] == s and r["shot"] == k])
                 for s in protocol["seeds"] for k in protocol["shots"]}
    category = {c: mean("tcrr_delta_pixel_ap", [r for r in rows if r["category"] == c]) for c in CATS}

    rng = np.random.default_rng(protocol["bootstrap"]["seed"])
    by_cat = {c: np.asarray([r["tcrr_delta_pixel_ap"] for r in rows if r["category"] == c]) for c in CATS}
    boot = np.empty(protocol["bootstrap"]["resamples"], dtype=np.float64)
    for i in range(len(boot)):
        sampled = rng.choice(CATS, size=len(CATS), replace=True)
        boot[i] = np.mean(np.concatenate([by_cat[c] for c in sampled]))
    ci = [float(x) for x in np.quantile(boot, [0.025, 0.975])]

    g = protocol["gate"]
    checks = {
        "macro_pixel_ap_gain_ge_0015": gain >= g["macro_pixel_ap_gain_ge"],
        "positive_seed_shots_ge_5": sum(v > 0 for v in seed_shot.values()) >= g["positive_seed_shot_count_ge"],
        "positive_category_configs_ge_24": sum(r["tcrr_delta_pixel_ap"] > 0 for r in rows) >= g["positive_category_seed_shot_count_ge"],
        "worst_category_gain_ge_minus002": min(category.values()) >= g["worst_category_macro_pixel_ap_gain_ge"],
        "macro_pixel_auroc_gain_ge_0": auc_gain >= g["macro_pixel_auroc_gain_ge"],
        "spatial_control_separation_ge_001": separation >= g["genuine_gain_minus_best_control_gain_ge"],
        "cluster_bootstrap_ci_lower_gt_0": ci[0] > g["cluster_bootstrap_95ci_lower_gt"],
    }
    passed = all(checks.values())
    report = {"program": protocol["program"], "phase": protocol["phase"],
              "created_at_utc": datetime.now(timezone.utc).isoformat(), "protocol": protocol,
              "sources": source_files, "rows": rows,
              "summary": {"macro_pixel_ap_gain": gain, "macro_pixel_auroc_gain": auc_gain,
                          "seed_shot_pixel_ap_gain": seed_shot, "category_pixel_ap_gain": category,
                          "control_pixel_ap_gain": controls, "genuine_minus_best_control_gain": separation,
                          "positive_category_configs": sum(r["tcrr_delta_pixel_ap"] > 0 for r in rows),
                          "cluster_bootstrap_95ci": ci},
              "gate_checks": checks, "gate_passed": passed}
    args.outdir.mkdir(parents=True, exist_ok=True)
    (args.outdir / "R2_RESULT.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    lines = ["# TCRR R2 seed1/2 confirmation", "", f"- macro Pixel-AP gain: {gain:+.6f}",
             f"- macro Pixel-AUROC gain: {auc_gain:+.6f}", f"- seed-shot gains: {seed_shot}",
             f"- category gains: {category}", f"- spatial-control separation: {separation:+.6f}",
             f"- category-cluster bootstrap 95% CI: [{ci[0]:+.6f}, {ci[1]:+.6f}]",
             f"- positive category-configs: {report['summary']['positive_category_configs']}/{expected}",
             f"- gate: {'PASS — candidate contribution' if passed else 'FAIL — archive/redesign'}", "",
             *[f"- {k}: {v}" for k, v in checks.items()]]
    (args.outdir / "R2_DECISION.md").write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines))
    return 0 if passed else 2


if __name__ == "__main__":
    raise SystemExit(main())
