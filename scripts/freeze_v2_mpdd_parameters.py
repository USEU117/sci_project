"""Freeze V2 parameters after leakage-safe MPDD development selection."""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
EVIDENCE = (
    "experiments/dynamic_fusion/v2/mpdd_router_selection/seed0/report.json",
    "experiments/dynamic_fusion/v2/mpdd_router_selection/seed12_validation/report.json",
    "experiments/dynamic_fusion/v2/mpdd_router_selection/gate_diagnostics.json",
    "experiments/dynamic_fusion/v2/mpdd_router_selection/seed0_pixel_independence_fix/report.json",
    "experiments/dynamic_fusion/v2/mpdd_router_selection/seed12_pixel_independence_fix/report.json",
    "experiments/dynamic_fusion/v2/mpdd_router_selection/final_all_seeds_apro200/report.json",
    "experiments/dynamic_fusion/v2/branch_cache_queue/runtime/completion_audit.json",
    "experiments/dynamic_fusion/v2/mpdd_prediction_gate_a/runtime/gate_a_audit.json",
    "experiments/dynamic_fusion/v2/mpdd_prediction_matrix/runtime/status.json",
    "src/industrial_ad/fusion/v2_router.py",
    "tests/test_dynamic_fusion_v2.py",
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--verify", action="store_true")
    args = parser.parse_args()
    files = []
    for relative in EVIDENCE:
        path = ROOT / relative
        if not path.is_file():
            raise SystemExit(f"freeze evidence missing: {path}")
        files.append({"relative_path": relative, "size_bytes": path.stat().st_size, "sha256": sha256(path)})
    if args.verify:
        current = json.loads(args.output.read_text(encoding="utf-8"))
        passed = current.get("evidence_files") == files and current.get("status") == "parameters_frozen"
        print(json.dumps({"status": "passed" if passed else "failed", "files": len(files)}))
        return 0 if passed else 1

    final_report = json.loads((ROOT / EVIDENCE[5]).read_text(encoding="utf-8"))
    summaries = {row["candidate"]: row for row in final_report["summaries"]}
    csv_path = ROOT / "experiments/dynamic_fusion/v2/mpdd_router_selection/final_all_seeds_apro200/report.csv"
    import csv

    rows = list(csv.DictReader(csv_path.open("r", encoding="utf-8-sig")))
    per_seed = {}
    for seed in (0, 1, 2):
        visual = [float(row["aupro"]) for row in rows if row["candidate"] == "visual_only" and int(row["seed"]) == seed]
        pixel = [float(row["aupro"]) for row in rows if row["candidate"] == "pixel_only_w15" and int(row["seed"]) == seed]
        per_seed[str(seed)] = sum(pixel) / len(pixel) - sum(visual) / len(visual)
    positive_seeds = sum(value > 0 for value in per_seed.values())
    if positive_seeds >= 2:
        raise SystemExit("repeatability condition changed; visual-only freeze requires fewer than two positive seeds")
    payload = {
        "schema_version": 1,
        "status": "parameters_frozen",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "development_dataset": "mpdd",
        "holdout_dataset": "btad",
        "selection_protocol": "three seeds and 1/2/4-shot; localization gain must be positive on at least two of three seeds with no material image degradation",
        "selected_candidate": "visual_only_safe_fallback",
        "router_parameters": {
            "support_tolerance": 3.0,
            "minimum_disagreement": 0.05,
            "uncertainty_margin": 0.05,
            "concentration_tolerance": 0.10,
            "max_image_text_weight": 0.0,
            "max_pixel_text_weight": 0.0,
            "smooth_pixel_weights": True,
        },
        "dynamic_text_assistance_enabled": False,
        "decision_reason": "pixel_only_w15 improved aggregate AUPRO but was positive on only one of three seeds; the gain did not repeat",
        "final_apro_steps": 200,
        "aggregate_delta_aupro_pixel_w15_vs_visual": summaries["pixel_only_w15"]["delta_aupro_vs_visual"],
        "per_seed_delta_aupro_pixel_w15_vs_visual": per_seed,
        "positive_seed_count": positive_seeds,
        "test_labels_used_by_router": False,
        "mpdd_development_labels_used_for_selection": True,
        "btad_predictions_used": False,
        "btad_labels_used": False,
        "btad_metrics_read": False,
        "holdout_metrics_allowed_after_this_freeze": True,
        "evidence_files": files,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"status": payload["status"], "selected": payload["selected_candidate"], "positive_seed_count": positive_seeds, "output": str(args.output)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
