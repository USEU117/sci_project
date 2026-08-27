"""P1 acceptance check: verifies all four P1 deliverables are present and valid.

Checks (evidence under submission_repro_20260827/evidence/p1/):
  P1-A bootstrap CI JSON (36 configs, sanity within 5e-4, shot-wise mean+/-std)
  P1-B failure boundaries MD + failure samples CSV
  P1-C efficiency JSON/CSV/MD
  P1-D fairness table JSON/CSV/MD

Writes p1_acceptance.json and exits 0 iff everything passes.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
P1 = ROOT / "submission_repro_20260827" / "evidence" / "p1"
EXPECTED_CONFIGS = 36
SANITY_TOLERANCE = 5e-4


def main() -> int:
    checks = {}
    p1a = P1 / "p1_a_bootstrap_ci.json"
    checks["p1_a_bootstrap_ci_json_exists"] = p1a.is_file()
    if p1a.is_file():
        payload = json.loads(p1a.read_text(encoding="utf-8"))
        checks["p1_a_n_configs"] = len(payload["configs"]) == EXPECTED_CONFIGS
        checks["p1_a_sanity_all_within_tolerance"] = bool(payload["sanity_all_within_tolerance"])
        checks["p1_a_bootstrap_B"] = payload["bootstrap"]["B"] >= 1000
        checks["p1_a_shot_wise_present"] = len(payload["shot_wise"]) == 12
        checks["p1_a_dataset_wise_present"] = len(payload["dataset_wise"]) == 4
        checks["p1_a_each_config_has_both_cis"] = all(
            r["category_bootstrap"].get("lo") is not None and r["image_bootstrap"].get("lo") is not None
            for r in payload["configs"])

    checks["p1_b_failure_boundaries_md"] = (P1 / "p1_b_failure_boundaries.md").is_file()
    checks["p1_b_failure_samples_csv"] = (P1 / "p1_b_failure_samples.csv").is_file()
    checks["p1_c_efficiency_json"] = (P1 / "p1_c_efficiency.json").is_file()
    checks["p1_c_efficiency_csv"] = (P1 / "p1_c_efficiency.csv").is_file()
    if checks["p1_c_efficiency_json"]:
        eff = json.loads((P1 / "p1_c_efficiency.json").read_text(encoding="utf-8"))
        category_counts = {"mpdd": 6, "btad": 3, "visa": 12, "mvtec": 15}
        checks["p1_c_bank_counts_match_grids"] = all(
            r["reference_images"] == category_counts[r["dataset"]] * r["shot"]
            and r["dino_bank_patches"] > r["reference_images"]
            and r["clip_bank_patches"] > r["reference_images"]
            and r["concat_bank_patches"] == r["dino_bank_patches"]
            and abs(r["dino_bank_mb_f32"] - round(r["dino_bank_patches"] * 768 * 4 / 1e6, 2)) < 1e-9
            and abs(r["clip_bank_mb_f32"] - round(r["clip_bank_patches"] * 768 * 4 / 1e6, 2)) < 1e-9
            and abs(r["concat_bank_mb_f32"] - round(r["concat_bank_patches"] * 1536 * 4 / 1e6, 2)) < 1e-9
            for r in eff["memory_bank_float32"]
        )
        checks["p1_c_peak_ram_present"] = eff.get("peak_ram_mb") is not None
        checks["p1_c_steady_state_benchmark_present"] = eff.get("steady_state_end_to_end_benchmark") is not None
    checks["p1_d_fairness_table_json"] = (P1 / "p1_d_fairness_table.json").is_file()
    checks["p1_d_fairness_table_csv"] = (P1 / "p1_d_fairness_table.csv").is_file()
    p1e = P1 / "p1_e_complete_metrics.json"
    checks["p1_e_complete_metrics_json"] = p1e.is_file()
    if p1e.is_file():
        complete = json.loads(p1e.read_text(encoding="utf-8"))
        checks["p1_e_36_reports_six_metrics"] = bool(
            complete.get("checks", {}).get("all_passed")
            and complete.get("checks", {}).get("reports") == 36
            and complete.get("checks", {}).get("config_method_rows") == 72
        )

    # cross-check P1-A dataset means against the paper table (main_results.json)
    paper = json.loads((ROOT / "submission_repro_20260827" / "evidence" / "paper_tables" /
                        "main_results.json").read_text(encoding="utf-8"))
    paper_delta = {}
    for row in paper["rows"]:
        # mpdd/btad have an explicit "minus matched feature-DINO-only" row; visa/mvtec
        # report the same matched-delta inside the "A1 concat (frozen w=0.5)" row.
        if row["method"].startswith("A1 concat minus feature-DINO-only"):
            paper_delta[row["dataset"]] = float(row["mean_delta_ap"])
    for row in paper["rows"]:
        if row["dataset"] not in paper_delta and row["method"].startswith("A1 concat (frozen"):
            paper_delta[row["dataset"]] = float(row["mean_delta_ap"])
    if p1a.is_file():
        payload = json.loads(p1a.read_text(encoding="utf-8"))
        for d in payload["dataset_wise"]:
            ref = paper_delta.get(d["dataset"])
            checks[f"p1_a_match_paper_{d['dataset']}"] = ref is not None and abs(d["mean_delta_ap"] - ref) <= SANITY_TOLERANCE

    passed = all(v is True for v in checks.values())
    report = {
        "schema_version": 1,
        "kind": "p1_acceptance",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "deliverables": {
            "P1-A statistics (bootstrap CI + shot-wise)": p1a.is_file(),
            "P1-B failure boundaries + failure samples": (P1 / "p1_b_failure_samples.csv").is_file(),
            "P1-C efficiency table": bool(
                checks.get("p1_c_efficiency_json")
                and checks.get("p1_c_bank_counts_match_grids")
                and checks.get("p1_c_peak_ram_present")
                and checks.get("p1_c_steady_state_benchmark_present")
            ),
            "P1-D fairness table": (P1 / "p1_d_fairness_table.json").is_file(),
            "P1-E complete image/pixel metric tables": bool(
                checks.get("p1_e_complete_metrics_json") and checks.get("p1_e_36_reports_six_metrics")
            ),
        },
        "checks": checks,
        "p1_complete": passed,
    }
    (P1 / "p1_acceptance.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"status": "passed" if passed else "failed", "p1_complete": passed,
                      "checks": checks}))
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
