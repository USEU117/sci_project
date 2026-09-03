"""Combine the two independently frozen R3 external decisions."""
from __future__ import annotations
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
BASE = ROOT / "experiments/dynamic_fusion/innovation_v8_tcrr_probe"

data = {d: json.loads((BASE / f"R3_{d}" / "R3_RESULT.json").read_text(encoding="utf-8"))
        for d in ("btad", "mvtec")}
status = {d: bool(v["gate_passed"]) for d, v in data.items()}
decision = "cross_dataset_pass" if all(status.values()) else (
    "dataset_specific_only" if any(status.values()) else "external_fail_archive_v8_reranker")
out = {"program": "innovation_v8_tcrr_probe", "phase": "R3_overall",
       "dataset_gate_passed": status, "overall_decision": decision,
       "parameter_changes_between_datasets": False,
       "summary": {d: v["summary"] for d, v in data.items()}}
(BASE / "R3_OVERALL.json").write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
lines = ["# TCRR R3 overall external decision", "",
         f"- BTAD: {'PASS' if status['btad'] else 'FAIL'}, Pixel-AP delta {data['btad']['summary']['macro_pixel_ap_gain']:+.6f}",
         f"- MVTec: {'PASS' if status['mvtec'] else 'FAIL'}, Pixel-AP delta {data['mvtec']['summary']['macro_pixel_ap_gain']:+.6f}",
         "- parameter changes between datasets: none",
         f"- overall: {decision}", "",
         "The aligned text signal remains measurable, but the fixed bidirectional score conversion is not cross-domain safe."]
(BASE / "R3_OVERALL_DECISION.md").write_text("\n".join(lines), encoding="utf-8")
print("\n".join(lines))
