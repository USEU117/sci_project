"""Static and evidence audit for DynamicFusion V3 leakage boundaries."""

from __future__ import annotations

import ast
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
V3_SOURCES = [
    ROOT / "src/industrial_ad/fusion/v3_contracts.py",
    ROOT / "src/industrial_ad/fusion/v3_calibration.py",
    ROOT / "src/industrial_ad/fusion/v3_router.py",
    ROOT / "src/industrial_ad/fusion/v3_adaptclip_mpdd.py",
]
FORBIDDEN_ROUTER_ARGUMENTS = {
    "label", "labels", "mask", "masks", "category", "class_name",
    "test_statistics", "test_mean", "test_std",
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    digest.update(path.read_bytes())
    return digest.hexdigest()


def public_arguments(path: Path) -> dict[str, list[str]]:
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    found = {}
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and not node.name.startswith("_"):
            found[node.name] = [arg.arg for arg in node.args.args + node.args.kwonlyargs]
    return found


def main() -> int:
    router_arguments = public_arguments(ROOT / "src/industrial_ad/fusion/v3_router.py")
    forbidden_hits = {
        function: sorted(FORBIDDEN_ROUTER_ARGUMENTS.intersection(arguments))
        for function, arguments in router_arguments.items()
        if FORBIDDEN_ROUTER_ARGUMENTS.intersection(arguments)
    }
    gate_a2 = json.loads((ROOT / "experiments/dynamic_fusion/v3/gate_a2_reliability_predictability/report.json").read_text(encoding="utf-8"))
    checks = {
        "router_public_api_has_no_forbidden_arguments": not forbidden_hits,
        "gate_a2_test_labels_used_by_router_false": gate_a2.get("test_labels_used_by_router") is False,
        "gate_a2_test_masks_used_by_router_false": gate_a2.get("test_masks_used_by_router") is False,
        "gate_a2_test_set_statistics_used_by_router_false": gate_a2.get("test_set_statistics_used_by_router") is False,
        "gate_a2_btad_accessed_false": gate_a2.get("btad_accessed") is False,
        "all_v3_sources_present": all(path.is_file() for path in V3_SOURCES),
    }
    report = {
        "schema_version": 1,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "run_id": "v3_leakage_provenance_audit_20260812_v1",
        "status": "passed" if all(checks.values()) else "failed",
        "checks": checks,
        "forbidden_router_argument_hits": forbidden_hits,
        "router_public_arguments": router_arguments,
        "source_sha256": {str(path.relative_to(ROOT)).replace("\\", "/"): sha256(path) for path in V3_SOURCES},
        "gate_a2_report_sha256": sha256(ROOT / "experiments/dynamic_fusion/v3/gate_a2_reliability_predictability/report.json"),
        "scientific_note": "Passing this audit establishes interface/provenance compliance, not predictive benefit; Gate A2 remains scientifically failed.",
    }
    output = ROOT / "experiments/dynamic_fusion/v3/leakage_provenance_audit.json"
    output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps({"status": report["status"], "checks": checks}, indent=2))
    return 0 if report["status"] == "passed" else 2


if __name__ == "__main__":
    raise SystemExit(main())
