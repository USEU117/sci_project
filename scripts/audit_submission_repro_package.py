"""Audit the compact submission reproducibility package for frozen A1.

Read-only unless --output is supplied. This never runs feature extraction or
evaluation; it distinguishes versioned evidence from live runnable inputs.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]

EVIDENCE_FILES = [
    "docs/CURRENT_DYNAMIC_FUSION_STATUS.md",
    "experiments/dynamic_fusion/main_results_20260818/main_results.json",
    "experiments/dynamic_fusion/main_results_20260818/main_results.csv",
    "experiments/dynamic_fusion/main_results_20260818/per_category_results.csv",
    "experiments/dynamic_fusion/main_results_20260818/metric_definition.md",
    "experiments/dynamic_fusion/freeze/a1_mpdd_w05/freeze_manifest.json",
    "experiments/dynamic_fusion/freeze/a1_mpdd_w05/freeze_verification.json",
    "experiments/dynamic_fusion/freeze/a1_mpdd_w05/METHOD_CARD.md",
    "experiments/dynamic_fusion/freeze/a1_mpdd_w05/REPRODUCE.md",
    "experiments/dynamic_fusion/v3_direction_a/a1_matrix_20260817/matrix_summary.json",
    "experiments/dynamic_fusion/v3_direction_a/a1_matrix_20260817/matrix_audit.json",
    "experiments/dynamic_fusion/v3_direction_a/a1_vitb14_btad_20260819/btad_summary.json",
    "experiments/dynamic_fusion/v3_direction_a/a1_vitb14_btad_20260819/btad_audit.json",
    "experiments/dynamic_fusion/v3_direction_a/a1_visa_20260818/visa_summary.json",
    "experiments/dynamic_fusion/v3_direction_a/a1_visa_20260818/visa_audit.json",
    "experiments/dynamic_fusion/v3_direction_a/a1_mvtec_20260818/mvtec_summary.json",
    "experiments/dynamic_fusion/v3_direction_a/a1_mvtec_20260818/mvtec_audit.json",
    "experiments/dynamic_fusion/v4_vision_text_20260819/00_g0_audit/modality_semantics_audit.json",
    "experiments/dynamic_fusion/v4_vision_text_20260819/04_gate_decision/gate_decision.md",
    "experiments/dynamic_fusion/v4_vision_text_20260819/06_v2_g2_audit/g2_audit_report.json",
    "docs/submission_reproducibility_20260826/CPU_REGRESSION_20260826.json",
]

CODE_FILES = [
    "scripts/export_anomalydino_mpdd_features.py",
    "scripts/export_anomalyclip_mpdd_features.py",
    "scripts/export_a1_mpdd_ref_only.py",
    "scripts/export_a1_visa_features.py",
    "scripts/export_a1_visa_ref_only.py",
    "scripts/evaluate_a1_feature_fusion.py",
    "scripts/evaluate_a1_visa_frozen.py",
    "scripts/evaluate_a1_complete_metrics.py",
    "scripts/freeze_a1_mpdd.py",
    "scripts/build_main_results_table.py",
    "scripts/smoke_a1_one_class_one_image.py",
    "scripts/p0_3_evaluate_a1_rebuild.py",
    "src/utils.py",
]

DATA_ROOTS = {
    "mpdd": "data/mpdd_raw/MPDD",
    "btad": "data/btad_raw",
    "visa": "data/visa_raw",
    "mvtec": "data/mvtec",
}
SPLIT_MANIFESTS = {name: f"data/splits/{name}/manifest.json" for name in DATA_ROOTS}
WEIGHTS = {
    "anomalyclip": "methods/AnomalyCLIP-main/checkpoints/9_12_4_multiscale_visa/epoch_15.pth",
    "dinov2_vitb14": str(Path.home() / ".cache/torch/hub/checkpoints/dinov2_vitb14_pretrain.pth"),
}
CACHE_ROOT = ROOT / "outputs" / "dynamic_fusion"
REBUILD_REPORT_ROOT = ROOT / "experiments" / "dynamic_fusion" / "v3_direction_a" / "p0_rebuild_20260826"
SMOKE_REPORT = ROOT / "outputs" / "p0_2_smoke" / "smoke_report.json"
PACKAGE_ROOT = ROOT / "submission_repro_20260827"

EXPECTED_CACHE_LAYOUT = {
    "mpdd_dino": ("features_vitb14_s*_k*/anomalydino_visual/*.npz", 54),
    "mpdd_clip": ("features_s*_k*/anomalyclip_text/*.npz", 54),
    "btad_dino": ("features_vitb14_btad_s*_k*/anomalydino_visual/*.npz", 27),
    "btad_clip": ("features_btad_s*_k*/anomalyclip_text/*.npz", 27),
    "visa_dino": ("visa_features_vitb14/s*_k*/anomalydino_visual/*.npz", 108),
    "visa_clip": ("visa_features/s*_k*/anomalyclip_text/*.npz", 108),
    "mvtec_dino": ("mvtec_features_vitb14/s*_k*/anomalydino_visual/*.npz", 135),
    "mvtec_clip": ("mvtec_features/s*_k*/anomalyclip_text/*.npz", 135),
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def describe_file(relative_or_absolute: str) -> dict:
    path = Path(relative_or_absolute)
    if not path.is_absolute():
        path = ROOT / path
    exists = path.is_file()
    return {
        "path": relative_or_absolute.replace("\\", "/"),
        "exists": exists,
        "size_bytes": path.stat().st_size if exists else None,
        "sha256": sha256(path) if exists else None,
    }


def describe_dir(relative: str) -> dict:
    path = ROOT / relative
    exists = path.is_dir()
    return {
        "path": relative,
        "exists": exists,
        "file_count": sum(1 for item in path.rglob("*") if item.is_file()) if exists else 0,
    }


def git_head() -> str | None:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True, stderr=subprocess.DEVNULL
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def pytest_available(python_path: Path) -> bool:
    if not python_path.is_file():
        return False
    result = subprocess.run(
        [str(python_path), "-c", "import importlib.util; print(int(importlib.util.find_spec('pytest') is not None))"],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=False,
    )
    return result.returncode == 0 and result.stdout.strip() == "1"


def tracked_by_git(relative: str) -> bool:
    result = subprocess.run(
        ["git", "ls-files", "--error-unmatch", relative],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=False,
    )
    return result.returncode == 0


def audit_cache_matrix() -> dict:
    root = CACHE_ROOT / "v3_direction_a"
    rows = {}
    for name, (pattern, expected) in EXPECTED_CACHE_LAYOUT.items():
        actual = len(list(root.glob(pattern))) if root.is_dir() else 0
        rows[name] = {"pattern": pattern, "expected": expected, "actual": actual, "passed": actual == expected}
    return {"branches": rows, "expected_total": 648, "actual_total": sum(r["actual"] for r in rows.values()),
            "passed": all(r["passed"] for r in rows.values())}


def audit_rebuild_reports() -> dict:
    expected_categories = {"mpdd": 6, "btad": 3, "visa": 12, "mvtec": 15}
    issues, counts = [], {name: 0 for name in expected_categories}
    for path in sorted(REBUILD_REPORT_ROOT.glob("*_s*_k*.json")):
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except Exception as exc:  # noqa: BLE001 - audit reports malformed evidence
            issues.append(f"{path.name}: unreadable: {exc}")
            continue
        dataset = payload.get("dataset")
        if dataset not in expected_categories:
            issues.append(f"{path.name}: invalid dataset")
            continue
        counts[dataset] += 1
        if len(payload.get("per_category", [])) != expected_categories[dataset]:
            issues.append(f"{path.name}: wrong category count")
        if payload.get("baseline_source") != "matched feature-level dino-only KNN (same pipeline)":
            issues.append(f"{path.name}: wrong baseline source")
        if payload.get("mean_delta_ap_vs_feature_dino") is None:
            issues.append(f"{path.name}: missing delta")
    passed = counts == {name: 9 for name in expected_categories} and not issues
    return {"counts": counts, "expected_per_dataset": 9, "issues": issues, "passed": passed}


def audit_smoke() -> dict:
    if not SMOKE_REPORT.is_file():
        return {"path": str(SMOKE_REPORT.relative_to(ROOT)), "exists": False, "passed": False}
    payload = json.loads(SMOKE_REPORT.read_text(encoding="utf-8"))
    checks = payload.get("checks", {})
    passed = bool(
        checks.get("sample_id_match")
        and checks.get("ref_match")
        and checks.get("dino_dim") == 768
        and checks.get("clip_dim") == 768
        and checks.get("concat_dim_resolved") == 1536
        and checks.get("no_nan_inf_dino")
        and checks.get("no_nan_inf_clip")
        and payload.get("repeat_run_identical")
        and payload.get("all_leakage_flags_false")
    )
    return {"path": str(SMOKE_REPORT.relative_to(ROOT)), "exists": True, "passed": passed,
            "concat_dim": checks.get("concat_dim_resolved")}


def audit_package() -> dict:
    checksum_path = PACKAGE_ROOT / "SHA256SUMS"
    checksum_errors, checked = [], 0
    if checksum_path.is_file():
        for line in checksum_path.read_text(encoding="utf-8").splitlines():
            try:
                expected, relative = line.split("  ", 1)
            except ValueError:
                checksum_errors.append(f"malformed line: {line}")
                continue
            path = PACKAGE_ROOT / relative
            if not path.is_file():
                checksum_errors.append(f"missing: {relative}")
            elif sha256(path) != expected.lower():
                checksum_errors.append(f"hash mismatch: {relative}")
            else:
                checked += 1
    else:
        checksum_errors.append("SHA256SUMS missing")

    compact_root = PACKAGE_ROOT / "predictions_compact"
    raw_compact = [p for p in compact_root.rglob("*") if p.is_file() and p.suffix.lower() in {".npz", ".npy", ".jsonl", ".parquet"}]
    package_recompute = PACKAGE_ROOT / "recompute_tables.py"
    source_release_pointer = PACKAGE_ROOT / "SOURCE_COMMIT.txt"
    return {
        "exists": PACKAGE_ROOT.is_dir(),
        "checksum_files_checked": checked,
        "checksum_errors": checksum_errors,
        "checksum_passed": checksum_path.is_file() and not checksum_errors,
        "compact_prediction_files": len(raw_compact),
        "package_local_recompute_script": package_recompute.is_file(),
        "source_commit_pointer": source_release_pointer.is_file(),
        "standalone_cpu_recompute_ready": bool(raw_compact) and package_recompute.is_file(),
    }


def audit_historical_freeze_presence() -> dict:
    manifest_path = ROOT / "experiments" / "dynamic_fusion" / "freeze" / "a1_mpdd_w05" / "freeze_manifest.json"
    if not manifest_path.is_file():
        return {"manifest_exists": False, "exact_live_inputs_present": False}
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    missing, size_mismatch, declared = 0, 0, 0
    for group in ("feature_caches", "baseline_prediction_caches"):
        for branches in payload.get(group, {}).values():
            for entries in branches.values():
                for entry in entries:
                    declared += 1
                    path = ROOT / entry["relative_path"]
                    if not path.is_file():
                        missing += 1
                    elif path.stat().st_size != entry["size_bytes"]:
                        size_mismatch += 1
    return {"manifest_exists": True, "declared_cache_files": declared, "missing": missing,
            "size_mismatch": size_mismatch, "exact_live_inputs_present": missing == 0 and size_mismatch == 0}


def build_report() -> dict:
    evidence = [describe_file(path) for path in EVIDENCE_FILES]
    code = [describe_file(path) for path in CODE_FILES]
    data = {name: describe_dir(path) for name, path in DATA_ROOTS.items()}
    splits = {name: describe_file(path) for name, path in SPLIT_MANIFESTS.items()}
    weights = {name: describe_file(path) for name, path in WEIGHTS.items()}
    cache_npz = list(CACHE_ROOT.rglob("*.npz")) if CACHE_ROOT.is_dir() else []
    patchcore_python = ROOT / ".venv-patchcore" / "Scripts" / "python.exe"
    anomalyclip_python = ROOT / ".venv-anomalyclip" / "Scripts" / "python.exe"

    evidence_complete = all(item["exists"] for item in evidence)
    code_complete = all(item["exists"] for item in code)
    inputs_complete = (
        all(item["exists"] and item["file_count"] > 0 for item in data.values())
        and all(item["exists"] for item in splits.values())
        and all(item["exists"] for item in weights.values())
    )
    cache_matrix = audit_cache_matrix()
    rebuild_reports = audit_rebuild_reports()
    smoke = audit_smoke()
    package = audit_package()
    historical_freeze = audit_historical_freeze_presence()
    patchcore_pytest = pytest_available(patchcore_python)
    anomalyclip_pytest = pytest_available(anomalyclip_python)
    test_runner_available = patchcore_pytest or anomalyclip_pytest

    blockers = []
    if not evidence_complete:
        blockers.append("one or more versioned evidence files are missing")
    if not code_complete:
        blockers.append("one or more frozen-method code files are missing")
    if not inputs_complete:
        blockers.append("one or more datasets, split manifests, or weights are missing")
    if not cache_matrix["passed"]:
        blockers.append("the exact 648-file four-dataset rebuild feature matrix is incomplete")
    if not rebuild_reports["passed"]:
        blockers.append("the 36 per-config rebuild reports are incomplete or malformed")
    if not smoke["passed"]:
        blockers.append("P0-2 smoke evidence is missing or failed")
    if not package["checksum_passed"]:
        blockers.append("compact package SHA256 verification failed")
    if not package["standalone_cpu_recompute_ready"]:
        blockers.append("compact package has summaries only; no compact prediction payload plus package-local recompute script")
    if not package["source_commit_pointer"]:
        blockers.append("compact package has no source commit pointer")
    if not test_runner_available:
        blockers.append("pytest is unavailable in both A1 environments")

    gates = {
        "P0A_versioned_evidence_indexed": evidence_complete and code_complete,
        "P0B_live_inputs_available": inputs_complete,
        "P0C_four_dataset_rebuild_matrix_ready": cache_matrix["passed"],
        "P0D_test_environment_ready": test_runner_available,
        "P0E_smoke_dimension_and_leakage_passed": smoke["passed"],
        "P0F_36_rebuild_reports_passed": rebuild_reports["passed"],
        "P0G_package_checksum_integrity": package["checksum_passed"],
        "P0H_standalone_cpu_recompute_ready": package["standalone_cpu_recompute_ready"],
        "P0I_source_release_identified": package["source_commit_pointer"],
    }
    gates["research_rebuild_complete"] = all(gates[key] for key in list(gates) if key <= "P0G_package_checksum_integrity")
    gates["submission_repro_package_complete"] = all(gates[key] for key in list(gates) if key.startswith("P0"))
    return {
        "schema_version": 1,
        "audit_kind": "submission_reproducibility_p0_live_audit",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "git_head": git_head(),
        "paper_method": "A1 dual-encoder visual patch fusion, fixed w=0.5, normal-memory KNN",
        "method_semantics": {
            "dynamic_router": False,
            "explicit_text_features_used_at_inference": False,
            "paper_safe_label": "dual-encoder visual feature fusion",
        },
        "versioned_evidence": evidence,
        "frozen_method_code": code,
        "datasets": data,
        "split_manifests": splits,
        "weights": weights,
        "runtime": {
            "dynamic_fusion_cache_root_exists": CACHE_ROOT.is_dir(),
            "dynamic_fusion_npz_count": len(cache_npz),
            "rebuild_cache_matrix": cache_matrix,
            "rebuild_reports": rebuild_reports,
            "smoke": smoke,
            "historical_freeze_exact_live_check": historical_freeze,
            "patchcore_python_exists": patchcore_python.is_file(),
            "anomalyclip_python_exists": anomalyclip_python.is_file(),
            "pytest_in_patchcore_env": patchcore_pytest,
            "pytest_in_anomalyclip_env": anomalyclip_pytest,
        },
        "compact_package": package,
        "release_tracking": {
            "new_smoke_script_tracked": tracked_by_git("scripts/smoke_a1_one_class_one_image.py"),
            "new_rebuild_script_tracked": tracked_by_git("scripts/p0_3_evaluate_a1_rebuild.py"),
            "compact_package_tracked": tracked_by_git("submission_repro_20260827/README.md"),
        },
        "gates": gates,
        "blockers": blockers,
        "interpretation": (
            "The four-dataset research rebuild and checksums can pass while the publishable compact package remains "
            "incomplete. Historical freeze byte identity, rebuilt numerical equivalence, and standalone release "
            "reproducibility are separate claims. This audit never authorizes tuning or dynamic-fusion experiments."
        ),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()
    report = build_report()
    rendered = json.dumps(report, ensure_ascii=False, indent=2) + "\n"
    if args.output is not None:
        output = args.output if args.output.is_absolute() else ROOT / args.output
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(rendered, encoding="utf-8")
    print(rendered, end="")
    return 0 if report["gates"]["submission_repro_package_complete"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
