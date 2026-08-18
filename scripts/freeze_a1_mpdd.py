"""Freeze the A1 MPDD fixed configuration (docs 阶段六).

Frozen candidate: A1 feature-level fusion `concat + KNN memory bank`,
DINO dinov2_vitb14 + AnomalyCLIP ViT-L/14@336, pca_dim=0, whiten=0,
dino_weight=0.5, pixel stride=8.

Modes (mutually exclusive, exactly one required):
  --create : recompute all hashes and write freeze_manifest.json.
  --verify : strictly read-only full verification of the existing manifest.
             Never writes; reports missing / size mismatch / hash mismatch /
             extra undeclared .npz; exit code 0 only when everything matches.

S1 fix (2026-08-18): previously --verify recomputed and *overwrote* the
manifest before checking, so it could not prove the freeze was unchanged,
and it only checked a subset of entries. Now --create/--verify are
exclusive, --verify is read-only and verifies every declared entry plus
detects extra undeclared cache files.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]

CODE_FILES = [
    "scripts/evaluate_a1_feature_fusion.py",
    "scripts/export_anomalydino_mpdd_features.py",
    "scripts/export_anomalyclip_mpdd_features.py",
    "scripts/export_a1_mpdd_ref_only.py",
    "scripts/run_a1_mpdd_matrix.py",
    "scripts/summarize_a1_mpdd_matrix.py",
    "scripts/audit_a1_mpdd_matrix.py",
    "scripts/evaluate_a1_dynamic_vs_fixed.py",
    "src/industrial_ad/fusion/v3_3_clean.py",
]

CHECKPOINTS = [
    "methods/AnomalyCLIP-main/checkpoints/9_12_4_multiscale_visa/epoch_15.pth",
    f"{Path.home()}/.cache/torch/hub/checkpoints/dinov2_vitb14_pretrain.pth",
]

MANIFESTS = [
    "data/splits/mpdd/manifest.json",
]

EVALUATOR_FILES = [
    "scripts/evaluate_unified.py",
]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def file_entry(relative: str) -> dict:
    path = ROOT / relative
    if not path.is_file():
        raise SystemExit(f"missing file: {path}")
    return {"relative_path": relative, "size_bytes": path.stat().st_size, "sha256": sha256(path)}


def dir_npz_hashes(relative_dir: str) -> list[dict]:
    """Hash every .npz under a directory (sorted by name)."""
    base = ROOT / relative_dir
    if not base.is_dir():
        raise SystemExit(f"missing dir: {base}")
    entries = []
    for path in sorted(base.rglob("*.npz")):
        rel = str(path.relative_to(ROOT)).replace("\\", "/")
        entries.append({"relative_path": rel, "size_bytes": path.stat().st_size, "sha256": sha256(path)})
    return entries


def collect_payload() -> dict:
    """Compute all hashes for the frozen artifact set (never writes anything)."""
    code = [file_entry(p) for p in CODE_FILES]
    checkpoints = [file_entry(p) for p in CHECKPOINTS]
    manifests = [file_entry(p) for p in MANIFESTS]
    evaluators = [file_entry(p) for p in EVALUATOR_FILES]

    feature_caches = {}
    for seed in (0, 1, 2):
        for shot in (1, 2, 4):
            key = f"s{seed}_k{shot}"
            feature_caches[key] = {
                "anomalydino_visual": dir_npz_hashes(
                    f"outputs/dynamic_fusion/v3_direction_a/features_vitb14_s{seed}_k{shot}/anomalydino_visual"
                ),
                "anomalyclip_text": dir_npz_hashes(
                    f"outputs/dynamic_fusion/v3_direction_a/features_s{seed}_k{shot}/anomalyclip_text"
                ),
            }

    baseline_caches = {}
    for seed in (0, 1, 2):
        for shot in (1, 2, 4):
            key = f"s{seed}_k{shot}"
            baseline_caches[key] = {
                "anomalydino_visual": dir_npz_hashes(
                    f"outputs/dynamic_fusion/v2_mpdd_predictions/v2_mpdd_s{seed}_k{shot}_full_v1/anomalydino_visual"
                ),
                "anomalyclip_text": dir_npz_hashes(
                    f"outputs/dynamic_fusion/v2_mpdd_predictions/v2_mpdd_s{seed}_k{shot}_full_v1/anomalyclip_text"
                ),
            }

    return {
        "schema_version": 1,
        "status": "frozen",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "development_dataset": "mpdd",
        "dataset_role": "development",
        "method": "A1_feature_level_fusion_concat_knn_memory_bank",
        "frozen_config": {
            "branches": ["anomalydino_visual (dinov2_vitb14)", "anomalyclip_text (ViT-L/14@336px)"],
            "fusion": "L2-normalize each branch -> concat -> L2-normalize -> KNN (k=1) memory bank on K normal-reference patches -> distance/2 = anomaly map",
            "pca_dim": 0,
            "whiten": False,
            "dino_weight": 0.5,
            "clip_weight": 0.5,
            "pixel_stride": 8,
            "map_size": 448,
            "clip_grid_resize_to_dino_grid": True,
            "K": [1, 2, 4],
            "seeds": [0, 1, 2],
        },
        "selection_evidence": {
            "matrix_9_configs_all_positive": True,
            "mean_delta_ap_vs_dino_9_configs": 0.048558777777777784,
            "matrix_audit_all_pass": True,
            "weight_scan_best": 0.4,
            "weight_scan_frozen_0_5_delta_gap": 0.000927,
            "weight_decision": "keep w=0.5 (equal, no hyperparameter, symmetric); w=0.4 gain ~+0.0009 within noise",
            "dynamic_vs_fixed": {
                "dynamic_mean_delta_ap": 0.04951190740740739,
                "fixed_mean_delta_ap": 0.0485977037037037,
                "dynamic_minus_fixed": 0.0009141851851851853,
                "dynamic_wins_fraction": 0.4444444444444444,
                "decision": "dynamic router does not exceed best fixed fusion; freeze fixed w=0.5",
            },
        },
        "leakage_flags": {
            "test_predictions_used": False,
            "test_labels_used": False,
            "test_masks_used": False,
            "test_dataset_statistics_used": False,
            "test_normal_selection_used": False,
        },
        "code": code,
        "checkpoints": checkpoints,
        "manifests": manifests,
        "evaluators": evaluators,
        "feature_caches": feature_caches,
        "baseline_prediction_caches": baseline_caches,
    }


def _declared_npz_paths(payload: dict) -> set[str]:
    declared = set()
    for group in ("feature_caches", "baseline_prediction_caches"):
        for branches in payload.get(group, {}).values():
            for entries in branches.values():
                declared.update(e["relative_path"] for e in entries)
    return declared


def _cache_dirs(payload: dict) -> list[str]:
    dirs = []
    for group in ("feature_caches", "baseline_prediction_caches"):
        for branches in payload.get(group, {}).values():
            for entries in branches.values():
                if entries:
                    rel = Path(entries[0]["relative_path"]).parent.as_posix()
                    if rel not in dirs:
                        dirs.append(rel)
    return dirs


def extra_undeclared_npz(payload: dict) -> list[str]:
    """Scan the declared cache directories and return .npz not declared in the manifest."""
    declared = _declared_npz_paths(payload)
    extra = []
    for relative_dir in _cache_dirs(payload):
        base = ROOT / relative_dir
        if not base.is_dir():
            continue
        for path in sorted(base.rglob("*.npz")):
            rel = str(path.relative_to(ROOT)).replace("\\", "/")
            if rel not in declared:
                extra.append(rel)
    return extra


def create_manifest(output: Path) -> dict:
    """Recompute all hashes and write the manifest (create mode only)."""
    payload = collect_payload()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return payload


def verify_manifest(manifest_path: Path) -> dict:
    """Strictly read-only full verification of an existing manifest."""
    if not manifest_path.is_file():
        return {"mode": "verify", "all_ok": False, "error": f"manifest missing: {manifest_path}"}
    manifest_sha_before = sha256(manifest_path)
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except Exception as exc:  # noqa: BLE001 - report any read/parse failure
        return {"mode": "verify", "all_ok": False, "error": f"cannot read manifest: {exc}"}

    missing, size_mismatch, hash_mismatch, verified = [], [], [], []

    def check_group(group: str, entries: list[dict]) -> None:
        for entry in entries:
            rel = entry["relative_path"]
            path = ROOT / rel
            if not path.is_file():
                missing.append({"group": group, "relative_path": rel, "reason": "missing"})
                continue
            size = path.stat().st_size
            if size != entry["size_bytes"]:
                size_mismatch.append({
                    "group": group, "relative_path": rel,
                    "expected_size": entry["size_bytes"], "actual_size": size,
                })
                continue
            if sha256(path) != entry["sha256"]:
                hash_mismatch.append({"group": group, "relative_path": rel})
                continue
            verified.append(rel)

    for group in ("code", "checkpoints", "manifests", "evaluators"):
        check_group(group, manifest.get(group, []))

    for group in ("feature_caches", "baseline_prediction_caches"):
        for key, branches in manifest.get(group, {}).items():
            for branch, entries in branches.items():
                check_group(f"{group}/{key}/{branch}", entries)

    extra = extra_undeclared_npz(manifest)
    all_ok = not (missing or size_mismatch or hash_mismatch or extra)
    return {
        "mode": "verify",
        "manifest_path": str(manifest_path),
        "manifest_sha256_before": manifest_sha_before,
        "status": manifest.get("status", "unknown"),
        "verified_entries": len(verified),
        "missing": missing,
        "size_mismatch": size_mismatch,
        "hash_mismatch": hash_mismatch,
        "extra_undeclared_npz": extra,
        "all_ok": all_ok,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="A1 MPDD freeze manifest creator/verifier (S1 read-only fix)")
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "experiments" / "dynamic_fusion" / "freeze" / "a1_mpdd_w05" / "freeze_manifest.json",
    )
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--create", action="store_true", help="recompute all hashes and write the manifest")
    mode.add_argument("--verify", action="store_true", help="read-only full verification of the existing manifest")
    args = parser.parse_args(argv)

    if args.create:
        payload = create_manifest(args.output)
        feature_npz = sum(
            len(entries)
            for branches in payload["feature_caches"].values()
            for entries in branches.values()
        )
        print(json.dumps({
            "status": payload["status"],
            "code_files": len(payload["code"]),
            "checkpoints": len(payload["checkpoints"]),
            "feature_cache_npz": feature_npz,
            "output": str(args.output),
        }))
        return 0

    report = verify_manifest(args.output)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0 if report.get("all_ok") else 1


if __name__ == "__main__":
    raise SystemExit(main())
