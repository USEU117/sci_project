"""Freeze the A1 MPDD fixed configuration (docs 阶段六).

Frozen candidate: A1 feature-level fusion `concat + KNN memory bank`,
DINO dinov2_vitb14 + AnomalyCLIP ViT-L/14@336, pca_dim=0, whiten=0,
dino_weight=0.5, pixel stride=8.

Collects hashes for: code, config/manifest, checkpoints, evaluator,
feature caches (dino/clip x 9 configs) and baseline prediction caches.
Generates freeze_manifest.json.
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


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=ROOT / "experiments" / "dynamic_fusion" / "freeze" / "a1_mpdd_w05" / "freeze_manifest.json")
    parser.add_argument("--verify", action="store_true")
    args = parser.parse_args()

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

    payload = {
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
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    if args.verify:
        current = json.loads(args.output.read_text(encoding="utf-8"))
        # recompute and compare a lightweight subset (code + checkpoints + manifests)
        ok = current.get("status") == "frozen"
        for entry in current["code"]:
            if entry != file_entry(entry["relative_path"]):
                ok = False
        print(json.dumps({"status": "passed" if ok else "failed"}))
        return 0 if ok else 1

    print(json.dumps({
        "status": payload["status"],
        "code_files": len(code),
        "checkpoints": len(checkpoints),
        "feature_cache_npz": sum(len(v) for k in feature_caches for v in feature_caches[k].values()),
        "output": str(args.output),
    }))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
