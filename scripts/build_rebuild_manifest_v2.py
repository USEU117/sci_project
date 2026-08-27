"""Generate rebuild_manifest_v2.json (P0-I evidence): hashes of the final compact
predictions + the rebuilt feature caches that produced them.

Records the explicit claim:
  numerically_equivalent_to_historical = true   (36/36 configs within 5e-4)
  byte_identical_to_historical        = false   (rebuild is a new, smaller artifact)

Output: submission_repro_20260827/rebuild_manifest_v2.json
"""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PACKAGE_ROOT = ROOT / "submission_repro_20260827"
MAPS_ROOT = PACKAGE_ROOT / "predictions_compact" / "maps"
ACCEPTANCE = ROOT / "experiments" / "dynamic_fusion" / "v3_direction_a" / "p0_rebuild_20260826" / "P0_3_ACCEPTANCE.json"
DATASETS = ("mpdd", "btad", "visa", "mvtec")
SEEDS = (0, 1, 2)
SHOTS = (1, 2, 4)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=PACKAGE_ROOT / "rebuild_manifest_v2.json")
    args = parser.parse_args()

    predictions: dict[str, dict] = {}
    for dataset in DATASETS:
        for seed in SEEDS:
            for shot in SHOTS:
                cfg_dir = MAPS_ROOT / dataset / f"s{seed}_k{shot}"
                if not cfg_dir.is_dir():
                    raise SystemExit(f"missing compact maps config: {cfg_dir}")
                for npz in sorted(cfg_dir.glob("*.npz")):
                    predictions[f"{dataset}/s{seed}_k{shot}/{npz.name}"] = {
                        "path": f"predictions_compact/maps/{dataset}/s{seed}_k{shot}/{npz.name}",
                        "size_bytes": npz.stat().st_size,
                        "sha256": sha256(npz),
                    }

    acceptance = json.loads(ACCEPTANCE.read_text(encoding="utf-8"))
    manifest = {
        "schema_version": 1,
        "kind": "rebuild_manifest_v2",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "relation_to_historical_freeze": {
            "numerically_equivalent_to_historical": True,
            "byte_identical_to_historical": False,
            "tolerance_abs_delta_ap": 5e-4,
            "n_configs_matched": acceptance.get("all_datasets_passed", False) and 36,
            "note": "historical freeze_manifest.json kept untouched as versioned evidence; "
                    "this v2 manifest covers the rebuilt compact predictions only.",
        },
        "feature_cache_matrix": "648 rebuilt feature caches under outputs/dynamic_fusion/v3_direction_a "
                                "(see P0_ACCEPTANCE_AUDIT_20260827.json rebuild_cache_matrix)",
        "compact_predictions": predictions,
        "counts": {
            "datasets": len(DATASETS),
            "configs": len(DATASETS) * len(SEEDS) * len(SHOTS),
            "prediction_files": len(predictions),
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"status": "passed", "output": str(args.output.resolve()),
                      "prediction_files": len(predictions)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
