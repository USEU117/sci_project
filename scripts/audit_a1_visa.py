"""Audit the A1 VisA/MVTec post-freeze validation (schema, alignment, leakage flags, NaN)."""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

DATASET_INFO = {
    "visa": {
        "exp": ROOT / "experiments" / "dynamic_fusion" / "v3_direction_a" / "a1_visa_20260818",
        "manifest": ROOT / "data" / "splits" / "visa" / "manifest.json",
        "data_root": ROOT / "data" / "visa_raw",
        "dino_prefix": "visa_features_vitb14",
        "clip_prefix": "visa_features",
    },
    "mvtec": {
        "exp": ROOT / "experiments" / "dynamic_fusion" / "v3_direction_a" / "a1_mvtec_20260818",
        "manifest": ROOT / "data" / "splits" / "mvtec" / "manifest.json",
        "data_root": ROOT / "data" / "mvtec",
        "dino_prefix": "mvtec_features_vitb14",
        "clip_prefix": "mvtec_features",
    },
}
FEATURES_ROOT = ROOT / "outputs" / "dynamic_fusion" / "v3_direction_a"
SEEDS = [0, 1, 2]
SHOTS = [1, 2, 4]
MODES = ["concat", "dino", "clip"]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=("visa", "mvtec"), required=True)
    args = parser.parse_args()
    info = DATASET_INFO[args.dataset]
    EXPERIMENT_ROOT = info["exp"]
    MANIFEST = info["manifest"]
    DATA_ROOT = info["data_root"]
    DINO_PREFIX = info["dino_prefix"]
    CLIP_PREFIX = info["clip_prefix"]

    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    categories = sorted(manifest["categories"])
    errors = []
    checks = []

    def check(name: str, ok: bool, detail: str = "") -> None:
        checks.append({"check": name, "passed": bool(ok), "detail": detail})
        if not ok:
            errors.append(f"{name}: {detail}")

    # 1. All 27 reports exist, 12 categories each, no NaN, delta consistency.
    for seed in SEEDS:
        for shot in SHOTS:
            for mode in MODES:
                path = EXPERIMENT_ROOT / f"seed{seed}_k{shot}" / f"{mode}_pca0_whiten0_w0.5_report.json"
                if not path.is_file():
                    check(f"s{seed}_k{shot}_{mode}_exists", False, str(path))
                    continue
                r = json.loads(path.read_text(encoding="utf-8"))
                cats = [c["category"] for c in r["per_category"]]
                check(f"s{seed}_k{shot}_{mode}_categories", cats == categories, f"{cats} vs {categories}")
                if mode == "concat":
                    check(f"s{seed}_k{shot}_{mode}_baseline_source",
                          r.get("baseline_source", "").startswith("feature_level_dino_only_knn"),
                          r.get("baseline_source", ""))
                for c in r["per_category"]:
                    for key in ("pixel_ap", "pixel_auroc", "pixel_aupro"):
                        v = c["fused"][key]
                        if not np.isfinite(v):
                            check(f"s{seed}_k{shot}_{mode}_{c['category']}_finite_{key}", False, str(v))
                mean = r["mean_fused"]["pixel_ap"]
                recomputed = float(np.mean([c["fused"]["pixel_ap"] for c in r["per_category"]]))
                check(f"s{seed}_k{shot}_{mode}_mean_consistent",
                      abs(mean - recomputed) < 1e-6, f"{mean} vs {recomputed}")

    # 2. Feature-cache leakage flags (all export reports must be false).
    export_dirs = []
    for seed in SEEDS:
        for shot in SHOTS:
            export_dirs.append(FEATURES_ROOT / DINO_PREFIX / f"s{seed}_k{shot}" / "anomalydino_visual")
            export_dirs.append(FEATURES_ROOT / CLIP_PREFIX / f"s{seed}_k{shot}" / "anomalyclip_text")
    for d in export_dirs:
        rep_path = d / "export_report.json"
        if not rep_path.is_file():
            check(f"export_report_{d.name}", False, str(rep_path))
            continue
        rep = json.loads(rep_path.read_text(encoding="utf-8"))
        if rep.get("status") != "passed":
            check(f"export_status_{d.name}", False, rep.get("status", ""))
            continue
        for flag in ("test_predictions_used_for_parameter_fit",
                     "test_labels_used_for_parameter_fit",
                     "test_set_statistics_used_for_calibration"):
            check(f"leak_{d.parent.name}/{d.name}_{flag}", rep.get(flag) is False, str(rep.get(flag)))

    # 3. Feature-cache sample alignment: sample_ids match between dino and clip, and match the dataset index order.
    from v2_mpdd_prediction_common import index_dataset

    indexed = index_dataset(args.dataset, DATA_ROOT)
    for cat in categories:
        expected_ids = [s.sample_id for s in indexed[cat]]
        dino0 = FEATURES_ROOT / DINO_PREFIX / "s0_k1" / "anomalydino_visual" / f"{cat}.npz"
        clip0 = FEATURES_ROOT / CLIP_PREFIX / "s0_k1" / "anomalyclip_text" / f"{cat}.npz"
        if not dino0.is_file() or not clip0.is_file():
            check(f"align_{cat}_files", False, f"{dino0.is_file()} {clip0.is_file()}")
            continue
        with np.load(dino0, allow_pickle=False) as a, np.load(clip0, allow_pickle=False) as b:
            ids_d = list(a["sample_ids"])
            ids_c = list(b["sample_ids"])
            masks_d = a["imgs_masks"]
            masks_c = b["imgs_masks"]
            gt_d = a["gt_sp"]
            gt_c = b["gt_sp"]
            refs_d = a["ref_patch_features"]
            refs_c = b["ref_patch_features"]
        n = len(ids_d)
        # Branch-convention mask sizes: dino map_size=448, clip image_size=518.
        # Evaluation only ever consumes the dino masks (map is 448); clip masks are
        # placeholder-only, so equality is checked on the per-sample labels instead.
        check(f"align_{cat}_mask_sizes",
              tuple(masks_d.shape) == (n, 448, 448) and tuple(masks_c.shape) == (n, 518, 518),
              f"dino {masks_d.shape} clip {masks_c.shape}")
        check(f"align_{cat}_sample_ids", ids_d == ids_c == expected_ids, f"{len(ids_d)} vs {len(ids_c)} vs {len(expected_ids)}")
        check(f"align_{cat}_labels", np.array_equal(gt_d, gt_c), "dino/clip labels differ")
        check(f"align_{cat}_refs_nonempty", int(refs_d.shape[0]) >= 1 and int(refs_c.shape[0]) >= 1,
              f"dino refs {refs_d.shape[0]}, clip refs {refs_c.shape[0]}")
        # reference sources must be exactly the manifest s0/k1 normal images
        expected_refs = manifest["categories"][cat]["0"]["1"]
        check(f"align_{cat}_ref_count", int(refs_d.shape[0]) == len(expected_refs),
              f"{refs_d.shape[0]} vs {len(expected_refs)}")

    # 4. Grid consistency across a category (dino test grid must match ref grid in every cache).
    for seed in SEEDS:
        for shot in SHOTS:
            dino_dir = FEATURES_ROOT / DINO_PREFIX / f"s{seed}_k{shot}" / "anomalydino_visual"
            for cat in categories:
                p = dino_dir / f"{cat}.npz"
                if not p.is_file():
                    continue
                with np.load(p, allow_pickle=False) as data:
                    tg = tuple(int(v) for v in data["grid_size"])
                    rg = data["ref_patch_features"].shape[1:3]
                check(f"grid_s{seed}_k{shot}_{cat}", tg == rg, f"{tg} vs {rg}")

    report = {
        "schema_version": 1,
        "run_id": f"a1_{args.dataset}_post_freeze_audit_20260818",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": "passed" if not errors else "failed",
        "dataset": args.dataset,
        "dataset_role": "holdout",
        "n_checks": len(checks),
        "n_failed": len(errors),
        "errors": errors,
        "checks": checks,
    }
    out = EXPERIMENT_ROOT / f"{args.dataset}_audit.json"
    out.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"status": report["status"], "n_checks": len(checks), "n_failed": len(errors)}, indent=2))
    if errors:
        print("\n".join(errors[:40]))
    return 0 if not errors else 1


if __name__ == "__main__":
    raise SystemExit(main())
