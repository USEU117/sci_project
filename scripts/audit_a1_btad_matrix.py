"""Audit the A1 BTAD post-freeze validation (schema, alignment, leakage, NaN, grid).

BTAD specifics vs the VisA/MVTec audit:
  - 9 configs = 3 seeds x 1/2/4-shot, but K1 lives in the old dirs
    (a1_vitb14_btad_fusion / a1_vitb14_btad_dino) while K2/K4 live in
    a1_vitb14_btad_20260819.
  - only concat + dino modes (no standalone CLIP-only report).
  - baseline key is `anomalydino_visual` (legacy v2 dino score cache), not
    `anomalydino_visual_feature_knn`.
  - the DINO grid is non-square for category 03 (32x42), so mask-size checks
    are read from the npz grid instead of hardcoded square sizes.
"""

from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

EXP = ROOT / "experiments" / "dynamic_fusion" / "v3_direction_a"
FUSION = EXP / "a1_vitb14_btad_fusion"
DINO_K1 = EXP / "a1_vitb14_btad_dino"
MATRIX = EXP / "a1_vitb14_btad_20260819"
FEATURES_ROOT = ROOT / "outputs" / "dynamic_fusion" / "v3_direction_a"

SEEDS = [0, 1, 2]
SHOTS = [1, 2, 4]
MODES = ["concat", "dino"]
CATEGORIES = ["01", "02", "03"]
EXPECTED_TEST_COUNTS = {"01": 70, "02": 230, "03": 441}
# DINO vitb14 grid per category (CLIP is 37x37 for all categories).
DINO_GRIDS = {"01": (32, 32), "02": (32, 32), "03": (32, 42)}

LEAK_FLAGS = (
    "test_predictions_used_for_parameter_fit",
    "test_labels_used_for_parameter_fit",
    "test_set_statistics_used_for_calibration",
)


def report_path(seed: int, shot: int, mode: str) -> Path:
    fname = f"{mode}_pca0_whiten0_w0.5_report.json"
    if shot == 1:
        base = FUSION if mode == "concat" else DINO_K1
        return base / f"seed{seed}" / fname
    return MATRIX / f"seed{seed}_k{shot}" / fname


def dino_dir(seed: int, shot: int) -> Path:
    return FEATURES_ROOT / f"features_vitb14_btad_s{seed}_k{shot}" / "anomalydino_visual"


def clip_dir(seed: int, shot: int) -> Path:
    return FEATURES_ROOT / f"features_btad_s{seed}_k{shot}" / "anomalyclip_text"


def main() -> int:
    errors: list[str] = []
    checks: list[dict] = []

    def check(name: str, ok: bool, detail: str = "") -> None:
        checks.append({"check": name, "passed": bool(ok), "detail": detail})
        if not ok:
            errors.append(f"{name}: {detail}")

    # 1. Reports: existence, 3 categories, NaN-free, delta consistency,
    #    matched legacy baseline across concat/dino.
    for seed in SEEDS:
        for shot in SHOTS:
            cr_path = report_path(seed, shot, "concat")
            dr_path = report_path(seed, shot, "dino")
            for mode, path in (("concat", cr_path), ("dino", dr_path)):
                if not path.is_file():
                    check(f"s{seed}_k{shot}_{mode}_exists", False, str(path))
                    continue
                r = json.loads(path.read_text(encoding="utf-8"))
                cats = [c["category"] for c in r["per_category"]]
                check(f"s{seed}_k{shot}_{mode}_categories", cats == CATEGORIES, f"{cats} vs {CATEGORIES}")
                for c in r["per_category"]:
                    for key in ("pixel_ap", "pixel_auroc", "pixel_aupro"):
                        v = c["fused"][key]
                        if not np.isfinite(v):
                            check(f"s{seed}_k{shot}_{mode}_{c['category']}_finite_{key}", False, str(v))
                    dino_base = c["baselines"]["anomalydino_visual"]["pixel_ap"]
                    check(
                        f"s{seed}_k{shot}_{mode}_{c['category']}_delta_consistency",
                        abs(c["delta_ap"] - (c["fused"]["pixel_ap"] - dino_base)) < 1e-5,
                        f"{c['delta_ap']} vs {c['fused']['pixel_ap'] - dino_base}",
                    )
                check(
                    f"s{seed}_k{shot}_{mode}_mean_delta_consistency",
                    abs(r["mean_delta_ap_vs_dino"] - (r["mean_fused"]["pixel_ap"] - r["mean_dino_baseline_ap"])) < 1e-5,
                    f"{r['mean_delta_ap_vs_dino']} vs {r['mean_fused']['pixel_ap'] - r['mean_dino_baseline_ap']}",
                )
                recomputed_mean = float(np.mean([c["fused"]["pixel_ap"] for c in r["per_category"]]))
                check(
                    f"s{seed}_k{shot}_{mode}_mean_consistent",
                    abs(r["mean_fused"]["pixel_ap"] - recomputed_mean) < 1e-6,
                    f"{r['mean_fused']['pixel_ap']} vs {recomputed_mean}",
                )
            # matched legacy baseline must be identical between concat and dino
            if cr_path.is_file() and dr_path.is_file():
                cr = json.loads(cr_path.read_text(encoding="utf-8"))
                dr = json.loads(dr_path.read_text(encoding="utf-8"))
                check(
                    f"s{seed}_k{shot}_legacy_baseline_match",
                    abs(cr["mean_dino_baseline_ap"] - dr["mean_dino_baseline_ap"]) < 1e-6,
                    f"{cr['mean_dino_baseline_ap']} vs {dr['mean_dino_baseline_ap']}",
                )

    # 2. Feature-cache leakage flags.
    export_dirs = []
    for seed in SEEDS:
        for shot in SHOTS:
            export_dirs.append(dino_dir(seed, shot))
            export_dirs.append(clip_dir(seed, shot))
    for d in export_dirs:
        rep_path = d / "export_report.json"
        if not rep_path.is_file():
            check(f"export_report_{d.parent.name}/{d.name}", False, str(rep_path))
            continue
        rep = json.loads(rep_path.read_text(encoding="utf-8"))
        if rep.get("status") != "passed":
            check(f"export_status_{d.parent.name}/{d.name}", False, rep.get("status", ""))
            continue
        for flag in LEAK_FLAGS:
            check(f"leak_{d.parent.name}/{d.name}_{flag}", rep.get(flag) is False, str(rep.get(flag)))

    # 3. Feature-cache alignment: dino vs clip sample_ids/labels/ref-count/test-count.
    for seed in SEEDS:
        for shot in SHOTS:
            for cat in CATEGORIES:
                dp = dino_dir(seed, shot) / f"{cat}.npz"
                cp = clip_dir(seed, shot) / f"{cat}.npz"
                if not dp.is_file() or not cp.is_file():
                    check(f"align_{cat}_s{seed}_k{shot}_files", False, f"{dp.is_file()} {cp.is_file()}")
                    continue
                with np.load(dp, allow_pickle=False) as a, np.load(cp, allow_pickle=False) as b:
                    ids_d = list(a["sample_ids"])
                    ids_c = list(b["sample_ids"])
                    gt_d = a["gt_sp"]
                    gt_c = b["gt_sp"]
                    refs_d = a["ref_patch_features"]
                    refs_c = b["ref_patch_features"]
                    grid_d = tuple(int(v) for v in a["grid_size"])
                    grid_c = tuple(int(v) for v in b["grid_size"])
                    mask_d = a["imgs_masks"]
                check(f"align_{cat}_s{seed}_k{shot}_sample_ids", ids_d == ids_c, f"{len(ids_d)} vs {len(ids_c)}")
                check(f"align_{cat}_s{seed}_k{shot}_test_count", len(ids_d) == EXPECTED_TEST_COUNTS[cat],
                      f"{len(ids_d)} vs {EXPECTED_TEST_COUNTS[cat]}")
                check(f"align_{cat}_s{seed}_k{shot}_labels", np.array_equal(gt_d, gt_c), "dino/clip labels differ")
                check(f"align_{cat}_s{seed}_k{shot}_ref_count",
                      int(refs_d.shape[0]) == shot and int(refs_c.shape[0]) == shot,
                      f"dino refs {refs_d.shape[0]}, clip refs {refs_c.shape[0]}, shot {shot}")
                check(f"align_{cat}_s{seed}_k{shot}_grid_dino", grid_d == DINO_GRIDS[cat], f"{grid_d} vs {DINO_GRIDS[cat]}")
                check(f"align_{cat}_s{seed}_k{shot}_grid_clip", grid_c == (37, 37), f"{grid_c} vs (37, 37)")
                check(f"align_{cat}_s{seed}_k{shot}_mask_n",
                      int(mask_d.shape[0]) == len(ids_d), f"mask {mask_d.shape[0]} vs {len(ids_d)}")

    # 4. Grid consistency: dino test grid == dino ref grid across all caches.
    for seed in SEEDS:
        for shot in SHOTS:
            for cat in CATEGORIES:
                p = dino_dir(seed, shot) / f"{cat}.npz"
                if not p.is_file():
                    continue
                with np.load(p, allow_pickle=False) as data:
                    tg = tuple(int(v) for v in data["grid_size"])
                    rg = tuple(int(v) for v in data["ref_patch_features"].shape[1:3])
                check(f"grid_s{seed}_k{shot}_{cat}_dino_test_ref", tg == rg, f"{tg} vs {rg}")

    report = {
        "schema_version": 1,
        "run_id": "a1_btad_post_freeze_audit_20260819",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": "passed" if not errors else "failed",
        "dataset": "btad",
        "dataset_role": "external_frozen_validation",
        "n_checks": len(checks),
        "n_failed": len(errors),
        "errors": errors,
        "checks": checks,
    }
    out = MATRIX / "btad_audit.json"
    out.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"status": report["status"], "n_checks": len(checks), "n_failed": len(errors)}, indent=2))
    if errors:
        print("\n".join(errors[:40]))
    return 0 if not errors else 1


if __name__ == "__main__":
    raise SystemExit(main())
