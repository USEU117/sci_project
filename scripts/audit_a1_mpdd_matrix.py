"""Audit the A1 MPDD 3x3 matrix (docs 阶段五 5.2): 9/9 must pass schema,
alignment, and NaN/Inf checks. CPU only. Reads feature caches + reports.
"""

from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

FEATURES_ROOT = ROOT / "outputs" / "dynamic_fusion" / "v3_direction_a"
MATRIX_ROOT = ROOT / "experiments" / "dynamic_fusion" / "v3_direction_a" / "a1_matrix_20260817"
CATS = ["bracket_black", "bracket_brown", "bracket_white", "connector", "metal_plate", "tubes"]


def audit_features(path: Path) -> dict:
    with np.load(path, allow_pickle=False) as data:
        required = {
            "patch_features",
            "ref_patch_features",
            "gt_sp",
            "imgs_masks",
            "sample_ids",
            "grid_size",
            "seed",
            "shot",
        }
        missing = required.difference(data.files)
        if missing:
            return {"ok": False, "error": f"missing {sorted(missing)}"}
        pf = np.asarray(data["patch_features"])
        rf = np.asarray(data["ref_patch_features"])
        masks = np.asarray(data["imgs_masks"])
        checks = {
            "patch_features_ndim": pf.ndim == 4,
            "ref_ndim": rf.ndim == 4,
            "n_match": pf.shape[0] == len(data["sample_ids"]),
            "grid_consistent": tuple(pf.shape[1:3]) == tuple(int(v) for v in data["grid_size"]),
            "ref_grid_same": rf.shape[1:3] == pf.shape[1:3],
            "mask_n_match": masks.shape[0] == pf.shape[0],
            "no_nan_inf_patch": bool(np.isfinite(pf).all()),
            "no_nan_inf_ref": bool(np.isfinite(rf).all()),
            "no_nan_inf_mask": bool(np.isfinite(masks).all()),
            "finite_gt": bool(np.isfinite(data["gt_sp"]).all()),
        }
        return {"ok": all(checks.values()), "checks": checks, "shape": pf.shape}


def main() -> int:
    results = {}
    all_ok = True
    for seed in (0, 1, 2):
        for shot in (1, 2, 4):
            key = f"s{seed}_k{shot}"
            dino_dir = FEATURES_ROOT / f"features_vitb14_s{seed}_k{shot}" / "anomalydino_visual"
            clip_dir = FEATURES_ROOT / f"features_s{seed}_k{shot}" / "anomalyclip_text"
            report_path = MATRIX_ROOT / f"seed{seed}_k{shot}" / "concat_pca0_whiten0_w0.5_report.json"
            cat_audits = {}
            for cat in CATS:
                dino_audit = audit_features(dino_dir / f"{cat}.npz")
                clip_audit = audit_features(clip_dir / f"{cat}.npz")
                cat_audits[cat] = {
                    "dino": dino_audit,
                    "clip": clip_audit,
                    "ok": bool(dino_audit["ok"] and clip_audit["ok"]),
                }
            report_ok = report_path.is_file()
            if report_ok:
                r = json.loads(report_path.read_text(encoding="utf-8"))
                n_cat = len(r.get("per_category", []))
                report_ok = n_cat == len(CATS)
            cat_ok = all(cat_audits[c]["ok"] for c in CATS)
            ok = cat_ok and report_ok
            all_ok = all_ok and ok
            results[key] = {
                "ok": ok,
                "categories_ok": int(sum(cat_audits[c]["ok"] for c in CATS)),
                "report_exists_and_complete": report_ok,
                "categories": cat_audits,
            }

    report = {
        "schema_version": 1,
        "run_id": "a1_mpdd_matrix_audit_20260817",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "stage": "phase_5_2_matrix_audit",
        "all_9_configs_pass": all_ok,
        "results": results,
    }
    out = MATRIX_ROOT / "matrix_audit.json"
    out.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"all_9_configs_pass": all_ok, "results": {k: v["ok"] for k, v in results.items()}}))
    return 0 if all_ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
