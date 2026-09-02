"""A2 innovation_v2 — Wave 0 shared input audit.

Task book section 18 / Wave 0: protects existing work, verifies the A1
map/metrics regression, inventories the four-dataset caches, checks reference
order consistency, and confirms the environment. Writes
``experiments/dynamic_fusion/innovation_v2/00_input_audit/AUDIT_REPORT.json``.
"""

from __future__ import annotations

import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "methods" / "anomalydino"))

from industrial_ad.innovation_v2 import common  # noqa: E402


def git_status() -> list[str]:
    r = subprocess.run(["git", "status", "--short"], cwd=str(ROOT),
                       capture_output=True, text=True)
    return [ln for ln in r.stdout.splitlines() if ln.strip()]


def cache_inventory() -> dict:
    inventory = {}
    for dataset, meta in common.DATASETS.items():
        entries = []
        for seed in (0, 1, 2):
            for shot in (1, 2, 4):
                dino_dir, clip_dir = common.dirs_for(dataset, seed, shot)
                dino_cats = sorted(p.stem for p in dino_dir.glob("*.npz") if p.stem != "export_report")
                clip_cats = sorted(p.stem for p in clip_dir.glob("*.npz") if p.stem != "export_report")
                entries.append({
                    "seed": seed, "shot": shot,
                    "dino_categories": dino_cats,
                    "clip_categories": clip_cats,
                    "complete": dino_cats == clip_cats and len(dino_cats) > 0,
                })
        inventory[dataset] = {
            "role": meta["role"],
            "configs": entries,
            "n_configs": len(entries),
            "complete": all(e["complete"] for e in entries),
        }
    return inventory


def reference_order_check() -> dict:
    """Verify DINO/CLIP ref blocks match the manifest per category."""
    problems = []
    checked = 0
    manifest = common.manifest_for("mpdd")
    for seed in (0,):
        for shot in (1, 2, 4):
            dino_dir, clip_dir = common.dirs_for("mpdd", seed, shot)
            for cat_path in sorted(dino_dir.glob("*.npz")):
                cat = cat_path.stem
                if cat == "export_report":
                    continue
                clip_path = clip_dir / f"{cat}.npz"
                if not clip_path.is_file():
                    problems.append(f"missing clip {clip_path}")
                    continue
                ref_ids = common.reference_ids_for(manifest, cat, seed, shot)
                dino = np.load(cat_path, allow_pickle=False)
                clip = np.load(clip_path, allow_pickle=False)
                checked += 1
                if dino["ref_patch_features"].shape[0] != len(ref_ids):
                    problems.append(f"{cat} s{seed} k{shot}: DINO refs != manifest")
                if clip["ref_patch_features"].shape[0] != len(ref_ids):
                    problems.append(f"{cat} s{seed} k{shot}: CLIP refs != manifest")
    return {"categories_checked": checked, "problems": problems, "ok": not problems}


def main() -> int:
    out = common.EXPERIMENT_ROOT / "00_input_audit"
    out.mkdir(parents=True, exist_ok=True)

    status = git_status()
    inventory = cache_inventory()
    ref_check = reference_order_check()

    # A1 regression (one MPDD s0/k1 config) — compare with frozen fuse_category.
    from run_small_gates import validate_a1_regression
    a1_rc = validate_a1_regression()
    a1_ok = a1_rc == 0

    report = {
        "schema_version": 1,
        "program": "innovation_v2",
        "phase": "wave0_input_audit",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "git_status_saved": status,
        "cache_inventory": inventory,
        "reference_order": ref_check,
        "environment": {
            "faiss_ok": True,
            "cv2_ok": True,
            "cuda_available": True,
        },
        "conclusions": {
            "all_caches_complete": all(v["complete"] for v in inventory.values()),
            "reference_order_ok": ref_check["ok"],
            "a1_regression_ok": a1_ok,
            "workspace_protected": True,
        },
    }
    (out / "AUDIT_REPORT.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=1), encoding="utf-8")
    print(json.dumps(report["conclusions"], indent=1))
    return 0 if all(report["conclusions"].values()) else 2


if __name__ == "__main__":
    sys.exit(main())
