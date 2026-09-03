"""V12-EARLY-FUSION Stage 0 formal deliverables builder (doc 23 8.1).

Builds (under experiments/dynamic_fusion/innovation_v12_early_fusion/):
  01_multilayer_cache/CACHE_MANIFEST.json   - aggregates outputs/.../ml_* export reports
  01_multilayer_cache/ALIGNMENT_REPORT.json - dino<->clip sample-id order, grids, refs==shot
  01_multilayer_cache/DEEPEST_PARITY_REPORT.json - raw feature maxabs (ml cache vs frozen A1
      cache) + map-level pooled Pixel-AP diff (probe concat_D11C24 vs A1_REFERENCE_MAPS) + verdict

Run (.venv-patchcore, CPU) AFTER the stage0 probe finished (heavy transient loads):
  .venv-patchcore\\Scripts\\python.exe scripts\\innovation_v12_new_observables\\run_r3_ef_stage0_deliverables.py
"""

from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

CATEGORIES = ["bracket_black", "bracket_brown", "bracket_white", "connector", "metal_plate", "tubes"]
SHOTS = [1, 2, 4]
ML_ROOT = ROOT / "outputs/dynamic_fusion/v12_early_fusion"
BASE = ROOT / "outputs/dynamic_fusion/v3_direction_a"
EXP = ROOT / "experiments/dynamic_fusion/innovation_v12_early_fusion"
PROBE_DIR = EXP / "02_stage0_probe"


def build_cache_manifest() -> dict:
    entries = []
    for branch in ("dino", "clip"):
        for shot in SHOTS:
            rep_path = ML_ROOT / f"ml_{branch}_s0_k{shot}/export_report.json"
            rep = json.loads(rep_path.read_text(encoding="utf-8"))
            entries.append({
                "cache_dir": f"ml_{branch}_s0_k{shot}",
                "branch": branch, "shot": shot,
                "layer_ids": rep["layer_ids"],
                "manifest_sha256": rep["manifest_sha256"],
                "categories": [
                    {k: c[k] for k in ("category", "layers", "n", "refs", "grid", "output", "sha256")}
                    for c in rep["categories"]
                ],
                "created_at_utc": rep["created_at_utc"],
            })
    return {
        "protocol": "experiments/dynamic_fusion/innovation_v12_early_fusion/00_protocol/PROTOCOL_FROZEN.yaml",
        "dataset": "MPDD development seed0",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "cache_sets": entries,
        "n_categories": len(CATEGORIES),
        "n_shots": len(SHOTS),
    }


def build_alignment_report() -> dict:
    checks = []
    ok = True
    for shot in SHOTS:
        for cat in CATEGORIES:
            zd = np.load(ML_ROOT / f"ml_dino_s0_k{shot}/{cat}.npz", allow_pickle=False)
            zc = np.load(ML_ROOT / f"ml_clip_s0_k{shot}/{cat}.npz", allow_pickle=False)
            sid_d = np.asarray(zd["sample_ids"])
            sid_c = np.asarray(zc["sample_ids"])
            gd = tuple(int(v) for v in zd["grid_size"])
            gc = tuple(int(v) for v in zc["grid_size"])
            same_order = bool(np.array_equal(sid_d, sid_c))
            # ref_patch_features: [n_layers, K, H, W, D] -> refs are on dim 1
            refs_d = np.asarray(zd["ref_patch_features"]).shape[1]
            refs_c = np.asarray(zc["ref_patch_features"]).shape[1]
            masks = np.asarray(zd["imgs_masks"])
            m448 = masks.shape[1] == 448 and masks.shape[2] == 448
            row_ok = same_order and refs_d == shot and refs_c == shot and m448
            ok = ok and row_ok
            checks.append({
                "category": cat, "shot": shot,
                "sample_order_dino_vs_clip_identical": same_order,
                "grids": {"dino": gd, "clip": gc},
                "refs": {"dino": refs_d, "clip": refs_c, "shot": shot},
                "masks_448": m448,
                "pass": row_ok,
            })
            del zd, zc, sid_d, sid_c, masks
    return {"alignment_pass": bool(ok), "checks": checks}


def raw_parity(shot: int, cat: str) -> dict:
    """Deepest layer raw feature parity ml-cache vs frozen A1 caches."""
    from industrial_ad.innovation_v10_portfolio.common import load_features

    z = np.load(ML_ROOT / f"ml_dino_s0_k{shot}/{cat}.npz", allow_pickle=False)
    d_feat = np.asarray(z["patch_features"]); d_ref = np.asarray(z["ref_patch_features"])
    del z
    zc = np.load(ML_ROOT / f"ml_clip_s0_k{shot}/{cat}.npz", allow_pickle=False)
    c_feat = np.asarray(zc["patch_features"]); c_ref = np.asarray(zc["ref_patch_features"])
    del zc
    a1_d = load_features(BASE / f"features_vitb14_s0_k{shot}/anomalydino_visual/{cat}.npz")
    a1_c = load_features(BASE / f"features_s0_k{shot}/anomalyclip_text/{cat}.npz")
    out = {
        "category": cat, "shot": shot,
        "dino_L11_maxabs": float(np.abs(d_feat[:, 2] - np.asarray(a1_d["patch_features"])).max()),
        "dino_L11_ref_maxabs": float(np.abs(d_ref[2] - np.asarray(a1_d["ref_patch_features"])).max()),
        "clip_L24_maxabs": float(np.abs(c_feat[:, 3] - np.asarray(a1_c["patch_features"])).max()),
        "clip_L24_ref_maxabs": float(np.abs(c_ref[3] - np.asarray(a1_c["ref_patch_features"])).max()),
    }
    del d_feat, d_ref, c_feat, c_ref, a1_d, a1_c
    return out


def build_deepest_parity_report() -> dict:
    a1ref = json.loads((PROBE_DIR / "A1_REFERENCE_MAPS.json").read_text(encoding="utf-8"))
    a1map = {f"{r['category']}|k{r['shot']}": r["pixel_ap_56"] for r in a1ref["rows"]}
    rows = []
    for shot in SHOTS:
        # probe internal A1-equivalent AP (concat_D11C24 built from ml caches)
        sfile = PROBE_DIR / f"STAGE0_RESULT_k{shot}.json"
        ofile = PROBE_DIR / f"ORACLE_HEADROOM_k{shot}.json"
        if not (sfile.exists() and ofile.exists()):
            continue  # shot not probed yet
        oracle = json.loads(ofile.read_text(encoding="utf-8"))
        for cat in CATEGORIES:
            par = raw_parity(shot, cat)
            probe_cat = next(r for r in oracle["per_category"] if r["category"] == cat)
            a1_ap = probe_cat["a1_ap"]
            harness_ap = a1map[f"{cat}|k{shot}"]
            d_ap = (a1_ap - harness_ap) if (a1_ap is not None and harness_ap == harness_ap) else None
            rows.append({
                **par,
                "probe_a1_ap_56": a1_ap,
                "harness_a1_ap_56": round(harness_ap, 6),
                "map_ap_diff": round(d_ap, 6) if d_ap is not None else None,
            })
    raw_max = {"dino_L11": max(r["dino_L11_maxabs"] for r in rows),
               "clip_L24": max(r["clip_L24_maxabs"] for r in rows)}
    map_max = max((abs(r["map_ap_diff"]) for r in rows if r["map_ap_diff"] is not None), default=1.0)
    return {
        "raw_parity_max_abs": {k: round(v, 8) for k, v in raw_max.items()},
        "raw_gate_lt_1e-5": bool(raw_max["dino_L11"] < 1e-5 and raw_max["clip_L24"] < 1e-5),
        "map_parity_max_abs": round(map_max, 6),
        "map_gate_lt_1e-4": bool(map_max < 1e-4),
        "rows": rows,
        "note": ("GPU forward determinism is ~1e-3 across sessions (observed); raw 1e-5 gate is "
                 "expected to FAIL for cross-session caches; map-level Pixel-AP parity is the "
                 "operational gate per doc 23 relaxation."),
    }


def main() -> int:
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--fast", action="store_true",
                        help="rebuild CACHE_MANIFEST + ALIGNMENT only (skip raw parity recompute)")
    args = parser.parse_args()
    c1 = EXP / "01_multilayer_cache"
    c1.mkdir(parents=True, exist_ok=True)
    (c1 / "CACHE_MANIFEST.json").write_text(
        json.dumps(build_cache_manifest(), ensure_ascii=False, indent=1), encoding="utf-8")
    (c1 / "ALIGNMENT_REPORT.json").write_text(
        json.dumps(build_alignment_report(), ensure_ascii=False, indent=1), encoding="utf-8")
    if not args.fast:
        (c1 / "DEEPEST_PARITY_REPORT.json").write_text(
            json.dumps(build_deepest_parity_report(), ensure_ascii=False, indent=1), encoding="utf-8")
    print("deliverables written", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
