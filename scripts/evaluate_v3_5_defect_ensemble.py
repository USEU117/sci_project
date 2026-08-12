"""Direction B evaluation: Compare defect ensemble text branch vs original text branch.

This script is CPU-only — it loads cached NPZ predictions and evaluates
V3.3 60:40 static fusion using either the original or defect-ensemble text branch.

Run AFTER `export_anomalyclip_defect_ensemble.py` has completed on GPU.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from evaluate_unified import aupro_fast
from industrial_ad.fusion.alignment import build_alignment_plan
from run_dynamic_fusion_v2_cache import load_cache, resize_maps
from sklearn.metrics import average_precision_score, roc_auc_score


STRIDE: int = 8


def compute_metrics(pixel_maps: np.ndarray, gt_masks: np.ndarray) -> dict:
    maps_s = pixel_maps[:, ::STRIDE, ::STRIDE]
    masks_s = gt_masks[:, ::STRIDE, ::STRIDE]
    flat_maps = maps_s.ravel()
    flat_labels = (masks_s.ravel() > 0.5).astype(np.int32)
    return {
        "pixel_auroc": float(roc_auc_score(flat_labels, flat_maps)),
        "pixel_ap": float(average_precision_score(flat_labels, flat_maps)),
        "pixel_aupro": float(aupro_fast(masks_s, maps_s)),
    }


def weighted_fusion(dino_maps, text_maps, masks, dino_w=0.60):
    """V3.3-style z-score calibrate + weighted average."""
    # Robust stats from normal images
    normal = ~masks.astype(bool).any(axis=(1, 2))
    if normal.sum() == 0:
        normal = slice(None)
    
    # DINO calibration
    d_vals = dino_maps[normal].ravel()
    d_center = float(np.median(d_vals))
    d_scale = float(np.subtract(*np.percentile(d_vals, [75, 25]))) + 1e-8
    dino_z = (dino_maps.astype(np.float64) - d_center) / d_scale
    
    # Text calibration
    t_vals = text_maps[normal].ravel()
    t_center = float(np.median(t_vals))
    t_scale = float(np.subtract(*np.percentile(t_vals, [75, 25]))) + 1e-8
    text_z = (text_maps.astype(np.float64) - t_center) / t_scale
    
    fused = dino_w * dino_z + (1.0 - dino_w) * text_z
    fused = fused * d_scale + d_center
    return fused


def load_categories() -> List[str]:
    manifest = json.loads(
        (ROOT / "data" / "splits" / "mpdd" / "manifest.json").read_text(encoding="utf-8")
    )
    return sorted(manifest["categories"])


def main() -> int:
    parser = argparse.ArgumentParser(description="Direction B: Evaluate defect ensemble text branch")
    parser.add_argument("--defect-ensemble-dir", type=Path, required=True,
                        help="Output dir from export_anomalyclip_defect_ensemble.py")
    parser.add_argument("--seed", type=int, default=0, choices=[0, 1, 2])
    parser.add_argument("--dino-weight", type=float, default=0.60)
    args = parser.parse_args()

    seed = args.seed
    v2_root = ROOT / "outputs" / "dynamic_fusion" / "v2_mpdd_predictions" / f"v2_mpdd_s{seed}_k1_full_v1"
    categories = load_categories()

    print(f"Direction B: Defect Ensemble Text Branch Evaluation")
    print(f"  Seed: {seed}, DINO weight: {args.dino_weight}")
    print(f"  Defect ensemble dir: {args.defect_ensemble_dir}")
    print()

    # Load original baselines from V3.3 report for comparison
    v3_3_report_path = ROOT / "experiments" / "dynamic_fusion" / "v3_3" / f"s{seed}_k1" / "report.json"
    v3_3_baseline = {}
    if v3_3_report_path.exists():
        v3_3 = json.loads(v3_3_report_path.read_text(encoding="utf-8"))
        if "baselines" in v3_3:
            v3_3_baseline = v3_3["baselines"]

    results_original = {}
    results_defect = {}

    for cat in categories:
        print(f"[{cat}]", end=" ", flush=True)

        # --- Load DINO (same for both) ---
        dino_path = v2_root / "anomalydino_visual" / f"{cat}.npz"
        if not dino_path.exists():
            print("SKIP (no DINO)")
            continue
        dino = load_cache(dino_path)
        ref_ids = np.asarray(dino["sample_ids"])
        dino_maps = np.asarray(dino["anomaly_maps"], dtype=np.float32)
        dino_masks = np.asarray(dino["imgs_masks"], dtype=np.uint8)
        ref_shape = dino_maps.shape[1:]

        dino_metrics = compute_metrics(dino_maps.astype(np.float64), dino_masks)

        # --- Load ORIGINAL text branch ---
        orig_path = v2_root / "anomalyclip_text" / f"{cat}.npz"
        orig_ap = None
        if orig_path.exists():
            orig = load_cache(orig_path)
            alignment = build_alignment_plan(ref_ids, np.asarray(orig["sample_ids"]))
            order = alignment.candidate_order
            orig_maps = resize_maps(
                np.asarray(orig["anomaly_maps"], dtype=np.float32)[order], ref_shape
            )
            orig_fused = weighted_fusion(dino_maps, orig_maps, dino_masks, args.dino_weight)
            orig_m = compute_metrics(orig_fused, dino_masks)
            orig_ap = orig_m["pixel_ap"]

        # --- Load DEFECT ENSEMBLE text branch ---
        defect_path = args.defect_ensemble_dir / f"{cat}.npz"
        defect_ap = None
        if defect_path.exists():
            defect = load_cache(defect_path)
            alignment = build_alignment_plan(ref_ids, np.asarray(defect["sample_ids"]))
            order = alignment.candidate_order
            defect_maps = resize_maps(
                np.asarray(defect["anomaly_maps"], dtype=np.float32)[order], ref_shape
            )
            defect_fused = weighted_fusion(dino_maps, defect_maps, dino_masks, args.dino_weight)
            defect_m = compute_metrics(defect_fused, dino_masks)
            defect_ap = defect_m["pixel_ap"]

        results_original[cat] = {
            "dino_ap": dino_metrics["pixel_ap"],
            "fused_ap": orig_ap,
            "delta_ap": round(orig_ap - dino_metrics["pixel_ap"], 6) if orig_ap is not None else None,
        }
        results_defect[cat] = {
            "dino_ap": dino_metrics["pixel_ap"],
            "fused_ap": defect_ap,
            "delta_ap": round(defect_ap - dino_metrics["pixel_ap"], 6) if defect_ap is not None else None,
        }

        orig_str = f"orig={orig_ap:.4f}" if orig_ap else "orig=N/A"
        defect_str = f"defect={defect_ap:.4f}" if defect_ap else "defect=N/A"
        delta_str = ""
        if orig_ap and defect_ap:
            delta_str = f" Δ={defect_ap - orig_ap:+.6f}"
        print(f"DINO={dino_metrics['pixel_ap']:.4f} {orig_str} {defect_str}{delta_str}", flush=True)

    # Summary
    print(f"\n{'='*80}")
    print(f"  DIRECTION B SUMMARY — Seed {seed}")
    print(f"{'='*80}")
    print(f"\n{'Category':<18} {'DINO AP':>8} {'Orig ΔAP':>10} {'Defect ΔAP':>10} {'Gain':>10}")
    print("-" * 60)

    orig_deltas, defect_deltas = [], []
    for cat in categories:
        o = results_original.get(cat, {})
        d = results_defect.get(cat, {})
        o_dap = o.get("delta_ap")
        d_dap = d.get("delta_ap")
        gain = ""
        if o_dap is not None and d_dap is not None:
            orig_deltas.append(o_dap)
            defect_deltas.append(d_dap)
            gain = f"{d_dap - o_dap:+.6f}"
        print(f"{cat:<18} {o.get('dino_ap',0):>8.4f} {o_dap or 'N/A':>10} {d_dap or 'N/A':>10} {gain:>10}")

    if orig_deltas:
        print("-" * 60)
        print(f"{'MEAN':<18} {'':>8} {np.mean(orig_deltas):>10.6f} {np.mean(defect_deltas):>10.6f} "
              f"{np.mean(defect_deltas) - np.mean(orig_deltas):>+10.6f}")

    # Save
    report = {
        "direction": "B_defect_ensemble",
        "seed": seed,
        "dino_weight": args.dino_weight,
        "original": results_original,
        "defect_ensemble": results_defect,
        "mean_original_dap": round(float(np.mean(orig_deltas)), 6) if orig_deltas else None,
        "mean_defect_dap": round(float(np.mean(defect_deltas)), 6) if defect_deltas else None,
        "gain_vs_original": (
            round(float(np.mean(defect_deltas) - np.mean(orig_deltas)), 6)
            if orig_deltas and defect_deltas else None
        ),
    }
    out_path = ROOT / "experiments" / "dynamic_fusion" / "v3_5_defect_ensemble" / f"s{seed}_eval.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nSaved: {out_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
