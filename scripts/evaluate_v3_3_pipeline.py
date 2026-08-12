"""V3.3 Automated Fusion Pipeline — compare 3 strategies end-to-end.

Loads all cached branch predictions (zero extra GPU inference), runs every
strategy variant, performs leave-one-out cross-validation, and outputs a
structured comparison report.

Strategies:
  1. weighted_ensemble  — z-score calibrate + weighted average
  2. max_z_selection    — per-pixel most-confident branch
  3. two_stage_calibrated — AdaptCLIP internal → calibrate → merge with DINO
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

# --- project root & path setup ---
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from evaluate_unified import aupro_fast
from industrial_ad.fusion.alignment import build_alignment_plan
from run_dynamic_fusion_v2_cache import load_cache, resize_maps
from sklearn.metrics import average_precision_score, roc_auc_score

from industrial_ad.fusion.v3_3_strategies import (
    BranchData,
    build_strategy_variants,
    run_fusion,
)


# ======================================================================
# Constants
# ======================================================================

STRIDE: int = 8  # spatial stride for metric computation


def _get_source_roots(seed: int) -> dict:
    """Build SOURCE_ROOTS dict for a specific seed."""
    v2_dir = ROOT / "outputs" / "dynamic_fusion" / "v2_mpdd_predictions" / f"v2_mpdd_s{seed}_k1_full_v1"
    v3_dir = ROOT / "outputs" / "dynamic_fusion" / "v3_2_branches" / f"v3_2_mpdd_s{seed}_k1"
    return {
        "anomalydino_visual": v2_dir,
        "anomalyclip_text": v2_dir,
        "adaptclip_textual_adapter": v3_dir,
        "adaptclip_visual_adapter": v3_dir,
        "adaptclip_pq_adapter": v3_dir,
        "adaptclip_fused": v3_dir,
    }


# Subdirectory mapping within each source root
BRANCH_SUBDIRS = {
    "anomalydino_visual": "anomalydino_visual",
    "anomalyclip_text": "anomalyclip_text",
    "adaptclip_textual_adapter": "adaptclip_textual_adapter",
    "adaptclip_visual_adapter": "adaptclip_visual_adapter",
    "adaptclip_pq_adapter": "adaptclip_pq_adapter",
    "adaptclip_fused": "",  # root-level NPZ
}


# ======================================================================
# Branch loading
# ======================================================================

def load_categories() -> List[str]:
    manifest = json.loads(
        (ROOT / "data" / "splits" / "mpdd" / "manifest.json").read_text(encoding="utf-8")
    )
    return sorted(manifest["categories"])


def load_branches_for_category(category: str, seed: int = 0) -> Dict[str, BranchData]:
    """Load all available branch caches for one category, aligned to the
    anomalydino_visual reference order.

    Returns dict of branch_name -> BranchData (all with same sample ordering).
    """
    src = _get_source_roots(seed)
    branches: Dict[str, BranchData] = {}

    # ---- Step 1: load DINO visual (reference) ----
    dino_root = src["anomalydino_visual"]
    dino_path = dino_root / BRANCH_SUBDIRS["anomalydino_visual"] / f"{category}.npz"
    if not dino_path.exists():
        print(f"  [SKIP] missing DINO cache: {dino_path}", flush=True)
        return {}
    dino = load_cache(dino_path)
    ref_ids = np.asarray(dino["sample_ids"])
    dino_maps = np.asarray(dino["anomaly_maps"], dtype=np.float32)
    dino_masks = np.asarray(dino["imgs_masks"], dtype=np.uint8)
    dino_gts = np.asarray(dino["gt_sp"], dtype=np.uint8)

    branches["anomalydino_visual"] = BranchData(
        name="anomalydino_visual",
        anomaly_maps=dino_maps,
        image_scores=np.zeros(len(ref_ids), dtype=np.float32),
        gt_labels=dino_gts,
        gt_masks=dino_masks,
        sample_ids=ref_ids,
    )

    ref_shape = dino_maps.shape[1:]  # (H, W) for DINO

    # ---- Step 2: load AnomalyCLIP text (same source root, different subdir) ----
    aclip_root = src["anomalyclip_text"]
    aclip_path = aclip_root / BRANCH_SUBDIRS["anomalyclip_text"] / f"{category}.npz"
    if aclip_path.exists():
        aclip = load_cache(aclip_path)
        alignment = build_alignment_plan(ref_ids, np.asarray(aclip["sample_ids"]))
        order = alignment.candidate_order
        maps_raw = np.asarray(aclip["anomaly_maps"], dtype=np.float32)[order]
        maps_resized = resize_maps(maps_raw, ref_shape)
        branches["anomalyclip_text"] = BranchData(
            name="anomalyclip_text",
            anomaly_maps=np.asarray(maps_resized, dtype=np.float32),
            image_scores=np.zeros(len(ref_ids), dtype=np.float32),
            gt_labels=dino_gts,
            gt_masks=dino_masks,
            sample_ids=ref_ids,
        )
    else:
        print(f"  [WARN] missing AnomalyCLIP text cache: {aclip_path}", flush=True)

    # ---- Step 3: load AdaptCLIP 3 adapter branches ----
    adapt_root = src["adaptclip_textual_adapter"]
    for bname in ["adaptclip_textual_adapter", "adaptclip_visual_adapter", "adaptclip_pq_adapter"]:
        bpath = adapt_root / BRANCH_SUBDIRS[bname] / f"{category}.npz"
        if not bpath.exists():
            print(f"  [WARN] missing {bname} cache: {bpath}", flush=True)
            continue
        cache = load_cache(bpath)
        alignment = build_alignment_plan(ref_ids, np.asarray(cache["sample_ids"]))
        order = alignment.candidate_order
        maps_raw = np.asarray(cache["anomaly_maps"], dtype=np.float32)[order]
        maps_resized = resize_maps(maps_raw, ref_shape)
        branches[bname] = BranchData(
            name=bname,
            anomaly_maps=np.asarray(maps_resized, dtype=np.float32),
            image_scores=np.zeros(len(ref_ids), dtype=np.float32),
            gt_labels=dino_gts,
            gt_masks=dino_masks,
            sample_ids=ref_ids,
        )

    # ---- Step 4: load AdaptCLIP fused (root-level) ----
    ac_fused_root = src["adaptclip_fused"]
    ac_fused_path = ac_fused_root / f"{category}.npz"
    if ac_fused_path.exists():
        cache = load_cache(ac_fused_path)
        alignment = build_alignment_plan(ref_ids, np.asarray(cache["sample_ids"]))
        order = alignment.candidate_order
        maps_raw = np.asarray(cache["anomaly_maps"], dtype=np.float32)[order]
        maps_resized = resize_maps(maps_raw, ref_shape)
        branches["adaptclip_fused"] = BranchData(
            name="adaptclip_fused",
            anomaly_maps=np.asarray(maps_resized, dtype=np.float32),
            image_scores=np.zeros(len(ref_ids), dtype=np.float32),
            gt_labels=dino_gts,
            gt_masks=dino_masks,
            sample_ids=ref_ids,
        )

    return branches


# ======================================================================
# Metrics
# ======================================================================

def compute_metrics(pixel_maps: np.ndarray, gt_masks: np.ndarray) -> dict:
    """Compute AUROC, AP, AUPRO for one category."""
    maps_strided = pixel_maps[:, ::STRIDE, ::STRIDE]
    masks_strided = gt_masks[:, ::STRIDE, ::STRIDE]

    flat_maps = maps_strided.ravel()
    flat_labels = (masks_strided.ravel() > 0.5).astype(np.int32)

    auroc = float(roc_auc_score(flat_labels, flat_maps))
    ap = float(average_precision_score(flat_labels, flat_maps))
    aupro = float(aupro_fast(masks_strided, maps_strided))

    return {"pixel_auroc": auroc, "pixel_ap": ap, "pixel_aupro": aupro}


def compute_baseline_metrics(branches: Dict[str, BranchData]) -> dict:
    """Compute single-branch baselines for comparison."""
    dino = branches.get("anomalydino_visual")
    aclip = branches.get("anomalyclip_text")
    results = {}
    if dino is not None:
        results["anomalydino_visual"] = compute_metrics(
            dino.anomaly_maps.astype(np.float64), dino.gt_masks
        )
    if aclip is not None:
        results["anomalyclip_text"] = compute_metrics(
            aclip.anomaly_maps.astype(np.float64), aclip.gt_masks
        )
    return results


# ======================================================================
# Per-category evaluation
# ======================================================================

def evaluate_category(
    category: str,
    branches: Dict[str, BranchData],
    variants: List[dict],
) -> List[dict]:
    """Run all strategy variants on one category, return list of result dicts."""
    dino = branches.get("anomalydino_visual")
    if dino is None:
        return []

    results = []
    gt_masks = dino.gt_masks

    for vi, variant in enumerate(variants):
        try:
            fused = run_fusion(variant, branches)
        except Exception as e:
            print(f"  [ERR] {variant['variant_name']}: {e}", flush=True)
            continue

        metrics = compute_metrics(fused, gt_masks)
        dino_metrics = compute_metrics(
            dino.anomaly_maps.astype(np.float64), gt_masks
        )

        results.append({
            "category": category,
            "strategy": variant["strategy"],
            "variant": variant["variant_name"],
            **metrics,
            "delta_auroc": round(metrics["pixel_auroc"] - dino_metrics["pixel_auroc"], 6),
            "delta_ap": round(metrics["pixel_ap"] - dino_metrics["pixel_ap"], 6),
            "delta_aupro": round(metrics["pixel_aupro"] - dino_metrics["pixel_aupro"], 6),
        })

    return results


# ======================================================================
# Leave-one-out cross-validation
# ======================================================================

def leave_one_out_select(
    all_results: Dict[str, List[dict]],
    categories: List[str],
) -> Tuple[dict, dict]:
    """For each held-out category, select best variant on the other 5 categories.

    Returns:
      selections: {category: {strategy, variant, ...}}
      heldout_metrics: {category: metrics of selected variant}
    """
    selections = {}
    heldout_metrics = {}

    for heldout in categories:
        # Aggregate metrics over OTHER categories
        train_results = []
        for cat in categories:
            if cat == heldout:
                continue
            train_results.extend(all_results.get(cat, []))

        if not train_results:
            selections[heldout] = {"strategy": "baseline", "variant": "anomalydino_visual"}
            continue

        # Group by variant and compute mean delta_ap
        by_variant: Dict[str, List[float]] = defaultdict(list)
        for r in train_results:
            key = f"{r['strategy']}::{r['variant']}"
            by_variant[key].append(r["delta_ap"])

        best_key = max(by_variant, key=lambda k: np.mean(by_variant[k]))
        best_strategy, best_variant = best_key.split("::", 1)

        # Find the heldout result for this variant
        heldout_result = None
        for r in all_results.get(heldout, []):
            if r["strategy"] == best_strategy and r["variant"] == best_variant:
                heldout_result = r
                break

        selections[heldout] = {
            "strategy": best_strategy,
            "variant": best_variant,
            "mean_train_delta_ap": round(float(np.mean(by_variant[best_key])), 6),
        }
        heldout_metrics[heldout] = heldout_result

    return selections, heldout_metrics


# ======================================================================
# Report generation
# ======================================================================

def build_report(
    categories: List[str],
    all_results: Dict[str, List[dict]],
    baselines: Dict[str, dict],
    selections: dict,
    heldout_metrics: dict,
) -> dict:
    """Build structured report dict and print summary."""

    # Collect held-out deltas
    heldout_deltas = []
    for cat in categories:
        hm = heldout_metrics.get(cat)
        if hm:
            heldout_deltas.append(hm["delta_ap"])
        else:
            heldout_deltas.append(0.0)

    mean_delta = float(np.mean(heldout_deltas))
    positive_count = sum(1 for d in heldout_deltas if d > 0)

    # Strategy-level aggregation: best variant per strategy averaged across categories
    strategy_summary: Dict[str, dict] = {}
    for strategy in ["weighted_ensemble", "max_z_selection", "two_stage_calibrated"]:
        # For each category, find the variant with max delta_ap for this strategy
        strategy_deltas = []
        for cat in categories:
            cat_results = [r for r in all_results.get(cat, []) if r["strategy"] == strategy]
            if cat_results:
                best = max(cat_results, key=lambda r: r["delta_ap"])
                strategy_deltas.append(best["delta_ap"])
            else:
                strategy_deltas.append(0.0)
        strategy_summary[strategy] = {
            "mean_delta_ap": round(float(np.mean(strategy_deltas)), 6),
            "positive_categories": sum(1 for d in strategy_deltas if d > 0),
            "per_category": {cat: round(d, 6) for cat, d in zip(categories, strategy_deltas)},
        }

    report = {
        "pipeline": "v3_3",
        "dataset": "mpdd",
        "categories": categories,
        "stride": STRIDE,
        "baselines": baselines,
        "leave_one_out_selections": selections,
        "heldout_summary": {
            "mean_delta_ap": round(mean_delta, 6),
            "positive_categories": positive_count,
            "per_category_delta_ap": {
                cat: round(heldout_deltas[i], 6) for i, cat in enumerate(categories)
            },
        },
        "strategy_summary": strategy_summary,
        "all_results": all_results,
    }

    return report


def print_summary(report: dict):
    """Print a human-readable summary table."""
    categories = report["categories"]
    bl = report["baselines"]

    print("\n" + "=" * 90)
    print("  V3.3 FUSION PIPELINE — COMPARISON REPORT")
    print("=" * 90)

    # Baselines
    print(f"\n{'Branch':<30} {'AUROC':>8} {'AP':>8} {'AUPRO':>8}")
    print("-" * 56)
    for bname, bm in bl.items():
        print(f"{bname:<30} {bm['pixel_auroc']:>8.4f} {bm['pixel_ap']:>8.4f} {bm['pixel_aupro']:>8.4f}")

    # Leave-one-out selections
    print(f"\n{'='*90}")
    print("  LEAVE-ONE-OUT CROSS-VALIDATION SELECTIONS")
    print(f"{'='*90}")
    print(f"\n{'Category':<20} {'Selected Strategy':<28} {'Selected Variant':<30} {'Train ΔAP':>10} {'Heldout ΔAP':>10}")
    print("-" * 100)
    for cat in categories:
        sel = report["leave_one_out_selections"].get(cat, {})
        hm = report["heldout_summary"]["per_category_delta_ap"].get(cat, 0)
        print(
            f"{cat:<20} {sel.get('strategy','?'):<28} {sel.get('variant','?'):<30} "
            f"{sel.get('mean_train_delta_ap',0):>10.6f} {hm:>10.6f}"
        )

    hs = report["heldout_summary"]
    print("-" * 100)
    print(f"{'HELDOUT MEAN':<20} {'':<28} {'':<30} {'':>10} {hs['mean_delta_ap']:>10.6f}")
    print(f"Positive categories: {hs['positive_categories']}/{len(categories)}")
    gate = "PASSED" if hs["positive_categories"] >= len(categories) // 2 else "FAILED"
    print(f"Gate B: {gate}")

    # Strategy summary
    print(f"\n{'='*90}")
    print("  STRATEGY-LEVEL COMPARISON (best variant per strategy)")
    print(f"{'='*90}")
    print(f"\n{'Strategy':<30} {'Mean ΔAP':>10} {'Pos. Cats':>12}")
    print("-" * 56)
    ss = report["strategy_summary"]
    for sname, sinfo in ss.items():
        print(f"{sname:<30} {sinfo['mean_delta_ap']:>10.6f} {sinfo['positive_categories']:>12}")

    # Per-category per-strategy detail
    print(f"\n{'='*90}")
    print("  PER-CATEGORY DETAIL (ΔAP vs DINO visual baseline)")
    print(f"{'='*90}")
    header = f"{'Category':<18}" + "".join(
        f" {s:<28}" for s in ss
    )
    print(f"\n{header}")
    print("-" * (18 + 28 * len(ss)))
    for cat in categories:
        row = f"{cat:<18}"
        for sname in ss:
            row += f" {ss[sname]['per_category'].get(cat, 0):>27.6f}"
        print(row)


# ======================================================================
# Main
# ======================================================================

def main() -> int:
    parser = argparse.ArgumentParser(description="V3.3 Fusion Pipeline")
    parser.add_argument("--seed", type=int, default=0, choices=[0, 1, 2],
                        help="Seed for train sample selection (default: 0)")
    args = parser.parse_args()
    seed: int = args.seed

    categories = load_categories()
    print(f"Dataset: mpdd, seed={seed}, k=1, categories: {categories}", flush=True)
    variants = build_strategy_variants()
    print(f"Strategy variants to evaluate: {len(variants)}", flush=True)

    # --- Phase 1: load all branch caches ---
    print("\n[Phase 1] Loading branch caches...", flush=True)
    all_branches: Dict[str, Dict[str, BranchData]] = {}
    for cat in categories:
        print(f"  {cat}...", end=" ", flush=True)
        branches = load_branches_for_category(cat, seed=seed)
        all_branches[cat] = branches
        print(f"{len(branches)} branches loaded", flush=True)

    # --- Phase 2: compute baselines ---
    print("\n[Phase 2] Computing single-branch baselines...", flush=True)
    all_baselines: Dict[str, dict] = {}
    for cat in categories:
        all_baselines[cat] = compute_baseline_metrics(all_branches[cat])

    # Aggregate baselines across categories
    baseline_agg: Dict[str, dict] = {}
    for bname in ["anomalydino_visual", "anomalyclip_text"]:
        metrics_keys = ["pixel_auroc", "pixel_ap", "pixel_aupro"]
        agg = {}
        for key in metrics_keys:
            vals = [all_baselines[cat].get(bname, {}).get(key, 0) for cat in categories]
            agg[key] = round(float(np.mean(vals)), 6)
        baseline_agg[bname] = agg

    # --- Phase 3: run all strategy variants per category ---
    print(f"\n[Phase 3] Running {len(variants)} strategy variants x {len(categories)} categories...", flush=True)
    all_results: Dict[str, List[dict]] = {}
    total = len(variants) * len(categories)
    done = 0
    for cat in categories:
        print(f"  {cat}...", end=" ", flush=True)
        results = evaluate_category(cat, all_branches[cat], variants)
        all_results[cat] = results
        done += len(variants)
        # Show best delta for this category
        if results:
            best = max(results, key=lambda r: r["delta_ap"])
            print(f"{len(results)} variants, best: {best['variant']} (ΔAP={best['delta_ap']:.6f})", flush=True)
        else:
            print("no results", flush=True)

    # --- Phase 4: leave-one-out cross-validation ---
    print("\n[Phase 4] Leave-one-out cross-validation...", flush=True)
    selections, heldout_metrics = leave_one_out_select(all_results, categories)

    # --- Phase 5: build report ---
    print("\n[Phase 5] Building report...", flush=True)
    report = build_report(categories, all_results, baseline_agg, selections, heldout_metrics)

    # Save JSON report
    out_dir = ROOT / "experiments" / "dynamic_fusion" / "v3_3" / f"s{seed}_k1"
    out_dir.mkdir(parents=True, exist_ok=True)
    report_path = out_dir / "report.json"
    report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Report saved to: {report_path}")

    # Print summary
    print_summary(report)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
