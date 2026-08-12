"""V3.5 Image-level Hierarchical Fusion — Evaluation Script.

Tests Direction C strategies on MPDD.

Key difference from V3.3:
  - Loads pr_sp as image_scores (previously ignored)
  - Per-image weight gating instead of global static weight
  - Includes oracle upper bound for image-level gating
  - Evaluates 3 strategies: discrete_gate, continuous_gate, agreement_gate
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from evaluate_unified import aupro_fast
from industrial_ad.fusion.alignment import build_alignment_plan
from run_dynamic_fusion_v2_cache import load_cache, resize_maps
from sklearn.metrics import average_precision_score, roc_auc_score

from industrial_ad.fusion.v3_5_strategies import (
    BranchData,
    build_v3_5_variants,
    oracle_image_gate_fusion,
    run_fusion,
)


STRIDE: int = 8


# ======================================================================
# Data loading
# ======================================================================

def load_categories() -> List[str]:
    manifest = json.loads(
        (ROOT / "data" / "splits" / "mpdd" / "manifest.json").read_text(encoding="utf-8")
    )
    return sorted(manifest["categories"])


def _get_v2_root(seed: int) -> Path:
    return ROOT / "outputs" / "dynamic_fusion" / "v2_mpdd_predictions" / f"v2_mpdd_s{seed}_k1_full_v1"


def load_branches_for_category(category: str, seed: int = 0) -> Dict[str, BranchData]:
    """Load DINO and AnomalyCLIP branches with image_scores from pr_sp."""
    v2_root = _get_v2_root(seed)
    branches: Dict[str, BranchData] = {}

    # ---- DINO visual (reference) ----
    dino_path = v2_root / "anomalydino_visual" / f"{category}.npz"
    if not dino_path.exists():
        print(f"  [SKIP] missing DINO cache: {dino_path}", flush=True)
        return {}
    dino = load_cache(dino_path)
    ref_ids = np.asarray(dino["sample_ids"])
    dino_maps = np.asarray(dino["anomaly_maps"], dtype=np.float32)
    dino_masks = np.asarray(dino["imgs_masks"], dtype=np.uint8)
    dino_gts = np.asarray(dino["gt_sp"], dtype=np.uint8)
    dino_img_scores = np.asarray(dino["pr_sp"], dtype=np.float32)

    branches["anomalydino_visual"] = BranchData(
        name="anomalydino_visual",
        anomaly_maps=dino_maps,
        image_scores=dino_img_scores,
        gt_labels=dino_gts,
        gt_masks=dino_masks,
        sample_ids=ref_ids,
    )

    ref_shape = dino_maps.shape[1:]

    # ---- AnomalyCLIP text ----
    aclip_path = v2_root / "anomalyclip_text" / f"{category}.npz"
    if aclip_path.exists():
        aclip = load_cache(aclip_path)
        alignment = build_alignment_plan(ref_ids, np.asarray(aclip["sample_ids"]))
        order = alignment.candidate_order
        maps_raw = np.asarray(aclip["anomaly_maps"], dtype=np.float32)[order]
        maps_resized = resize_maps(maps_raw, ref_shape)
        aclip_img_scores = np.asarray(aclip["pr_sp"], dtype=np.float32)[order]

        branches["anomalyclip_text"] = BranchData(
            name="anomalyclip_text",
            anomaly_maps=np.asarray(maps_resized, dtype=np.float32),
            image_scores=aclip_img_scores,
            gt_labels=dino_gts,
            gt_masks=dino_masks,
            sample_ids=ref_ids,
        )
    else:
        print(f"  [WARN] missing AnomalyCLIP text cache: {aclip_path}", flush=True)

    return branches


# ======================================================================
# Metrics
# ======================================================================

def compute_metrics(pixel_maps: np.ndarray, gt_masks: np.ndarray) -> dict:
    maps_strided = pixel_maps[:, ::STRIDE, ::STRIDE]
    masks_strided = gt_masks[:, ::STRIDE, ::STRIDE]
    flat_maps = maps_strided.ravel()
    flat_labels = (masks_strided.ravel() > 0.5).astype(np.int32)
    return {
        "pixel_auroc": float(roc_auc_score(flat_labels, flat_maps)),
        "pixel_ap": float(average_precision_score(flat_labels, flat_maps)),
        "pixel_aupro": float(aupro_fast(masks_strided, maps_strided)),
    }


def compute_baseline_metrics(branches: Dict[str, BranchData]) -> dict:
    results = {}
    for bname in ["anomalydino_visual", "anomalyclip_text"]:
        bdata = branches.get(bname)
        if bdata is not None:
            results[bname] = compute_metrics(
                bdata.anomaly_maps.astype(np.float64), bdata.gt_masks
            )
    return results


# ======================================================================
# Evaluate one category
# ======================================================================

def evaluate_category(
    category: str,
    branches: Dict[str, BranchData],
    variants: List[dict],
) -> Tuple[List[dict], dict]:
    """Run all V3.5 variants + oracle on one category."""
    dino = branches.get("anomalydino_visual")
    text = branches.get("anomalyclip_text")
    if dino is None or text is None:
        return [], {}

    results = []
    gt_masks = dino.gt_masks
    dino_metrics = compute_metrics(dino.anomaly_maps.astype(np.float64), gt_masks)

    for vi, variant in enumerate(variants):
        try:
            fused = run_fusion(variant, dino, text)
        except Exception as e:
            print(f"  [ERR] {variant['variant_name']}: {e}", flush=True)
            continue

        metrics = compute_metrics(fused, gt_masks)
        results.append({
            "category": category,
            "strategy": variant["strategy"],
            "variant": variant["variant_name"],
            **metrics,
            "delta_auroc": round(metrics["pixel_auroc"] - dino_metrics["pixel_auroc"], 6),
            "delta_ap": round(metrics["pixel_ap"] - dino_metrics["pixel_ap"], 6),
            "delta_aupro": round(metrics["pixel_aupro"] - dino_metrics["pixel_aupro"], 6),
        })

    # ---- Oracle upper bound ----
    oracle_maps, oracle_stats = oracle_image_gate_fusion(dino, text)
    oracle_metrics = compute_metrics(oracle_maps, gt_masks)
    oracle_result = {
        "category": category,
        "strategy": "oracle",
        "variant": "oracle_image_gate",
        **oracle_metrics,
        "delta_auroc": round(oracle_metrics["pixel_auroc"] - dino_metrics["pixel_auroc"], 6),
        "delta_ap": round(oracle_metrics["pixel_ap"] - dino_metrics["pixel_ap"], 6),
        "delta_aupro": round(oracle_metrics["pixel_aupro"] - dino_metrics["pixel_aupro"], 6),
        "oracle_stats": oracle_stats,
    }

    return results, oracle_result


# ======================================================================
# Leave-one-out CV
# ======================================================================

def leave_one_out_select(
    all_results: Dict[str, List[dict]],
    categories: List[str],
) -> Tuple[dict, dict]:
    selections = {}
    heldout_metrics = {}

    for heldout in categories:
        train_results = []
        for cat in categories:
            if cat == heldout:
                continue
            train_results.extend(all_results.get(cat, []))

        if not train_results:
            selections[heldout] = {"strategy": "baseline", "variant": "anomalydino_visual"}
            continue

        by_variant: Dict[str, List[float]] = defaultdict(list)
        for r in train_results:
            key = f"{r['strategy']}::{r['variant']}"
            by_variant[key].append(r["delta_ap"])

        best_key = max(by_variant, key=lambda k: np.mean(by_variant[k]))
        best_strategy, best_variant = best_key.split("::", 1)

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
# Report
# ======================================================================

def build_report(
    categories: List[str],
    all_results: Dict[str, List[dict]],
    all_oracles: Dict[str, dict],
    baselines: Dict[str, dict],
    selections: dict,
    heldout_metrics: dict,
    seed: int,
) -> dict:
    # Held-out summary
    heldout_deltas = []
    for cat in categories:
        hm = heldout_metrics.get(cat)
        heldout_deltas.append(hm["delta_ap"] if hm else 0.0)
    mean_delta = float(np.mean(heldout_deltas))
    positive_count = sum(1 for d in heldout_deltas if d > 0)

    # V3.3 static baseline (first variant, strategy=v3_3_static)
    v3_3_deltas = []
    for cat in categories:
        for r in all_results.get(cat, []):
            if r["strategy"] == "v3_3_static":
                v3_3_deltas.append(r["delta_ap"])
                break
        else:
            v3_3_deltas.append(0.0)
    v3_3_mean = float(np.mean(v3_3_deltas))

    # Oracle summary
    oracle_deltas = []
    for cat in categories:
        o = all_oracles.get(cat, {})
        oracle_deltas.append(o.get("delta_ap", 0.0))
    oracle_mean = float(np.mean(oracle_deltas))

    # Strategy-level summary
    strategy_summary: Dict[str, dict] = {}
    for strategy in ["discrete_gate", "continuous_gate", "agreement_gate"]:
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
        "pipeline": "v3_5",
        "direction": "C_image_level_hierarchical_fusion",
        "dataset": "mpdd",
        "seed": seed,
        "categories": categories,
        "stride": STRIDE,
        "baselines": baselines,
        "v3_3_static_baseline": {"mean_delta_ap": v3_3_mean},
        "oracle_upper_bound": {
            "mean_delta_ap": oracle_mean,
            "text_potential_accessed": f"{oracle_mean / 0.4717 * 100:.1f}% of Gate A1 oracle",
        },
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
        "oracle_results": all_oracles,
    }
    return report


def print_summary(report: dict):
    categories = report["categories"]
    bl = report["baselines"]

    print("\n" + "=" * 90)
    print("  V3.5 IMAGE-LEVEL HIERARCHICAL FUSION — EVALUATION REPORT")
    print("=" * 90)

    # Baselines
    print(f"\n{'Branch':<30} {'AUROC':>8} {'AP':>8} {'AUPRO':>8}")
    print("-" * 56)
    for bname, bm in bl.items():
        print(f"{bname:<30} {bm['pixel_auroc']:>8.4f} {bm['pixel_ap']:>8.4f} {bm['pixel_aupro']:>8.4f}")

    # V3.3 vs V3.5 comparison
    print(f"\n{'='*90}")
    print("  COMPARISON: V3.3 STATIC vs V3.5 HIERARCHICAL")
    print(f"{'='*90}")
    v3_3 = report["v3_3_static_baseline"]["mean_delta_ap"]
    hs = report["heldout_summary"]
    oracle = report["oracle_upper_bound"]["mean_delta_ap"]
    print(f"  V3.3 static (60:40):     ΔAP = {v3_3:+.6f}")
    print(f"  V3.5 hierarchical (LOO): ΔAP = {hs['mean_delta_ap']:+.6f}")
    print(f"  Oracle upper bound:      ΔAP = {oracle:+.6f}")
    print(f"  V3.5 vs V3.3:            {hs['mean_delta_ap'] - v3_3:+.6f}")
    print(f"  Gate B: {'PASSED' if hs['positive_categories'] >= len(categories) // 2 else 'FAILED'} "
          f"({hs['positive_categories']}/{len(categories)})")

    # Strategy summary
    print(f"\n{'='*90}")
    print("  STRATEGY-LEVEL COMPARISON (best variant per strategy)")
    print(f"{'='*90}")
    print(f"\n{'Strategy':<25} {'Mean ΔAP':>10} {'Pos. Cats':>12}")
    print("-" * 50)
    ss = report["strategy_summary"]
    for sname, sinfo in ss.items():
        print(f"{sname:<25} {sinfo['mean_delta_ap']:>10.6f} {sinfo['positive_categories']:>12}")

    # Per-category detail
    print(f"\n{'='*90}")
    print("  PER-CATEGORY DETAIL (ΔAP vs DINO visual baseline)")
    print(f"{'='*90}")
    strategies = list(ss.keys())
    header = f"{'Category':<18}" + "".join(f" {s:<22}" for s in strategies)
    print(f"\n{header}")
    print("-" * (18 + 22 * len(strategies)))
    for cat in categories:
        row = f"{cat:<18}"
        for sname in strategies:
            row += f" {ss[sname]['per_category'].get(cat, 0):>21.6f}"
        print(row)

    # Leave-one-out selections
    print(f"\n{'='*90}")
    print("  LEAVE-ONE-OUT SELECTIONS")
    print(f"{'='*90}")
    print(f"\n{'Category':<20} {'Selected':<40} {'Train ΔAP':>10} {'Heldout ΔAP':>10}")
    print("-" * 82)
    for cat in categories:
        sel = report["leave_one_out_selections"].get(cat, {})
        hm = report["heldout_summary"]["per_category_delta_ap"].get(cat, 0)
        label = f"{sel.get('strategy','?')}::{sel.get('variant','?')}"[:38]
        print(f"{cat:<20} {label:<40} {sel.get('mean_train_delta_ap',0):>10.6f} {hm:>10.6f}")


# ======================================================================
# Main
# ======================================================================

def main() -> int:
    parser = argparse.ArgumentParser(description="V3.5 Image-level Hierarchical Fusion")
    parser.add_argument("--seed", type=int, default=0, choices=[0, 1, 2])
    parser.add_argument("--max-variants", type=int, default=0,
                        help="Limit variants for quick test (0=all)")
    args = parser.parse_args()
    seed: int = args.seed

    categories = load_categories()
    variants = build_v3_5_variants()
    if args.max_variants > 0:
        variants = variants[:args.max_variants]

    print(f"Dataset: mpdd, seed={seed}, k=1, categories: {categories}", flush=True)
    print(f"Strategy variants: {len(variants)}", flush=True)

    # Phase 1: Load
    print("\n[Phase 1] Loading branch caches (with image_scores)...", flush=True)
    all_branches: Dict[str, Dict[str, BranchData]] = {}
    for cat in categories:
        print(f"  {cat}...", end=" ", flush=True)
        branches = load_branches_for_category(cat, seed=seed)
        all_branches[cat] = branches
        dino = branches.get("anomalydino_visual")
        if dino is not None:
            print(f"DINO imgs={dino.n_samples}, img_score range=[{dino.image_scores.min():.4f}, {dino.image_scores.max():.4f}]", flush=True)
        else:
            print("no DINO data", flush=True)

    # Phase 2: Baselines
    print("\n[Phase 2] Computing baselines...", flush=True)
    all_baselines: Dict[str, dict] = {}
    for cat in categories:
        all_baselines[cat] = compute_baseline_metrics(all_branches[cat])

    baseline_agg: Dict[str, dict] = {}
    for bname in ["anomalydino_visual", "anomalyclip_text"]:
        agg = {}
        for key in ["pixel_auroc", "pixel_ap", "pixel_aupro"]:
            vals = [all_baselines[cat].get(bname, {}).get(key, 0) for cat in categories]
            agg[key] = round(float(np.mean(vals)), 6)
        baseline_agg[bname] = agg

    # Phase 3: Run variants + oracle
    print(f"\n[Phase 3] Running {len(variants)} variants x {len(categories)} categories...", flush=True)
    all_results: Dict[str, List[dict]] = {}
    all_oracles: Dict[str, dict] = {}

    for cat in categories:
        t0 = time.time()
        results, oracle = evaluate_category(cat, all_branches[cat], variants)
        all_results[cat] = results
        all_oracles[cat] = oracle
        elapsed = time.time() - t0

        if results:
            best = max(results, key=lambda r: r["delta_ap"])
            v3_3_ref = next((r for r in results if r["strategy"] == "v3_3_static"), None)
            v3_3_dap = v3_3_ref["delta_ap"] if v3_3_ref else 0.0
            print(f"  {cat}: {len(results)} variants, best={best['variant']} "
                  f"(ΔAP={best['delta_ap']:+.6f}), V3.3_ref={v3_3_dap:+.6f}, "
                  f"oracle={oracle['delta_ap']:+.6f}, {elapsed:.1f}s", flush=True)

    # Phase 4: Leave-one-out CV
    print("\n[Phase 4] Leave-one-out cross-validation...", flush=True)
    selections, heldout_metrics = leave_one_out_select(all_results, categories)

    # Phase 5: Report
    print("\n[Phase 5] Building report...", flush=True)
    report = build_report(
        categories, all_results, all_oracles, baseline_agg,
        selections, heldout_metrics, seed,
    )

    out_dir = ROOT / "experiments" / "dynamic_fusion" / "v3_5_hierarchical"
    out_dir.mkdir(parents=True, exist_ok=True)
    report_path = out_dir / f"s{seed}_report.json"
    report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nReport saved to: {report_path}")

    print_summary(report)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
