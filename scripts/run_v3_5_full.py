"""V3.5 full evaluation - all categories, all variants. Outputs to JSON."""
import sys, json, time
from pathlib import Path
from collections import defaultdict
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from evaluate_unified import aupro_fast
from run_dynamic_fusion_v2_cache import load_cache, resize_maps
from industrial_ad.fusion.alignment import build_alignment_plan
from sklearn.metrics import average_precision_score, roc_auc_score
from industrial_ad.fusion.v3_5_strategies import *

STRIDE = 8
OUT_DIR = ROOT / "experiments" / "dynamic_fusion" / "v3_5_hierarchical"
OUT_DIR.mkdir(parents=True, exist_ok=True)

manifest = json.loads((ROOT / "data" / "splits" / "mpdd" / "manifest.json").read_text(encoding="utf-8"))
CATEGORIES = sorted(manifest["categories"])

def compute_metrics(pixel_maps, gt_masks):
    maps_s = pixel_maps[:, ::STRIDE, ::STRIDE]
    masks_s = gt_masks[:, ::STRIDE, ::STRIDE]
    flat_maps = maps_s.ravel()
    flat_labels = (masks_s.ravel() > 0.5).astype(np.int32)
    return {
        "auroc": float(roc_auc_score(flat_labels, flat_maps)),
        "ap": float(average_precision_score(flat_labels, flat_maps)),
        "aupro": float(aupro_fast(masks_s, maps_s)),
    }

def run_seed(seed: int):
    v2_root = ROOT / "outputs" / "dynamic_fusion" / "v2_mpdd_predictions" / f"v2_mpdd_s{seed}_k1_full_v1"
    variants = build_v3_5_variants()
    
    all_results = {}
    all_oracles = {}
    all_baselines = {}
    
    for cat in CATEGORIES:
        print(f"\n{'='*60}")
        print(f"  [{seed}] {cat}")
        print(f"{'='*60}", flush=True)
        
        # Load branches
        dino_path = v2_root / "anomalydino_visual" / f"{cat}.npz"
        if not dino_path.exists():
            print(f"  SKIP: no DINO", flush=True)
            continue
        dino = load_cache(dino_path)
        ref_ids = np.asarray(dino["sample_ids"])
        dino_maps = np.asarray(dino["anomaly_maps"], dtype=np.float32)
        dino_img = np.asarray(dino["pr_sp"], dtype=np.float32)
        dino_masks = np.asarray(dino["imgs_masks"], dtype=np.uint8)
        dino_gts = np.asarray(dino["gt_sp"], dtype=np.uint8)
        
        aclip_path = v2_root / "anomalyclip_text" / f"{cat}.npz"
        if not aclip_path.exists():
            print(f"  SKIP: no text", flush=True)
            continue
        aclip = load_cache(aclip_path)
        alignment = build_alignment_plan(ref_ids, np.asarray(aclip["sample_ids"]))
        order = alignment.candidate_order
        text_maps = resize_maps(np.asarray(aclip["anomaly_maps"], dtype=np.float32)[order], dino_maps.shape[1:])
        text_img = np.asarray(aclip["pr_sp"], dtype=np.float32)[order]
        
        db = BranchData("anomalydino_visual", dino_maps, dino_img, dino_gts, dino_masks, ref_ids)
        tb = BranchData("anomalyclip_text", text_maps, text_img, dino_gts, dino_masks, ref_ids)
        
        # Baselines
        dino_m = compute_metrics(dino_maps.astype(np.float64), dino_masks)
        text_m = compute_metrics(text_maps.astype(np.float64), dino_masks)
        all_baselines[cat] = {"anomalydino_visual": dino_m, "anomalyclip_text": text_m}
        print(f"  DINO AP={dino_m['ap']:.4f}  Text AP={text_m['ap']:.4f}", flush=True)
        
        # Run all variants
        cat_results = []
        t0 = time.time()
        errors = 0
        
        for vi, variant in enumerate(variants):
            try:
                fused = run_fusion(variant, db, tb)
            except Exception as e:
                errors += 1
                continue
            
            m = compute_metrics(fused, dino_masks)
            cat_results.append({
                "category": cat,
                "strategy": variant["strategy"],
                "variant": variant["variant_name"],
                **m,
                "delta_ap": round(m["ap"] - dino_m["ap"], 6),
                "delta_auroc": round(m["auroc"] - dino_m["auroc"], 6),
                "delta_aupro": round(m["aupro"] - dino_m["aupro"], 6),
            })
        
        elapsed = time.time() - t0
        if errors:
            print(f"  {len(cat_results)}/{len(variants)} variants OK, {errors} errors, {elapsed:.1f}s", flush=True)
        else:
            print(f"  {len(cat_results)} variants OK, {elapsed:.1f}s", flush=True)
        
        # Oracle
        oracle_maps, oracle_stats = oracle_image_gate_fusion(db, tb)
        oracle_m = compute_metrics(oracle_maps, dino_masks)
        oracle_m["delta_ap"] = round(oracle_m["ap"] - dino_m["ap"], 6)
        oracle_m["oracle_stats"] = oracle_stats
        print(f"  Oracle dAP={oracle_m['delta_ap']:+.6f} (text={oracle_stats['text_ratio']:.1%})", flush=True)
        
        all_results[cat] = cat_results
        all_oracles[cat] = oracle_m
    
    # ---- Aggregate ----
    strategies = ["v3_3_static", "discrete_gate", "continuous_gate", "agreement_gate"]
    summary = {}
    
    for strat in strategies:
        deltas = []
        for cat in CATEGORIES:
            best = 0.0
            for r in all_results.get(cat, []):
                if r["strategy"] == strat:
                    best = max(best, r["delta_ap"])
            deltas.append(best)
        summary[strat] = {
            "mean_delta_ap": round(float(np.mean(deltas)), 6),
            "positive": sum(1 for d in deltas if d > 0),
            "per_category": {c: round(d, 6) for c, d in zip(CATEGORIES, deltas)},
        }
    
    oracle_deltas = []
    for cat in CATEGORIES:
        o = all_oracles.get(cat, {})
        oracle_deltas.append(o.get("delta_ap", 0.0))
    summary["oracle"] = {
        "mean_delta_ap": round(float(np.mean(oracle_deltas)), 6),
        "positive": sum(1 for d in oracle_deltas if d > 0),
    }
    
    # Baseline aggregate
    baseline_agg = {}
    for bname in ["anomalydino_visual", "anomalyclip_text"]:
        agg = {}
        for key in ["auroc", "ap", "aupro"]:
            vals = [all_baselines[cat].get(bname, {}).get(key, 0) for cat in CATEGORIES]
            agg[key] = round(float(np.mean(vals)), 6)
        baseline_agg[bname] = agg
    
    # LOO
    loo_selections = {}
    loo_deltas = {}
    for heldout in CATEGORIES:
        train = [r for cat in CATEGORIES if cat != heldout for r in all_results.get(cat, [])]
        by_variant = defaultdict(list)
        for r in train:
            by_variant[f"{r['strategy']}::{r['variant']}"].append(r["delta_ap"])
        if not by_variant:
            loo_selections[heldout] = "baseline"
            loo_deltas[heldout] = 0.0
            continue
        best_key = max(by_variant, key=lambda k: np.mean(by_variant[k]))
        loo_selections[heldout] = best_key
        # Find heldout result
        for r in all_results.get(heldout, []):
            if f"{r['strategy']}::{r['variant']}" == best_key:
                loo_deltas[heldout] = r["delta_ap"]
                break
        else:
            loo_deltas[heldout] = 0.0
    
    loo_mean = float(np.mean(list(loo_deltas.values())))
    loo_pos = sum(1 for d in loo_deltas.values() if d > 0)
    
    # ---- Report ----
    report = {
        "pipeline": "v3_5",
        "direction": "C_image_level_hierarchical_fusion",
        "seed": seed,
        "categories": CATEGORIES,
        "stride": STRIDE,
        "variants_tested": len(variants),
        "baselines": baseline_agg,
        "v3_3_static_mean_dap": summary["v3_3_static"]["mean_delta_ap"],
        "strategy_summary": {k: v for k, v in summary.items() if k != "v3_3_static"},
        "oracle_summary": summary["oracle"],
        "leave_one_out": {
            "mean_delta_ap": round(loo_mean, 6),
            "positive_categories": loo_pos,
            "selections": loo_selections,
            "per_category": {c: round(d, 6) for c, d in loo_deltas.items()},
        },
        "per_category_baselines": all_baselines,
        "all_results": all_results,
        "oracles": all_oracles,
    }
    
    rpath = OUT_DIR / f"s{seed}_report.json"
    rpath.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nSaved: {rpath}", flush=True)
    
    # Print summary
    print(f"\n{'='*70}")
    print(f"  V3.5 IMAGE-LEVEL HIERARCHICAL FUSION — SEED {seed}")
    print(f"{'='*70}")
    print(f"\n  V3.3 static (60:40) baseline: ΔAP = {summary['v3_3_static']['mean_delta_ap']:+.6f}")
    print(f"  LOO hierarchical best:      ΔAP = {loo_mean:+.6f}")
    print(f"  Oracle image-gate:          ΔAP = {summary['oracle']['mean_delta_ap']:+.6f}")
    print(f"\n  Strategy comparison (best variant per strategy):")
    for strat in ["discrete_gate", "continuous_gate", "agreement_gate"]:
        s = summary[strat]
        print(f"    {strat:<20} ΔAP={s['mean_delta_ap']:+.6f}  pos={s['positive']}/{len(CATEGORIES)}")
    print(f"\n  Per-category (best variant):")
    for cat in CATEGORIES:
        d_vals = {s: summary[s]["per_category"][cat] for s in strategies[1:]}
        best_s = max(d_vals, key=d_vals.get)
        print(f"    {cat:<18} best={best_s:<22} dAP={d_vals[best_s]:+.6f}  "
              f"(v3.3={summary['v3_3_static']['per_category'][cat]:+.6f})")
    
    return report


if __name__ == "__main__":
    for seed in [0, 1, 2]:
        run_seed(seed)
    print("\nAll seeds complete!")
