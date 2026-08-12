"""Minimal V3.5 evaluation - one seed, one category, all variants."""
import sys, json, time
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from evaluate_unified import aupro_fast
from industrial_ad.fusion.alignment import build_alignment_plan
from run_dynamic_fusion_v2_cache import load_cache, resize_maps
from sklearn.metrics import average_precision_score, roc_auc_score
from industrial_ad.fusion.v3_5_strategies import (
    BranchData, build_v3_5_variants, oracle_image_gate_fusion, run_fusion,
)

STRIDE = 8
SEED = 0
v2_root = ROOT / "outputs" / "dynamic_fusion" / "v2_mpdd_predictions" / f"v2_mpdd_s{SEED}_k1_full_v1"

manifest = json.loads((ROOT / "data" / "splits" / "mpdd" / "manifest.json").read_text(encoding="utf-8"))
categories = sorted(manifest["categories"])

variants = build_v3_5_variants()
print(f"Variants: {len(variants)}", flush=True)
for v in variants:
    print(f"  {v['strategy']}::{v['variant_name']}", flush=True)

def compute_metrics(pixel_maps, gt_masks):
    maps_s = pixel_maps[:, ::STRIDE, ::STRIDE]
    masks_s = gt_masks[:, ::STRIDE, ::STRIDE]
    flat_maps = maps_s.ravel()
    flat_labels = (masks_s.ravel() > 0.5).astype(np.int32)
    return {
        "pixel_auroc": float(roc_auc_score(flat_labels, flat_maps)),
        "pixel_ap": float(average_precision_score(flat_labels, flat_maps)),
        "pixel_aupro": float(aupro_fast(masks_s, maps_s)),
    }

all_results = {}
all_oracles = {}

for cat in categories:
    print(f"\n=== {cat} ===", flush=True)
    
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
    
    dino_metrics = compute_metrics(dino_maps.astype(np.float64), dino_masks)
    print(f"  DINO AP={dino_metrics['pixel_ap']:.4f}", flush=True)
    
    cat_results = []
    t0 = time.time()
    
    for vi, variant in enumerate(variants):
        try:
            fused = run_fusion(variant, db, tb)
        except Exception as e:
            print(f"  [ERR] {variant['variant_name']}: {e}", flush=True)
            continue
        
        m = compute_metrics(fused, dino_masks)
        r = {
            "category": cat,
            "strategy": variant["strategy"],
            "variant": variant["variant_name"],
            **m,
            "delta_ap": round(m["pixel_ap"] - dino_metrics["pixel_ap"], 6),
        }
        cat_results.append(r)
        
        if (vi + 1) % 5 == 0 or vi == len(variants) - 1:
            print(f"  [{vi+1}/{len(variants)}] {time.time()-t0:.1f}s", flush=True)
    
    # Oracle
    oracle_maps, oracle_stats = oracle_image_gate_fusion(db, tb)
    oracle_m = compute_metrics(oracle_maps, dino_masks)
    oracle_m["delta_ap"] = round(oracle_m["pixel_ap"] - dino_metrics["pixel_ap"], 6)
    oracle_m["oracle_stats"] = oracle_stats
    print(f"  Oracle: dAP={oracle_m['delta_ap']:+.6f}, text_ratio={oracle_stats['text_ratio']:.2f}", flush=True)
    
    all_results[cat] = cat_results
    all_oracles[cat] = oracle_m

# Summary
print("\n" + "=" * 80)
print("SUMMARY")
print("=" * 80)

for strategy in ["v3_3_static", "discrete_gate", "continuous_gate", "agreement_gate"]:
    deltas = []
    for cat in categories:
        cat_best = 0.0
        for r in all_results.get(cat, []):
            if r["strategy"] == strategy:
                cat_best = max(cat_best, r["delta_ap"])
        deltas.append(cat_best)
    mean_d = np.mean(deltas)
    pos = sum(1 for d in deltas if d > 0)
    print(f"  {strategy:<20}: mean_dAP={mean_d:+.6f}, pos={pos}/{len(categories)}", flush=True)

oracle_deltas = [all_oracles[c]["delta_ap"] for c in categories]
print(f"  {'oracle':<20}: mean_dAP={np.mean(oracle_deltas):+.6f}", flush=True)

# Save
out_dir = ROOT / "experiments" / "dynamic_fusion" / "v3_5_hierarchical"
out_dir.mkdir(parents=True, exist_ok=True)
report = {
    "seed": SEED, "variants": len(variants),
    "all_results": all_results,
    "oracles": all_oracles,
}
with open(out_dir / f"s{SEED}_report.json", "w", encoding="utf-8") as f:
    json.dump(report, f, indent=2, ensure_ascii=False)
print(f"\nSaved to {out_dir / f's{SEED}_report.json'}", flush=True)
