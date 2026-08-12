"""Master: run BTAD holdout. Each (seed,cat) runs as isolated subprocess writing to file."""
import json, subprocess, sys, time
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SEEDS, CATS = [0, 1, 2], ["01", "02", "03"]
SCRIPT = ROOT / "scripts" / "_btad_eval_file.py"
OUT_DIR = ROOT / "experiments" / "dynamic_fusion" / "v3_3" / "btad_holdout"
OUT_DIR.mkdir(parents=True, exist_ok=True)

all_entries = []
for seed in SEEDS:
    for cat in CATS:
        tmp = OUT_DIR / f"_tmp_s{seed}_{cat}.json"
        tmp.unlink(missing_ok=True)
        print(f"s{seed}/{cat}...", end=" ", flush=True)
        t0 = time.time()
        p = subprocess.Popen(
            [sys.executable, str(SCRIPT), str(seed), cat, str(tmp)],
            cwd=str(ROOT),
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
        )
        p.wait(timeout=120)
        elapsed = time.time() - t0

        if p.returncode != 0 or not tmp.exists():
            print(f"FAILED rc={p.returncode} ({elapsed:.0f}s)", flush=True)
            continue

        entry = json.loads(tmp.read_text(encoding="utf-8"))
        all_entries.append(entry)
        tmp.unlink(missing_ok=True)
        dap = entry["delta_ap"]
        sign = "+" if dap >= 0 else ""
        print(f"DINO={entry['dino']['pixel_ap']:.4f} Fused={entry['fused']['pixel_ap']:.4f} Δ={sign}{dap:.4f} ({elapsed:.0f}s)", flush=True)

# Summaries
per_seed_summary = {}
for seed in SEEDS:
    entries = [e for e in all_entries if e["seed"] == seed]
    if not entries:
        continue
    deltas = [e["delta_ap"] for e in entries]
    per_seed_summary[str(seed)] = {
        "mean_delta_ap": round(float(np.mean(deltas)), 6),
        "positive_categories": sum(1 for d in deltas if d > 0),
        "gate_b": "PASSED" if sum(1 for d in deltas if d > 0) >= 2 else "FAILED",
    }

all_deltas = [e["delta_ap"] for e in all_entries]
mean_overall = float(np.mean(all_deltas))
pos_overall = sum(1 for d in all_deltas if d > 0)
seeds_passed = sum(1 for v in per_seed_summary.values() if v["gate_b"] == "PASSED")
seeds_pos = sum(1 for v in per_seed_summary.values() if v["mean_delta_ap"] > 0)

per_category = {}
for cat in CATS:
    cat_entries = [e for e in all_entries if e["category"] == cat]
    per_category[cat] = {str(e["seed"]): e for e in cat_entries}

report = {
    "pipeline": "v3_3_btad_holdout",
    "strategy": "weighted_ensemble",
    "weights": {"anomalydino_visual": 0.60, "anomalyclip_text": 0.40},
    "calibrate": True, "tuned_on": "mpdd", "evaluated_on": "btad_holdout",
    "categories": CATS, "seeds": SEEDS, "k": 1, "stride": 8,
    "per_seed_summary": per_seed_summary,
    "per_category": per_category,
    "aggregate": {
        "mean_delta_ap": round(mean_overall, 6),
        "positive_entries": pos_overall,
        "total_entries": len(all_entries),
        "seeds_passed_gate_b": seeds_passed,
        "seeds_with_positive_mean": seeds_pos,
    },
    "verdict": {
        "holds_out": seeds_passed >= 2 and mean_overall > 0,
        "conclusion": (
            "PASSED — frozen weights generalize to BTAD holdout"
            if seeds_passed >= 2 and mean_overall > 0
            else "PARTIAL" if seeds_pos >= 1 and mean_overall > 0
            else "FAILED"
        ),
    },
}

(OUT_DIR / "report.json").write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")

print(f"\n{'=' * 60}")
print(f"BTAD HOLDOUT — FINAL SUMMARY")
print(f"{'=' * 60}")
for seed in SEEDS:
    s = per_seed_summary.get(str(seed), {})
    entries = [e for e in all_entries if e["seed"] == seed]
    if not entries:
        continue
    print(f"Seed {seed}: mean ΔAP={s.get('mean_delta_ap',0):.6f} pos={s.get('positive_categories',0)}/{len(entries)} {s.get('gate_b','?')}")
    for e in entries:
        sign = "+" if e["delta_ap"] >= 0 else ""
        print(f"  {e['category']}: DINO={e['dino']['pixel_ap']:.4f} Fused={e['fused']['pixel_ap']:.4f} Δ={sign}{e['delta_ap']:.4f}")
print(f"\nOverall mean ΔAP: {mean_overall:.6f} | {pos_overall}/{len(all_entries)} positive | {seeds_passed}/3 seeds passed")
print(f"Verdict: {report['verdict']['conclusion']}")
