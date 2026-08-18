"""A1 MPDD weight scan across the 9-config matrix (docs 阶段五 5.2 -> 阶段六).

Pre-registered grid over dino_weight (CPU + faiss only, serial, marker-resumable):
    {0.3, 0.4, 0.5, 0.6, 0.7}
w=0.5 is the frozen config; the scan checks that it stays optimal (or near-optimal)
across all 9 (seed, shot) configs before freezing.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

PYTHON = str(ROOT / ".venv-patchcore" / "Scripts" / "python.exe")
FEATURES_ROOT = ROOT / "outputs" / "dynamic_fusion" / "v3_direction_a"
BASELINE_ROOT = ROOT / "outputs" / "dynamic_fusion" / "v2_mpdd_predictions"
EXPERIMENT_ROOT = ROOT / "experiments" / "dynamic_fusion" / "v3_direction_a" / "a1_weight_scan_20260817"

WEIGHTS = [0.3, 0.4, 0.5, 0.6, 0.7]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    parser.add_argument("--shots", type=int, nargs="+", default=[1, 2, 4])
    parser.add_argument("--weights", type=float, nargs="+", default=WEIGHTS)
    parser.add_argument("--validate-only", action="store_true")
    args = parser.parse_args()

    EXPERIMENT_ROOT.mkdir(parents=True, exist_ok=True)
    jobs = []
    for seed in args.seeds:
        for shot in args.shots:
            dino_dir = FEATURES_ROOT / f"features_vitb14_s{seed}_k{shot}" / "anomalydino_visual"
            clip_dir = FEATURES_ROOT / f"features_s{seed}_k{shot}" / "anomalyclip_text"
            baseline_dir = BASELINE_ROOT / f"v2_mpdd_s{seed}_k{shot}_full_v1"
            out_dir = EXPERIMENT_ROOT / f"seed{seed}_k{shot}"
            for w in args.weights:
                marker = out_dir / f"concat_pca0_whiten0_w{w:g}_report.json"
                jobs.append(
                    {
                        "seed": seed,
                        "shot": shot,
                        "weight": w,
                        "dino_dir": dino_dir,
                        "clip_dir": clip_dir,
                        "baseline_dir": baseline_dir,
                        "out_dir": out_dir,
                        "marker": marker,
                    }
                )

    # validate inputs
    missing = []
    seen = set()
    for job in jobs:
        key = (job["seed"], job["shot"])
        if key in seen:
            continue
        seen.add(key)
        for name in ("dino_dir", "clip_dir"):
            n_npz = len(list(job[name].glob("*.npz")))
            if n_npz < 6:
                missing.append(f"{name} {job[name]} only {n_npz} npz")
        bd = job["baseline_dir"]
        for sub in ("anomalydino_visual", "anomalyclip_text"):
            n_npz = len(list((bd / sub).glob("*.npz")))
            if n_npz < 6:
                missing.append(f"baseline {sub} {bd} only {n_npz} npz")
    if missing:
        raise SystemExit("missing inputs:\n  " + "\n  ".join(missing))
    if args.validate_only:
        print(json.dumps({"status": "passed", "mode": "validate_only", "jobs": len(jobs)}))
        return 0

    completed, failed = [], []
    for job in jobs:
        if job["marker"].is_file():
            completed.append({"seed": job["seed"], "shot": job["shot"], "w": job["weight"], "status": "cached"})
            continue
        cmd = [
            PYTHON,
            str(ROOT / "scripts" / "evaluate_a1_feature_fusion.py"),
            "--dino-features", str(job["dino_dir"]),
            "--clip-features", str(job["clip_dir"]),
            "--baseline-dir", str(job["baseline_dir"]),
            "--dataset", "mpdd",
            "--seed", str(job["seed"]),
            "--mode", "concat",
            "--pca-dim", "0",
            "--whiten", "0",
            "--dino-weight", str(job["weight"]),
            "--output-dir", str(job["out_dir"]),
        ]
        print(f"[s{job['seed']}/k{job['shot']}/w{job['weight']}] running...", flush=True)
        proc = subprocess.run(cmd, cwd=ROOT)
        if proc.returncode != 0 or not job["marker"].is_file():
            failed.append({"seed": job["seed"], "shot": job["shot"], "w": job["weight"], "exit": proc.returncode})
        else:
            completed.append({"seed": job["seed"], "shot": job["shot"], "w": job["weight"], "status": "ok"})

    print(json.dumps({"completed": len(completed), "failed": failed}))
    return 0 if not failed else 1


if __name__ == "__main__":
    raise SystemExit(main())
