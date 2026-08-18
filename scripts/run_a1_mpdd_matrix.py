"""Run the full MPDD A1 development matrix (docs 阶段五 5.2): 3 seeds x 3 shots.

CPU + faiss only (no GPU). Uses the frozen vitb14 DINO + CLIP feature caches
(k1 test features; k2/k4 ref-only exports). Serial, resumable by marker file.

Config: mode=concat, pca_dim=0, whiten=0, dino_weight=0.5 (the frozen A1 config).
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
EXPERIMENT_ROOT = ROOT / "experiments" / "dynamic_fusion" / "v3_direction_a" / "a1_matrix_20260817"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    parser.add_argument("--shots", type=int, nargs="+", default=[1, 2, 4])
    parser.add_argument("--dino-weight", type=float, default=0.5)
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
            marker = out_dir / f"concat_pca0_whiten0_w{args.dino_weight:g}_report.json"
            jobs.append(
                {
                    "seed": seed,
                    "shot": shot,
                    "dino_dir": dino_dir,
                    "clip_dir": clip_dir,
                    "baseline_dir": baseline_dir,
                    "out_dir": out_dir,
                    "marker": marker,
                }
            )

    # ---- validate all inputs exist ----
    missing = []
    for job in jobs:
        for name in ("dino_dir", "clip_dir"):
            path = job[name]
            n_npz = len(list(path.glob("*.npz")))
            if n_npz < 6:
                missing.append(f"{name} {path} only {n_npz} npz")
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
            completed.append({"seed": job["seed"], "shot": job["shot"], "status": "cached"})
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
            "--dino-weight", str(args.dino_weight),
            "--output-dir", str(job["out_dir"]),
        ]
        print(f"[{job['seed']}/{job['shot']}] running...", flush=True)
        proc = subprocess.run(cmd, cwd=ROOT)
        if proc.returncode != 0 or not job["marker"].is_file():
            failed.append({"seed": job["seed"], "shot": job["shot"], "exit": proc.returncode})
        else:
            completed.append({"seed": job["seed"], "shot": job["shot"], "status": "ok"})

    print(json.dumps({"completed": len(completed), "failed": failed}))
    return 0 if not failed else 1


if __name__ == "__main__":
    raise SystemExit(main())
