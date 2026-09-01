"""RCEC v1 — single (dataset, seed, shot, candidate) evaluation entry point.

Use for smoke checks and to reproduce one report:
  .venv-patchcore/Scripts/python.exe scripts/evaluate_rcec_cached.py \
      --dataset mpdd --seed 0 --shot 1 --candidate dino_to_clip --k 3 --lambda 0.25 \
      --config configs/rcec_v1.yaml --output-dir ...
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "src"))

from rcec_common import evaluate_config, load_config  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True, choices=("mpdd", "btad", "visa", "mvtec"))
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--shot", type=int, required=True)
    parser.add_argument("--candidate", required=True, choices=("dino_to_clip", "symmetric"))
    parser.add_argument("--k", type=int, required=True)
    parser.add_argument("--lambda", dest="lam", type=float, required=True)
    parser.add_argument("--config", type=Path, default=ROOT / "configs" / "rcec_v1.yaml")
    parser.add_argument("--output-dir", type=Path, default=None)
    args = parser.parse_args()

    cfg = load_config(args.config)
    cand = {"direction": args.candidate, "k": args.k, "lambda": args.lam}
    report = evaluate_config(args.dataset, args.seed, args.shot, cand, cfg)
    if args.output_dir:
        out = args.output_dir
        out.mkdir(parents=True, exist_ok=True)
        (out / "report.json").write_text(
            json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(report, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
