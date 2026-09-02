"""Wave 0 runner — amplified CASF gate probe -> frozen Gset (task book 15).

Usage:
    python scripts/innovation_v5_casf/run_gate_probe.py [--category CAT]

Loads the pre-registered config (configs/innovation_v5_casf/gate_probe.json),
asserts module constants match it (drift guard), runs the probe on all 6 MPDD
development categories x 3 family seeds, and writes:
  experiments/dynamic_fusion/innovation_v5_casf/Wave0_gate_probe/
      PROBE_SUMMARY.json   (6-category full table incl. negative classes)
      GSET.json            (frozen category gate set)
      <cat>_s0_k2_seed<ix>.json  (per-category per-seed raw rows)
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
for p in (str(ROOT / "src"), str(ROOT / "scripts")):
    if p not in sys.path:
        sys.path.insert(0, p)

from industrial_ad.innovation_v4_diagnostics import common  # noqa: E402
from industrial_ad.innovation_v5_casf import gate_probe as gp  # noqa: E402

CONFIG_PATH = ROOT / "configs" / "innovation_v5_casf" / "gate_probe.json"
OUT_ROOT = (ROOT / "experiments" / "dynamic_fusion" / "innovation_v5_casf"
            / "Wave0_gate_probe")


def assert_config_matches() -> None:
    cfg = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    checks = {
        "episodes_train_per_family": gp.N_TRAIN_EP,
        "episodes_eval_asym": gp.N_EVAL_EP,
        "gamma": gp.GAMMA,
        "logreg_C": gp.LOGREG_C,
        "family_seeds": list(gp.FAMILY_SEEDS),
        "shot": gp.SHOT,
    }
    drift = {k: (v, cfg[k]) for k, v in checks.items() if cfg.get(k) != v}
    if drift:
        raise RuntimeError(f"gate-probe config drift vs code: {drift}")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--category", default=None)
    parser.add_argument("--out-root", type=Path, default=OUT_ROOT)
    args = parser.parse_args()

    common.assert_development_only()
    assert_config_matches()

    manifest = common.manifest_for("mpdd")
    cats = [args.category] if args.category else sorted(manifest["categories"])
    out_root = args.out_root
    out_root.mkdir(parents=True, exist_ok=True)

    t0 = time.time()
    rows = []
    for cat in cats:
        for ix in gp.FAMILY_SEEDS:
            print(f"[GATE-PROBE] {cat} s0/k{gp.SHOT} family-seed {ix}", flush=True)
            rep = gp.category_probe_seed(cat, ix)
            rows.append(rep)
            (out_root / f"{cat}_s0_k{gp.SHOT}_seed{ix}.json").write_text(
                json.dumps(rep, ensure_ascii=False, indent=1), encoding="utf-8")
            a = rep["trained_asym_eval"]; b = rep["trained_sym_eval"]
            print(f"      asym={a} sym={b} headroom={rep['headroom']}", flush=True)

    summary = gp.summarize(rows)
    summary["created_at_utc"] = datetime.now(timezone.utc).isoformat()
    summary["elapsed_s"] = round(time.time() - t0, 1)
    summary["n_categories"] = len(cats)
    (out_root / "PROBE_SUMMARY.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=1), encoding="utf-8")
    (out_root / "GSET.json").write_text(
        json.dumps({"schema_version": 1,
                    "program": "innovation_v5_casf",
                    "phase": "Wave0_gate_probe",
                    "dataset": "mpdd", "role": "development",
                    "rule": "mean headroom >= 0.02 AND >= 2/3 family seeds >= 0.02",
                    "Gset": summary["Gset"],
                    "Gset_frozen_at_utc": summary["created_at_utc"]},
                   ensure_ascii=False, indent=1), encoding="utf-8")

    print(json.dumps({"Gset": summary["Gset"]}, indent=1))
    for cat in cats:
        d = summary["by_category"][cat]
        print(f"  {cat}: mean_hr={d['mean_headroom']} votes={d['seed_votes']}/"
              f"{gp.N_SEEDS} active={d['active']}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
