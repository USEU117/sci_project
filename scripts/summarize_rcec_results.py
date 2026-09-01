"""RCEC v1 — summarize results, run the final gate, write FINAL_RCEC_DECISION.md.

Usage:
  .venv-patchcore/Scripts/python.exe scripts/summarize_rcec_results.py \
      --experiment-root experiments/dynamic_fusion/rcec_v1
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "src"))

from rcec_common import final_gate_pass  # noqa: E402

REPORTS = ROOT / "experiments" / "dynamic_fusion" / "rcec_v1"


def _load_json(p: Path) -> dict:
    return json.loads(p.read_text(encoding="utf-8"))


def _write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment-root", type=Path, default=REPORTS)
    args = parser.parse_args()
    root = args.experiment_root if args.experiment_root.is_absolute() else ROOT / args.experiment_root

    # ---- Development summary ----
    small = _load_json(root / "development_mpdd" / "SMALL_GATE_REPORT.json")
    dev_rows = []
    for cid, r in small["results"].items():
        dev_rows.append({
            "candidate": cid,
            "passed_small_gate": r["passed"],
            "mean_delta_pixel_ap": r["detail"].get("mean_delta"),
            "n_positive_shots": r["detail"].get("n_positive_shots"),
        })
    _write_csv(root / "reports" / "small_gate_summary.csv", dev_rows)

    # ---- Frozen validation + final gate ----
    val_path = root / "frozen_validation" / "FROZEN_VALIDATION_REPORT.json"
    decision = {"status": "BLOCKED", "decision": "RCEC not promoted"}
    if val_path.is_file():
        val = _load_json(val_path)
        validation_reports = {}
        for ds in ("btad", "mvtec", "visa"):
            rows = []
            for seed in (0, 1, 2):
                for shot in (1, 2, 4):
                    p = root / "frozen_validation" / ds / f"s{seed}_k{shot}" / "report.json"
                    if p.is_file():
                        rows.append(_load_json(p))
            if rows:
                validation_reports[ds] = rows
        ok, detail = final_gate_pass(validation_reports)
        decision = {
            "status": "PROMOTE" if ok else "ARCHIVE",
            "decision": ("RCEC upgraded to paper main method" if ok
                         else "RCEC archived as development negative result; A1 remains"),
            "final_gate": ok,
            "final_gate_detail": detail,
        }
    else:
        decision["note"] = "frozen validation not run (no selection pool winner)"

    # ---- Write FINAL_RCEC_DECISION.md ----
    md = _decision_md(root, decision)
    (root / "FINAL_RCEC_DECISION.md").write_text(md, encoding="utf-8")
    (root / "FINAL_RCEC_DECISION.json").write_text(
        json.dumps(decision, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(decision, ensure_ascii=False, indent=2))
    return 0 if decision["status"] != "BLOCKED" else 1


def _decision_md(root: Path, decision: dict) -> str:
    lines = [
        "# FINAL_RCEC_DECISION",
        "",
        f"- **Status: `{decision['status']}`**",
        f"- Decision: {decision['decision']}",
        "",
        "## Questions answered",
        "",
        "1. **Does RCEC beat A1 (not just DINO)?** See development mean Pixel-AP "
        "deltas vs A1 in `development_mpdd/SMALL_GATE_REPORT.json` and the "
        "frozen-validation summary below.",
        "2. **Which gates passed/failed?** See gate detail in this directory's "
        "JSON reports.",
        "3. **Validation-set tuning or leakage?** All reports carry five leakage "
        "flags; the frozen runner only reads the frozen config and rejects CLI "
        "overrides.",
        "4. **Does pairing shuffle support the consistency explanation?** "
        "See `ablations/` (Phase 4).",
        "5. **Which datasets / categories / shots gain or regress?** "
        "Per-config JSON under `development_mpdd/` and `frozen_validation/`.",
        "6. **Is the gain worth the added complexity?** Discussed in the paper "
        "draft if promoted; otherwise archived as a negative result.",
        "7. **Main method: RCEC or A1?** ",
        f"   -> `{decision['status']}`",
        "8. **Evidence pointers** are the JSON/CSV files referenced above.",
        "",
    ]
    return "\n".join(lines)


if __name__ == "__main__":
    raise SystemExit(main())
