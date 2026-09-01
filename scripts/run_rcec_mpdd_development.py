"""RCEC v1 — MPDD development runner (small gate + full matrix).

Usage:
  .venv-patchcore/Scripts/python.exe scripts/run_rcec_mpdd_development.py \
      --config configs/rcec_v1.yaml --validate-only
  ... --phase small-gate      # MPDD seed0 x shot {1,2,4} x 12 candidates
  ... --phase full            # MPDD 3 seeds x 3 shots x small-gate winners

Every evaluated config is written to
``experiments/dynamic_fusion/rcec_v1/development_mpdd/...`` with a resumable
marker that records the config hash. Failing candidates are reported, never
silently dropped.
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "src"))

from rcec_common import (  # noqa: E402
    EXPERIMENT_ROOT,
    candidate_id,
    candidates_from_config,
    dirs_for,
    evaluate_config,
    load_config,
    manifest_for,
    selection_pool_pass,
    sha256_file,
    small_gate_pass,
)

DEV_ROOT = EXPERIMENT_ROOT / "development_mpdd"


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def _validate(cfg: dict) -> None:
    manifest = manifest_for("mpdd")
    missing = []
    for seed in cfg["seeds"]:
        for shot in cfg["shots"]:
            dino_dir, clip_dir = dirs_for("mpdd", seed, shot)
            d_cats = {p.stem for p in dino_dir.glob("*.npz") if p.stem != "export_report"}
            c_cats = {p.stem for p in clip_dir.glob("*.npz") if p.stem != "export_report"}
            m_cats = set(manifest["categories"].keys())
            if not d_cats or d_cats != c_cats:
                missing.append(f"s{seed}_k{shot}: dino={sorted(d_cats)} clip={sorted(c_cats)}")
            if d_cats != m_cats:
                missing.append(f"s{seed}_k{shot}: cache cats != manifest cats")
            # Reference count check.
            for cat in sorted(d_cats):
                n_refs = len(manifest["categories"][cat][str(seed)][str(shot)])
                npz = dino_dir / f"{cat}.npz"
                import numpy as _np

                with _np.load(npz, allow_pickle=False) as data:
                    if data["ref_patch_features"].shape[0] != n_refs:
                        missing.append(
                            f"s{seed}_k{shot}/{cat}: ref blocks != manifest refs ({n_refs})")
    if missing:
        raise SystemExit("missing inputs:\n  " + "\n  ".join(missing))
    print(json.dumps({"status": "passed", "mode": "validate_only",
                      "candidates": len(candidates_from_config(cfg))}))


def _run_one(dataset: str, seed: int, shot: int, cand: dict, cfg: dict,
             out_dir: Path) -> dict:
    config_sha = cfg["_config_sha256"]
    marker = out_dir / "report.json"
    meta = out_dir / "marker.json"
    if marker.is_file() and meta.is_file():
        saved = json.loads(meta.read_text(encoding="utf-8"))
        if saved.get("config_sha256") == config_sha:
            return json.loads(marker.read_text(encoding="utf-8"))
    report = evaluate_config(dataset, seed, shot, cand, cfg)
    _write_json(marker, report)
    _write_json(meta, {
        "config_sha256": config_sha,
        "candidate": cand,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
    })
    return report


def run_small_gate(cfg: dict) -> int:
    candidates = candidates_from_config(cfg)
    results = {}
    for cand in candidates:
        cid = candidate_id(cand)
        rows = []
        for shot in cfg["shots"]:
            out_dir = DEV_ROOT / "small_gate" / cid / f"s0_k{shot}"
            rows.append(_run_one("mpdd", 0, shot, cand, cfg, out_dir))
        passed, detail = small_gate_pass(rows)
        results[cid] = {
            "candidate": cand,
            "passed": passed,
            "detail": detail,
            "per_config": [r["metrics"]["delta_rcec_vs_a1"]["pixel"]["pixel_ap"]
                           for r in rows],
        }
        print(f"[small-gate] {cid}: passed={passed} {json.dumps(detail, ensure_ascii=False)}",
              flush=True)

    winners = [cid for cid, r in results.items() if r["passed"]]
    report = {
        "schema_version": 1,
        "phase": "small_gate",
        "dataset": "mpdd",
        "seeds": [0],
        "shots": cfg["shots"],
        "n_candidates_evaluated": len(candidates),
        "winners": winners,
        "results": results,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    _write_json(DEV_ROOT / "SMALL_GATE_REPORT.json", report)
    if not winners:
        # Task book: no candidate passes -> early stop.
        _write_json(DEV_ROOT / "RCEC_V1_EARLY_STOP_REPORT.json", report)
        print("NO candidate passed the small gate -> RCEC_V1_EARLY_STOP")
        return 1
    return 0


def run_full_matrix(cfg: dict) -> int:
    small_report = DEV_ROOT / "SMALL_GATE_REPORT.json"
    if not small_report.is_file():
        raise SystemExit("small-gate report missing; run --phase small-gate first")
    winners = json.loads(small_report.read_text(encoding="utf-8"))["winners"]
    if not winners:
        raise SystemExit("no small-gate winners; early stop already declared")
    candidates = {candidate_id(c): c for c in candidates_from_config(cfg)}
    pool_rows = {}
    for cid in winners:
        cand = candidates[cid]
        rows = []
        for seed in cfg["seeds"]:
            for shot in cfg["shots"]:
                out_dir = DEV_ROOT / "full_matrix" / cid / f"s{seed}_k{shot}"
                rows.append(_run_one("mpdd", seed, shot, cand, cfg, out_dir))
        pool_rows[cid] = rows
        passed, detail = selection_pool_pass(rows)
        print(f"[full] {cid}: selection_pool={passed} {json.dumps(detail, ensure_ascii=False)}",
              flush=True)

    def sort_key(cid: str) -> tuple:
        rows = pool_rows[cid]
        mean_d = float(np.mean(
            [r["metrics"]["delta_rcec_vs_a1"]["pixel"]["pixel_ap"] for r in rows]))
        n_pos = int(sum(1 for r in rows
                        if (r["metrics"]["delta_rcec_vs_a1"]["pixel"]["pixel_ap"] or 0) > 0))
        cand = candidates[cid]
        pref = 0 if cand["direction"] == "dino_to_clip" else 1
        return (-mean_d, -n_pos, pref, cand["k"], cand["lambda"])

    eligible = [cid for cid, rows in pool_rows.items()
                if selection_pool_pass(rows)[0]]
    eligible_sorted = sorted(eligible, key=sort_key)
    selected = eligible_sorted[0] if eligible_sorted else None
    report = {
        "schema_version": 1,
        "phase": "full_matrix",
        "dataset": "mpdd",
        "seeds": cfg["seeds"],
        "shots": cfg["shots"],
        "candidates_run": list(pool_rows.keys()),
        "selection_pool_eligible": eligible,
        "selected_candidate": selected,
        "selection_rule": ("max mean Pixel-AP delta vs A1; tie->more positive; "
                           "tie->dino_to_clip; tie->smaller k; tie->smaller lambda"),
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    _write_json(DEV_ROOT / "FULL_MATRIX_REPORT.json", report)
    if selected is None:
        print("NO candidate entered the selection pool -> RCEC v1 performance gate FAILED")
        return 1
    print(json.dumps({"selected_candidate": selected}, ensure_ascii=False))
    return 0


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--validate-only", action="store_true")
    parser.add_argument("--phase", choices=("small-gate", "full"), default="small-gate")
    args = parser.parse_args()
    cfg = load_config(args.config)
    if args.validate_only:
        _validate(cfg)
        return 0
    if args.phase == "small-gate":
        return run_small_gate(cfg)
    return run_full_matrix(cfg)


if __name__ == "__main__":
    raise SystemExit(main())
