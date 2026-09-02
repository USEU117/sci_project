"""A2 innovation_v2 — Full MPDD Gate runner (3 seeds x 3 shots, 9 configs).

Task book section 3.2 (Full Gate) / Wave 4. Runs the surviving small-gate
candidate(s) of one route on the MPDD 9-config matrix, evaluates the six
metrics vs frozen A1, applies the full-gate conditions, and records the
per-candidate decision under experiments/dynamic_fusion/innovation_v2/02_full_mpdd/.
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from run_small_gates import (  # noqa: E402
    candidate_id, evaluate_config, route_spec,
)
from industrial_ad.innovation_v2 import common  # noqa: E402

SEEDS = [0, 1, 2]
SHOTS = [1, 2, 4]


def full_gate_pass(config_rows: list[dict], control_rows: list[dict] | None = None) -> tuple[bool, dict]:
    """Task book 3.2 Full MPDD Gate conditions (9 configs, one candidate)."""
    if len(config_rows) != 9:
        return False, {"reason": f"expected 9 configs, got {len(config_rows)}"}
    deltas = [r["metrics"]["delta_vs_a1"]["pixel"]["pixel_ap"] for r in config_rows]
    if any(d is None for d in deltas):
        return False, {"reason": "missing pixel_ap delta"}
    mean_d = float(np.mean(deltas))
    n_pos = int(sum(1 for d in deltas if d > 0))
    worst = min(deltas)

    per_cat = {}
    for r in config_rows:
        for pc in r["per_category"]:
            per_cat.setdefault(pc["category"], []).append(
                pc["metrics"]["delta_vs_a1"]["pixel"]["pixel_ap"])
    cat_means = {c: float(np.mean(v)) for c, v in per_cat.items()}
    n_cat_ok = int(sum(1 for v in cat_means.values() if v >= 0.0))
    worst_cat = min(cat_means.values())

    img_ap = [r["metrics"]["delta_vs_a1"]["image"]["image_ap"] for r in config_rows]
    img_ap = [v for v in img_ap if v is not None]
    mean_img_ap = float(np.mean(img_ap)) if img_ap else None
    img_f1 = [r["metrics"]["delta_vs_a1"]["image"]["image_f1_max"] for r in config_rows]
    img_f1 = [v for v in img_f1 if v is not None]
    mean_img_f1 = float(np.mean(img_f1)) if img_f1 else None

    control_mean = None
    if control_rows is not None and len(control_rows) == 9:
        ctrl = [r["metrics"]["delta_vs_a1"]["pixel"]["pixel_ap"] for r in control_rows]
        ctrl = [c for c in ctrl if c is not None]
        control_mean = float(np.mean(ctrl)) if ctrl else None

    ok = (
        mean_d >= 0.005
        and n_pos >= 7
        and worst >= -0.010
        and n_cat_ok >= 4
        and worst_cat >= -0.015
        and (mean_img_ap is None or mean_img_ap >= -0.005)
        and (mean_img_f1 is None or mean_img_f1 >= -0.010)
        and all(r["checks"]["all_finite"] for r in config_rows)
        and all(not any(r["leakage_flags"].values()) for r in config_rows)
        and (control_mean is None or mean_d > control_mean)
    )
    detail = {
        "mean_delta_pixel_ap": round(mean_d, 6),
        "n_positive_configs": n_pos,
        "worst_config_delta": round(worst, 6),
        "n_categories_non_negative": n_cat_ok,
        "n_categories_total": len(cat_means),
        "worst_category_mean_delta": round(worst_cat, 6),
        "mean_image_ap_delta": round(mean_img_ap, 6) if mean_img_ap is not None else None,
        "mean_image_f1_max_delta": round(mean_img_f1, 6) if mean_img_f1 is not None else None,
        "control_mean_delta_pixel_ap": round(control_mean, 6) if control_mean is not None else None,
    }
    return ok, detail


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--route", required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--candidate-ids", nargs="*", required=True)
    parser.add_argument("--out-root", type=Path,
                        default=common.EXPERIMENT_ROOT / "02_full_mpdd")
    parser.add_argument("--seeds", nargs="*", type=int, default=SEEDS)
    args = parser.parse_args()

    common.assert_development_only("mpdd")
    cfg = common.load_config(args.config)
    spec = route_spec(args.route)

    winners = []
    rows = []
    for cid in args.candidate_ids:
        candidate = next(c for c in cfg["candidates"]
                         if candidate_id(args.route, c) == cid)
        config_rows = []
        control_rows = []
        for seed in args.seeds:
            for shot in SHOTS:
                out_dir = args.out_root / args.route / cid / f"s{seed}_k{shot}"
                out_dir.mkdir(parents=True, exist_ok=True)
                report_path = out_dir / "report.json"
                marker_path = out_dir / "marker.json"
                cfg_sha = cfg["_config_sha256"]
                fresh = True
                if report_path.is_file() and marker_path.is_file():
                    m = json.loads(marker_path.read_text(encoding="utf-8"))
                    fresh = not (m.get("config_sha256") == cfg_sha and m.get("candidate_id") == cid)
                if not fresh:
                    config_rows.append(json.loads(report_path.read_text(encoding="utf-8")))
                    ctrl_path = out_dir / "_control" / "report.json"
                    if ctrl_path.is_file():
                        control_rows.append(json.loads(ctrl_path.read_text(encoding="utf-8")))
                    print(f"[resume] {args.route} {cid} s{seed}_k{shot}")
                    continue

                cfg_eval = dict(cfg)
                cfg_eval["_shot"] = shot
                report = evaluate_config(args.route, cfg_eval, candidate, shot, spec["score_fn"],
                                         spec.get("map_fn"), seed=seed)
                report["seed"] = seed
                marker_path.write_text(json.dumps(
                    {"config_sha256": cfg_sha, "candidate_id": cid,
                     "seed": seed, "shot": shot,
                     "done_utc": datetime.now(timezone.utc).isoformat()}, ensure_ascii=False),
                    encoding="utf-8")
                report_path.write_text(json.dumps(report, ensure_ascii=False, indent=1),
                                       encoding="utf-8")
                config_rows.append(report)
                print(f"[done] {args.route} {cid} s{seed}_k{shot} "
                      f"dPixelAP={report['metrics']['delta_vs_a1']['pixel']['pixel_ap']}")

                if spec.get("control_fn") is not None:
                    ctrl_dir = out_dir / "_control"
                    ctrl_dir.mkdir(parents=True, exist_ok=True)
                    ctrl_path = ctrl_dir / "report.json"
                    ctrl_marker = ctrl_dir / "marker.json"
                    if not (ctrl_path.is_file() and ctrl_marker.is_file()
                            and json.loads(ctrl_marker.read_text(encoding="utf-8"))
                            .get("config_sha256") == cfg_sha):
                        ctrl_cfg = dict(cfg)
                        ctrl_cfg["_shot"] = shot
                        ctrl_report = evaluate_config(
                            args.route, ctrl_cfg, candidate, shot, spec["control_fn"],
                            spec.get("map_fn"), seed=seed)
                        ctrl_marker.write_text(json.dumps(
                            {"config_sha256": cfg_sha, "candidate_id": cid,
                             "seed": seed, "shot": shot, "control": spec["control_name"]},
                            ensure_ascii=False), encoding="utf-8")
                        ctrl_path.write_text(json.dumps(ctrl_report, ensure_ascii=False, indent=1),
                                             encoding="utf-8")
                    else:
                        ctrl_report = json.loads(ctrl_path.read_text(encoding="utf-8"))
                    control_rows.append(ctrl_report)

        ctrl_for_gate = control_rows if len(control_rows) == 9 else None
        passed, detail = full_gate_pass(config_rows, ctrl_for_gate)
        summary = {"route": args.route, "candidate_id": cid, "candidate": candidate,
                   "full_gate_passed": passed, **detail,
                   "config_sha256": cfg["_config_sha256"]}
        rows.append(summary)
        (args.out_root / args.route / cid / "full_gate_summary.json").write_text(
            json.dumps(summary, ensure_ascii=False, indent=1), encoding="utf-8")
        if passed:
            winners.append(cid)
        print(f"[gate] {args.route} {cid}: passed={passed} "
              f"mean={detail['mean_delta_pixel_ap']} worst={detail['worst_config_delta']} "
              f"worst_cat={detail['worst_category_mean_delta']}")

    decision = {
        "schema_version": 1,
        "program": "innovation_v2",
        "phase": "full_mpdd_gate",
        "route": args.route,
        "dataset": "mpdd",
        "role": "development",
        "seeds": args.seeds,
        "shots": SHOTS,
        "config_sha256": cfg["_config_sha256"],
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "winners": winners,
        "candidates": rows,
    }
    (args.out_root / args.route / "FULL_GATE_DECISION.json").write_text(
        json.dumps(decision, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"[summary] {args.route}: {len(winners)}/{len(rows)} passed full gate")
    return 0 if winners else 1


if __name__ == "__main__":
    sys.exit(main())
