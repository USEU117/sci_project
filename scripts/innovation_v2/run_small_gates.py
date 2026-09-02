"""A2 innovation_v2 — Small Gate runner (MPDD seed0 x shot {1,2,4}).

Task book sections 3.2 / Wave 1. Evaluates every pre-registered candidate of one
route on MPDD seed0 x shot {1,2,4} across all categories, computes the six
metrics vs the frozen A1, runs the small-gate conditions, and compares against
the route-specific sham/control. Per-candidate marker files are bound to the
config SHA256 (resume-safe, no stale reuse).
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
sys.path.insert(0, str(ROOT / "methods" / "anomalydino"))

from industrial_ad.innovation_v2 import common  # noqa: E402
from industrial_ad.innovation_v2 import local_density  # noqa: E402
from industrial_ad.innovation_v2 import deformable_spatial_memory as dsam  # noqa: E402
from industrial_ad.innovation_v2 import consensus_query_adaptation as cecqa  # noqa: E402
from industrial_ad.innovation_v2 import feature_graph_refinement as fagr  # noqa: E402
from industrial_ad.innovation_v2 import equivariant_augmentation as deva  # noqa: E402
from industrial_ad.innovation_v2 import predictive_adapter as ncpra  # noqa: E402

SHOTS = [1, 2, 4]
SEED0 = 0
DEVA_CACHE = ROOT / "outputs" / "dynamic_fusion" / "innovation_v2_deva"


def candidate_id(route: str, cand: dict) -> str:
    if route == "A_LNDC":
        return f"k{cand['k']}"
    if route == "B_DSAM":
        return f"{cand['alignment']}_r{cand['r']}"
    if route == "C_CEQA":
        return f"q{cand['q']:g}_eta{cand['eta']:g}"
    if route == "F_FAGR":
        return f"mu{cand['mu']:g}_iter{cand['iters']}"
    if route == "D_DEVA":
        return f"{cand['pack']}_tau{cand['tau']:g}"
    if route == "E_NCPRA":
        return f"r{cand['r']}_lam{cand['lambda']:g}"
    return json.dumps(cand, sort_keys=True)


def candidates_from_config(cfg: dict) -> list[dict]:
    return [dict(c) for c in cfg["candidates"]]


def route_spec(route: str) -> dict:
    if route == "A_LNDC":
        return {"score_fn": local_density.score_lndc,
                "control_fn": local_density.score_lndc_global_sham,
                "control_name": "global_density_sham"}
    if route == "B_DSAM":
        return {"score_fn": dsam.score_dsam,
                "control_fn": dsam.score_dsam_fixed_loc,
                "control_name": "no_alignment_local_window"}
    if route == "C_CEQA":
        return {"score_fn": cecqa.score_cecqa,
                "control_fn": cecqa.score_cecqa_a1_rank_only,
                "control_name": "a1_rank_only"}
    if route == "F_FAGR":
        return {"score_fn": fagr.score_fagr,
                "map_fn": fagr.bilinear_map,
                "control_fn": fagr.score_fagr_uniform,
                "control_name": "uniform_smoothing"}
    if route == "D_DEVA":
        return {"score_fn": _deva_score(tau_from_candidate=True),
                "control_fn": _deva_score(tau=-1.0),
                "control_name": "unfiltered_augmentation"}
    if route == "E_NCPRA":
        return {"score_fn": ncpra.score_ncpra,
                "control_fn": None,
                "control_name": "linear_ridge"}
    raise common.InnovationError(f"small gate not implemented for route {route}")


def _deva_score(tau_from_candidate: bool = False, tau: float | None = None):
    """Factory returning a DEVA score_fn reading the pre-extracted cache.

    tau_from_candidate=True reads ``candidate["tau"]``; otherwise a fixed tau
    (used for the unfiltered control, tau=-1.0).
    """
    def _score(aligned, candidate, cfg):
        pack = candidate["pack"]
        shot = int(cfg["_shot"])
        eff_tau = candidate["tau"] if tau_from_candidate else tau
        cache = DEVA_CACHE / f"{aligned.category}_k{shot}_{pack}.npz"
        if not cache.is_file():
            raise common.InnovationError(
                f"missing DEVA cache {cache}; run export_deva_references.py first")
        with np.load(cache, allow_pickle=False) as d:
            aug_d = d["aug_d"]
            aug_c = d["aug_c"]
            vd = d["valid_d"]
            vc = d["valid_c"]
            src = d["source_ref"]
        ident = DEVA_CACHE / f"{aligned.category}_k{shot}_identity.npz"
        clip_ref_grid = None
        if ident.is_file():
            with np.load(ident, allow_pickle=False) as i:
                ic = i["identity_c"]
            if ic.ndim == 3:  # (n_refs, H*W, D) -> (n_refs, H, W, D)
                side = int(round((ic.shape[1]) ** 0.5))
                if side * side == ic.shape[1]:
                    ic = ic.reshape(ic.shape[0], side, side, -1)
            if ic.ndim == 4 and ic.shape[0] == aligned.n_references:
                clip_ref_grid = np.ascontiguousarray(ic, dtype=np.float32)
        d_full, c_full, meta = deva.build_augmented_memory(
            aligned, aug_d, aug_c, vd, vc, src, float(eff_tau),
            clip_ref_grid=clip_ref_grid)
        s = deva.score_deva_memory(aligned, d_full, c_full)
        diag = {"pack": pack, "tau": float(eff_tau), **meta}
        return s, diag
    return _score


def evaluate_config(
    route: str, cfg: dict, candidate: dict, shot: int,
    score_fn, map_fn, seed: int = SEED0,
) -> dict:
    """Evaluate every MPDD category for (candidate, shot, seed)."""
    dataset = "mpdd"
    dino_dir, clip_dir = common.dirs_for(dataset, seed, shot)
    manifest = common.manifest_for(dataset)
    cat_paths = sorted(p for p in dino_dir.glob("*.npz") if p.stem != "export_report")
    per_category = []
    cfg_eval = dict(cfg)
    cfg_eval["_shot"] = shot

    for cat_path in cat_paths:
        cat = cat_path.stem
        clip_path = clip_dir / f"{cat}.npz"
        if not clip_path.is_file():
            raise common.InnovationError(f"missing clip features: {clip_path}")
        ref_ids = common.reference_ids_for(manifest, cat, seed, shot)
        dino = common.load_features(cat_path)
        clip = common.load_features(clip_path)
        aligned = common.align_features(dino, clip, ref_ids)
        aligned.category = cat
        report = common.evaluate_category_generic(
            aligned=aligned, ref_ids=ref_ids, seed=seed, shot=shot,
            route_id=route, candidate=candidate, cfg=cfg_eval, category=cat,
            score_fn=score_fn, map_fn=map_fn)
        common.attach_metrics(report, np.asarray(dino["imgs_masks"], dtype=np.uint8),
                              np.asarray(dino["gt_sp"]))
        report["candidate_id"] = candidate_id(route, candidate)
        per_category.append(report)

    return aggregate_config(route, cfg, candidate, dataset, seed, shot, per_category)


def aggregate_config(route, cfg, candidate, dataset, seed, shot, per_category) -> dict:
    def mean_metric(group: str, key: str, method: str = "new") -> float | None:
        vals = [r["metrics"][method][group][key] for r in per_category]
        vals = [v for v in vals if v is not None]
        return None if not vals else round(float(np.mean(vals)), 6)

    def mean_delta(group: str, key: str) -> float | None:
        vals = [r["metrics"]["delta_vs_a1"][group][key] for r in per_category]
        vals = [v for v in vals if v is not None]
        return None if not vals else round(float(np.mean(vals)), 6)

    def block(method: str) -> dict:
        return {
            "pixel": {k: mean_metric("pixel", k, method)
                      for k in ("pixel_auroc", "pixel_ap", "pixel_aupro")},
            "image": {k: mean_metric("image", k, method)
                      for k in ("image_auroc", "image_ap", "image_f1_max")},
        }

    first = per_category[0]
    return {
        "schema_version": 1,
        "program": "innovation_v2",
        "route": route,
        "candidate_id": candidate_id(route, candidate),
        "candidate": candidate,
        "dataset": dataset,
        "dataset_role": "development",
        "seed": seed,
        "shot": shot,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "config_sha256": cfg["_config_sha256"],
        "code_sha256": common.sha256_file(Path(__file__)),
        "inputs": {
            "manifest_sha256": common.sha256_file(ROOT / "data" / "splits" / dataset / "manifest.json"),
            "dino_cache_sha256": common.sha256_file(
                common.dirs_for(dataset, seed, shot)[0] / f"{first['category']}.npz"),
            "clip_cache_sha256": common.sha256_file(
                common.dirs_for(dataset, seed, shot)[1] / f"{first['category']}.npz"),
        },
        "leakage_flags": {
            "test_labels_used_by_method": False,
            "test_masks_used_by_method": False,
            "test_distribution_used_for_calibration": False,
            "validation_dataset_used_for_tuning": False,
            "category_specific_test_rules_used": False,
        },
        "metrics": {
            "new": block("new"),
            "a1": block("a1"),
            "dino": block("dino"),
            "delta_vs_a1": {
                "pixel": {k: mean_delta("pixel", k)
                          for k in ("pixel_auroc", "pixel_ap", "pixel_aupro")},
                "image": {k: mean_delta("image", k)
                          for k in ("image_auroc", "image_ap", "image_f1_max")},
            },
        },
        "per_category": per_category,
        "checks": {"all_finite": all(r["checks"]["no_nan_inf_scores"] for r in per_category)},
    }


def small_gate_decision(route: str, cfg: dict, shot_reports: list[dict],
                        control_shot_reports: list[dict] | None) -> tuple[bool, dict]:
    """Task book 3.2 small-gate conditions for one candidate."""
    if len(shot_reports) != 3:
        return False, {"reason": f"expected 3 shots, got {len(shot_reports)}"}
    deltas = [r["metrics"]["delta_vs_a1"]["pixel"]["pixel_ap"] for r in shot_reports]
    if any(d is None for d in deltas):
        return False, {"reason": "missing pixel_ap delta"}
    mean_d = float(np.mean(deltas))
    n_pos = int(sum(1 for d in deltas if d > 0))
    worst = min(deltas)

    gate = cfg.get("small_gate", {})
    mean_min = float(gate.get("mean_delta_min", 0.003))
    n_pos_min = int(gate.get("n_positive_shots_min", 2))
    worst_min = float(gate.get("worst_shot_delta_min", -0.005))

    control_mean = None
    control_beats_candidate = False
    if control_shot_reports is not None:
        ctrl = [r["metrics"]["delta_vs_a1"]["pixel"]["pixel_ap"] for r in control_shot_reports]
        ctrl = [c for c in ctrl if c is not None]
        control_mean = float(np.mean(ctrl)) if ctrl else None
        control_beats_candidate = (control_mean is not None and control_mean >= mean_d)

    ok = (
        mean_d >= mean_min
        and n_pos >= n_pos_min
        and worst >= worst_min
        and all(r["checks"]["all_finite"] for r in shot_reports)
        and all(not any(r["leakage_flags"].values()) for r in shot_reports)
        and not control_beats_candidate
    )
    detail = {
        "mean_delta_pixel_ap": round(mean_d, 6),
        "n_positive_shots": n_pos,
        "worst_shot_delta": round(worst, 6),
        "deltas": [round(d, 6) for d in deltas],
        "control_mean_delta_pixel_ap": round(control_mean, 6) if control_mean is not None else None,
        "control_name": cfg.get("_control_name"),
        "candidate_beats_control": (control_mean is not None and mean_d > control_mean),
    }
    return ok, detail


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--route", required=True,
                        choices=("A_LNDC", "B_DSAM", "C_CEQA", "D_DEVA", "E_NCPRA", "F_FAGR"))
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--out-root", type=Path, default=common.EXPERIMENT_ROOT / "01_small_gates")
    parser.add_argument("--candidate-ids", nargs="*", default=None,
                        help="restrict to these candidate ids (default: all)")
    parser.add_argument("--validate-only", action="store_true",
                        help="Wave 0: reproduce one MPDD s0/k1 A1 config and exit")
    args = parser.parse_args()

    common.assert_development_only("mpdd")
    cfg = common.load_config(args.config)
    spec = route_spec(args.route)

    if args.validate_only:
        return validate_a1_regression()

    out_root = args.out_root / args.route
    out_root.mkdir(parents=True, exist_ok=True)
    candidates = candidates_from_config(cfg)
    if args.candidate_ids:
        wanted = set(args.candidate_ids)
        candidates = [c for c in candidates if candidate_id(args.route, c) in wanted]

    winners = []
    rows = []
    for cand in candidates:
        cid = candidate_id(args.route, cand)
        shot_reports = []
        control_reports = []
        for shot in SHOTS:
            shot_dir = out_root / cid / f"s0_k{shot}"
            shot_dir.mkdir(parents=True, exist_ok=True)
            report_path = shot_dir / "report.json"
            ctrl_dir = shot_dir / "_control"
            ctrl_path = ctrl_dir / "report.json"
            cfg_sha = cfg["_config_sha256"]

            def _fresh(report: Path, marker_path: Path) -> bool:
                if not (report.is_file() and marker_path.is_file()):
                    return True
                m = json.loads(marker_path.read_text(encoding="utf-8"))
                return not (m.get("config_sha256") == cfg_sha and m.get("candidate_id") == cid)

            if not _fresh(report_path, shot_dir / "marker.json"):
                shot_reports.append(json.loads(report_path.read_text(encoding="utf-8")))
                ctrl_marker = ctrl_dir / "marker.json"
                if ctrl_path.is_file() and ctrl_marker.is_file() and json.loads(
                        ctrl_marker.read_text(encoding="utf-8")).get("config_sha256") == cfg_sha:
                    control_reports.append(json.loads(ctrl_path.read_text(encoding="utf-8")))
                print(f"[resume] {args.route} {cid} s0_k{shot}")
                continue

            report = evaluate_config(args.route, cfg, cand, shot, spec["score_fn"],
                                     spec.get("map_fn"))
            (shot_dir / "marker.json").write_text(json.dumps(
                {"config_sha256": cfg_sha, "candidate_id": cid, "shot": shot,
                 "done_utc": datetime.now(timezone.utc).isoformat()}, ensure_ascii=False),
                encoding="utf-8")
            report_path.write_text(json.dumps(report, ensure_ascii=False, indent=1), encoding="utf-8")
            shot_reports.append(report)
            print(f"[done] {args.route} {cid} s0_k{shot} "
                  f"dPixelAP={report['metrics']['delta_vs_a1']['pixel']['pixel_ap']}")

            ctrl_dir.mkdir(parents=True, exist_ok=True)
            if spec.get("control_fn") is None:
                continue
            if not (ctrl_path.is_file() and (ctrl_dir / "marker.json").is_file()
                    and json.loads((ctrl_dir / "marker.json").read_text(encoding="utf-8"))
                    .get("config_sha256") == cfg_sha):
                ctrl_cfg = dict(cfg)
                ctrl_cfg["_shot"] = shot
                ctrl_report = evaluate_config(
                    args.route, ctrl_cfg, cand, shot, spec["control_fn"], spec.get("map_fn"))
                (ctrl_dir / "marker.json").write_text(json.dumps(
                    {"config_sha256": cfg_sha, "candidate_id": cid, "shot": shot,
                     "control": spec["control_name"]}, ensure_ascii=False), encoding="utf-8")
                ctrl_path.write_text(json.dumps(ctrl_report, ensure_ascii=False, indent=1),
                                     encoding="utf-8")
            else:
                ctrl_report = json.loads(ctrl_path.read_text(encoding="utf-8"))
            control_reports.append(ctrl_report)

        cfg["_control_name"] = spec["control_name"]
        ctrl_for_gate = control_reports if len(control_reports) == 3 else None
        passed, detail = small_gate_decision(args.route, cfg, shot_reports, ctrl_for_gate)
        summary = {
            "route": args.route,
            "candidate_id": cid,
            "candidate": cand,
            "small_gate_passed": passed,
            **detail,
            "config_sha256": cfg["_config_sha256"],
        }
        rows.append(summary)
        (out_root / cid / "small_gate_summary.json").write_text(
            json.dumps(summary, ensure_ascii=False, indent=1), encoding="utf-8")
        if passed:
            winners.append(cid)
        print(f"[gate] {args.route} {cid}: passed={passed} "
              f"mean={detail['mean_delta_pixel_ap']} worst={detail['worst_shot_delta']} "
              f"ctrl={detail['control_mean_delta_pixel_ap']}")

    decision = {
        "schema_version": 1,
        "program": "innovation_v2",
        "phase": "small_gate",
        "route": args.route,
        "dataset": "mpdd",
        "role": "development",
        "shots": SHOTS,
        "config_sha256": cfg["_config_sha256"],
        "control": spec["control_name"],
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "winners": winners,
        "candidates": rows,
        "stop_rule": "all candidates failed -> route early stop, do not run full matrix",
    }
    (out_root / "SMALL_GATE_DECISION.json").write_text(
        json.dumps(decision, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"[summary] {args.route}: {len(winners)}/{len(candidates)} passed")
    return 0 if winners else 1


def validate_a1_regression() -> int:
    """Wave 0: reproduce one MPDD s0/k1 A1 config and compare with fuse_category."""
    from evaluate_a1_feature_fusion import fuse_category

    dataset, seed, shot = "mpdd", 0, 1
    manifest = common.manifest_for(dataset)
    dino_dir, clip_dir = common.dirs_for(dataset, seed, shot)
    cat_paths = sorted(p for p in dino_dir.glob("*.npz") if p.stem != "export_report")
    cat = cat_paths[0].stem
    ref_ids = common.reference_ids_for(manifest, cat, seed, shot)
    dino = common.load_features(dino_dir / f"{cat}.npz")
    clip = common.load_features(clip_dir / f"{cat}.npz")

    aligned = common.align_features(dino, clip, ref_ids)
    s_a1 = common.a1_grid(aligned)
    maps_new = common.grids_to_maps(s_a1, (448, 448))

    maps_frozen = fuse_category(dino, clip, "concat", pca_dim=0, whiten=False,
                                map_size=(448, 448), dino_weight=0.5)
    err = float(np.max(np.abs(maps_new - maps_frozen)))
    print(json.dumps({
        "category": cat,
        "max_abs_map_error": err,
        "a1_regression_ok": err < 1e-4,
        "n_test_images": int(dino["patch_features"].shape[0]),
    }, indent=1))
    return 0 if err < 1e-4 else 2


if __name__ == "__main__":
    sys.exit(main())
