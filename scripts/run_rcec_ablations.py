"""RCEC v1 — interpretability ablations for the selected candidate (Phase 4).

Runs on the selected MPDD candidate:
  * correct pairing (already in development_mpdd);
  * three shuffled-pairing seeds (fixed RNG seeds {0,1,2});
  * DINO duplicate 1536-D control;
  * A1 / DINO-only / CLIP-only reference rows.

Interpretability gate (task book 5.2): correct-pairing mean Pixel-AP must be
>= +0.003 above the mean of the three shuffled runs, and dimension duplication
must not produce spurious gains.
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

from evaluate_a1_feature_fusion import load_features, resize_patches  # noqa: E402
from rcec_common import (  # noqa: E402
    DEV_MPDD_ROOT,
    candidate_id as rcec_candidate_id,
    candidates_from_config,
    dirs_for,
    load_config,
    manifest_for,
    reference_ids_for,
    sha256_file,
)
from src.utils import dists2map  # noqa: E402
from industrial_ad.fusion import rcec  # noqa: E402

ABL_ROOT = DEV_MPDD_ROOT.parent / "ablations"


def _pixel_ap(maps: np.ndarray, masks: np.ndarray) -> float:
    from evaluate_a1_feature_fusion import compute_metrics

    return compute_metrics(maps.astype(np.float64), masks)["pixel_ap"]


def _evaluate_config_abl(dataset: str, seed: int, shot: int, cand: dict, cfg: dict,
                         shuffle_seed: int | None) -> dict:
    """Like rcec_common.evaluate_config but with optional shuffled pairing."""
    from rcec_common import evaluate_category

    dino_dir, clip_dir = dirs_for(dataset, seed, shot)
    manifest = manifest_for(dataset)
    per_category = []
    for cat_path in sorted(dino_dir.glob("*.npz")):
        if cat_path.stem == "export_report":
            continue
        cat = cat_path.stem
        clip_path = clip_dir / f"{cat}.npz"
        dino = load_features(cat_path)
        clip = load_features(clip_path)
        ref_ids = reference_ids_for(manifest, cat, seed, shot)
        # build aligned features + memory with optional shuffled CLIP pairing
        aligned = rcec.align_and_normalize_paired_features(
            dino_patch=dino["patch_features"], clip_patch=clip["patch_features"],
            dino_ref=dino["ref_patch_features"], clip_ref=clip["ref_patch_features"],
            dino_sample_ids=dino["sample_ids"], clip_sample_ids=clip["sample_ids"],
            dino_grid=dino["grid_size"], resize_fn=resize_patches)
        memory = rcec.build_paired_reference_memory(aligned["d_ref"], aligned["c_ref"], len(ref_ids))
        if shuffle_seed is not None:
            rng = np.random.default_rng(shuffle_seed)
            memory = rcec.shuffled_paired_memory(memory, rng)

        # recompute scores with the (possibly shuffled) memory
        d_feat, c_feat = aligned["d_feat"], aligned["c_feat"]
        grid = aligned["grid"]
        s_a1 = rcec.compute_a1_dists(d_feat, c_feat, memory)
        s_dino = rcec.compute_dino_dists(d_feat, memory)
        s_dup = rcec.compute_dino_duplicate_dists(d_feat, memory)
        r_cd, r_dc = rcec.compute_conditional_scores(d_feat, c_feat, memory, cand["direction"], cand["k"])
        r_cond = r_cd if cand["direction"] == "dino_to_clip" else 0.5 * (r_cd + r_dc)
        loo = rcec.compute_reference_loo_statistics(
            aligned["d_ref"], aligned["c_ref"], len(ref_ids),
            direction=cand["direction"], k=cand["k"], shot=shot)
        stats_a1 = rcec.compute_reference_stats(loo["a1_loo"])
        stats_cond = rcec.compute_reference_stats(loo["cond_loo"])
        s_rcec = rcec.combine_rcec_scores(s_a1, r_cond, stats_a1, stats_cond, cand["lambda"])

        n = d_feat.shape[0]
        h, w = grid
        def _maps(scores):
            return np.stack([dists2map(scores.reshape(n, h, w)[i], (448, 448))
                             for i in range(n)]).astype(np.float32)

        per_category.append({
            "category": cat,
            "rcec": _pixel_ap(_maps(s_rcec), dino["imgs_masks"]),
            "a1": _pixel_ap(_maps(s_a1), dino["imgs_masks"]),
            "dino": _pixel_ap(_maps(s_dino), dino["imgs_masks"]),
            "dino_dup": _pixel_ap(_maps(s_dup), dino["imgs_masks"]),
        })
    out = {k: float(np.mean([r[k] for r in per_category])) for k in ("rcec", "a1", "dino", "dino_dup")}
    out["per_category"] = per_category
    return out


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=ROOT / "configs" / "rcec_v1.yaml")
    args = parser.parse_args()
    cfg = load_config(args.config)

    full = json.loads((DEV_MPDD_ROOT / "FULL_MATRIX_REPORT.json").read_text(encoding="utf-8"))
    selected = full["selected_candidate"]
    if not selected:
        raise SystemExit("no selected candidate; ablations require a winner")
    cand = next(c for c in candidates_from_config(cfg) if rcec_candidate_id(c) == selected)

    report = {"selected_candidate": selected, "config_sha256": cfg["_config_sha256"],
              "created_at_utc": datetime.now(timezone.utc).isoformat(), "datasets": {}}
    rows_by_ds = {}
    for dataset in ("mpdd",):
        ds_rows = []
        for seed in cfg["seeds"]:
            for shot in cfg["shots"]:
                correct = _evaluate_config_abl(dataset, seed, shot, cand, cfg, shuffle_seed=None)
                shuffled = [_evaluate_config_abl(dataset, seed, shot, cand, cfg, shuffle_seed=s)
                            for s in (0, 1, 2)]
                ds_rows.append({
                    "seed": seed, "shot": shot,
                    "correct_rcec_pixel_ap": correct["rcec"],
                    "a1_pixel_ap": correct["a1"],
                    "dino_pixel_ap": correct["dino"],
                    "dino_dup_pixel_ap": correct["dino_dup"],
                    "shuffle0_pixel_ap": shuffled[0]["rcec"],
                    "shuffle1_pixel_ap": shuffled[1]["rcec"],
                    "shuffle2_pixel_ap": shuffled[2]["rcec"],
                })
        rows_by_ds[dataset] = ds_rows
        report["datasets"][dataset] = ds_rows

    mean = {k: float(np.mean([r[k] for r in rows_by_ds["mpdd"]]))
            for k in ("correct_rcec_pixel_ap", "a1_pixel_ap", "dino_pixel_ap",
                      "dino_dup_pixel_ap", "shuffle0_pixel_ap", "shuffle1_pixel_ap",
                      "shuffle2_pixel_ap")}
    shuffle_mean = float(np.mean([mean[f"shuffle{i}_pixel_ap"] for i in (0, 1, 2)]))
    report["mpdd_mean"] = mean
    report["mpdd_mean_shuffled_pairing"] = shuffle_mean
    report["interpretability_delta_correct_minus_shuffle"] = round(
        mean["correct_rcec_pixel_ap"] - shuffle_mean, 6)
    report["interpretability_gate_passed"] = bool(
        mean["correct_rcec_pixel_ap"] - shuffle_mean >= 0.003)
    report["dimension_control_passed"] = bool(
        abs(mean["dino_pixel_ap"] - mean["dino_dup_pixel_ap"]) < 1e-5)

    out = ABL_ROOT / "pairing_ablation.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
