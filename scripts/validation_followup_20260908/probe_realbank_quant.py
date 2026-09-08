"""Validation step 2: real-bank FP16/INT8 storage-roundtrip QA on the frozen A1 concat.

Pre-registered engineering QA (doc: validation follow-up step 2). Uses the exact
frozen A1 concat pipeline (pca0/whiten0/w0.5, faiss-normalize -> L2 k=1 equivalent
1-max-cos, 448 maps, Pixel-AP stride 8) on REAL MPDD test seed0 k2/k4. Only the
memory bank is cast to FP16 or symmetric per-channel INT8 and decoded back to
float32 before retrieval; queries stay float32. Report worst category, pixel AP,
NN identity, distance error p99, storage bytes ratio and decode cost.

Discipline: no test-label fitting/selection; candidates fixed before run; the
control arm must reproduce the frozen macro AP (k2 0.343706 / k4 0.388328).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "2")
os.environ.setdefault("MKL_NUM_THREADS", "2")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "2")

ROOT = Path(__file__).resolve().parents[2]
for p in (str(ROOT / "src"), str(ROOT / "scripts"), str(ROOT / "methods" / "anomalydino")):
    sys.path.insert(0, p)

import numpy as np  # noqa: E402
from sklearn.preprocessing import normalize  # noqa: E402

import evaluate_a1_feature_fusion as A1  # noqa: E402
from src.utils import dists2map  # noqa: E402

FEAT = ROOT / "outputs/dynamic_fusion/v3_direction_a"
OUT = ROOT / "experiments/dynamic_fusion/validation_followup_20260908/realbank_quant"
CATEGORIES = ("bracket_black", "bracket_brown", "bracket_white", "connector",
              "metal_plate", "tubes")
METHODS = ("control_f32", "fp16_roundtrip", "int8_pc_roundtrip")
MAP_SIZE = (448, 448)
EPS = np.float32(1e-12)

# Frozen macro reference (REAL_D5 C0 / A1 freeze), for parity only.
REF_MACRO_AP = {2: 0.343706, 4: 0.388328}

# Pre-registered engineering gates (both shots must pass for a method to be usable).
GATES = {
    "fp16_roundtrip": {"macro_ap_delta": -0.0005, "worst_cat_ap_delta": -0.002,
                       "nn_identity": 0.999, "storage_ratio": 0.51},
    "int8_pc_roundtrip": {"macro_ap_delta": -0.002, "worst_cat_ap_delta": -0.01,
                          "nn_identity": 0.99, "storage_ratio": 0.30},
}


def _utc() -> str:
    return datetime.now(timezone.utc).isoformat()


def _fin(x):
    return None if x is None else float(x)


def _jsonable(x):
    if isinstance(x, dict):
        return {str(k): _jsonable(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [_jsonable(v) for v in x]
    if isinstance(x, np.ndarray):
        return _jsonable(x.tolist())
    if isinstance(x, (np.bool_, bool)):
        return bool(x)
    if isinstance(x, (np.integer, int)):
        return int(x)
    if isinstance(x, (np.floating, float)):
        return _fin(x)
    return x


def _git_commit() -> str | None:
    try:
        return subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, check=True,
                              capture_output=True, text=True).stdout.strip()
    except Exception:
        return None


def _unit(x):
    return np.ascontiguousarray(normalize(np.asarray(x, dtype=np.float32).reshape(-1, x.shape[-1])))


def load_concat(cat: str, shot: int) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, tuple]:
    """Return (feat_unit [n,G,G,D], ref_unit [nref,G,G,D], masks, sample_ids, grid)."""
    dino_path = FEAT / f"features_vitb14_s0_k{shot}/anomalydino_visual/{cat}.npz"
    clip_path = FEAT / f"features_s0_k{shot}/anomalyclip_text/{cat}.npz"
    with np.load(dino_path, allow_pickle=False) as z:
        dino_feat = np.asarray(z["patch_features"], dtype=np.float32).copy()
        dino_ref = np.asarray(z["ref_patch_features"], dtype=np.float32).copy()
        masks = np.asarray(z["imgs_masks"], dtype=np.uint8).copy()
        ids = np.asarray(z["sample_ids"]).copy()
        grid = tuple(int(v) for v in z["grid_size"])
    with np.load(clip_path, allow_pickle=False) as z:
        clip_feat = np.asarray(z["patch_features"], dtype=np.float32).copy()
        clip_ref = np.asarray(z["ref_patch_features"], dtype=np.float32).copy()
        clip_ids = np.asarray(z["sample_ids"]).copy()
    if np.array_equal(ids, clip_ids):
        order = None
    else:
        order = np.asarray(A1.build_alignment_plan(ids, clip_ids).candidate_order, dtype=np.int64)
        clip_feat = clip_feat[order]
    clip_feat = A1.resize_patches(clip_feat, grid)
    clip_ref = A1.resize_patches(clip_ref, grid)

    w = 0.5
    df = _unit(dino_feat).reshape(dino_feat.shape)
    dr = _unit(dino_ref).reshape(dino_ref.shape)
    cf = _unit(clip_feat.reshape(-1, clip_feat.shape[-1])).reshape(clip_feat.shape)
    cr = _unit(clip_ref.reshape(-1, clip_ref.shape[-1])).reshape(clip_ref.shape)
    feat = np.concatenate([w * df, (1.0 - w) * cf], axis=-1)
    ref = np.concatenate([w * dr, (1.0 - w) * cr], axis=-1)
    n, G, Gx, D = feat.shape
    nref = ref.shape[0]
    feat_flat = _unit(feat.reshape(n * G * G, D)).reshape(n, G, G, D)
    ref_flat = _unit(ref.reshape(nref * G * G, D)).reshape(nref, G, G, D)
    return feat_flat, ref_flat, masks, ids, grid


def prepare(method: str, bank_flat: np.ndarray) -> tuple[np.ndarray, dict, float]:
    """bank_flat: unit rows [R, D]. Storage-only roundtrip; runtime decoded f32."""
    t0 = time.perf_counter()
    rows, dim = bank_flat.shape
    control_bytes = rows * dim * 4
    if method == "control_f32":
        runtime = np.asarray(bank_flat, dtype=np.float32).copy()
        meta = {"storage_bytes": runtime.nbytes, "scale_bytes": 0, "dtype": "float32",
                "storage_ratio_vs_control": 1.0, "prepare_seconds": None}
    elif method == "fp16_roundtrip":
        stored = np.asarray(bank_flat, dtype=np.float16)
        decoded = _unit(stored.astype(np.float32))
        runtime = decoded
        meta = {"storage_bytes": int(stored.nbytes), "scale_bytes": 0, "dtype": "float16_storage",
                "storage_ratio_vs_control": float(stored.nbytes / control_bytes),
                "prepare_seconds": None}
    elif method == "int8_pc_roundtrip":
        base = np.asarray(bank_flat, dtype=np.float32)
        scale = np.max(np.abs(base), axis=0).astype(np.float32) / np.float32(127.0)
        scale = np.maximum(scale, EPS).astype(np.float32)
        quant = np.clip(np.rint(base / scale), -127, 127).astype(np.int8)
        decoded = _unit(quant.astype(np.float32) * scale[None, :])
        runtime = decoded
        meta = {"storage_bytes": int(quant.nbytes + scale.nbytes), "scale_bytes": int(scale.nbytes),
                "dtype": "int8_storage+float32_scale",
                "storage_ratio_vs_control": float((quant.nbytes + scale.nbytes) / control_bytes),
                "prepare_seconds": None}
    else:
        raise ValueError(method)
    meta["prepare_seconds"] = time.perf_counter() - t0
    meta["rows"] = int(rows)
    meta["dim"] = int(dim)
    meta["control_storage_bytes"] = int(control_bytes)
    return runtime, meta, meta["prepare_seconds"]


def score(q_flat: np.ndarray, bank: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """q_flat/bank rows unit -> dists=1-max_cos [nq], nn argmax."""
    dots = q_flat @ bank.T
    nn = np.argmax(dots, axis=1).astype(np.int32)
    dists = (1.0 - dots[np.arange(dots.shape[0]), nn]).astype(np.float32)
    return dists, nn


def maps_from_dists(dists: np.ndarray, n: int, grid: tuple, map_size: tuple) -> np.ndarray:
    d = dists.reshape(n, *grid)
    return np.stack([dists2map(x, map_size) for x in d]).astype(np.float32)


def run_shot(shot: int, cats: list[str]) -> dict:
    cat_rows = []
    for cat in cats:
        feat, ref, masks, ids, grid = load_concat(cat, shot)
        n, G, Gx, D = feat.shape
        q_flat = feat.reshape(-1, D)
        ref_flat = ref.reshape(-1, D)
        bank_row_count = ref_flat.shape[0]
        # control reference scores once
        ctrl_dist, ctrl_nn = score(q_flat, ref_flat)
        ctrl_maps = maps_from_dists(ctrl_dist, n, grid, MAP_SIZE)
        ctrl_metrics = A1.compute_metrics(ctrl_maps.astype(np.float64), masks)
        per_method = {}
        for method in METHODS:
            bank, meta, prep_s = prepare(method, ref_flat)
            t_s = time.perf_counter()
            if method == "control_f32":
                cand_dist, cand_nn = ctrl_dist, ctrl_nn
                cand_maps = ctrl_maps
                cand_metrics = ctrl_metrics
            else:
                cand_dist, cand_nn = score(q_flat, bank)
                cand_maps = maps_from_dists(cand_dist, n, grid, MAP_SIZE)
                cand_metrics = A1.compute_metrics(cand_maps.astype(np.float64), masks)
            score_s = time.perf_counter() - t_s
            err = np.abs(cand_dist.astype(np.float64) - ctrl_dist.astype(np.float64))
            identity = float(np.mean(cand_nn == ctrl_nn))
            per_method[method] = {
                "pixel_ap": _fin(cand_metrics["pixel_ap"]),
                "pixel_auroc": _fin(cand_metrics["pixel_auroc"]),
                "pixel_aupro": _fin(cand_metrics["pixel_aupro"]),
                "ap_delta_vs_control": _fin(cand_metrics["pixel_ap"] - ctrl_metrics["pixel_ap"]),
                "nn_identity": identity,
                "distance_error": {"mean": _fin(err.mean()), "p99": _fin(float(np.percentile(err, 99))),
                                   "max": _fin(float(err.max()))},
                "storage": {k: meta[k] for k in ("storage_bytes", "scale_bytes", "control_storage_bytes",
                                                 "storage_ratio_vs_control", "rows", "dim")},
                "prepare_seconds": _fin(prep_s), "score_seconds": _fin(score_s),
            }
        cat_rows.append({"category": cat, "n_test": int(n), "memory_rows": int(bank_row_count),
                         "grid": list(grid), "control_full_metrics": {k: _fin(v) for k, v in ctrl_metrics.items()},
                         "methods": per_method})
        print(f"  [{cat}] shot={shot} ctrlAP={ctrl_metrics['pixel_ap']:.4f} | "
              + " | ".join(f"{m}:{per_method[m]['ap_delta_vs_control']:+.5f}"
                           for m in METHODS if m != "control_f32"), flush=True)
        del feat, ref, masks, ids

    def mean(vals):
        return _fin(float(np.mean([v for v in vals if v is not None]))) if vals else None

    aggregate = {}
    for method in METHODS:
        ap_d = [r["methods"][method]["ap_delta_vs_control"] for r in cat_rows]
        ap_c = [r["methods"][method]["pixel_ap"] for r in cat_rows]
        worst = [r["methods"][method]["ap_delta_vs_control"] for r in cat_rows]
        agg = {"macro_pixel_ap": mean(ap_c), "macro_ap_delta": mean(ap_d),
               "worst_cat_ap_delta": _fin(float(np.min(worst))) if worst else None,
               "nn_identity": mean([r["methods"][method]["nn_identity"] for r in cat_rows]),
               "dist_p99_worst_cat": _fin(float(np.max([r["methods"][method]["distance_error"]["p99"]
                                                       for r in cat_rows]))) if cat_rows else None,
               "storage_ratio": mean([r["methods"][method]["storage"]["storage_ratio_vs_control"]
                                      for r in cat_rows]),
               "prepare_seconds": mean([r["methods"][method]["prepare_seconds"] for r in cat_rows]),
               "score_seconds": mean([r["methods"][method]["score_seconds"] for r in cat_rows])}
        agg["gates"] = {}
        spec = GATES.get(method)
        if spec is not None:
            ok_macro = agg["macro_ap_delta"] is not None and agg["macro_ap_delta"] >= spec["macro_ap_delta"]
            ok_worst = agg["worst_cat_ap_delta"] is not None and agg["worst_cat_ap_delta"] >= spec["worst_cat_ap_delta"]
            ok_nn = agg["nn_identity"] is not None and agg["nn_identity"] >= spec["nn_identity"]
            ok_storage = agg["storage_ratio"] is not None and agg["storage_ratio"] <= spec["storage_ratio"]
            agg["gates"] = {"macro_ap": ok_macro, "worst_cat_ap": ok_worst, "nn_identity": ok_nn,
                            "storage": ok_storage, "all": bool(ok_macro and ok_worst and ok_nn and ok_storage)}
        aggregate[method] = agg

    control_macro = aggregate["control_f32"]["macro_pixel_ap"]
    parity = {"control_macro_pixel_ap": control_macro, "frozen_ref": REF_MACRO_AP[shot],
              "parity_ok": abs(control_macro - REF_MACRO_AP[shot]) <= 3e-4}
    return {"seed": 0, "shot": shot, "created_utc": _utc(), "parity": parity,
            "aggregate": aggregate, "per_category": cat_rows}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--shot", type=int, required=True, choices=[2, 4])
    ap.add_argument("--cats", default=None)
    args = ap.parse_args()
    cats = [c.strip() for c in args.cats.split(",")] if args.cats else list(CATEGORIES)
    result = run_shot(args.shot, cats)
    result["protocol"] = {
        "script": str(Path(__file__).resolve().relative_to(ROOT)),
        "git_commit": _git_commit(), "seed": 0, "shot": args.shot, "cats": cats,
        "frozen_config": "A1 concat pca0 whiten0 w0.5; faiss k=1 (1-max-cos); 448 maps; Pixel-AP stride8",
        "quantization": "memory-only roundtrip; fp16 astype; int8 symmetric per-channel scale max(|.|)/127 rint clip; query f32",
        "test_labels_fit_or_select": False,
    }
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / f"RESULTS_s0_k{args.shot}.json").write_text(
        json.dumps(_jsonable(result), ensure_ascii=False, indent=1), encoding="utf-8")
    print(json.dumps({"shot": args.shot, "parity": result["parity"],
                      "macro": {m: {"ap": result["aggregate"][m]["macro_ap_delta"],
                                    "worst": result["aggregate"][m]["worst_cat_ap_delta"],
                                    "nn": result["aggregate"][m]["nn_identity"],
                                    "gates": result["aggregate"][m]["gates"]}
                                for m in ("fp16_roundtrip", "int8_pc_roundtrip")}},
                     ensure_ascii=False, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
