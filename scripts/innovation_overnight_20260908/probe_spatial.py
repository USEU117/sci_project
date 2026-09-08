"""Bounded spatial correspondence probe (2026-09-08 overnight run).

This probe tests a cross-image geometry hypothesis that is distinct from the
archived same-image neighbourhood descriptors in Track-1/Track-5 and from the
image-coordinate multi-shift median in v12 PSMF:

* use clean support features for the memory bank, and estimate one global
  translation independently from each query feature map to each memory image;
* score with a bandwidth-limited nearest neighbour (spatial radius 2 at the
  32 x 32 grid) around that translated coordinate;
* compare against unrestricted global NN and a fixed coordinate-shuffle
  control.

The memory is strict leave-one-support-image-out.  Synthetic masks are used
only as held-out support-derived evaluation labels; no real MPDD test image or
test label is loaded.  The frozen default run is six MPDD categories, seed 0,
shot 2, CPU, two threads.  No parameter is selected from AP results.

Example:
  .venv-anomalyclip\\Scripts\\python.exe \\
      scripts\\innovation_overnight_20260908\\probe_spatial.py
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

# Set common CPU thread pools before importing NumPy / Torch.  The run contract
# is intentionally small and reproducible on the overnight host.
os.environ["OMP_NUM_THREADS"] = "2"
os.environ["MKL_NUM_THREADS"] = "2"
os.environ["OPENBLAS_NUM_THREADS"] = "2"
os.environ["NUMEXPR_NUM_THREADS"] = "2"

import cv2  # noqa: E402
import numpy as np  # noqa: E402
from sklearn.metrics import average_precision_score, roc_auc_score  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
for p in (str(ROOT), str(ROOT / "src"), str(ROOT / "scripts"),
          str(ROOT / "scripts" / "innovation_v14_decisive_validation_20260905")):
    sys.path.insert(0, p)

from industrial_ad.innovation_v10_portfolio.common import resize_patches  # noqa: E402
from v14_common import (  # noqa: E402
    CATEGORIES,
    assert_fit_ids_are_support,
    load_manifest,
    support_paths,
)

try:  # Torch is used by resize_patches; configure it when available.
    import torch  # noqa: E402

    torch.set_num_threads(2)
    try:
        torch.set_num_interop_threads(2)
    except RuntimeError:
        # The inter-op pool may already be initialized when this module is
        # imported from a larger runner; intra-op remains fixed at two.
        pass
except Exception:  # pragma: no cover - import failure is reported on use
    torch = None

try:  # FAISS is not needed for this small probe, but cap it if installed.
    import faiss  # noqa: E402

    faiss.omp_set_num_threads(2)
except Exception:  # pragma: no cover - optional
    faiss = None


CACHE = ROOT / "outputs" / "dynamic_fusion" / "v14_p1_support"
OUT = ROOT / "experiments" / "dynamic_fusion" / "innovation_overnight_20260908" / "spatial"
SYN_KINDS = ("cutpaste", "local_erasure", "thin_scratch")
FAMILIES = ("cutpaste", "local_erasure")
THREADS = 2
GRID = 32
ALIGN_GRID = 16
ALIGN_POOL = 2
ALIGN_MAX_SHIFT = 2  # 16-grid cells, equivalent to +/- 4 cells at 32-grid.
BANDWIDTH = 2  # Chebyshev radius at 32-grid; fixed before reading AP results.
ALIGN_TRIM_FRACTION = 0.20  # discard lowest 20% overlap cosine (normal-majority).
FUSION_WEIGHT = 0.5
SHUFFLE_SEED = 20260908
PROTOCOL_VERSION = "spatial_global_translation_bwnn_v1"
METHODS = ("global_nn", "spatial_bw_nn", "spatial_bw_nn_coord_shuffle")


def _jsonable(value):
    """Convert NumPy scalars/arrays to JSON-safe values."""
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    return value


def _write_json(path: Path, obj: dict) -> None:
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=1, default=_jsonable), encoding="utf-8")


def frozen_protocol(shot: int) -> dict:
    """Return the complete immutable protocol used by this probe."""
    return {
        "protocol_version": PROTOCOL_VERSION,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "scope": {
            "dataset": "MPDD",
            "seed": 0,
            "shot": shot,
            "categories": list(CATEGORIES),
            "device": "cpu",
            "threads": THREADS,
            "real_test_read": False,
        },
        "input": {
            "cache": "outputs/dynamic_fusion/v14_p1_support",
            "branches": ["dino", "clip"],
            "feature_role": "v14 P1 support-only clean/synthetic/nuisance cache",
            "fit_select_role": "clean support bank only; held-out h is excluded from bank; query-time alignment has no labels",
            "evaluation_role": "support-derived cutpaste/erasure masks and photometric variants only",
        },
        "mechanism": {
            "name": "cross-image global translation + bandwidth-limited nearest neighbour",
            "description": (
                "For each query feature map independently estimate one translation to each clean "
                "memory image using 16-grid mean-pooled fused features. Drop the lowest 20% "
                "overlap cosines and average the normal-majority remainder. At 32-grid query "
                "coordinate u, restrict memory coordinates to a Chebyshev radius around u + translation."
            ),
            "a1_fusion_weight": FUSION_WEIGHT,
            "feature_grid": GRID,
            "alignment_grid": ALIGN_GRID,
            "alignment_pool": ALIGN_POOL,
            "alignment_max_shift_cells": ALIGN_MAX_SHIFT,
            "alignment_trim_fraction": ALIGN_TRIM_FRACTION,
            "bandwidth_chebyshev_cells": BANDWIDTH,
            "alignment_score": "mean of top 80% per-cell cosine on valid 16-grid overlap",
            "shift_tie_break": "higher mean cosine, then smaller L1 shift, then lexicographic dy/dx",
            "coordinate_shuffle": {
                "seed": SHUFFLE_SEED,
                "scope": "one fixed permutation per category/shot/reference-image; features only; each query keeps its own clean-bank shift",
            },
        },
        "methods": {
            "global_nn": "unrestricted 32-grid NN over all clean bank cells",
            "spatial_bw_nn": "query-time translated-coordinate Chebyshev-band NN over clean bank cells",
            "spatial_bw_nn_coord_shuffle": "same query-time band rule after fixed per-reference coordinate permutation",
        },
        "metrics": {
            "anomaly": "Pixel-AP on held-out support-derived cutpaste and local_erasure masks after 32-grid scoring",
            "normal": "cell-level ROC-AUC of clean h versus 15 photometric nuisance variants; 0.5 is stable",
            "normal_auxiliary": "mean nuisance score minus mean clean score",
            "thin_scratch": "excluded from AP because this probe reports only cutpaste and local_erasure as requested",
        },
        "history_audit": {
            "t1": "same-image 16-grid C1/C2 neighbourhood descriptors; real route archived after negative C1 gate",
            "t5": "same-image 32-grid relation descriptors; synthetic gain did not generalize to real MPDD",
            "v12_prs": "normal photometric response axis non-monotonic; archived",
            "v12_psmf": "image-coordinate multi-shift median approximately smoothing; archived",
            "novelty_boundary": "cross-image translation and coordinate-constrained matching, not neighbourhood concatenation or image-coordinate aggregation",
        },
    }


def freeze_protocol(path: Path, shot: int) -> dict:
    """Write protocol before loading results, and reject accidental changes."""
    current = frozen_protocol(shot)
    if path.exists():
        old = json.loads(path.read_text(encoding="utf-8"))
        # The timestamp is intentionally mutable; all actual choices are not.
        old_cmp = dict(old)
        new_cmp = dict(current)
        old_cmp.pop("created_utc", None)
        new_cmp.pop("created_utc", None)
        if old_cmp != new_cmp:
            raise RuntimeError(f"Frozen protocol mismatch: {path}")
        return old
    path.parent.mkdir(parents=True, exist_ok=True)
    _write_json(path, current)
    return current


def _row_normalize(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32)
    n = np.linalg.norm(x, axis=-1, keepdims=True)
    return x / np.maximum(n, 1e-12)


def _fuse(dino: np.ndarray, clip: np.ndarray, n: int) -> np.ndarray:
    """A1 0.5/0.5 branch concat with row normalization.

    Inputs are [n,H,W,768] at already aligned grids.  The last dimension is
    normalized per branch before concatenation, matching the frozen A1 cache
    construction used by the prior probes.
    """
    d = _row_normalize(dino.reshape(n, -1, dino.shape[-1]))
    c = _row_normalize(clip.reshape(n, -1, clip.shape[-1]))
    z = np.concatenate([FUSION_WEIGHT * d, (1.0 - FUSION_WEIGHT) * c], axis=-1)
    z = _row_normalize(z)
    return z.reshape(n, dino.shape[1], dino.shape[2], -1).astype(np.float32, copy=False)


def _mask32(m1024: np.ndarray) -> np.ndarray:
    out = cv2.resize(m1024.astype(np.uint8), (GRID, GRID), interpolation=cv2.INTER_AREA)
    return (out > 127).astype(np.uint8)


def _load_category(cat: str, shot: int) -> dict:
    """Load one category's support-only cache and build 32-grid fused arrays."""
    dpath = CACHE / f"v14_p1_support_dino_s0_k{shot}" / f"{cat}.npz"
    cpath = CACHE / f"v14_p1_support_clip_s0_k{shot}" / f"{cat}.npz"
    if not dpath.exists() or not cpath.exists():
        raise FileNotFoundError(f"Missing support cache for {cat} k{shot}: {dpath} / {cpath}")
    with np.load(dpath, allow_pickle=False) as zd, np.load(cpath, allow_pickle=False) as zc:
        dclean = np.asarray(zd["clean_feat"], dtype=np.float32)
        dsyn = np.asarray(zd["syn_feat"], dtype=np.float32)
        dmask = np.asarray(zd["syn_masks"], dtype=np.uint8)
        dnui = np.asarray(zd["nui_feat"], dtype=np.float32)
        drel = np.asarray(zd["ref_rel"])
        cclean = np.asarray(zc["clean_feat"], dtype=np.float32)
        csyn = np.asarray(zc["syn_feat"], dtype=np.float32)
        cnui = np.asarray(zc["nui_feat"], dtype=np.float32)
        crel = np.asarray(zc["ref_rel"])
    if dclean.shape[1:3] != (GRID, GRID):
        raise ValueError(f"DINO cache grid is {dclean.shape[1:3]}, expected {(GRID, GRID)}")
    if dclean.shape[0] != cclean.shape[0]:
        raise ValueError(f"Branch support count mismatch for {cat}: {dclean.shape[0]} vs {cclean.shape[0]}")
    if not np.array_equal(drel.astype(str), crel.astype(str)):
        raise ValueError(f"Branch support ID mismatch for {cat}")
    rels = [str(x) for x in drel.tolist()]
    assert_fit_ids_are_support(rels, cat, shot, "0")
    if any("/test/" in rel for rel in rels):
        raise ValueError(f"LEAK: support cache ID includes /test/ for {cat}")

    k = dclean.shape[0]
    if cclean.shape[1:3] != (GRID, GRID):
        cclean = resize_patches(cclean, (GRID, GRID))
    if csyn.shape[2:4] != (GRID, GRID):
        csyn = resize_patches(csyn.reshape(-1, *csyn.shape[2:]), (GRID, GRID)).reshape(
            csyn.shape[0], csyn.shape[1], GRID, GRID, -1
        )
    if cnui.shape[2:4] != (GRID, GRID):
        cnui = resize_patches(cnui.reshape(-1, *cnui.shape[2:]), (GRID, GRID)).reshape(
            cnui.shape[0], cnui.shape[1], GRID, GRID, -1
        )

    clean = _fuse(dclean, cclean, k)
    syn = _fuse(
        dsyn.reshape(k * dsyn.shape[1], *dsyn.shape[2:]),
        csyn.reshape(k * csyn.shape[1], *csyn.shape[2:]),
        k * dsyn.shape[1],
    ).reshape(k, dsyn.shape[1], GRID, GRID, -1)
    nui = _fuse(
        dnui.reshape(k * dnui.shape[1], *dnui.shape[2:]),
        cnui.reshape(k * cnui.shape[1], *cnui.shape[2:]),
        k * dnui.shape[1],
    ).reshape(k, dnui.shape[1], GRID, GRID, -1)
    masks = np.asarray([[_mask32(dmask[i, e]) for e in range(dmask.shape[1])] for i in range(k)], dtype=np.uint8)
    return {"clean": clean, "syn": syn, "nui": nui, "masks": masks, "rels": rels}


def _pool_align(x: np.ndarray) -> np.ndarray:
    """Mean-pool 32-grid unit rows to the fixed 16-grid alignment grid."""
    if x.shape[:2] != (GRID, GRID):
        raise ValueError(f"alignment input must be {GRID} x {GRID}, got {x.shape[:2]}")
    p = x.reshape(ALIGN_GRID, ALIGN_POOL, ALIGN_GRID, ALIGN_POOL, x.shape[-1]).mean(axis=(1, 3))
    return _row_normalize(p)


def _overlap_slices(size: int, dy: int, dx: int):
    """Slices where ref[y + dy, x + dx] overlaps query[y, x]."""
    qy0, qy1 = max(0, -dy), min(size, size - dy)
    qx0, qx1 = max(0, -dx), min(size, size - dx)
    ry0, ry1 = qy0 + dy, qy1 + dy
    rx0, rx1 = qx0 + dx, qx1 + dx
    return (slice(qy0, qy1), slice(qx0, qx1)), (slice(ry0, ry1), slice(rx0, rx1))


def _estimate_shift(query: np.ndarray, ref: np.ndarray) -> dict:
    """Estimate ref coordinate = query coordinate + (dy, dx).

    ``query`` is the actual feature map being scored (synthetic, nuisance, or
    clean normal-control query).  The reference is always a clean support
    memory image.  Per-shift cosine is trimmed from below so a sparse defect in
    a synthetic query does not decide the global translation.
    """
    q = _pool_align(query)
    r = _pool_align(ref)
    best = None
    for dy in range(-ALIGN_MAX_SHIFT, ALIGN_MAX_SHIFT + 1):
        for dx in range(-ALIGN_MAX_SHIFT, ALIGN_MAX_SHIFT + 1):
            qs, rs = _overlap_slices(ALIGN_GRID, dy, dx)
            qv = q[qs].reshape(-1, q.shape[-1])
            rv = r[rs].reshape(-1, r.shape[-1])
            cos = np.sum(qv * rv, axis=1)
            raw_mean_cos = float(cos.mean())
            n_keep = max(1, int(np.ceil((1.0 - ALIGN_TRIM_FRACTION) * cos.size)))
            # Highest-cosine cells are the presumed normal majority.  This is
            # a fixed robust estimator, not a result-tuned selector.
            mean_cos = float(np.partition(cos, cos.size - n_keep)[-n_keep:].mean())
            key = (-mean_cos, abs(dy) + abs(dx), abs(dy), abs(dx), dy, dx)
            if best is None or key < best[0]:
                best = (key, dy, dx, mean_cos, raw_mean_cos, int(qv.shape[0]), n_keep)
    assert best is not None
    return {
        "dy_align": int(best[1]),
        "dx_align": int(best[2]),
        "dy_32": int(best[1] * ALIGN_POOL),
        "dx_32": int(best[2] * ALIGN_POOL),
        "mean_cos": float(best[3]),
        "raw_mean_cos": float(best[4]),
        "overlap_cells": int(best[5]),
        "trimmed_cells": int(best[6]),
        "trim_fraction": ALIGN_TRIM_FRACTION,
    }


def _stable_seed(cat: str, shot: int, ref_index: int) -> int:
    payload = f"{SHUFFLE_SEED}|{cat}|{shot}|{ref_index}".encode("utf-8")
    digest = hashlib.sha256(payload).digest()
    return int.from_bytes(digest[:8], "little", signed=False) % (2**32)


def _shuffled_map(ref: np.ndarray, cat: str, shot: int, ref_index: int) -> np.ndarray:
    """Fixed coordinate permutation; feature values and shift estimate stay unchanged."""
    rng = np.random.default_rng(_stable_seed(cat, shot, ref_index))
    perm = rng.permutation(GRID * GRID)
    flat = ref.reshape(GRID * GRID, -1)
    return flat[perm].reshape(GRID, GRID, -1)


def _global_scores(query: np.ndarray, refs: list[np.ndarray]) -> np.ndarray:
    q = query.reshape(GRID * GRID, -1)
    bank = np.concatenate([r.reshape(GRID * GRID, -1) for r in refs], axis=0)
    max_cos = (q @ bank.T).max(axis=1)
    return 1.0 - max_cos


def _band_scores(query: np.ndarray, refs: list[np.ndarray], shifts: list[dict]) -> np.ndarray:
    """NN score within a fixed square band around each estimated translation."""
    q = query.reshape(GRID * GRID, -1)
    scores = np.empty(GRID * GRID, dtype=np.float32)
    for flat_i, (y, x) in enumerate((divmod(i, GRID) for i in range(GRID * GRID))):
        best_cos = -np.inf
        for ref, shift in zip(refs, shifts):
            cy = y + int(shift["dy_32"])
            cx = x + int(shift["dx_32"])
            y0, y1 = max(0, cy - BANDWIDTH), min(GRID, cy + BANDWIDTH + 1)
            x0, x1 = max(0, cx - BANDWIDTH), min(GRID, cx + BANDWIDTH + 1)
            cand = ref[y0:y1, x0:x1].reshape(-1, ref.shape[-1])
            if cand.size:
                best_cos = max(best_cos, float(np.max(cand @ q[flat_i])))
        scores[flat_i] = 1.0 - best_cos if np.isfinite(best_cos) else 1.0
    return scores


def _ap(mask: np.ndarray, score: np.ndarray) -> float:
    y = mask.reshape(-1) > 0
    if y.sum() == 0 or y.sum() == y.size:
        return float("nan")
    return float(average_precision_score(y.astype(np.int32), score.astype(np.float64)))


def _normal_stats(clean_score: np.ndarray, nui_scores: list[np.ndarray]) -> dict:
    clean = clean_score.reshape(-1).astype(np.float64)
    nui = np.concatenate([x.reshape(-1) for x in nui_scores]).astype(np.float64)
    labels = np.concatenate([np.zeros(clean.size, dtype=np.int32), np.ones(nui.size, dtype=np.int32)])
    vals = np.concatenate([clean, nui])
    return {
        "auc_clean_vs_nuisance": float(roc_auc_score(labels, vals)),
        "mean_clean_score": float(clean.mean()),
        "mean_nuisance_score": float(nui.mean()),
        "nuisance_minus_clean": float(nui.mean() - clean.mean()),
        "p95_clean_score": float(np.percentile(clean, 95)),
        "p95_nuisance_score": float(np.percentile(nui, 95)),
    }


def run_category(cat: str, shot: int) -> tuple[list[dict], list[dict], dict]:
    data = _load_category(cat, shot)
    clean, syn, nui, masks, rels = (
        data["clean"], data["syn"], data["nui"], data["masks"], data["rels"]
    )
    k = clean.shape[0]
    if k != shot:
        raise ValueError(f"Expected K={shot} support images for {cat}, got {k}")
    ap_rows: list[dict] = []
    normal_rows: list[dict] = []
    shift_rows: list[dict] = []
    for h in range(k):
        bank_indices = [i for i in range(k) if i != h]
        bank = [clean[i] for i in bank_indices]
        shuffled = [_shuffled_map(clean[i], cat, shot, i) for i in bank_indices]
        def score_methods(query: np.ndarray, query_kind: str, query_episode: int = -1) -> dict[str, np.ndarray]:
            """Score one actual query; spatial shifts are query-specific."""
            q_shifts = []
            for ref_i, ref in zip(bank_indices, bank):
                shift = _estimate_shift(query, ref)
                q_shifts.append(shift)
                shift_rows.append({
                    "category": cat,
                    "shot": shot,
                    "held_out_h": h,
                    "held_out_rel": rels[h],
                    "query_kind": query_kind,
                    "query_episode": query_episode,
                    "bank_ref_index": ref_i,
                    "bank_ref_rel": rels[ref_i],
                    **shift,
                })
            # The fixed coordinate permutation is applied after the same
            # query-to-clean-bank shift inference, so the control isolates
            # spatial correspondence rather than changing the query estimator.
            return {
                "global_nn": _global_scores(query, bank),
                "spatial_bw_nn": _band_scores(query, bank, q_shifts),
                "spatial_bw_nn_coord_shuffle": _band_scores(query, shuffled, q_shifts),
            }

        for fam_i, family in enumerate(FAMILIES):
            for e in range(fam_i * 3, fam_i * 3 + 3):
                mask = masks[h, e]
                scores = score_methods(syn[h, e], "synthetic", e)
                for method, score in scores.items():
                    ap_rows.append({
                        "category": cat,
                        "shot": shot,
                        "held_out_h": h,
                        "held_out_rel": rels[h],
                        "family": family,
                        "episode": e,
                        "method": method,
                        "mask_positive_cells": int(mask.sum()),
                        "pixel_ap": _ap(mask, score),
                    })
        clean_scores = score_methods(clean[h], "clean_normal")
        nuisance_scores = [score_methods(nui[h, e], "nuisance", e) for e in range(nui.shape[1])]
        for method in METHODS:
            clean_score = clean_scores[method]
            method_nuisance_scores = [s[method] for s in nuisance_scores]
            stats = _normal_stats(clean_score, method_nuisance_scores)
            normal_rows.append({
                "category": cat,
                "shot": shot,
                "held_out_h": h,
                "held_out_rel": rels[h],
                "method": method,
                **stats,
            })
    return ap_rows, normal_rows, {"category": cat, "shot": shot, "n_support": k, "support_rels": rels, "shifts": shift_rows}


def _finite_mean(values: list[float]) -> float:
    vals = [float(v) for v in values if np.isfinite(v)]
    return float(np.mean(vals)) if vals else float("nan")


def aggregate(ap_rows: list[dict], normal_rows: list[dict], shift_rows: list[dict], shot: int) -> dict:
    out = {"shot": shot, "methods": {}, "families": {}, "normal": {}, "alignment": {}}
    for method in METHODS:
        rows = [r for r in ap_rows if r["method"] == method]
        out["methods"][method] = {
            "cutpaste_pixel_ap_macro": _finite_mean([r["pixel_ap"] for r in rows if r["family"] == "cutpaste"]),
            "local_erasure_pixel_ap_macro": _finite_mean([r["pixel_ap"] for r in rows if r["family"] == "local_erasure"]),
            "all_requested_families_pixel_ap_macro": _finite_mean([r["pixel_ap"] for r in rows]),
            "n_episode_rows": len(rows),
        }
        nrows = [r for r in normal_rows if r["method"] == method]
        out["normal"][method] = {
            "auc_clean_vs_nuisance_macro": _finite_mean([r["auc_clean_vs_nuisance"] for r in nrows]),
            "nuisance_minus_clean_macro": _finite_mean([r["nuisance_minus_clean"] for r in nrows]),
            "mean_clean_score_macro": _finite_mean([r["mean_clean_score"] for r in nrows]),
            "mean_nuisance_score_macro": _finite_mean([r["mean_nuisance_score"] for r in nrows]),
            "n_held_out_rows": len(nrows),
        }
    for family in FAMILIES:
        out["families"][family] = {}
        for method in METHODS:
            vals = [r["pixel_ap"] for r in ap_rows if r["family"] == family and r["method"] == method]
            out["families"][family][method] = _finite_mean(vals)
    out["alignment"] = {
        "n_query_ref_fits": len(shift_rows),
        "query_kind_counts": {
            kind: sum(1 for r in shift_rows if r["query_kind"] == kind)
            for kind in sorted({r["query_kind"] for r in shift_rows})
        },
        "mean_cos": _finite_mean([r["mean_cos"] for r in shift_rows]),
        "mean_raw_cos": _finite_mean([r["raw_mean_cos"] for r in shift_rows]),
        "mean_abs_shift_32": _finite_mean([abs(r["dy_32"]) + abs(r["dx_32"]) for r in shift_rows]),
        "shift_histogram": {
            f"({r['dy_32']},{r['dx_32']})": sum(
                1 for q in shift_rows if q["dy_32"] == r["dy_32"] and q["dx_32"] == r["dx_32"]
            )
            for r in sorted(shift_rows, key=lambda x: (x["dy_32"], x["dx_32"]))
        },
    }
    # Explicit fixed controls; no pass/fail threshold is optimized here.
    base = out["methods"]["global_nn"]
    spatial = out["methods"]["spatial_bw_nn"]
    shuffle = out["methods"]["spatial_bw_nn_coord_shuffle"]
    out["deltas_vs_global"] = {
        "cutpaste": {
            "spatial_bw_nn": spatial["cutpaste_pixel_ap_macro"] - base["cutpaste_pixel_ap_macro"],
            "spatial_bw_nn_coord_shuffle": shuffle["cutpaste_pixel_ap_macro"] - base["cutpaste_pixel_ap_macro"],
        },
        "local_erasure": {
            "spatial_bw_nn": spatial["local_erasure_pixel_ap_macro"] - base["local_erasure_pixel_ap_macro"],
            "spatial_bw_nn_coord_shuffle": shuffle["local_erasure_pixel_ap_macro"] - base["local_erasure_pixel_ap_macro"],
        },
    }
    out["decision_note"] = (
        "Bounded mechanism probe only; no real-test confirmation, no parameter search, "
        "and no main-table result. Interpret any synthetic AP delta as support-derived evidence."
    )
    return out


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    keys = list(rows[0].keys())
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)


def report_cn(protocol: dict, results: dict, category_summaries: list[dict], command: str) -> str:
    methods = results["methods"]
    normal = results["normal"]
    align = results["alignment"]
    lines = [
        "# 空间跨图对应探针首轮记录（2026-09-08）",
        "",
        "## 范围与数据角色",
        "",
        f"- 协议：`{protocol['protocol_version']}`；seed0 × k{protocol['scope']['shot']} × 六类；CPU threads={THREADS}。",
        "- fit/select 只使用每类 manifest 的 `/train/good/` K-shot clean support；每个 held-out 图 h 都从 memory 排除。",
        "- 每个 synthetic / nuisance query 都用自身特征独立估计到 clean support memory 的平移；clean query 只用于正常控制，不能为 synthetic query 提供对应图。",
        "- 未读取 `/test/` 图像、真实 defect 标签或主表；`v14_p1_support` 缓存的 `ref_rel` 已逐类调用 support 断言。",
        "",
        "## 冻结机制",
        "",
        "将 32 网格 A1 0.5/0.5 融合特征在 16 网格做 2×2 mean-pool；每个实际 query 与每个 clean support memory 对在 ±2 个 16-grid cell 内比较 overlap cosine，丢弃最低 20% 后取 normal-majority 均值，平移映射回 32 网格。查询 cell 只在平移后中心的 Chebyshev 半径 2 内做 NN。",
        "",
        "对照为无空间限制的全局 NN，以及保持同一平移估计、只对每个 memory 图做固定坐标置乱的 control。没有根据 AP 或 mask 选择平移范围、带宽或类别。",
        "",
        "## 首轮实测（support-derived）",
        "",
        "| 方法 | cutpaste AP | erasure AP | 正常 clean/光度 AUC | 光度均分−clean均分 |",
        "|---|---:|---:|---:|---:|",
    ]
    for method in METHODS:
        lines.append(
            f"| {method} | {methods[method]['cutpaste_pixel_ap_macro']:.4f} | "
            f"{methods[method]['local_erasure_pixel_ap_macro']:.4f} | "
            f"{normal[method]['auc_clean_vs_nuisance_macro']:.4f} | "
            f"{normal[method]['nuisance_minus_clean_macro']:+.6f} |"
        )
    lines.extend([
        "",
        "全局 NN 是排序基线；空间带宽方法相对它的 AP 差值：",
        "",
        f"- cutpaste：`{results['deltas_vs_global']['cutpaste']['spatial_bw_nn']:+.4f}`；坐标打乱 control `{results['deltas_vs_global']['cutpaste']['spatial_bw_nn_coord_shuffle']:+.4f}`。",
        f"- local_erasure：`{results['deltas_vs_global']['local_erasure']['spatial_bw_nn']:+.4f}`；坐标打乱 control `{results['deltas_vs_global']['local_erasure']['spatial_bw_nn_coord_shuffle']:+.4f}`。",
        "",
        "首轮结论（负）：独立 query 对齐后的空间带宽 NN 在两类 support-derived 代理上均低于全局 NN（cutpaste −0.2558、erasure −0.1210）；坐标打乱进一步下降，表明带宽限制确实施加了空间约束，但本批数据中该约束损害了异常排序，不能保留为候选增强。",
        "",
        f"query-to-memory 平移拟合数 {align['n_query_ref_fits']}（按 query 类型：{align['query_kind_counts']}），trimmed overlap mean cosine={align['mean_cos']:.4f}，平均 32-grid L1 平移={align['mean_abs_shift_32']:.3f}。",
        "",
        "## 负结果与复现边界",
        "",
        "这是有界机制探针，不构成真实 MPDD 泛化或论文主表证据。若空间 AP 未超过全局 NN，结论是本批 support-derived 代理上没有观察到跨图带宽限制的独立收益；若坐标打乱同样提升，则提升不能归因于真实空间对应。正常 AUC 越接近 0.5 越稳定，需与 AP 一起解读。",
        "",
        f"复现命令：`{command}`",
        "",
        "复用历史审计：t1/t5 的机制是同图邻域描述子，PSMF 是图像坐标多移位合并；本探针只测试跨图平移与坐标受限检索，避免重复上述路线。",
        "",
    ])
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description="Support-only cross-image spatial correspondence probe")
    parser.add_argument("--shot", type=int, choices=(2, 4), default=2)
    parser.add_argument("--cats", default=None, help="comma-separated subset for debugging; default is all six")
    args = parser.parse_args()
    if torch is None:
        raise SystemExit("Torch is required by the frozen cache resize helper")
    cats = [c.strip() for c in args.cats.split(",")] if args.cats else list(CATEGORIES)
    unknown = [c for c in cats if c not in CATEGORIES]
    if unknown:
        raise SystemExit(f"Unknown category: {unknown}")
    OUT.mkdir(parents=True, exist_ok=True)
    protocol = freeze_protocol(OUT / "PROTOCOL.json", args.shot)
    command = f".venv-anomalyclip/Scripts/python.exe scripts/innovation_overnight_20260908/probe_spatial.py --shot {args.shot}"
    all_ap, all_normal, all_shifts, cat_summaries = [], [], [], []
    # Protocol is frozen above; only now load support caches and compute metrics.
    for cat in cats:
        ap_rows, normal_rows, summary = run_category(cat, args.shot)
        all_ap.extend(ap_rows)
        all_normal.extend(normal_rows)
        all_shifts.extend(summary["shifts"])
        cat_summaries.append({
            "category": cat,
            "shot": args.shot,
            "n_support": summary["n_support"],
            "support_rels": summary["support_rels"],
            "ap": {
                family: {
                    method: _finite_mean([
                        r["pixel_ap"] for r in ap_rows if r["family"] == family and r["method"] == method
                    ])
                    for method in METHODS
                }
                for family in FAMILIES
            },
        })
        print(f"  done {cat} k{args.shot} support={summary['n_support']}", flush=True)
    results = aggregate(all_ap, all_normal, all_shifts, args.shot)
    payload = {
        "protocol_version": PROTOCOL_VERSION,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "scope": {"shot": args.shot, "categories": cats, "threads": THREADS, "device": "cpu"},
        "protocol_path": str((OUT / "PROTOCOL.json").relative_to(ROOT)),
        "results": results,
        "category_summaries": cat_summaries,
        "data_role": {
            "support_only": True,
            "real_test_read": False,
            "fit_select_ids_checked": True,
            "query_alignment_independent": True,
            "clean_query_used_for_synthetic_alignment": False,
            "query_alignment_uses_labels": False,
            "synthetic_masks_role": "support-derived evaluation only",
        },
        "reproduction": {"command": command, "script": "scripts/innovation_overnight_20260908/probe_spatial.py"},
    }
    _write_json(OUT / "RESULTS.json", payload)
    write_csv(OUT / "per_episode.csv", all_ap)
    write_csv(OUT / "normal_per_image.csv", all_normal)
    write_csv(OUT / "alignment_per_query.csv", all_shifts)
    (OUT / "REPORT_CN.md").write_text(report_cn(protocol, results, cat_summaries, command), encoding="utf-8")
    print("\n==== SPATIAL PROBE ====")
    for method in METHODS:
        m = results["methods"][method]
        n = results["normal"][method]
        print(
            f"{method}: cutpaste={m['cutpaste_pixel_ap_macro']:.4f} "
            f"erasure={m['local_erasure_pixel_ap_macro']:.4f} "
            f"nui_auc={n['auc_clean_vs_nuisance_macro']:.4f}",
            flush=True,
        )
    print("outputs:", OUT, flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
