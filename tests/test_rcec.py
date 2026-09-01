"""RCEC v1 unit & technical tests (task book section 7).

Run with the patchcore env:
    .\\.venv-patchcore\\Scripts\\python.exe -m pytest tests/test_rcec.py -q
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from evaluate_a1_feature_fusion import fuse_category, load_features, resize_patches  # noqa: E402
from industrial_ad.fusion import rcec  # noqa: E402
from src.utils import dists2map  # noqa: E402

FEATURES_ROOT = ROOT / "outputs" / "dynamic_fusion" / "v3_direction_a"
DINO_MPDD_S0K1 = FEATURES_ROOT / "features_vitb14_s0_k1" / "anomalydino_visual" / "metal_plate.npz"
CLIP_MPDD_S0K1 = FEATURES_ROOT / "features_s0_k1" / "anomalyclip_text" / "metal_plate.npz"


def _normalize(arr: np.ndarray) -> np.ndarray:
    flat = arr.reshape(-1, arr.shape[-1]).astype(np.float32)
    flat = flat / np.linalg.norm(flat, axis=1, keepdims=True)
    return flat.reshape(arr.shape)


def _synth_pair(s_images: int = 2, grid=(4, 5), dino_d=16, clip_d=12,
                n_test: int = 6, seed: int = 0) -> dict:
    """Synthetic paired cache (features already normalised per patch)."""
    rng = np.random.default_rng(seed)
    h, w = grid
    dino_ref = _normalize(rng.normal(size=(s_images, h, w, dino_d)))
    clip_ref = _normalize(rng.normal(size=(s_images, h, w, clip_d)))
    dino_feat = _normalize(rng.normal(size=(n_test, h, w, dino_d)))
    clip_feat = _normalize(rng.normal(size=(n_test, h, w, clip_d)))
    return {
        "dino_patch": dino_feat,
        "clip_patch": clip_feat,
        "dino_ref": dino_ref,
        "clip_ref": clip_ref,
        "dino_sample_ids": np.asarray([f"c-good-{i:03d}" for i in range(n_test)]),
        "clip_sample_ids": np.asarray([f"c-good-{i:03d}" for i in range(n_test)]),
        "grid": grid,
    }


# ---------------------------------------------------------------------------
# 1. sample-ID alignment (reorder + mismatch errors)
# ---------------------------------------------------------------------------

def test_alignment_reorders_and_detects_mismatch():
    ids_a = np.asarray(["a-good-000", "a-good-001", "a-good-002"])
    ids_b = np.asarray(["a-good-002", "a-good-000", "a-good-001"])
    from industrial_ad.fusion.alignment import build_alignment_plan

    plan = build_alignment_plan(ids_a, ids_b)
    assert list(plan.candidate_order) == [1, 2, 0]
    assert plan.order_already_equal is False

    with pytest.raises(rcec.AlignmentError):
        rcec.align_and_normalize_paired_features(
            dino_patch=np.zeros((3, 4, 4, 8), dtype=np.float32),
            clip_patch=np.zeros((3, 4, 4, 8), dtype=np.float32),
            dino_ref=np.zeros((1, 4, 4, 8), dtype=np.float32),
            clip_ref=np.zeros((1, 4, 4, 8), dtype=np.float32),
            dino_sample_ids=ids_a,
            clip_sample_ids=np.asarray(["a-good-000", "a-good-001"]),  # missing ID
            dino_grid=(4, 4),
            resize_fn=resize_patches,
        )
    with pytest.raises(rcec.AlignmentError):
        rcec.align_and_normalize_paired_features(
            dino_patch=np.zeros((3, 4, 4, 8), dtype=np.float32),
            clip_patch=np.zeros((3, 4, 4, 8), dtype=np.float32),
            dino_ref=np.zeros((1, 4, 4, 8), dtype=np.float32),
            clip_ref=np.zeros((1, 4, 4, 8), dtype=np.float32),
            dino_sample_ids=ids_a,
            clip_sample_ids=np.asarray(["a-good-000", "a-good-000", "a-good-001"]),  # duplicate
            dino_grid=(4, 4),
            resize_fn=resize_patches,
        )


# ---------------------------------------------------------------------------
# 2. non-square CLIP grid resize
# ---------------------------------------------------------------------------

def test_non_square_clip_grid_resized():
    s = _synth_pair(s_images=1, grid=(4, 5), clip_d=12)
    aligned = rcec.align_and_normalize_paired_features(
        dino_patch=s["dino_patch"],
        clip_patch=s["clip_patch"],
        dino_ref=s["dino_ref"],
        clip_ref=s["clip_ref"],
        dino_sample_ids=s["dino_sample_ids"],
        clip_sample_ids=s["clip_sample_ids"],
        dino_grid=s["grid"],
        resize_fn=resize_patches,
    )
    assert aligned["c_feat"].shape == (s["clip_patch"].shape[0],) + tuple(s["grid"]) + (s["clip_patch"].shape[-1],)
    assert aligned["c_ref"].shape == s["clip_ref"].shape
    norms = np.linalg.norm(aligned["c_feat"].reshape(-1, aligned["c_feat"].shape[-1]), axis=1)
    assert np.allclose(norms, 1.0, atol=1e-5)


# ---------------------------------------------------------------------------
# 3. paired metadata (image / row / column) correctness
# ---------------------------------------------------------------------------

def test_memory_metadata_pairing():
    s = _synth_pair(s_images=2, grid=(2, 3))
    mem = rcec.build_paired_reference_memory(s["dino_ref"], s["clip_ref"], 2)
    assert mem.size == 2 * 2 * 3
    # First image block: ref_image == 0, rows/cols enumerate (r, c).
    assert list(mem.ref_image[:6]) == [0] * 6
    assert list(mem.ref_image[6:]) == [1] * 6
    assert list(mem.row[:6]) == [0, 0, 0, 1, 1, 1]
    assert list(mem.col[:6]) == [0, 1, 2, 0, 1, 2]
    # Pairing: (d_i, c_i) come from the same (image, row, col).
    q = 0  # image 0, row 0, col 0
    assert float(np.max(np.abs(mem.d[q] - s["dino_ref"][0, 0, 0]))) < 1e-6
    assert float(np.max(np.abs(mem.c[q] - s["clip_ref"][0, 0, 0]))) < 1e-6


# ---------------------------------------------------------------------------
# 4 & 5. conditional score searches only within DINO top-k CLIP items
# ---------------------------------------------------------------------------

def test_conditional_uses_only_dino_topk_handcrafted():
    # Handcrafted: DINO memory with one clearly closest patch whose CLIP vector
    # is far; the global CLIP min would come from another patch. r_C|D must
    # equal the CLIP distance of the DINO top-1 neighbour, not the global min.
    d = 768
    d_ref = np.zeros((1, 1, 3, d), dtype=np.float32)
    c_ref = np.zeros((1, 1, 3, d), dtype=np.float32)
    d_ref[0, 0, 0, 0] = 1.0
    d_ref[0, 0, 1, 0] = -1.0
    d_ref[0, 0, 2, 0] = 0.5
    d_ref[0, 0, 2, 1] = np.sqrt(0.75)
    c_ref[0, 0, 0, 0] = -1.0
    c_ref[0, 0, 1, 0] = 1.0
    c_ref[0, 0, 2, 0] = 1.0

    dq = np.zeros((1, 1, 1, d), dtype=np.float32)
    dq[0, 0, 0, 0] = 1.0
    cq = np.zeros((1, 1, 1, d), dtype=np.float32)
    cq[0, 0, 0, 0] = 1.0

    mem = rcec.build_paired_reference_memory(d_ref, c_ref, 1)
    r_cd, r_dc = rcec.compute_conditional_scores(dq, cq, mem, "dino_to_clip", k=1)
    # DINO top-1 neighbour is patch 0 (d=[1,0,...]) -> CLIP distance to c_ref[0]=[-1,0,...]
    expected = 0.5 * np.sum((cq.reshape(-1) - c_ref.reshape(3, d)[0]) ** 2)
    assert abs(float(r_cd[0]) - float(expected)) < 1e-6
    # Global CLIP nearest (patch 1) would be ~0, proving r_C|D is restricted to
    # the DINO top-k neighbourhood.
    assert float(r_cd[0]) > 1.0
    assert r_dc is None


def test_conditional_symmetric_returns_both():
    s = _synth_pair(s_images=1, grid=(3, 3), dino_d=16, clip_d=16)
    mem = rcec.build_paired_reference_memory(s["dino_ref"], s["clip_ref"], 1)
    r_cd, r_dc = rcec.compute_conditional_scores(
        s["dino_patch"], s["clip_patch"], mem, "symmetric", k=3)
    assert r_dc is not None
    expected = s["dino_patch"].shape[0] * 3 * 3  # per-patch scores
    assert r_cd.shape == (expected,)
    assert r_dc.shape == (expected,)


# ---------------------------------------------------------------------------
# 6. LOO exclusion rules
# ---------------------------------------------------------------------------

def test_loo_exclusion_shot_ge_2_leave_reference_image_out():
    s = _synth_pair(s_images=2, grid=(2, 2), dino_d=16, clip_d=16)
    loo = rcec.compute_reference_loo_statistics(
        s["dino_ref"], s["clip_ref"], 2, direction="dino_to_clip", k=1, shot=2)
    assert loo["exclusion_rule"] == "leave_one_reference_image_out"
    assert loo["n_ref_patches"] == 8

    # Manually verify one score: image-0 patch must NOT use any image-0 neighbour.
    mem = rcec.build_paired_reference_memory(s["dino_ref"], s["clip_ref"], 2)
    z_ref = rcec._concat_and_l2(mem.d, mem.c)
    q = 0  # image 0, (0,0)
    keep = np.nonzero(mem.ref_image != 0)[0]
    zq = z_ref[q : q + 1]
    expected = float(np.sum((z_ref[keep] - zq) ** 2, axis=-1).min() / 2.0)
    assert abs(float(loo["a1_loo"][q]) - expected) < 1e-6


def test_loo_exclusion_shot_1_self_and_radius1():
    s = _synth_pair(s_images=1, grid=(4, 4), dino_d=16, clip_d=16)
    loo = rcec.compute_reference_loo_statistics(
        s["dino_ref"], s["clip_ref"], 1, direction="dino_to_clip", k=1, shot=1)
    assert loo["exclusion_rule"] == "self_and_chebyshev_radius_1"
    mem = rcec.build_paired_reference_memory(s["dino_ref"], s["clip_ref"], 1)
    z_ref = rcec._concat_and_l2(mem.d, mem.c)
    # query = image0 (0,0): exclude patches with chebyshev distance <= 1
    q = 0
    allowed = [
        i for i in range(mem.size)
        if max(abs(mem.row[i] - 0), abs(mem.col[i] - 0)) > 1
    ]
    expected = float(np.sum((z_ref[allowed] - z_ref[q : q + 1]) ** 2, axis=-1).min() / 2.0)
    assert abs(float(loo["a1_loo"][q]) - expected) < 1e-6
    # Self-matching is impossible: the LOO score of the self patch must be > 0.
    assert float(loo["a1_loo"][q]) > 1e-8


def test_loo_candidate_shortage_fails():
    # 3x3 grid: the centre patch excludes itself + all 8 neighbours -> 0
    # candidates, which must fail loudly (task book 3.5).
    s = _synth_pair(s_images=1, grid=(3, 3), dino_d=16, clip_d=16)
    with pytest.raises(rcec.CandidateShortageError):
        rcec.compute_reference_loo_statistics(
            s["dino_ref"], s["clip_ref"], 1, direction="dino_to_clip", k=1, shot=1)


# ---------------------------------------------------------------------------
# 7. degenerate / shortage / NaN failures
# ---------------------------------------------------------------------------

def test_degenerate_shortage_nan_fail():
    with pytest.raises(rcec.CalibrationDegenerateError):
        rcec.compute_reference_stats(np.zeros(50))
    with pytest.raises(rcec.CalibrationDegenerateError):
        rcec.robust_z_from_reference(np.zeros(5), {"median": 0.0, "mad": 1e-9})
    with pytest.raises(rcec.RCECError):
        rcec.compute_reference_stats(np.array([1.0, np.nan, 2.0]))
    with pytest.raises(rcec.RCECError):
        rcec.compute_reference_stats(np.array([]))
    with pytest.raises(rcec.RCECError):
        rcec.compute_conditional_scores(
            np.zeros((4, 768), dtype=np.float32), np.zeros((4, 768), dtype=np.float32),
            rcec.build_paired_reference_memory(
                np.zeros((1, 2, 2, 768), dtype=np.float32),
                np.zeros((1, 2, 2, 768), dtype=np.float32), 1),
            "dino_to_clip", k=0)


# ---------------------------------------------------------------------------
# 8. robust z consumes reference statistics only
# ---------------------------------------------------------------------------

def test_robust_z_reference_stats_only():
    ref_scores = np.array([1.0, 2.0, 3.0, 4.0, 100.0])
    stats = rcec.compute_reference_stats(ref_scores)
    x = np.array([1.5, 200.0])
    z = rcec.robust_z_from_reference(x, stats)
    med, mad = stats["median"], stats["mad"]
    expected = np.clip((x - med) / (1.4826 * mad + 1e-6), -5.0, 10.0)
    assert np.allclose(z, expected, atol=1e-6)
    # z of the reference median itself is ~0 (not a test-set statistic).
    assert abs(float(rcec.robust_z_from_reference(np.array([med]), stats)[0])) < 1e-6


# ---------------------------------------------------------------------------
# 9. lambda = 0 => calibrated A1
# ---------------------------------------------------------------------------

def test_lambda_zero_equals_calibrated_a1():
    rng = np.random.default_rng(0)
    s_a1 = rng.normal(size=100)
    r_cond = rng.normal(size=100)
    stats = {"median": 0.5, "mad": 0.25}
    s = rcec.combine_rcec_scores(s_a1, r_cond, stats, stats, lam=0.0)
    expected = rcec.robust_z_from_reference(s_a1, stats)
    assert np.allclose(s, expected, atol=1e-9)


# ---------------------------------------------------------------------------
# 10 & 11. A1 / DINO-duplicate regression on a real MPDD category
# ---------------------------------------------------------------------------

def _load_real_mpdd_category():
    if not DINO_MPDD_S0K1.is_file() or not CLIP_MPDD_S0K1.is_file():
        pytest.skip("MPDD s0/k1 metal_plate caches not available")
    dino = load_features(DINO_MPDD_S0K1)
    clip = load_features(CLIP_MPDD_S0K1)
    return dino, clip


def test_a1_map_direct_regression_vs_frozen():
    dino, clip = _load_real_mpdd_category()
    maps_frozen = fuse_category(
        dino, clip, "concat", pca_dim=0, whiten=False,
        map_size=(448, 448), dino_weight=0.5).astype(np.float64)

    from rcec_common import evaluate_category

    ref_ids = ["x"] * int(dino["ref_patch_features"].shape[0])
    cfg = {"normal_calibration": {"epsilon": 1e-6, "z_clip": [-5.0, 10.0]},
           "postprocess": {"map_size": [448, 448]},
           "fixed": {"dino_weight": 0.5}}
    cand = {"direction": "dino_to_clip", "k": 1, "lambda": 0.0}
    report = evaluate_category(dino, clip, ref_ids, 0, 1, cand, cfg, category="metal_plate")
    # Reproduce the RCEC s_A1 map with the same API used inside evaluate_category.
    from industrial_ad.fusion import rcec as r
    from evaluate_a1_feature_fusion import resize_patches as rp

    aligned = r.align_and_normalize_paired_features(
        dino_patch=dino["patch_features"], clip_patch=clip["patch_features"],
        dino_ref=dino["ref_patch_features"], clip_ref=clip["ref_patch_features"],
        dino_sample_ids=dino["sample_ids"], clip_sample_ids=clip["sample_ids"],
        dino_grid=dino["grid_size"], resize_fn=rp)
    mem = r.build_paired_reference_memory(aligned["d_ref"], aligned["c_ref"], len(ref_ids))
    s_a1 = r.compute_a1_dists(aligned["d_feat"], aligned["c_feat"], mem)
    grid = aligned["grid"]
    n = aligned["d_feat"].shape[0]
    h, w = grid
    maps_rcec = np.stack(
        [dists2map(s_a1.reshape(n, h, w)[i], (448, 448)) for i in range(n)]).astype(np.float32)
    max_abs = float(np.max(np.abs(maps_rcec.astype(np.float64) - maps_frozen)))
    assert max_abs < 1e-5, f"A1 map regression max abs = {max_abs}"
    from evaluate_a1_feature_fusion import compute_metrics

    m1 = compute_metrics(maps_frozen, dino["imgs_masks"])
    m2 = compute_metrics(maps_rcec.astype(np.float64), dino["imgs_masks"])
    for key in m1:
        assert abs(m1[key] - m2[key]) < 1e-6, f"{key} delta {abs(m1[key]-m2[key])}"
    assert report["metrics"]["a1"]["pixel"]["pixel_ap"] is not None


def test_dino_duplicate_distance_preserving():
    rng = np.random.default_rng(1)
    d = rng.normal(size=(200, 768)).astype(np.float32)
    d = d / np.linalg.norm(d, axis=1, keepdims=True)
    z = np.concatenate([0.5 * d, 0.5 * d], axis=-1)
    z = z / np.linalg.norm(z, axis=1, keepdims=True)
    x, y = d[0], d[1]
    dist_d = 0.5 * np.sum((x - y) ** 2)
    dist_z = 0.5 * np.sum((z[0] - z[1]) ** 2)
    assert abs(dist_d - dist_z) < 1e-6


def test_dino_duplicate_real_map_close():
    dino, clip = _load_real_mpdd_category()
    from industrial_ad.fusion import rcec as r
    from evaluate_a1_feature_fusion import resize_patches as rp

    ref_ids = ["x"] * int(dino["ref_patch_features"].shape[0])
    aligned = r.align_and_normalize_paired_features(
        dino_patch=dino["patch_features"], clip_patch=clip["patch_features"],
        dino_ref=dino["ref_patch_features"], clip_ref=clip["ref_patch_features"],
        dino_sample_ids=dino["sample_ids"], clip_sample_ids=clip["sample_ids"],
        dino_grid=dino["grid_size"], resize_fn=rp)
    mem = r.build_paired_reference_memory(aligned["d_ref"], aligned["c_ref"], len(ref_ids))
    s_dino = r.compute_dino_dists(aligned["d_feat"], mem)
    s_dup = r.compute_dino_duplicate_dists(aligned["d_feat"], mem)
    grid = aligned["grid"]
    n = aligned["d_feat"].shape[0]
    h, w = grid
    maps_dino = np.stack(
        [dists2map(s_dino.reshape(n, h, w)[i], (448, 448)) for i in range(n)])
    maps_dup = np.stack(
        [dists2map(s_dup.reshape(n, h, w)[i], (448, 448)) for i in range(n)])
    assert float(np.max(np.abs(maps_dino - maps_dup))) < 1e-5
    from evaluate_a1_feature_fusion import compute_metrics

    m1 = compute_metrics(maps_dino.astype(np.float64), dino["imgs_masks"])
    m2 = compute_metrics(maps_dup.astype(np.float64), dino["imgs_masks"])
    # Task book 5.1: map error < 1e-5 and Pixel-AP error < 1e-6; AUPRO is a
    # 200-step integral and tolerates slightly larger float32 rounding noise.
    assert abs(m1["pixel_ap"] - m2["pixel_ap"]) < 1e-6
    assert abs(m1["pixel_auroc"] - m2["pixel_auroc"]) < 1e-6
    assert abs(m1["pixel_aupro"] - m2["pixel_aupro"]) < 1e-5


# ---------------------------------------------------------------------------
# 12. frozen validation rejects CLI overrides
# ---------------------------------------------------------------------------

def test_frozen_validation_rejects_overrides():
    from rcec_common import check_no_config_overrides

    frozen = {"direction": "dino_to_clip", "k": 3, "lambda": 0.25}
    check_no_config_overrides({}, frozen)  # no overrides -> OK
    with pytest.raises(rcec.RCECError):
        check_no_config_overrides({"k": 5}, frozen)
    with pytest.raises(rcec.RCECError):
        check_no_config_overrides({"direction": "symmetric"}, frozen)
    with pytest.raises(rcec.RCECError):
        check_no_config_overrides({"lambda": 0.5}, frozen)


# ---------------------------------------------------------------------------
# 13. output schema & leakage flags
# ---------------------------------------------------------------------------

def test_output_schema_and_leakage_flags():
    from rcec_common import evaluate_category

    s = _synth_pair(s_images=2, grid=(4, 4), dino_d=8, clip_d=8, n_test=5)
    gt_sp = np.array([0, 1, 0, 1, 0])
    masks = np.zeros((5, 448, 448), dtype=np.uint8)
    masks[1, 200:240, 200:240] = 1
    masks[3, 100:140, 320:360] = 1
    dino = {
        "patch_features": s["dino_patch"], "ref_patch_features": s["dino_ref"],
        "sample_ids": s["dino_sample_ids"],
        "gt_sp": gt_sp, "imgs_masks": masks,
        "grid_size": np.asarray(s["grid"], dtype=np.int64),
    }
    clip = {
        "patch_features": s["clip_patch"], "ref_patch_features": s["clip_ref"],
        "sample_ids": s["clip_sample_ids"],
        "gt_sp": gt_sp, "imgs_masks": masks,
        "grid_size": np.asarray((4, 4), dtype=np.int64),
    }
    cfg = {"normal_calibration": {"epsilon": 1e-6, "z_clip": [-5.0, 10.0]},
           "postprocess": {"map_size": [448, 448]},
           "fixed": {"dino_weight": 0.5}}
    cand = {"direction": "dino_to_clip", "k": 1, "lambda": 0.25}
    report = evaluate_category(dino, clip, ["a", "b"], 0, 2, cand, cfg, category="synth")
    assert report["schema_version"] == 1
    assert report["method"] == "rcec_v1"
    assert report["candidate"] == cand
    for flag, value in report["leakage_flags"].items():
        assert value is False, flag
    assert report["metrics"]["rcec"]["pixel"]["pixel_ap"] is not None
    assert report["checks"]["no_nan_inf_scores"] is True


# ---------------------------------------------------------------------------
# 14. chunked == non-chunked
# ---------------------------------------------------------------------------

def test_chunked_matches_nonchunked():
    s = _synth_pair(s_images=2, grid=(4, 5), dino_d=16, clip_d=16, n_test=12)
    mem = rcec.build_paired_reference_memory(s["dino_ref"], s["clip_ref"], 2)
    r_chunk, r_dc_chunk = rcec.compute_conditional_scores(
        s["dino_patch"], s["clip_patch"], mem, "symmetric", k=3, chunk=64)
    r_full, r_dc_full = rcec.compute_conditional_scores(
        s["dino_patch"], s["clip_patch"], mem, "symmetric", k=3, chunk=1 << 30)
    assert np.max(np.abs(r_chunk - r_full)) < 1e-6
    assert np.max(np.abs(r_dc_chunk - r_dc_full)) < 1e-6


# ---------------------------------------------------------------------------
# 15. image aggregation never changes the pixel map
# ---------------------------------------------------------------------------

def test_image_aggregation_preserves_pixel_map():
    rng = np.random.default_rng(2)
    maps = rng.normal(size=(10, 16, 16)).astype(np.float32)
    maps_before = maps.copy()
    scores = rcec.aggregate_image_score(maps, mode="max")
    assert np.allclose(scores, maps.reshape(10, -1).max(axis=1), atol=0)
    for q in (0.001, 0.01, 0.05):
        s = rcec.aggregate_image_score(maps, mode=f"top{q*100:g}pct", top_q=q)
        flat = maps.reshape(10, -1)
        k = max(1, int(round(flat.shape[1] * q)))
        expected = np.partition(flat, -k, axis=1)[:, -k:].mean(axis=1)
        assert np.allclose(s, expected, atol=1e-6)
    # top_q<0.1% still takes at least 1 patch per image.
    s_min = rcec.aggregate_image_score(maps, mode="top0.1pct", top_q=0.0005)
    assert s_min.shape == (10,)
    # Input pixel map untouched.
    assert np.array_equal(maps, maps_before)
    # Unknown mode raises.
    with pytest.raises(rcec.RCECError):
        rcec.aggregate_image_score(maps, mode="median")
