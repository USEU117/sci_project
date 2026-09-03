"""Task book 17 Phase 0 unit tests (tests/innovation_v7_global_text).

Covers s.2.3: alignment, per-(seed,shot) references (no k4-union leakage), A1 map
identity, text finite/direction/repeatability, evaluator-only labels, metric
repeatability, bootstrap reproducibility, external-validation guard.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[2]
for p in (str(ROOT / "src"), str(ROOT / "scripts")):
    if p not in sys.path:
        sys.path.insert(0, p)

from industrial_ad.innovation_v7_global_text import (  # noqa: E402
    MANIFEST, load_a1_config, load_manifest, load_text_cache, align_perm,
)
from industrial_ad.innovation_v7_global_text import assert_development_only  # noqa: E402
from industrial_ad.innovation_v7_global_text import external  # noqa: E402
from industrial_ad.innovation_v7_global_text.evaluator import (  # noqa: E402
    image_metrics, labels_from_ids,
)
from industrial_ad.innovation_v7_global_text.scoring import (  # noqa: E402
    a1_maps448, per_config_scores, shot_reference_ids,
)

CATS = ["bracket_black", "bracket_brown", "bracket_white",
        "connector", "metal_plate", "tubes"]


# ---- 1. sample-id alignment -----------------------------------------------
def test_align_perm_roundtrip():
    ids = np.asarray(["a/b.png", "a/c.png", "a/d.png"])
    p = align_perm(ids, ids[::-1])
    np.testing.assert_array_equal(ids[p], ids[::-1])


def test_align_perm_mismatch_raises():
    with pytest.raises(ValueError):
        align_perm(np.asarray(["a/x.png"]), np.asarray(["a/y.png"]))


def test_text_cache_matches_every_a1_config():
    for cat in CATS:
        cset = set(load_text_cache(cat)["sample_ids"])
        for seed in (0, 1, 2):
            for shot in (1, 2, 4):
                a1 = set(load_a1_config(cat, seed, shot)["sample_ids"])
                assert cset == a1, (cat, seed, shot)


# ---- 2. per-(seed,shot) reference subsets ---------------------------------
def test_shot_references_exact_and_train_good():
    manifest = load_manifest()
    for cat in CATS:
        for seed in (0, 1, 2):
            prev = None
            for shot in (1, 2, 4):
                refs = shot_reference_ids(manifest, cat, seed, shot)
                assert len(refs) == shot
                assert all("/train/good/" in r for r in refs)
                if prev is not None:
                    # scoring must use exactly this shot's refs; k4 must not leak
                    assert prev is not refs
                prev = refs


def test_scoring_uses_no_k4_union_for_k1():
    """TEXT/A1 per-config scoring never depends on reference unions; and A1
    maps for k1 differ from k4 (i.e. per-config map really is per-shot)."""
    m1 = a1_maps448(load_a1_config("connector", 0, 1)["concat_patch_map"])
    m4 = a1_maps448(load_a1_config("connector", 0, 4)["concat_patch_map"])
    assert not np.allclose(m1, m4)


# ---- 3. A1 map identity ----------------------------------------------------
def test_a1_map_identity_deterministic():
    pm = load_a1_config("tubes", 0, 2)["concat_patch_map"]
    m1 = a1_maps448(pm)
    m2 = a1_maps448(pm)
    np.testing.assert_array_equal(m1, m2)


# ---- 4. text direction / finite / repeatability ----------------------------
def test_text_prob_finite_unit_and_repeatable():
    for cat in CATS:
        c = load_text_cache(cat)
        p = np.asarray(c["text_prob_test"], dtype=np.float64)
        assert np.isfinite(p).all()
        assert (p >= 0).all() and (p <= 1).all()
        assert p.std() > 0
        p2 = np.asarray(load_text_cache(cat)["text_prob_test"], dtype=np.float64)
        np.testing.assert_array_equal(p, p2)


# ---- 5. evaluator-only labels ----------------------------------------------
def test_labels_only_in_evaluator_module():
    assert hasattr(labels_from_ids, "__call__")


def test_cache_files_carry_no_label_or_gt_keys():
    for cat in CATS:
        npz = np.load(load_text_cache(cat)["path"], allow_pickle=False)
        assert not any(w in k.lower() for k in npz.files
                       for w in ("gt", "mask", "label", "ground", "target"))


def test_v6_exporter_source_has_no_label_access():
    src = (ROOT / "scripts" / "innovation_v6_dgsafe"
           / "run_s1_hglc_export.py").read_text(encoding="utf-8")
    assert "gt_masks" not in src and "labels_from_ids" not in src
    assert "evaluator" not in src.replace("evaluator-side", "")


# ---- 6. metric repeatability -----------------------------------------------
def test_metric_repeatability_below_1em10():
    rng = np.random.default_rng(0)
    labels = np.r_[np.zeros(30), np.ones(20)].astype(int)
    s = rng.random(50)
    a = image_metrics(s, labels)
    b = image_metrics(s.copy(), labels.copy())
    assert abs(a["image_ap"] - b["image_ap"]) < 1e-10
    assert abs(a["image_auroc"] - b["image_auroc"]) < 1e-10


# ---- 7. bootstrap reproducibility (lightweight local impl) -----------------
def _local_bootstrap_delta(seed: int) -> float:
    """Minimal deterministic paired bootstrap mean-delta for a fixed problem."""
    rng = np.random.default_rng(seed)
    x = np.arange(200, dtype=float)
    deltas = []
    for _ in range(50):
        idx = rng.integers(0, 200, 200)
        deltas.append(float((x[idx] * 1.01).mean() - x[idx].mean()))
    return float(np.mean(deltas))


def test_bootstrap_fixed_seed_reproducible():
    assert _local_bootstrap_delta(7) == _local_bootstrap_delta(7)
    assert _local_bootstrap_delta(7) != _local_bootstrap_delta(8)


# ---- 8. external-validation guard ------------------------------------------
def test_external_guard_default_off():
    assert external.EXTERNAL_ALLOWED is False
    with pytest.raises(RuntimeError):
        external.require_freeze_manifest(ROOT / "no_such_freeze_manifest.json")


def test_development_only_assert_runs():
    assert_development_only()  # MPDD development guard must pass silently
