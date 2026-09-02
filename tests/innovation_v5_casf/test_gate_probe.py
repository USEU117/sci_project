"""Unit tests for the Wave-0 CASF gate probe (task book 15 section 2.4).

Pure-logic tests only (rule boundaries, config drift guard, RNG determinism,
statistics shape); no real MPDD feature load, no masks/GT anywhere.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[2]
for p in (str(ROOT / "src"), str(ROOT / "scripts")):
    if p not in sys.path:
        sys.path.insert(0, p)

from industrial_ad.innovation_v4_diagnostics import common  # noqa: E402
from industrial_ad.innovation_v4_diagnostics import diagnostics as diag  # noqa: E402
from industrial_ad.innovation_v5_casf import gate_probe as gp  # noqa: E402

CONFIG = ROOT / "configs" / "innovation_v5_casf" / "gate_probe.json"


def _rows_for(detail: dict[str, dict[int, float | None]]) -> list[dict]:
    rows = []
    for cat, seeds in detail.items():
        for ix, hr in seeds.items():
            rows.append({"category": cat, "seed_ix": ix, "headroom": hr})
    return rows


def test_derive_gset_rule_boundaries():
    # A: mean exactly 0.02, 2/3 votes -> active (>= boundaries inclusive)
    # B: mean below -> inactive
    # C: mean 0.0167 below -> inactive despite one strong seed
    # D: all strong -> active
    detail = {
        "A": {0: 0.03, 1: 0.02, 2: 0.01},
        "B": {0: 0.05, 1: 0.05, 2: -0.10},
        "C": {0: 0.03, 1: 0.01, 2: 0.01},
        "D": {0: 0.025, 1: 0.025, 2: 0.025},
    }
    gset, detail_out = gp.derive_gset(_rows_for(detail))
    assert gset == ["A", "D"]
    assert detail_out["B"]["active"] is False
    assert detail_out["C"]["active"] is False
    assert detail_out["A"]["mean_headroom"] == 0.02
    assert detail_out["A"]["seed_votes"] == 2


def test_derive_gset_tolerates_none():
    # X: no valid seed -> inactive. Y: seeds 0 and 2 valid (2 votes, mean 0.075)
    # -> active; a missing seed never blocks the rule.
    detail = {"X": {0: None, 1: None, 2: None}, "Y": {0: 0.1, 1: None, 2: 0.05}}
    gset, detail_out = gp.derive_gset(_rows_for(detail))
    assert gset == ["Y"]
    assert detail_out["X"]["active"] is False
    assert detail_out["Y"]["active"] is True
    assert gp.MIN_MEAN_HEADROOM == 0.02
    assert gp.MIN_SEED_VOTES == 2 and gp.N_SEEDS == len(gp.FAMILY_SEEDS) == 3


def test_config_matches_module_constants():
    cfg = json.loads(CONFIG.read_text(encoding="utf-8"))
    assert cfg["episodes_train_per_family"] == gp.N_TRAIN_EP
    assert cfg["episodes_eval_asym"] == gp.N_EVAL_EP
    assert cfg["gamma"] == gp.GAMMA
    assert cfg["logreg_C"] == gp.LOGREG_C
    assert cfg["family_seeds"] == list(gp.FAMILY_SEEDS)
    assert cfg["shot"] == gp.SHOT
    assert cfg["dataset"] == "mpdd" and cfg["role"] == "development"


def test_rng_deterministic_per_category_seed():
    a1 = gp.rng_for("bracket_brown", 0).random(16)
    a2 = gp.rng_for("bracket_brown", 0).random(16)
    b = gp.rng_for("bracket_brown", 1).random(16)
    assert np.array_equal(a1, a2)
    assert not np.array_equal(a1, b)


def test_d3_statistics_shape_and_semantics():
    rng = np.random.default_rng(0)
    h = w = 4
    zd = rng.normal(size=(h, w)).astype(np.float32)
    zc = rng.normal(size=(h, w)).astype(np.float32)
    dm = np.zeros((h, w, 4), dtype=np.float32)
    cm = np.zeros((h, w, 4), dtype=np.float32)
    feats = diag.d3_statistics(dm, cm, zd, zc)
    assert feats.shape == (h * w, 4)
    diff = zd - zc
    np.testing.assert_allclose(feats[:, 0], zd.ravel(), atol=1e-6)
    np.testing.assert_allclose(feats[:, 1], zc.ravel(), atol=1e-6)
    np.testing.assert_allclose(feats[:, 2], np.abs(diff).ravel(), atol=1e-6)
    np.testing.assert_allclose(feats[:, 3], diff.ravel(), atol=1e-6)


def test_development_dataset_only():
    assert common.development_dataset() == "mpdd"


def test_pre_registered_episode_counts_strict():
    # Task book 15 section 2.4 freezes the probe scale; guard against drift.
    assert gp.N_TRAIN_EP >= 24
    assert gp.N_EVAL_EP >= 24
