"""Route A CRAM: no-leakage tests (task book 19 §11.3).

- source scan of scoring modules for forbidden GT keys
- 'changing GT does not change output maps' style determinism check
"""

from __future__ import annotations

import ast
from pathlib import Path

import numpy as np
import pytest

SRC = Path(__file__).resolve().parents[2] / "src/industrial_ad/innovation_v10_portfolio"
FORBIDDEN = {"gt_masks", "gt_labels", "gt_sp"}
# the metrics evaluator is the ONLY place masks may enter (mirrors A1 evaluator)
ALLOWED_CONSUMERS = {"compute_pixel_metrics"}


def _ids_in_tree(node) -> set[str]:
    hits: set[str] = set()
    for n in ast.walk(node):
        if isinstance(n, ast.Name) and n.id in FORBIDDEN:
            hits.add(n.id)
        if isinstance(n, ast.Attribute) and n.attr in FORBIDDEN:
            hits.add(n.attr)
    return hits


def _forbidden_ids(src: str) -> set[str]:
    tree = ast.parse(src)
    found: set[str] = set()

    def scan_body(body):
        for node in body:
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                if node.name not in ALLOWED_CONSUMERS:
                    found.update(_ids_in_tree(node))
                    # still recurse for nested defs inside non-allowed functions
                    scan_body(node.body)
            else:
                found.update(_ids_in_tree(node))

    scan_body(tree.body)
    return found


@pytest.mark.parametrize(
    "module",
    ["common.py", "cram.py", "capm.py", "norc.py", "spectral.py"],
)
def test_scoring_source_has_no_gt_keys(module: str):
    src = (SRC / module).read_text(encoding="utf-8")
    assert _forbidden_ids(src) == set(), f"{module} uses forbidden GT keys outside evaluator"


def test_build_fused_signature_is_gt_free():
    import inspect

    from industrial_ad.innovation_v10_portfolio import common

    sig = inspect.signature(common.build_fused_blocks)
    assert not any("gt" in p for p in sig.parameters)
    sig2 = inspect.signature(common.per_reference_distances)
    assert not any("gt" in p for p in sig2.parameters)


def test_candidate_maps_deterministic_and_gt_independent():
    """Same input features -> identical maps; GT is never an input."""
    from industrial_ad.innovation_v10_portfolio import cram

    rng = np.random.default_rng(0)
    feat = rng.normal(size=(3, 4, 4, 8)).astype(np.float32)
    ref = rng.normal(size=(3, 4, 4, 8)).astype(np.float32)
    a = cram.candidate_maps(feat, ref, ("a0", "a1", "a2"), mad95_normal=0.1, map_size=(16, 16))
    b = cram.candidate_maps(feat, ref, ("a0", "a1", "a2"), mad95_normal=0.1, map_size=(16, 16))
    for name in a:
        assert a[name].shape == (3, 16, 16)
        assert np.array_equal(a[name], b[name]), name
        assert np.all(np.isfinite(a[name])), name


def test_a0_equals_pooled_min():
    """min over per-reference-image banks == single pooled index (math identity)."""
    from industrial_ad.innovation_v10_portfolio import common, cram

    rng = np.random.default_rng(1)
    feat = rng.normal(size=(2, 4, 4, 8)).astype(np.float32)
    ref = rng.normal(size=(3, 4, 4, 8)).astype(np.float32)
    d = feat.shape[-1]
    feat_flat = feat.reshape(-1, d)
    dr = common.per_reference_distances(feat_flat, ref)
    d_min = cram.agreement_stats(dr)["d_min"]
    pooled = cram.pooled_min_map(feat, ref).reshape(-1)
    np.testing.assert_allclose(d_min, pooled, rtol=1e-6, atol=1e-6)


def test_a1_degrades_to_a0_when_refs_identical():
    """If every reference image is identical, gap==0 -> A1 == A0 (no spurious gain)."""
    from industrial_ad.innovation_v10_portfolio import cram

    rng = np.random.default_rng(2)
    feat = rng.normal(size=(2, 4, 4, 8)).astype(np.float32)
    ref_img = rng.normal(size=(1, 4, 4, 8)).astype(np.float32)
    ref = np.repeat(ref_img, 4, axis=0)
    a0 = cram.candidate_maps(feat, ref, ("a0", "a1"), map_size=(16, 16))["a0"]
    a1 = cram.candidate_maps(feat, ref, ("a0", "a1"), map_size=(16, 16))["a1"]
    np.testing.assert_array_equal(a0, a1)
