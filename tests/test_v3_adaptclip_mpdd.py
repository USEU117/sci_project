import json
from pathlib import Path

import numpy as np
import pytest

from industrial_ad.fusion.v3_adaptclip_mpdd import (
    build_adaptclip_mpdd_metadata,
    validate_adaptclip_prediction_payload,
    write_adaptclip_prediction_cache,
)


def _fixture(tmp_path: Path):
    root = tmp_path / "MPDD"
    category = "part"
    refs = []
    for name in ("a.png", "b.png"):
        relative = f"{category}/train/good/{name}"
        path = root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"image")
        refs.append(relative)
    good = root / category / "test" / "good" / "g.png"
    bad = root / category / "test" / "scratch" / "x.png"
    mask = root / category / "ground_truth" / "scratch" / "x_mask.png"
    for path in (good, bad, mask):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"image")
    manifest = {"dataset": "mpdd", "seeds": [0], "shots": [1, 2],
                "categories": {category: {"0": {"1": refs[:1], "2": refs}}}}
    return root, manifest


def test_frozen_references_and_test_index_are_exact(tmp_path):
    root, manifest = _fixture(tmp_path)
    metadata, audit = build_adaptclip_mpdd_metadata(root, manifest, seed=0, shot=2)
    assert [row["img_path"] for row in metadata["train"]["part"]] == manifest["categories"]["part"]["0"]["2"]
    assert len(metadata["test"]["part"]) == 2
    assert audit["normal_references"] == 2
    assert audit["anomalous_test_images"] == 1
    assert audit["btad_accessed"] is False


def test_rejects_non_normal_reference(tmp_path):
    root, manifest = _fixture(tmp_path)
    manifest["categories"]["part"]["0"]["1"] = ["part/test/good/g.png"]
    with pytest.raises(ValueError, match="non-normal"):
        build_adaptclip_mpdd_metadata(root, manifest, seed=0, shot=1)


def test_rejects_unfrozen_seed_or_shot(tmp_path):
    root, manifest = _fixture(tmp_path)
    with pytest.raises(ValueError, match="outside"):
        build_adaptclip_mpdd_metadata(root, manifest, seed=1, shot=1)


def test_unified_prediction_cache_schema(tmp_path):
    output = tmp_path / "part.npz"
    audit = write_adaptclip_prediction_cache(
        output, category="part", seed=0, shot=1, sample_ids=["a", "b"],
        image_scores=np.asarray([0.1, 0.8]),
        pixel_maps=np.zeros((2, 4, 4), dtype=np.float32),
        labels=np.asarray([0, 1]), masks=np.zeros((2, 4, 4), dtype=np.uint8),
    )
    with np.load(output) as cache:
        assert set(("gt_sp", "pr_sp", "imgs_masks", "anomaly_maps", "sample_ids")).issubset(cache.files)
        assert cache["branch"].item() == "adaptclip_text_v3"
        assert cache["test_labels_used_by_router"].item() is False
    assert audit["labels_evaluator_only"] is True


def test_prediction_payload_rejects_nonfinite_and_misaligned():
    kwargs = dict(sample_ids=["a"], image_scores=np.asarray([np.nan]),
                  pixel_maps=np.zeros((1, 2, 2)), labels=np.asarray([0]),
                  masks=np.zeros((1, 2, 2)))
    with pytest.raises(ValueError, match="non-finite"):
        validate_adaptclip_prediction_payload(**kwargs)
    kwargs["image_scores"] = np.asarray([0.1])
    kwargs["masks"] = np.zeros((2, 2, 2))
    with pytest.raises(ValueError, match="align"):
        validate_adaptclip_prediction_payload(**kwargs)
