"""Shared per-config scoring + reference helpers (task book 17).

Only the A1-max / A1-top1% / TEXT signals are produced here (DINO CLS and
CLIP-global were seed0-only diagnostics already archived in v6). TEXT p_abn is
seed/shot independent; it is aligned per image to each A1 config by sample id.
"""

from __future__ import annotations

import numpy as np

from . import (A1_MAP_SIZE, load_a1_config, load_manifest, load_text_cache)
from .evaluator import labels_from_ids, top1_mean


def a1_maps448(patch_map: np.ndarray) -> np.ndarray:
    """(N,32,32) grid -> (N,448,448) via the frozen A1 map path (v6 maps.py)."""
    from industrial_ad.innovation_v6_dgsafe import maps as v6maps
    return v6maps.a1_maps448(patch_map)


def shot_reference_ids(manifest, cat: str, seed: int, shot: int):
    """Exact manifest reference ids for (cat, seed, shot); train/good only."""
    refs = list(manifest["categories"][cat][str(seed)][str(shot)])
    for r in refs:
        if "/train/good/" not in r:
            raise ValueError(f"reference not in train/good: {r}")
    return refs


def align_to_a1(cache: dict, a1: dict) -> np.ndarray:
    from . import align_perm
    return align_perm(np.asarray(cache["sample_ids"]),
                      np.asarray(a1["sample_ids"]))


def per_config_scores(cat: str, seed: int, shot: int) -> dict:
    """Return aligned scores + labels for one (cat, seed, shot) config.

    Keys: labels (N,), a1_max (N,), a1_top1 (N,), text (N,),
          sample_ids (N,), n_normal, n_anomaly.
    """
    cache = load_text_cache(cat)
    a1 = load_a1_config(cat, seed, shot)
    perm = align_to_a1(cache, a1)
    m = a1_maps448(a1["concat_patch_map"])
    labels = labels_from_ids(a1["sample_ids"])
    text = np.asarray(cache["text_prob_test"], dtype=np.float64)[perm]
    return {
        "sample_ids": a1["sample_ids"],
        "labels": labels,
        "a1_max": np.asarray(m.reshape(len(m), -1).max(axis=1), dtype=np.float64),
        "a1_top1": top1_mean(m),
        "text": text,
        "n_normal": int((labels == 0).sum()),
        "n_anomaly": int((labels == 1).sum()),
    }
