"""innovation_v7_global_text - task book 17 (2026-09-03).

Global text evidence confirmation (AnomalyCLIP image-level p_abn) and, if G1
passes, the GLSD task-decoupled system and the optional TCRR region-level route.

Houses shared constants, input loading, alignment and evaluator-side helpers.
Label / GT handling lives ONLY in evaluator-facing helpers; exporters never
import them.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
for p in (str(ROOT / "src"), str(ROOT / "scripts"), str(ROOT / "methods" / "anomalydino")):
    if p not in sys.path:
        sys.path.insert(0, p)

# Frozen inputs (task book 17 s.2)
V6_EXP = ROOT / "experiments" / "dynamic_fusion" / "innovation_v6_dgsafe"
S1_CACHE = V6_EXP / "s1_hglc" / "cache"
A1_MAPS_ROOT = ROOT / "submission_repro_20260827" / "predictions_compact" / "maps" / "mpdd"
MPDD_DATA_ROOT = ROOT / "data" / "mpdd_raw" / "MPDD"
MANIFEST = ROOT / "data" / "splits" / "mpdd" / "manifest.json"
EXPERIMENT_ROOT = ROOT / "experiments" / "dynamic_fusion" / "innovation_v7_global_text"
OUTPUTS_ROOT = ROOT / "outputs" / "dynamic_fusion" / "innovation_v7_global_text"

A1_MAP_SIZE = (448, 448)
A1_STRIDE = 8
TOP1_FRAC = 0.01
SEEDS = (0, 1, 2)
SHOTS = (1, 2, 4)
SIGNALS = ("a1_max", "a1_top1", "text")
CKPT_REL = ROOT / "methods" / "AnomalyCLIP-main" / "checkpoints"
ANOMALYCLIP_CKPT = CKPT_REL / "9_12_4_multiscale_visa" / "epoch_15.pth"
ANOMALYCLIP_CKPT_SHA256 = "415c5dcb52668b8c33fb9c1a351c686d632b919df5b384d63fa9ce7a2338ced4"

SEEDS_S = {str(s) for s in SEEDS}


def assert_development_only() -> None:
    """Full-MPDD development is allowed; external sets stay frozen."""
    from industrial_ad.innovation_v2.common import assert_development_only as _g
    _g("mpdd")


def sha256_file(path: Path) -> str:
    import hashlib
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def load_manifest() -> dict:
    return json.loads(MANIFEST.read_text(encoding="utf-8"))


def is_normal_id(sample_id: str) -> bool:
    return "/good/" in sample_id


def align_perm(ids_b: np.ndarray, ids_a: np.ndarray) -> np.ndarray:
    """Permutation reordering ids_b to match ids_a (raises on mismatch)."""
    if len(ids_b) != len(ids_a):
        raise ValueError(f"length mismatch {len(ids_b)} vs {len(ids_a)}")
    order = {str(s): i for i, s in enumerate(ids_b)}
    if len(order) != len(ids_b):
        raise ValueError("duplicate sample_ids in b")
    try:
        perm = np.asarray([order[str(s)] for s in ids_a], dtype=np.int64)
    except KeyError as e:
        raise ValueError(f"sample id in a not present in b: {e.args[0]}") from None
    if not np.array_equal(np.asarray(ids_b)[perm], np.asarray(ids_a)):
        raise ValueError("sample set mismatch after alignment")
    return perm


def load_a1_config(category: str, seed: int, shot: int) -> dict:
    path = A1_MAPS_ROOT / f"s{seed}_k{shot}" / f"{category}.npz"
    if not path.exists():
        raise FileNotFoundError(path)
    d = np.load(path, allow_pickle=False)
    return {
        "sample_ids": np.asarray([str(s) for s in d["sample_ids"]]),
        "concat_patch_map": np.asarray(d["concat_patch_map"], dtype=np.float32),
        "ref_ids": np.asarray([str(r) for r in d["ref_ids"]]),
        "seed": int(d["seed"]),
        "shot": int(d["shot"]),
        "path": str(path),
        "sha256": sha256_file(path),
    }


def load_text_cache(category: str) -> dict:
    path = S1_CACHE / f"{category}.npz"
    d = np.load(path, allow_pickle=False)
    return {
        "sample_ids": np.asarray([str(s) for s in d["sample_ids"]]),
        "clip_global_test": np.asarray(d["clip_global_test"], dtype=np.float64),
        "text_prob_test": np.asarray(d["text_prob_test"], dtype=np.float64),
        "ref_ids": np.asarray([str(r) for r in d["ref_ids"]]),
        "clip_global_refs": np.asarray(d["clip_global_refs"], dtype=np.float64),
        "checkpoint_sha256": str(d["checkpoint_sha256"]),
        "text_embedding_sha256": str(d["text_embedding_sha256"]),
        "path": str(path),
        "sha256": sha256_file(path),
    }
