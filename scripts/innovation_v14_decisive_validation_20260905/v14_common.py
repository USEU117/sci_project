"""v14 decisive-validation common helpers (doc28).

Leak gate: every fit/select ID must be a K-shot SUPPORT path from the MPDD
manifest; any '/test/' in a fit/select list raises. test/good images are NOT
allowed to enter fitting/selection/calibration (doc28 s4.2; V13 DATA_ROLES).
"""
from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
MANIFEST = ROOT / "data/splits/mpdd/manifest.json"
DATA_ROOT = ROOT / "data/mpdd_raw/MPDD"
CATEGORIES = ["bracket_black", "bracket_brown", "bracket_white", "connector", "metal_plate", "tubes"]


def load_manifest():
    return json.loads(MANIFEST.read_text(encoding="utf-8"))


def support_paths(cat: str, shot: int, seed: str = "0", manifest=None) -> list[str]:
    """K-shot support image REL paths (train/good only) for (cat, seed, shot)."""
    m = manifest or load_manifest()
    return list(m["categories"][cat][seed][str(shot)])


def assert_fit_ids_are_support(fit_ids: list[str], cat: str, shot: int, seed: str = "0"):
    """Fail hard if any fit/select ID is not exactly a support path for this shot."""
    m = load_manifest()
    sup = set(support_paths(cat, shot, seed, m))
    for rel in fit_ids:
        if "/test/" in rel:
            raise ValueError(f"LEAK: fit id {rel} contains /test/")
        if rel not in sup:
            raise ValueError(f"fit id {rel} not in support set of {cat} k{shot}")
    return True


def abs_path(rel: str) -> Path:
    return DATA_ROOT / rel
