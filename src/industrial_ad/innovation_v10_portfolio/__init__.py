"""innovation_v10_portfolio (task book 19) - shared paths + feature loading.

Exact A1 concat reproduction from the frozen v3 feature caches, with
per-reference distance computation for memory-side routes (CRAM/CAPM/NORC).
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import faiss
import numpy as np

ROOT = Path(__file__).resolve().parents[3]
for p in (str(ROOT / "src"), str(ROOT / "scripts"), str(ROOT / "methods" / "anomalydino")):
    if p not in sys.path:
        sys.path.insert(0, p)

from sklearn.preprocessing import normalize  # noqa: E402

FEATURES_ROOT = ROOT / "outputs" / "dynamic_fusion" / "v3_direction_a"
EXPERIMENT_ROOT = ROOT / "experiments" / "dynamic_fusion" / "innovation_v10_portfolio"
OUTPUTS_ROOT = ROOT / "outputs" / "dynamic_fusion" / "innovation_v10_portfolio"
MANIFEST = ROOT / "data" / "splits" / "mpdd" / "manifest.json"
MAP_SIZE = (448, 448)
GRID = (32, 32)
DINO_WEIGHT = 0.5
KNN_K = 1
CATS = ["bracket_black", "bracket_brown", "bracket_white",
        "connector", "metal_plate", "tubes"]

DINO_DIR = "features_vitb14_s{seed}_k{shot}/anomalydino_visual"
CLIP_DIR = "features_s{seed}_k{shot}/anomalyclip_text"


def load_raw_features(cat: str, seed: int, shot: int, branch: str) -> dict:
    """branch in {dino, clip}; reads the v3 feature cache npz."""
    from evaluate_a1_feature_fusion import load_features
    d = FEATURES_ROOT / (DINO_DIR if branch == "dino" else CLIP_DIR).format(
        seed=seed, shot=shot)
    return load_features(d / f"{cat}.npz")


def resize_patches(patches: np.ndarray, target_grid=(32, 32)) -> np.ndarray:
    from evaluate_a1_feature_fusion import resize_patches as _rp
    return _rp(patches, target_grid)


def concat_bank(cat: str, seed: int, shot: int) -> dict:
    """Return L2-normalised concat (dino w .5 + clip w .5) test/ref patch
    features on the 32x32 grid, plus aligned sample_ids and ref_ids."""
    import numpy as np
    dino = load_raw_features(cat, seed, shot, "dino")
    clip = load_raw_features(cat, seed, shot, "clip")
    if not np.array_equal(dino["sample_ids"], clip["sample_ids"]):
        raise ValueError(f"dino/clip sample order mismatch {cat} s{seed} k{shot}")
    clip_f = resize_patches(clip["patch_features"], GRID)
    clip_r = resize_patches(clip["ref_patch_features"], GRID)
    dino_f, dino_r = dino["patch_features"], dino["ref_patch_features"]
    def l2(x):
        return normalize(x.reshape(-1, x.shape[-1])).reshape(x.shape).astype(np.float32)
    dino_f, dino_r = l2(dino_f), l2(dino_r)
    clip_f, clip_r = l2(clip_f), l2(clip_r)
    feat = np.concatenate([DINO_WEIGHT * dino_f, (1 - DINO_WEIGHT) * clip_f], axis=-1)
    ref = np.concatenate([DINO_WEIGHT * dino_r, (1 - DINO_WEIGHT) * clip_r], axis=-1)
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    ref_ids = manifest["categories"][cat][str(seed)][str(shot)]
    if len(ref) != len(ref_ids):
        raise ValueError("ref bank count != manifest shot")
    return {"sample_ids": np.asarray(dino["sample_ids"]),
            "feat": feat, "ref": ref, "ref_ids": np.asarray(ref_ids),
            "n_ref": len(ref)}


def dist_to_ref_single(feat_flat: np.ndarray, ref_flat: np.ndarray) -> np.ndarray:
    """1 - cos distance from each query patch to nearest patch of ONE reference."""
    q = feat_flat.astype(np.float32).copy()
    r = ref_flat.astype(np.float32).copy()
    faiss.normalize_L2(q)
    faiss.normalize_L2(r)
    index = faiss.IndexFlatL2(r.shape[1])
    index.add(r)
    d, _ = index.search(q, k=1)
    return d[:, 0] / 2.0


def per_reference_dists(feat: np.ndarray, ref: np.ndarray,
                        grid=GRID, chunk: int = 8) -> np.ndarray:
    """(n,H,W,D) test, (k,H,W,D) refs -> (n, k, H, W) d_r arrays.

    Processes the test images in chunks to bound peak memory."""
    n = feat.shape[0]
    k = ref.shape[0]
    out = np.empty((n, k, grid[0], grid[1]), dtype=np.float32)
    Hp, Wp = grid
    for i0 in range(0, n, chunk):
        i1 = min(i0 + chunk, n)
        q = feat[i0:i1].reshape(-1, feat.shape[-1])
        for r in range(k):
            rref = ref[r].reshape(-1, feat.shape[-1])
            out[i0:i1, r] = dist_to_ref_single(q, rref).reshape(i1 - i0, Hp, Wp)
    return out
