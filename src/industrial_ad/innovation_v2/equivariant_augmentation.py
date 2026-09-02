"""Route D — DEVA: Dual-Encoder Equivariance-Validated Normal Memory Augmentation.

Task book section 8. Augment normal reference images only (geometry /
photometric / combined packs), re-extract DINO + CLIP features, inverse-warp the
augmented feature grids back to the original coordinates, drop boundary-invalid
patches, and keep an augmented paired descriptor only when BOTH encoders agree
with the original patch (min cosine equivariance >= tau). The accepted patches
extend the normal memory; scoring is the frozen A1 concat + KNN.

This module is pure numpy/cv2 (no model loading) so it is unit-testable on CPU;
the GPU feature re-export lives in scripts/innovation_v2/export_deva_references.py.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import cv2
import numpy as np

from industrial_ad.fusion import rcec
from industrial_ad.innovation_v2.common import AlignedFeatures, InnovationError


# ---------------------------------------------------------------------------
# Transforms (image space, cv2 conventions)
# ---------------------------------------------------------------------------

@dataclass
class Transform:
    """Relative transform; M depends on the preprocessing scale (448/518)."""
    name: str
    tx_frac: float = 0.0
    ty_frac: float = 0.0
    angle_deg: float = 0.0
    contrast: float | None = None
    brightness_offset: float | None = None

    def M_at(self, size: float) -> np.ndarray:
        """2x3 original->augmented warp matrix for a square preprocessing size."""
        M = np.float32([[1, 0, self.tx_frac * size], [0, 1, self.ty_frac * size]])
        if self.angle_deg:
            M = cv2.getRotationMatrix2D((size / 2.0, size / 2.0), self.angle_deg, 1.0)
        return M


def make_transforms(pack: str, cfg: dict, seed: int = 0) -> list[Transform]:
    """Fixed transform list for one pack (deterministic, seeds recorded)."""
    deva = cfg.get("deva", {})
    geom = deva.get("geometry", {})
    shift_frac = float(geom.get("shift_frac", 0.04))
    max_rot = float(geom.get("max_rotation_deg", 5.0))
    photo = deva.get("photometric", {})
    brightness = [float(v) for v in photo.get("brightness", [0.90, 1.10])]
    contrast = [float(v) for v in photo.get("contrast", [0.90, 1.10])]

    if pack == "geometry":
        out = [Transform("tx{:+d}".format(int(shift_frac * 100)), tx_frac=shift_frac),
               Transform("tx{:d}".format(-int(shift_frac * 100)), tx_frac=-shift_frac),
               Transform("ty{:+d}".format(int(shift_frac * 100)), ty_frac=shift_frac),
               Transform("ty{:d}".format(-int(shift_frac * 100)), ty_frac=-shift_frac),
               Transform("rot{:+g}".format(max_rot), angle_deg=max_rot),
               Transform("rot{:g}".format(-max_rot), angle_deg=-max_rot)]
        return out

    if pack == "photometric":
        return [Transform(f"b{b:g}_c{c:g}", contrast=c, brightness_offset=(b - 1.0) * 128.0)
                for b in brightness for c in contrast]

    if pack == "combined":
        return [Transform("combined", tx_frac=shift_frac, ty_frac=shift_frac,
                          angle_deg=max_rot, contrast=contrast[0],
                          brightness_offset=(brightness[1] - 1.0) * 128.0)]
    raise InnovationError(f"unknown DEVA pack: {pack}")


def apply_photometric(image: np.ndarray, contrast: float, brightness_offset: float) -> np.ndarray:
    out = image.astype(np.float32) * float(contrast) + float(brightness_offset)
    return np.clip(out, 0, 255).astype(np.uint8)


def apply_transform(image: np.ndarray, t: Transform, size: int) -> np.ndarray:
    """Warp a square (size, size) RGB image with the transform (or photometric)."""
    if t.contrast is not None:
        return apply_photometric(image, t.contrast, t.brightness_offset)
    return cv2.warpAffine(image, t.M_at(float(size)), (size, size),
                          flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)


# ---------------------------------------------------------------------------
# Inverse warp of a feature grid back to the original coordinates
# ---------------------------------------------------------------------------

def inverse_warp_features(
    feat_grid: np.ndarray, M_inv: np.ndarray, stride: float
) -> tuple[np.ndarray, np.ndarray]:
    """Warp an augmented feature grid back to the original grid.

    Output pixel (r, c) samples the augmented grid at the inverse-transformed
    location of the original patch top-left corner (r, c) (stride = grid spacing
    in pixels). Identity thus reproduces the grid exactly. Returns
    (warped [H,W,D] float32, valid [H,W] bool); invalid patches (sampled from
    outside the augmented image) are set to NaN and masked out.
    """
    h, w, d = feat_grid.shape
    rr, cc = np.mgrid[0:h, 0:w].astype(np.float64)
    x_px = cc * stride
    y_px = rr * stride
    pts = np.stack([x_px.ravel(), y_px.ravel(), np.ones(h * w, np.float64)], axis=0)
    src = M_inv.astype(np.float64) @ pts  # [2, n] pixels in the augmented image
    map_x = (src[0].reshape(h, w) / stride).astype(np.float32)  # grid units
    map_y = (src[1].reshape(h, w) / stride).astype(np.float32)
    in_bounds = (
        (map_x >= 0) & (map_x <= float(w - 1)) &
        (map_y >= 0) & (map_y <= float(h - 1))
    )
    out = np.empty((h, w, d), dtype=np.float32)
    for k in range(d):
        out[:, :, k] = cv2.remap(feat_grid[:, :, k], map_x, map_y, cv2.INTER_LINEAR)
    out[~in_bounds] = np.nan
    return out, in_bounds


# ---------------------------------------------------------------------------
# Equivariance filter
# ---------------------------------------------------------------------------

def equivariance_scores(
    original: np.ndarray, warped: np.ndarray, valid: np.ndarray
) -> np.ndarray:
    """cos(original_patch, warped_patch) per patch; NaN where invalid."""
    orig = original.reshape(-1, original.shape[-1]).astype(np.float32)
    war = warped.reshape(-1, warped.shape[-1]).astype(np.float32)
    cos = np.einsum("nd,nd->n", orig, war)  # rows already unit L2
    cos = cos.reshape(original.shape[:2]).astype(np.float32)
    cos[~valid] = np.nan
    return cos


# ---------------------------------------------------------------------------
# Memory construction and scoring
# ---------------------------------------------------------------------------

def build_augmented_memory(
    aligned: AlignedFeatures,
    aug_d: np.ndarray,   # (n_aug, H, W, 768) warped to original DINO grid
    aug_c: np.ndarray,   # (n_aug, 37, 37, 768) warped to original CLIP grid
    valid_d: np.ndarray, # (n_aug, H, W) bool, DINO-grid valid patch mask
    valid_c: np.ndarray, # (n_aug, 37, 37) bool, CLIP-grid valid patch mask
    source_ref: np.ndarray,  # (n_aug,) int, source reference image index
    tau: float,
    clip_ref_grid: np.ndarray | None = None,  # (n_refs, 37, 37, 768) original CLIP grid
) -> tuple[np.ndarray, np.ndarray, dict]:
    """Filter augmented paired descriptors by dual-equivariance and concatenate
    them with the original reference memory. Returns (d_all, c_all, meta).

    e_D uses the DINO grid; e_C uses the CLIP grid (before resizing). The
    accepted CLIP mask is then resized to the DINO grid with nearest sampling.
    clip_ref_grid holds the original-scale CLIP reference features (37x37 grid,
    from the {cat}_k{shot}_identity.npz cache) so e_C is computed at native
    CLIP resolution; when absent it falls back to the resized aligned.c_ref.
    """
    from evaluate_a1_feature_fusion import resize_patches

    n_refs = aligned.n_references
    d_ref_flat = np.ascontiguousarray(aligned.d_ref.reshape(-1, 768), dtype=np.float32)
    c_ref_flat = np.ascontiguousarray(aligned.c_ref.reshape(-1, 768), dtype=np.float32)
    d_ref_grid = aligned.d_ref.reshape(n_refs, *aligned.grid, 768)
    if clip_ref_grid is not None:
        if clip_ref_grid.shape[1:] != aug_c.shape[1:]:
            raise InnovationError(
                f"clip_ref_grid {clip_ref_grid.shape} != aug_c {aug_c.shape}")
        c_ref_grid = clip_ref_grid
    else:
        c_ref_grid = aligned.c_ref.reshape(n_refs, *aug_c.shape[1:3], 768)

    d_all = [d_ref_flat]
    c_all = [c_ref_flat]
    n_acc_total = 0
    n_valid_total = 0
    per_image = []
    for g in range(aug_d.shape[0]):
        s = int(source_ref[g])
        e_d = equivariance_scores(d_ref_grid[s], aug_d[g], valid_d[g])
        e_c = equivariance_scores(c_ref_grid[s], aug_c[g], valid_c[g])
        # separate per-branch acceptance, intersected on the DINO grid
        acc_d_d = (e_d >= float(tau)) & np.isfinite(e_d)
        acc_c_c = (e_c >= float(tau)) & np.isfinite(e_c)
        acc_c = _nearest_resize_bool(acc_c_c, aligned.grid)
        acc = acc_d_d & acc_c & valid_d[g]
        n_valid = int(valid_d[g].sum())
        n_acc = int(acc.sum())
        n_acc_total += n_acc
        n_valid_total += n_valid
        per_image.append({"n_valid": n_valid, "n_accepted": n_acc,
                          "acceptance_ratio": (n_acc / n_valid) if n_valid else 0.0})
        if n_acc:
            flat_idx = np.nonzero(acc.ravel())[0]
            c_resized = np.ascontiguousarray(
                resize_patches(aug_c[g][None], aligned.grid)[0], dtype=np.float32)
            d_all.append(np.ascontiguousarray(aug_d[g].reshape(-1, 768)[flat_idx], dtype=np.float32))
            c_all.append(np.ascontiguousarray(c_resized.reshape(-1, 768)[flat_idx], dtype=np.float32))

    d_full = np.concatenate(d_all, axis=0)
    c_full = np.concatenate(c_all, axis=0)
    meta = {
        "n_augmentations": int(aug_d.shape[0]),
        "n_valid_patches_total": n_valid_total,
        "n_accepted_patches_total": n_acc_total,
        "global_acceptance_ratio": round(n_acc_total / n_valid_total, 6) if n_valid_total else 0.0,
        "per_image": per_image,
        "tau": tau,
        "memory_growth": round(float(d_full.shape[0] / d_ref_flat.shape[0]), 4),
    }
    return d_full, c_full, meta


def _nearest_resize_bool(mask: np.ndarray, target: tuple[int, int]) -> np.ndarray:
    m = mask.astype(np.float32)
    m = cv2.resize(m, (target[1], target[0]), interpolation=cv2.INTER_NEAREST)
    return m > 0.5


def score_deva_memory(
    aligned: AlignedFeatures,
    d_full: np.ndarray,
    c_full: np.ndarray,
    dino_weight: float = 0.5,
) -> np.ndarray:
    """A1 concat + KNN score grid with the augmented memory."""
    n, h, w = aligned.d_feat.shape[0], *aligned.grid
    z_q = rcec._concat_and_l2(
        aligned.d_feat.reshape(-1, 768), aligned.c_feat.reshape(-1, 768), dino_weight)
    z_r = rcec._concat_and_l2(d_full, c_full, dino_weight)
    index = rcec._faiss_index(np.ascontiguousarray(z_r, dtype=np.float32))
    dists, _ = rcec._search_chunked(index, np.ascontiguousarray(z_q, dtype=np.float32),
                                    k=1, chunk=16384)
    return (dists[:, 0] / 2.0).reshape(n, h, w)
