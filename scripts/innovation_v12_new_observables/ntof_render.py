"""Shared NTOF render library (doc 22 s3.3 + pre-registered R0_PROTOCOL).

Deterministic photometric interventions and synthetic structure defects applied
to MPDD normal images at their native 1024x1024 resolution, BEFORE the frozen
per-branch preprocessing, so that only the intervention changes the features.

All strengths are frozen in R0_PROTOCOL.json (no test-driven adjustment).
"""

from __future__ import annotations

import numpy as np

# 5 families x 3 fit strengths (used on support/memory images to fit U)
FIT_FAMILIES = ["exposure", "gamma", "white_balance", "lr_brightness_gradient", "specular_blob"]
FIT_STRENGTHS = {
    "exposure": [0.70, 1.15, 1.40],
    "gamma": [0.80, 1.20, 1.50],
    "white_balance": [[0.92, 1.0, 1.08], [1.0, 1.0, 1.0], [1.08, 1.0, 0.92]],
    "lr_brightness_gradient": [-0.25, 0.0, 0.25],
    "specular_blob": [18.0, 36.0, 54.0],
}
# held-out strengths (one per family) - NEVER used to fit U
HELD_STRENGTHS = {
    "exposure": 0.55,
    "gamma": 1.8,
    "white_balance": [1.16, 1.0, 0.86],
    "lr_brightness_gradient": 0.35,
    "specular_blob": 80.0,
}
SYNTHETIC_KINDS = ["cutpaste", "local_erasure", "thin_scratch"]
FIT_VARIANT_KEYS = [f"{fam}_{i}" for fam in FIT_FAMILIES for i in range(3)]
HELD_VARIANT_KEYS = [f"held_{fam}" for fam in FIT_FAMILIES]
SPECULAR_SIGMA_FRAC = 0.12


def _seed_for(cat: str, rel: str) -> int:
    h = 0
    for ch in f"{cat}::{rel}":
        h = (h * 131 + ord(ch)) & 0xFFFFFFFF
    return h


def apply_exposure(x: np.ndarray, s: float) -> np.ndarray:
    return np.clip(x.astype(np.float32) * s, 0.0, 255.0).astype(np.uint8)


def apply_gamma(x: np.ndarray, g: float) -> np.ndarray:
    y = (x.astype(np.float32) / 255.0) ** (1.0 / g)
    return np.clip(y * 255.0, 0.0, 255.0).astype(np.uint8)


def apply_white_balance(x: np.ndarray, mult: list) -> np.ndarray:
    y = x.astype(np.float32).copy()
    for c, m in enumerate(mult):
        y[..., c] = y[..., c] * m
    return np.clip(y, 0.0, 255.0).astype(np.uint8)


def apply_lr_gradient(x: np.ndarray, delta: float) -> np.ndarray:
    h, w = x.shape[:2]
    ramp = 1.0 + delta * (np.arange(w, dtype=np.float32) / w - 0.5)
    y = x.astype(np.float32) * ramp[None, :, None]
    return np.clip(y, 0.0, 255.0).astype(np.uint8)


def apply_specular(x: np.ndarray, amp: float) -> np.ndarray:
    h, w = x.shape[:2]
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    cx, cy = w * 0.5, h * 0.42
    sigma = SPECULAR_SIGMA_FRAC * min(h, w)
    blob = amp * np.exp(-((xx - cx) ** 2 + (yy - cy) ** 2) / (2.0 * sigma ** 2))
    y = x.astype(np.float32) + blob[..., None]
    return np.clip(y, 0.0, 255.0).astype(np.uint8)


def render_fit_variant(x: np.ndarray, fam: str, idx: int) -> np.ndarray:
    s = FIT_STRENGTHS[fam][idx]
    if fam == "exposure":
        return apply_exposure(x, s)
    if fam == "gamma":
        return apply_gamma(x, s)
    if fam == "white_balance":
        return apply_white_balance(x, s)
    if fam == "lr_brightness_gradient":
        return apply_lr_gradient(x, s)
    if fam == "specular_blob":
        return apply_specular(x, s)
    raise ValueError(fam)


def render_held_variant(x: np.ndarray, fam: str) -> np.ndarray:
    s = HELD_STRENGTHS[fam]
    if fam == "exposure":
        return apply_exposure(x, s)
    if fam == "gamma":
        return apply_gamma(x, s)
    if fam == "white_balance":
        return apply_white_balance(x, s)
    if fam == "lr_brightness_gradient":
        return apply_lr_gradient(x, s)
    if fam == "specular_blob":
        return apply_specular(x, s)
    raise ValueError(fam)


def render_synthetic(x: np.ndarray, kind: str, seed: int) -> tuple[np.ndarray, np.ndarray]:
    """Return (defect image, 1024x1024 uint8 mask 255 inside defect)."""
    rng = np.random.RandomState(seed)
    h, w = x.shape[:2]
    mask = np.zeros((h, w), dtype=np.uint8)
    y = x.copy()
    if kind == "cutpaste":
        side = int(rng.randint(70, 116))
        sx = int(rng.randint(0, w // 2 - side))
        sy = int(rng.randint(0, h - side))
        tx = int(rng.randint(w // 2, w - side))
        ty = int(rng.randint(0, h - side))
        y[ty:ty + side, tx:tx + side] = y[sy:sy + side, sx:sx + side]
        mask[ty:ty + side, tx:tx + side] = 255
    elif kind == "local_erasure":
        bw = int(rng.randint(64, 140))
        bh = int(rng.randint(64, 140))
        bx = int(rng.randint(0, w - bw))
        by = int(rng.randint(0, h - bh))
        noise = rng.normal(128.0, 34.0, (bh, bw, 3)).clip(0, 255).astype(np.uint8)
        y[by:by + bh, bx:bx + bw] = noise
        mask[by:by + bh, bx:bx + bw] = 255
    elif kind == "thin_scratch":
        cx, cy = int(rng.randint(w // 4, 3 * w // 4)), int(rng.randint(h // 4, 3 * h // 4))
        length = int(rng.randint(180, 320))
        angle = rng.uniform(0.25, 0.75) * np.pi
        dx, dy = int(np.cos(angle) * length), int(np.sin(angle) * length)
        thickness = int(rng.randint(2, 5))
        pts = np.linspace(0.0, 1.0, int(length // 2) + 2)
        for t in pts:
            px = int(round(cx - dx / 2 + dx * t))
            py = int(round(cy - dy / 2 + dy * t))
            if 0 <= px < w and 0 <= py < h:
                r0 = max(0, py - thickness // 2)
                r1 = min(h, py + thickness // 2 + 1)
                c0 = max(0, px - thickness // 2)
                c1 = min(w, px + thickness // 2 + 1)
                y[r0:r1, c0:c1] = np.clip(y[r0:r1, c0:c1].astype(np.int16) - 110, 0, 255).astype(np.uint8)
                mask[r0:r1, c0:c1] = 255
    else:
        raise ValueError(kind)
    return y, mask


REF_KEYS = [f"{fam}_{i}" for fam in FIT_FAMILIES for i in range(3)]
HELD_KEYS = [f"held_{fam}" for fam in FIT_FAMILIES]
SYN_KEYS = SYNTHETIC_KINDS


def fit_key_to_family_idx(key: str) -> tuple[str, int]:
    fam, idx = key.rsplit("_", 1)
    return fam, int(idx)


def render_by_key(x: np.ndarray, key: str, cat: str, rel: str) -> tuple[np.ndarray, np.ndarray]:
    """Return (rendered image, 1024x1024 defect mask or None)."""
    if key.startswith("held_"):
        return render_held_variant(x, key[5:]), None
    if key in SYN_KEYS:
        img, mask = render_synthetic(x, key, _seed_for(cat, rel))
        return img, mask
    fam, idx = fit_key_to_family_idx(key)
    return render_fit_variant(x, fam, idx), None
