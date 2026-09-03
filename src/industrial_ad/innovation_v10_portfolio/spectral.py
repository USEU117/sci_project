"""Route E — STR: Support-Conditioned Texture Residual (task book 19 §8).

Training-free 2-level Haar spectral residual calibrated ONLY on normal reference
images (median/MAD of absolute subband coefficients per channel per band).

Residual semantics (pre-registered): defects add high-frequency energy, so a band
residual is the robust z of the absolute coefficient against reference statistics,
floored at 0 and clipped at 6. The per-pixel STR residual is the trimmed mean
(drop min+max) of the per-band channel-averaged z. R0 uses this purely as a
DIAGNOSTIC map (never fused with A1).
"""

from __future__ import annotations

import numpy as np
import pywt


def _channels(img: np.ndarray) -> dict:
    """img: uint8 RGB (H, W, 3) -> gray / R-G / B-Y float maps."""
    r = img[..., 0].astype(np.float32)
    g = img[..., 1].astype(np.float32)
    b = img[..., 2].astype(np.float32)
    return {
        "gray": 0.299 * r + 0.587 * g + 0.114 * b,
        "rg": r - g,
        "by": b - 0.5 * (r + g),
    }


def _dwt_detail(plane: np.ndarray) -> dict:
    """One 2-level Haar transform -> absolute detail bands at each level."""
    out = {}
    a1, (h1, v1, d1) = pywt.dwt2(plane, "haar")
    out["h1"] = np.abs(h1)
    out["v1"] = np.abs(v1)
    out["d1"] = np.abs(d1)
    _, (h2, v2, d2) = pywt.dwt2(a1, "haar")
    out["h2"] = np.abs(h2)
    out["v2"] = np.abs(v2)
    out["d2"] = np.abs(d2)
    return out


def _pad_to_even(x: np.ndarray) -> np.ndarray:
    h, w = x.shape
    return x[: h - (h % 2), : w - (w % 2)]


def band_stats_from_references(planes_by_ch: dict[str, list[np.ndarray]]) -> dict:
    """Global robust stats per (channel, band) over normal reference planes.

    planes_by_ch: channel name -> list of float planes from normal references.
    Returns stats[(channel, band)] = (median_abs, mad_abs_of_abs).
    """
    bands_per = ["h1", "v1", "d1", "h2", "v2", "d2"]
    stats: dict[tuple[str, str], tuple[float, float]] = {}
    for ch, planes in planes_by_ch.items():
        pooled: dict[str, list[np.ndarray]] = {b: [] for b in bands_per}
        for plane in planes:
            plane = _pad_to_even(plane)
            det = _dwt_detail(plane)
            for b in bands_per:
                pooled[b].append(det[b].ravel())
        for b in bands_per:
            vals = np.concatenate(pooled[b])
            med = float(np.median(vals))
            mad = float(np.median(np.abs(vals - med)))
            stats[(ch, b)] = (med, max(mad, 1e-6))
    return stats


def residual_map(img: np.ndarray, stats: dict, size: int = 448) -> np.ndarray:
    """STR residual for one image: [size, size] float32 at A1 map resolution.

    img: uint8 RGB at any resolution >= size (resized to size internally).
    """
    import cv2

    img = cv2.resize(img, (size, size), interpolation=cv2.INTER_AREA)
    planes = _channels(img)
    bands_per = ["h1", "v1", "d1", "h2", "v2", "d2"]
    # robust z per (channel, band), floored at 0, clipped at 6
    z_per_band: dict[str, np.ndarray] = {}
    for b in bands_per:
        zs = []
        for ch in ("gray", "rg", "by"):
            plane = _pad_to_even(planes[ch])
            det = _dwt_detail(plane)
            coeff = det[b]
            med, mad = stats[(ch, b)]
            z = np.maximum(0.0, (coeff - med) / (1.4826 * mad))
            zs.append(np.clip(z, 0.0, 6.0))
        # level-2 bands are 2x smaller: upsample to level-1 grid
        z_c = np.mean(zs, axis=0)
        if b in ("h2", "v2", "d2"):
            z_c = cv2.resize(z_c.astype(np.float32), None, fx=2, fy=2,
                             interpolation=cv2.INTER_LINEAR)
        z_per_band[b] = z_c
    # trimmed mean over bands (drop min + max of the 6 bands)
    stack = np.stack([z_per_band[b] for b in bands_per])  # [6, H, W]
    trimmed = (stack.sum(axis=0) - stack.min(axis=0) - stack.max(axis=0)) / 4.0
    # upscale band-level (coarse) values to full size
    return cv2.resize(trimmed.astype(np.float32), (size, size),
                      interpolation=cv2.INTER_CUBIC)


def gradient_magnitude_map(img: np.ndarray, size: int = 448) -> np.ndarray:
    """Control predictor: RGB gradient magnitude only (no reference calibration)."""
    import cv2

    img = cv2.resize(img, (size, size), interpolation=cv2.INTER_AREA)
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY).astype(np.float32)
    gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    return np.sqrt(gx * gx + gy * gy)


def random_phase_surrogate(img: np.ndarray, size: int = 448, seed: int = 0) -> np.ndarray:
    """Same magnitude spectrum, randomized phase -> RGB surrogate (control)."""
    import cv2

    img = cv2.resize(img, (size, size), interpolation=cv2.INTER_AREA).astype(np.float32)
    rng = np.random.default_rng(seed)
    out = np.empty_like(img)
    for c in range(3):
        f = np.fft.fft2(img[..., c])
        mag = np.abs(f)
        phase = np.exp(1j * rng.uniform(0, 2 * np.pi, size=f.shape))
        phase[0, 0] = 1.0
        recon = np.real(np.fft.ifft2(mag * phase))
        out[..., c] = recon
    return np.clip(out, 0, 255).astype(np.uint8)
