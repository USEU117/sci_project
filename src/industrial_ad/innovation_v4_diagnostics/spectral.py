"""Spectral descriptors for D1 / route J (SF-NM) — numpy/pywt only.

A fixed, parameter-free two-scale stationary-wavelet descriptor is computed per
DINO patch over luminance + two opponent-colour channels. Each patch of a
448x448 input covers a 14x14 window of 448-grid coefficients, so the descriptor
lives on the same 32x32 grid as the A1 fused features.

Descriptor channels (<= 32, fixed):
  18 = log-energy of {LH,HL,HH} x {scale1, scale2} x {lum, opp1, opp2}
   4 = log direction ratio per scale on luminance: log(EH/EV), log(ED/(EH+EV+ED)+eps)
   2 = adjacent-energy drop: per scale, log of (central window mean / 3x3-neighbour mean)
  => 24 dims total.  Only luminance enters the ratios; all bands per patch.

Scores are produced by robustly normalising against normal reference patches and
a FAISS 1-NN spectral memory — the same protocol as A1 but on the spectral
descriptor. CTRL-* variants (documented in task book 14 section 3.3) are exposed
for later route-J gating; D1 uses the full descriptor only.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
for p in (str(ROOT / "src"), str(ROOT / "scripts")):
    if p not in sys.path:
        sys.path.insert(0, p)


def rgb_to_channels(image: np.ndarray) -> np.ndarray:
    """RGB uint8 [H,W,3] -> float channels [C,H,W], C=3.

    luma = 0.299R+0.587G+0.114B; opp1 = R-G; opp2 = (R+G)/2 - B.
    """
    f = image.astype(np.float64)
    lum = 0.299 * f[..., 0] + 0.587 * f[..., 1] + 0.114 * f[..., 2]
    opp1 = f[..., 0] - f[..., 1]
    opp2 = 0.5 * (f[..., 0] + f[..., 1]) - f[..., 2]
    return np.stack([lum, opp1, opp2], axis=0)


def _haar_swt_bands(channel: np.ndarray, n_scales: int = 2) -> list:
    """Stationary (undecimated) Haar decomposition bands per scale.

    Returns [(cH, cV, cD) x n_scales]; pywt.swt2 keeps every array at the input
    size, so per-patch windowing is straightforward.
    """
    import pywt
    out = []
    coeffs = pywt.swt2(channel, "haar", level=n_scales)  # list len n_scales
    for k in range(n_scales):
        (_, detail) = coeffs[k]
        out.append(tuple(np.asarray(b, dtype=np.float64) for b in detail))
    return out


def swt_descriptor(
    image: np.ndarray, n_scales: int = 2,
    grid: tuple[int, int] = (32, 32), window_px: int = 14,
) -> np.ndarray:
    """Fixed spectral descriptor for one square RGB image.

    Returns [H, W, 24] on `grid` (default the DINO 32x32 grid).
    """
    h, w = image.shape[:2]
    assert h == w and h % window_px == 0, f"expected square multiple of {window_px}, got {h}x{w}"
    ch = rgb_to_channels(image)
    gh, gw = grid
    bands = []
    for c in range(ch.shape[0]):
        bands.append(_haar_swt_bands(ch[c], n_scales))
    # per-band 2D log-energy via uniform_filter over the window
    from scipy.ndimage import uniform_filter

    hw = window_px
    feats = []
    # 1) log-energy per (channel, scale, band)
    for c in range(ch.shape[0]):
        for s in range(n_scales):
            for b in range(3):
                coef = bands[c][s][b]
                e = uniform_filter(coef * coef, size=hw, mode="reflect")
                feats.append(np.log(1e-8 + e))
    # 2) luminance direction ratios per scale
    lum_bands = bands[0]
    for s in range(n_scales):
        eh = uniform_filter(lum_bands[s][0] * lum_bands[s][0], size=hw, mode="reflect")
        ev = uniform_filter(lum_bands[s][1] * lum_bands[s][1], size=hw, mode="reflect")
        ed = uniform_filter(lum_bands[s][2] * lum_bands[s][2], size=hw, mode="reflect")
        feats.append(np.log(1e-8 + eh) - np.log(1e-8 + ev))
        feats.append(np.log(1e-8 + ed) - np.log(1e-8 + eh + ev))
    # 3) luminance adjacent-energy drop per scale (centre vs 3x3-window neighbour mean)
    for s in range(n_scales):
        tot = sum(uniform_filter(lum_bands[s][b] * lum_bands[s][b], size=hw, mode="reflect")
                  for b in range(3))
        centre = tot
        neigh = uniform_filter(tot, size=3 * hw, mode="reflect")
        feats.append(np.log(1e-8 + centre) - np.log(1e-8 + neigh))
    desc = np.stack(feats, axis=-1)  # [448,448,24]
    # downsample each channel grid by window_px via block mean
    gh_, gw_ = gh, gw
    if desc.shape[0] == h:
        desc = desc.reshape(gh_, window_px, gw_, window_px, -1).mean(axis=(1, 3))
    return desc.astype(np.float32)


def spectral_scores(
    desc_ref: np.ndarray,   # (S, H, W, D) reference descriptors
    desc_query: np.ndarray, # (N, H, W, D) query descriptors
    robust: bool = True,
) -> np.ndarray:
    """1-NN L2 distance from each query patch to the robustly normalised
    spectral memory; higher = more anomalous."""
    import faiss

    d = desc_ref.shape[-1]
    r = desc_ref.reshape(-1, d).astype(np.float32)
    q = desc_query.reshape(-1, d).astype(np.float32)
    if robust:
        med = np.median(r, axis=0, keepdims=True)
        mad = np.median(np.abs(r - med), axis=0, keepdims=True) + 1e-6
        r = (r - med) / mad
        q = (q - med) / mad
    r = np.ascontiguousarray(r, dtype=np.float32)
    q = np.ascontiguousarray(q, dtype=np.float32)
    faiss.normalize_L2(r)
    faiss.normalize_L2(q)
    index = faiss.IndexFlatL2(r.shape[1])
    index.add(r)
    dists, _ = index.search(q, k=1)
    n, h, w = desc_query.shape[:3]
    return (dists[:, 0] / 2.0).astype(np.float32).reshape(n, h, w)


def spectral_descriptor_image(image_rgb: np.ndarray) -> np.ndarray:
    """Convenience: resize any RGB image to 448 square then descriptor [32,32,24]."""
    import cv2

    im = cv2.resize(image_rgb, (448, 448), interpolation=cv2.INTER_LINEAR)
    return swt_descriptor(im)
