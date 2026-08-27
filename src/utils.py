"""Shared utilities restored for the frozen A1 evaluation chain (P0 rebuild)."""

from __future__ import annotations

import cv2
import numpy as np
from scipy.ndimage import gaussian_filter


def dists2map(dists: np.ndarray, shape: tuple[int, int]) -> np.ndarray:
    return gaussian_filter(
        cv2.resize(dists, (shape[1], shape[0]), interpolation=cv2.INTER_LINEAR), sigma=4
    )
