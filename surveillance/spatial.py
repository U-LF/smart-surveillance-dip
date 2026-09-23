"""Technical Task 1 - Spatial filtering (mean, median, Laplacian and friends).

Two implementations are provided for the core filters:
  impl="numpy"  - written from first principles (sliding windows), for the
                  report / viva to show *how* the operation works;
  impl="cv2"    - OpenCV's SIMD-optimised version used in the real-time path.
Both are checked for numerical equivalence in tests/test_core.py and timed
against each other in experiments/exp1_spatial.py (resource-constraint analysis).
"""
from __future__ import annotations

import cv2
import numpy as np
from numpy.lib.stride_tricks import sliding_window_view


def _per_channel(fn, img, *a, **k):
    if img.ndim == 2:
        return fn(img, *a, **k)
    return np.dstack([fn(img[..., c], *a, **k) for c in range(img.shape[2])])


# ----------------------------------------------------------------------------
# From-scratch primitives
# ----------------------------------------------------------------------------
def convolve2d(img: np.ndarray, kernel: np.ndarray) -> np.ndarray:
    """2-D correlation with reflect-101 border (same convention as cv2.filter2D)."""
    def _one(ch):
        kh, kw = kernel.shape
        p = np.pad(ch.astype(np.float64), ((kh // 2, kh // 2), (kw // 2, kw // 2)), mode="reflect")
        win = sliding_window_view(p, (kh, kw))
        return np.einsum("ijkl,kl->ij", win, kernel.astype(np.float64))
    return _per_channel(_one, img)


def _median_numpy(ch, k):
    p = np.pad(ch, k // 2, mode="edge")  # cv2.medianBlur replicates the border
    return np.median(sliding_window_view(p, (k, k)), axis=(-1, -2)).astype(ch.dtype)


# ----------------------------------------------------------------------------
# Smoothing filters
# ----------------------------------------------------------------------------
def mean_filter(img, k: int = 3, impl: str = "cv2"):
    """Arithmetic mean (box) filter - good for Gaussian noise, blurs edges."""
    if impl == "numpy":
        out = convolve2d(img, np.full((k, k), 1.0 / (k * k)))
        return np.clip(np.round(out), 0, 255).astype(np.uint8)
    return cv2.blur(img, (k, k), borderType=cv2.BORDER_REFLECT_101)


def gaussian_filter(img, sigma: float = 1.0):
    k = int(2 * np.ceil(3 * sigma) + 1)
    return cv2.GaussianBlur(img, (k, k), sigma)


def median_filter(img, k: int = 3, impl: str = "cv2"):
    """Order-statistic filter - optimal for impulse (salt & pepper) noise."""
    if impl == "numpy":
        return _per_channel(_median_numpy, img, k)
    return cv2.medianBlur(img, k)


def adaptive_median_filter(img, s_max: int = 7):
    """Adaptive median filter (Gonzalez & Woods, Sec. 5.3), fully vectorised.

    Stage A: grow the window until the median is not an impulse
             (z_min < z_med < z_max) or s_max is reached.
    Stage B: keep the original pixel if it is not an impulse, else use z_med.
    Preserves detail far better than a fixed median at high noise densities.
    """
    def _one(ch):
        out = ch.copy()
        undecided = np.ones(ch.shape, bool)
        z = ch.astype(np.int16)
        for s in range(3, s_max + 1, 2):
            kernel = np.ones((s, s), np.uint8)
            zmin = cv2.erode(ch, kernel, borderType=cv2.BORDER_REFLECT).astype(np.int16)
            zmax = cv2.dilate(ch, kernel, borderType=cv2.BORDER_REFLECT).astype(np.int16)
            zmed = cv2.medianBlur(ch, s).astype(np.int16)
            stage_a = (zmed > zmin) & (zmed < zmax) & undecided
            not_impulse = (z > zmin) & (z < zmax)
            out[stage_a & not_impulse] = ch[stage_a & not_impulse]
            fix = stage_a & ~not_impulse
            out[fix] = zmed[fix].astype(np.uint8)
            undecided &= ~stage_a
            if not undecided.any():
                break
        if undecided.any():
            out[undecided] = cv2.medianBlur(ch, s_max)[undecided]
        return out
    return _per_channel(_one, img)


def bilateral_filter(img, sigma_noise: float = 20.0, d: int = 7):
    """Edge-preserving smoothing; sigma_color tied to estimated noise level."""
    return cv2.bilateralFilter(img, d, sigmaColor=2.0 * sigma_noise, sigmaSpace=d)


def nlm_filter(img, sigma_noise: float = 20.0):
    """Non-local means - best quality, slowest (quality profile only)."""
    h = max(3.0, 0.8 * sigma_noise)
    if img.ndim == 3:
        return cv2.fastNlMeansDenoisingColored(img, None, h, h, 7, 21)
    return cv2.fastNlMeansDenoising(img, None, h, 7, 21)


# ----------------------------------------------------------------------------
# Sharpening (second-derivative / Laplacian)
# ----------------------------------------------------------------------------
LAPLACIAN_4 = np.array([[0, 1, 0], [1, -4, 1], [0, 1, 0]], np.float64)
LAPLACIAN_8 = np.array([[1, 1, 1], [1, -8, 1], [1, 1, 1]], np.float64)


def laplacian(img, eight: bool = True, impl: str = "cv2"):
    k = LAPLACIAN_8 if eight else LAPLACIAN_4
    if impl == "numpy":
        return convolve2d(img, k)
    return cv2.filter2D(img.astype(np.float64), -1, k, borderType=cv2.BORDER_REFLECT_101)


def laplacian_sharpen(img, alpha: float = 0.5, eight: bool = True, impl: str = "cv2"):
    """g = f - alpha * lap(f)   (centre coefficient negative -> subtract)."""
    g = img.astype(np.float64) - alpha * laplacian(img, eight, impl)
    return np.clip(np.rint(g), 0, 255).astype(np.uint8)


def unsharp_mask(img, sigma: float = 1.5, amount: float = 0.8):
    """High-boost filtering: g = f + k (f - f_blur). Less noise-sensitive than Laplacian."""
    blur = cv2.GaussianBlur(img, (0, 0), sigma)
    return cv2.addWeighted(img, 1 + amount, blur, -amount, 0)


def sobel_magnitude(gray):
    gx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
    return np.hypot(gx, gy)


SPATIAL_FILTERS = {
    "mean3": lambda im, s: mean_filter(im, 3),
    "mean5": lambda im, s: mean_filter(im, 5),
    "gaussian": lambda im, s: gaussian_filter(im, max(0.8, s / 15.0)),
    "median3": lambda im, s: median_filter(im, 3),
    "median5": lambda im, s: median_filter(im, 5),
    "adaptive_median": lambda im, s: adaptive_median_filter(im, 7),
    "bilateral": lambda im, s: bilateral_filter(im, s),
    "nlm": lambda im, s: nlm_filter(im, s),
    "median3+laplacian": lambda im, s: laplacian_sharpen(median_filter(im, 3), 0.25),
    "nlm+unsharp": lambda im, s: unsharp_mask(nlm_filter(im, s), 1.0, 0.5),
}
