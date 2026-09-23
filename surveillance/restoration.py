"""Technical Task 3 - Image restoration and reconstruction.

Restoration  : inverse filter, Wiener filter, constrained least squares (CLS)
               for motion-blurred frames; low-light enhancement (gamma,
               histogram equalisation written from scratch, CLAHE).
Reconstruction: in-painting of blocks lost in transmission.
Diagnostics  : blind estimators (noise sigma, impulse density, blur, brightness,
               illumination non-uniformity) that let the pipeline decide
               *which* restoration to run on each frame.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass

import cv2
import numpy as np

from .degradation import pad_psf


def _to_gray(img):
    return img if img.ndim == 2 else cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)


# ----------------------------------------------------------------------------
# Blind frame diagnostics
# ----------------------------------------------------------------------------
def estimate_noise_sigma(img, edge_fraction: float = 0.3) -> float:
    """Edge-aware Immerkaer noise estimator.

    Immerkaer (1996): sigma = sqrt(pi/2) / (6 N) * sum |I * M|,
    M = [[1,-2,1],[-2,4,-2],[1,-2,1]] (difference of two Laplacians, ~zero on
    ramps). Plain Immerkaer over-estimates noise on textured street scenes, so
    following Tai & Yang (2008) the sum is restricted to homogeneous pixels:
    the `edge_fraction` strongest Sobel responses are excluded.
    Calibrated on Penn-Fudan: clean frames 0.6-8.1, sigma=15 noise >= 9.4.
    """
    g = _to_gray(img).astype(np.float64)
    M = np.array([[1, -2, 1], [-2, 4, -2], [1, -2, 1]], np.float64)
    conv = np.abs(cv2.filter2D(g, -1, M))
    mag = np.hypot(cv2.Sobel(g, cv2.CV_64F, 1, 0), cv2.Sobel(g, cv2.CV_64F, 0, 1))
    homo = mag <= np.percentile(mag, 100 * (1 - edge_fraction))
    homo[[0, -1], :] = False
    homo[:, [0, -1]] = False
    n = homo.sum()
    if n == 0:
        return 0.0
    return float(np.sqrt(np.pi / 2) * conv[homo].sum() / (6.0 * n))


def estimate_impulse_ratio(img, diff_thresh: int = 60) -> float:
    """Fraction of pixels that are extreme (0/255) *and* differ strongly from
    their 3x3 median -> isolated impulses (saturated sky is not counted)."""
    g = _to_gray(img)
    med = cv2.medianBlur(g, 3)
    extreme = (g <= 5) | (g >= 250)
    return float((extreme & (np.abs(g.astype(np.int16) - med) > diff_thresh)).mean())


def blur_score(img) -> float:
    """Variance of the Laplacian - low values indicate a blurred frame."""
    return float(cv2.Laplacian(_to_gray(img), cv2.CV_64F).var())


def dark_region_fraction(img, grid: int = 8, dark: float = 40.0) -> float:
    """Fraction of a coarse grid x grid tiling whose mean luminance is < `dark`.
    High values with an acceptable global mean = shadows / lamp pools
    (uneven illumination) rather than a uniformly dark night frame."""
    g = _to_gray(img).astype(np.float64)
    small = cv2.resize(g, (grid, grid), interpolation=cv2.INTER_AREA)
    return float((small < dark).mean())


def lost_block_mask(img, block: int = 16, dark_thresh: int = 3) -> np.ndarray:
    """Detect all-black blocks (packet loss signature)."""
    g = _to_gray(img)
    h, w = g.shape
    hb, wb = h // block, w // block
    mask = np.zeros((h, w), np.uint8)
    if hb == 0 or wb == 0:
        return mask
    blocks = g[:hb * block, :wb * block].reshape(hb, block, wb, block).max(axis=(1, 3))
    lost = (blocks <= dark_thresh).astype(np.uint8)
    mask[:hb * block, :wb * block] = np.kron(lost, np.ones((block, block), np.uint8))
    return mask


@dataclass
class FrameDiagnostics:
    noise_sigma: float
    impulse_ratio: float
    mean_luma: float
    blur: float
    dark_regions: float
    lost_block_ratio: float

    def as_dict(self):
        return asdict(self)


def diagnose(img) -> FrameDiagnostics:
    return FrameDiagnostics(
        noise_sigma=estimate_noise_sigma(img),
        impulse_ratio=estimate_impulse_ratio(img),
        mean_luma=float(_to_gray(img).mean()),
        blur=blur_score(img),
        dark_regions=dark_region_fraction(img),
        lost_block_ratio=float(lost_block_mask(img).mean()),
    )


# ----------------------------------------------------------------------------
# Deconvolution (known / estimated PSF)
# ----------------------------------------------------------------------------
def _deconv(img, psf, H_restore_fn, pad: int = 32):
    def _one(ch):
        p = np.pad(ch.astype(np.float64), pad, mode="reflect")
        H = np.fft.fft2(pad_psf(psf, p.shape))
        F_hat = H_restore_fn(H, p.shape) * np.fft.fft2(p)
        return np.real(np.fft.ifft2(F_hat))[pad:-pad, pad:-pad]
    out = _one(img) if img.ndim == 2 else np.dstack([_one(img[..., c]) for c in range(img.shape[2])])
    return np.clip(np.rint(out), 0, 255).astype(np.uint8)


def inverse_filter(img, psf, eps: float = 1e-3):
    """F_hat = G / H with a tiny floor on |H| - shows why naive inversion fails."""
    def R(H, shape):
        Hs = np.where(np.abs(H) < eps, eps, H)
        return 1.0 / Hs
    return _deconv(img, psf, R)


def wiener_filter(img, psf, K: float = 0.01):
    """F_hat = [H* / (|H|^2 + K)] G, K ~ noise-to-signal power ratio."""
    return _deconv(img, psf, lambda H, s: np.conj(H) / (np.abs(H) ** 2 + K))


def cls_filter(img, psf, gamma: float = 0.01):
    """Constrained least squares: F_hat = [H* / (|H|^2 + gamma |P|^2)] G,
    P = FFT of the Laplacian -> penalises non-smooth solutions."""
    lap = np.array([[0, -1, 0], [-1, 4, -1], [0, -1, 0]], np.float64)

    def R(H, shape):
        P = np.fft.fft2(pad_psf(lap, shape))
        return np.conj(H) / (np.abs(H) ** 2 + gamma * np.abs(P) ** 2)
    return _deconv(img, psf, R)


def wiener_K_from_noise(img, sigma: float) -> float:
    """Heuristic NSR: noise variance over signal variance."""
    return float(max(1e-4, (sigma ** 2) / (_to_gray(img).astype(np.float64).var() + 1e-9)))


# ----------------------------------------------------------------------------
# Low-light / contrast restoration
# ----------------------------------------------------------------------------
def gamma_correction(img, gamma: float = 0.5):
    lut = np.clip(np.rint(255.0 * (np.arange(256) / 255.0) ** gamma), 0, 255).astype(np.uint8)
    return cv2.LUT(img, lut)


def auto_gamma(img, target_mean: float = 110.0):
    """Choose gamma so the mean luminance maps to target_mean."""
    m = max(1.0, _to_gray(img).mean())
    gamma = np.log(target_mean / 255.0) / np.log(m / 255.0)
    return gamma_correction(img, float(np.clip(gamma, 0.25, 1.0)))


def histogram_equalization(gray: np.ndarray) -> np.ndarray:
    """From scratch: s_k = (L-1) * CDF(r_k)  (Gonzalez & Woods, Eq. 3-15)."""
    hist = np.bincount(gray.ravel(), minlength=256).astype(np.float64)
    cdf = hist.cumsum()
    cdf_min = cdf[cdf > 0][0]
    lut = np.round((cdf - cdf_min) / (cdf[-1] - cdf_min + 1e-9) * 255).clip(0, 255).astype(np.uint8)
    return lut[gray]


def equalize_color(img):
    """HE on the luminance channel only (avoids colour shifts)."""
    if img.ndim == 2:
        return histogram_equalization(img)
    ycc = cv2.cvtColor(img, cv2.COLOR_BGR2YCrCb)
    ycc[..., 0] = histogram_equalization(ycc[..., 0])
    return cv2.cvtColor(ycc, cv2.COLOR_YCrCb2BGR)


def clahe(img, clip: float = 2.5, tiles: int = 8):
    """Contrast-limited adaptive HE on the L channel of CIELAB."""
    op = cv2.createCLAHE(clipLimit=clip, tileGridSize=(tiles, tiles))
    if img.ndim == 2:
        return op.apply(img)
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    lab[..., 0] = op.apply(lab[..., 0])
    return cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)


def enhance_low_light(img, method: str = "gamma+clahe"):
    if method == "gamma":
        return auto_gamma(img)
    if method == "he":
        return equalize_color(img)
    if method == "clahe":
        return clahe(img)
    if method == "gamma+clahe":
        return clahe(auto_gamma(img), clip=2.0)
    raise ValueError(method)


# ----------------------------------------------------------------------------
# Reconstruction of lost blocks
# ----------------------------------------------------------------------------
def reconstruct_lost_blocks(img, mask=None, method: str = "telea", radius: int = 5):
    """Fill packet-loss holes by diffusion-based in-painting (Telea / Navier-Stokes)."""
    if mask is None:
        mask = lost_block_mask(img)
    if mask.sum() == 0:
        return img
    flag = cv2.INPAINT_TELEA if method == "telea" else cv2.INPAINT_NS
    return cv2.inpaint(img, (mask > 0).astype(np.uint8) * 255, radius, flag)
