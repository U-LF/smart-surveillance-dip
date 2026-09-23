"""Technical Task 2 - Frequency-domain filtering (Fourier transform).

Implements the textbook transfer functions H(u,v) (ideal / Butterworth /
Gaussian, low- and high-pass), an *automatic* notch-reject filter that finds
periodic-interference spikes in the spectrum, and homomorphic filtering for
uneven street lighting. Filtering pipeline for every function:

    f -> reflect-pad -> F = FFT(f) (centred) -> G = H . F -> g = IFFT(G) -> crop
"""
from __future__ import annotations

import cv2
import numpy as np


# ----------------------------------------------------------------------------
# Basics
# ----------------------------------------------------------------------------
def spectrum(gray: np.ndarray, window: bool = True) -> np.ndarray:
    """Centred log-magnitude spectrum log(1+|F|). Hann window suppresses the
    '+'-shaped artefact caused by image borders (used for peak detection)."""
    g = gray.astype(np.float64)
    if window:
        g = (g - g.mean()) * np.outer(np.hanning(g.shape[0]), np.hanning(g.shape[1]))
    return np.log1p(np.abs(np.fft.fftshift(np.fft.fft2(g))))


def distance_grid(shape) -> np.ndarray:
    P, Q = shape
    u = np.arange(P) - P // 2   # DC sits at index P//2 after fftshift
    v = np.arange(Q) - Q // 2
    V, U = np.meshgrid(v, u)
    return np.sqrt(U ** 2 + V ** 2)


def lowpass(shape, d0: float, kind: str = "gaussian", order: int = 2) -> np.ndarray:
    D = distance_grid(shape)
    if kind == "ideal":
        return (D <= d0).astype(np.float64)
    if kind == "butterworth":
        return 1.0 / (1.0 + (D / d0) ** (2 * order))
    if kind == "gaussian":
        return np.exp(-(D ** 2) / (2 * d0 ** 2))
    raise ValueError(kind)


def highpass(shape, d0: float, kind: str = "gaussian", order: int = 2) -> np.ndarray:
    return 1.0 - lowpass(shape, d0, kind, order)


def _pad_amount(n: int) -> int:
    return max(8, n // 8)


def apply_filter(img: np.ndarray, H_fn, pad: bool = True) -> np.ndarray:
    """Apply a centred transfer function to each channel.

    H_fn(shape) -> H so the filter is built for the *padded* size.
    """
    def _one(ch):
        ph, pw = (_pad_amount(ch.shape[0]), _pad_amount(ch.shape[1])) if pad else (0, 0)
        f = np.pad(ch.astype(np.float64), ((ph, ph), (pw, pw)), mode="reflect")
        F = np.fft.fftshift(np.fft.fft2(f))
        g = np.real(np.fft.ifft2(np.fft.ifftshift(F * H_fn(f.shape))))
        return g[ph:ph + ch.shape[0], pw:pw + ch.shape[1]]
    out = _one(img) if img.ndim == 2 else np.dstack([_one(img[..., c]) for c in range(img.shape[2])])
    return np.clip(np.rint(out), 0, 255).astype(np.uint8)


def frequency_lowpass(img, d0_ratio: float = 0.08, kind: str = "gaussian", order: int = 2):
    """d0 expressed as a fraction of the (padded) image size -> resolution independent."""
    return apply_filter(img, lambda s: lowpass(s, d0_ratio * min(s), kind, order))


def frequency_highpass_sharpen(img, d0_ratio: float = 0.05, k: float = 0.6):
    """High-frequency emphasis: H = 1 + k * HP  (sharpening in the Fourier domain)."""
    return apply_filter(img, lambda s: 1.0 + k * highpass(s, d0_ratio * min(s), "gaussian"))


# ----------------------------------------------------------------------------
# Periodic noise: automatic notch-reject filtering
# ----------------------------------------------------------------------------
def peak_prominence_map(gray: np.ndarray, r_in: int = 4, r_out: int = 9) -> np.ndarray:
    """Spike score for every frequency: mean log-magnitude of the 3x3 core minus
    the mean over an annulus r_in..r_out bins away (the local 1/f background).
    A pure sinusoid concentrates energy in the core (Hann main lobe ~3 bins)."""
    S = spectrum(gray).astype(np.float32)
    yy, xx = np.mgrid[-r_out:r_out + 1, -r_out:r_out + 1]
    rr = np.hypot(yy, xx)
    ring = ((rr >= r_in) & (rr <= r_out)).astype(np.float32)
    core = (rr <= 1.5).astype(np.float32)
    bg = cv2.filter2D(S, -1, ring / ring.sum(), borderType=cv2.BORDER_REFLECT)
    pk = cv2.filter2D(S, -1, core / core.sum(), borderType=cv2.BORDER_REFLECT)
    return (pk - bg).astype(np.float64)


def detect_periodic_peaks(gray: np.ndarray, thresh: float = 2.0, min_radius_ratio: float = 0.03,
                          max_peaks: int = 8):
    """Find conjugate-symmetric interference spikes in the spectrum.

    Calibrated on Penn-Fudan (experiments/exp2): clean images never exceed a
    prominence of ~2.0 (periodic textures such as fences come closest), while
    amplitude-40 interference scores 1.6-4.3. Returns offsets (du, dv) from the
    spectrum centre, one per conjugate pair.
    """
    Pm = peak_prominence_map(gray)
    P, Q = Pm.shape
    Pm[distance_grid(Pm.shape) < min_radius_ratio * min(P, Q)] = 0  # skip DC / scene energy
    local_max = Pm == cv2.dilate(Pm.astype(np.float32), np.ones((7, 7), np.uint8)).astype(np.float64)
    cand = np.argwhere(local_max & (Pm > thresh))
    cand = sorted(cand.tolist(), key=lambda rc: -Pm[rc[0], rc[1]])
    peaks = []
    cu, cv = P // 2, Q // 2
    for r, c in cand:
        du, dv = r - cu, c - cv
        if any(abs(du + a) <= 2 and abs(dv + b) <= 2 for a, b in peaks) or \
           any(abs(du - a) <= 2 and abs(dv - b) <= 2 for a, b in peaks):
            continue  # conjugate (or duplicate) of a peak we already have
        peaks.append((du, dv))
        if len(peaks) >= max_peaks:
            break
    return peaks


def notch_reject(shape, peaks_rel, d0: float = 4.0, order: int = 4) -> np.ndarray:
    """Butterworth notch-reject product filter. peaks_rel are offsets expressed
    as fractions of the analysed image size so they rescale to any padding."""
    P, Q = shape
    V, U = np.meshgrid(np.arange(Q) - Q // 2, np.arange(P) - P // 2)
    H = np.ones(shape)
    for ru, rv in peaks_rel:
        du, dv = ru * P, rv * Q
        for s in (1, -1):
            Dk = np.sqrt((U - s * du) ** 2 + (V - s * dv) ** 2)
            H *= 1.0 / (1.0 + (d0 / (Dk + 1e-9)) ** (2 * order))
    return H


def remove_periodic_noise(img, peaks=None, d0: float = 4.0):
    """Detect interference peaks (if not given) and suppress them with notches."""
    gray = img if img.ndim == 2 else cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    if peaks is None:
        peaks = detect_periodic_peaks(gray)
    if not peaks:
        return img, []
    P, Q = gray.shape
    rel = [(du / P, dv / Q) for du, dv in peaks]
    return apply_filter(img, lambda s: notch_reject(s, rel, d0)), peaks


# ----------------------------------------------------------------------------
# Homomorphic filtering (illumination-reflectance model f = i * r)
# ----------------------------------------------------------------------------
def homomorphic_filter(img, gamma_l: float = 0.4, gamma_h: float = 1.1, c: float = 1.0,
                       d0_ratio: float = 0.01, brighten: float = 0.3):
    """ln f = ln i + ln r; attenuate low frequencies (illumination, gamma_L<1)
    and keep/boost high frequencies (reflectance/detail, gamma_H>=1).

    Applied to the luminance channel only (colours preserved). Output mapping:
    the log-domain mean of the input is restored plus `brighten` (log units,
    0.3 ~ x1.35) to compensate the attenuated illumination. Parameters tuned
    in experiments/exp2 (SSIM 0.908 -> 0.927 on uneven illumination; a
    min-max/percentile stretch instead *lowered* SSIM).
    """
    if img.ndim == 3:
        ycc = cv2.cvtColor(img, cv2.COLOR_BGR2YCrCb)
        y = ycc[..., 0]
    else:
        y = img
    ph, pw = _pad_amount(y.shape[0]), _pad_amount(y.shape[1])
    z = np.log1p(np.pad(y.astype(np.float64), ((ph, ph), (pw, pw)), mode="reflect"))
    Z = np.fft.fftshift(np.fft.fft2(z))
    D = distance_grid(z.shape)
    d0 = d0_ratio * min(z.shape)
    H = (gamma_h - gamma_l) * (1 - np.exp(-c * D ** 2 / d0 ** 2)) + gamma_l
    s = np.real(np.fft.ifft2(np.fft.ifftshift(H * Z)))[ph:ph + y.shape[0], pw:pw + y.shape[1]]
    s = s - s.mean() + np.log1p(y.astype(np.float64)).mean() + brighten
    out = np.clip(np.rint(np.expm1(s)), 0, 255).astype(np.uint8)
    if img.ndim == 3:
        ycc = ycc.copy()
        ycc[..., 0] = out
        return cv2.cvtColor(ycc, cv2.COLOR_YCrCb2BGR)
    return out
