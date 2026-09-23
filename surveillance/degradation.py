"""Controlled degradation models for the Islamabad surveillance scenario.

Public datasets contain clean(ish) images, so to measure PSNR/SSIM we need a
known ground truth. We therefore take a clean frame f(x,y) and apply the
standard degradation model  g = H[f] + eta  (Gonzalez & Woods, Ch. 5) with
parameters chosen to mimic real CCTV failure modes:

  gaussian      sensor / thermal noise at night             (eta ~ N(0, sigma^2))
  salt_pepper   bit errors on the wireless link             (impulse noise)
  periodic      electrical interference from power lines    (sinusoidal pattern)
  motion_blur   fast vehicles / long exposure at night      (linear motion PSF)
  low_light     under-exposed night frames + shot noise
  uneven_light  street-lamp pools, shadows, headlights      (multiplicative field)
  block_loss    dropped packets in an H.264/MJPEG stream    (missing 16x16 blocks)
"""
from __future__ import annotations

import cv2
import numpy as np


def _clip(x: np.ndarray) -> np.ndarray:
    return np.clip(np.rint(x), 0, 255).astype(np.uint8)


def add_gaussian_noise(img, sigma: float = 20.0, rng=None):
    rng = rng or np.random.default_rng()
    return _clip(img.astype(np.float64) + rng.normal(0.0, sigma, img.shape))


def add_salt_pepper(img, amount: float = 0.05, salt_ratio: float = 0.5, rng=None):
    rng = rng or np.random.default_rng()
    out = img.copy()
    h, w = img.shape[:2]
    r = rng.random((h, w))
    salt = r < amount * salt_ratio
    pepper = (r >= amount * salt_ratio) & (r < amount)
    out[salt] = 255
    out[pepper] = 0
    return out


def add_periodic_noise(img, amplitude: float = 40.0, freqs=((0.12, 0.0), (0.05, 0.09)), rng=None):
    """Additive sinusoidal interference. freqs are (fx, fy) in cycles/pixel."""
    h, w = img.shape[:2]
    y, x = np.mgrid[0:h, 0:w].astype(np.float64)
    pattern = np.zeros((h, w))
    for fx, fy in freqs:
        pattern += np.sin(2 * np.pi * (fx * x + fy * y))
    pattern *= amplitude / len(freqs)
    if img.ndim == 3:
        pattern = pattern[..., None]
    return _clip(img.astype(np.float64) + pattern)


def motion_psf(length: int = 15, angle_deg: float = 0.0, size: int | None = None) -> np.ndarray:
    """Linear motion-blur point-spread function, normalised to sum 1."""
    size = size or (length if length % 2 == 1 else length + 1)
    psf = np.zeros((size, size), np.float64)
    c = size // 2
    theta = np.deg2rad(angle_deg)
    dx, dy = np.cos(theta), -np.sin(theta)
    for t in np.linspace(-(length - 1) / 2, (length - 1) / 2, length * 4):
        x, y = int(round(c + t * dx)), int(round(c + t * dy))
        if 0 <= x < size and 0 <= y < size:
            psf[y, x] = 1.0
    return psf / psf.sum()


def pad_psf(psf: np.ndarray, shape) -> np.ndarray:
    """Zero-pad PSF to `shape` with its centre moved to (0,0) (for FFT use)."""
    out = np.zeros(shape, np.float64)
    ph, pw = psf.shape
    out[:ph, :pw] = psf
    return np.roll(out, (-(ph // 2), -(pw // 2)), axis=(0, 1))


def blur_with_psf(img, psf, pad: int = 32):
    """Blur via FFT with reflect padding (avoids wrap-around seams)."""
    def _one(ch):
        p = np.pad(ch.astype(np.float64), pad, mode="reflect")
        H = np.fft.fft2(pad_psf(psf, p.shape))
        out = np.real(np.fft.ifft2(np.fft.fft2(p) * H))
        return out[pad:-pad, pad:-pad]
    if img.ndim == 2:
        return _clip(_one(img))
    return _clip(np.dstack([_one(img[..., c]) for c in range(img.shape[2])]))


def add_motion_blur(img, length: int = 15, angle: float = 0.0, noise_sigma: float = 2.0, rng=None):
    rng = rng or np.random.default_rng()
    blurred = blur_with_psf(img, motion_psf(length, angle))
    return add_gaussian_noise(blurred, noise_sigma, rng) if noise_sigma > 0 else blurred


def low_light(img, factor: float = 0.3, gamma: float = 1.4, noise_sigma: float = 6.0, rng=None):
    """Night-time exposure: compress intensities, then add sensor noise."""
    rng = rng or np.random.default_rng()
    dark = 255.0 * factor * (img.astype(np.float64) / 255.0) ** gamma
    return add_gaussian_noise(_clip(dark), noise_sigma, rng) if noise_sigma > 0 else _clip(dark)


def uneven_illumination(img, strength: float = 0.7, rng=None):
    """Multiply by a smooth illumination field i(x,y) in [1-strength, 1].

    Models f = i * r (illumination-reflectance), the exact assumption behind
    homomorphic filtering.
    """
    rng = rng or np.random.default_rng()
    h, w = img.shape[:2]
    y, x = np.mgrid[0:h, 0:w].astype(np.float64)
    cx, cy = rng.uniform(0.2, 0.8) * w, rng.uniform(0.2, 0.8) * h
    field = np.exp(-(((x - cx) / (0.6 * w)) ** 2 + ((y - cy) / (0.6 * h)) ** 2))  # street-lamp pool
    field = 0.5 * field + 0.5 * (x / w)  # plus a lateral shadow gradient
    field = (field - field.min()) / (np.ptp(field) + 1e-9)
    field = 1.0 - strength + strength * field
    if img.ndim == 3:
        field = field[..., None]
    return _clip(img.astype(np.float64) * field)


def block_loss(img, ratio: float = 0.03, block: int = 16, rng=None):
    """Zero out random blocks (lost packets). Returns (image, loss_mask)."""
    rng = rng or np.random.default_rng()
    out = img.copy()
    h, w = img.shape[:2]
    mask = np.zeros((h, w), np.uint8)
    by, bx = h // block, w // block
    n = max(1, int(ratio * by * bx))
    idx = rng.choice(by * bx, size=n, replace=False)
    for k in idx:
        r, c = divmod(int(k), bx)
        out[r * block:(r + 1) * block, c * block:(c + 1) * block] = 0
        mask[r * block:(r + 1) * block, c * block:(c + 1) * block] = 1
    return out, mask


# Scenario catalogue used by all experiments (name -> callable(img, rng) -> img)
SCENARIOS = {
    "gaussian_s15": lambda im, rng: add_gaussian_noise(im, 15, rng),
    "gaussian_s30": lambda im, rng: add_gaussian_noise(im, 30, rng),
    "saltpepper_5": lambda im, rng: add_salt_pepper(im, 0.05, rng=rng),
    "saltpepper_20": lambda im, rng: add_salt_pepper(im, 0.20, rng=rng),
    "periodic": lambda im, rng: add_periodic_noise(im, 40, rng=rng),
    "motion_blur": lambda im, rng: add_motion_blur(im, 15, 0, 2.0, rng),
    "low_light": lambda im, rng: low_light(im, rng=rng),
    "uneven_light": lambda im, rng: uneven_illumination(im, 0.7, rng),
    "block_loss": lambda im, rng: block_loss(im, 0.03, rng=rng)[0],
    # Compound "worst night": dark + noisy + a few bit errors
    "night_compound": lambda im, rng: add_salt_pepper(low_light(im, 0.35, 1.3, 10, rng), 0.01, rng=rng),
}
