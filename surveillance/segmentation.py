"""Technical Task 4 - Image segmentation (edge-based and region-based) + morphology.

Single images (Penn-Fudan, evaluated with IoU/Dice against pixel masks):
  Edge-based   : Canny (auto thresholds) / Sobel  -> morphological closing
                 -> hole filling -> largest connected component
  Region-based : Otsu thresholding (from scratch), seeded region growing,
                 marker-controlled watershed, GrabCut (graph-cut on colour GMMs)
All single-image methods run inside a (detected or given) person ROI, which is
how surveillance systems use segmentation: the detector says *where*, the
segmenter says *which pixels* (for privacy masking, silhouettes, counting).

Video (static CCTV camera):
  MotionSegmenter : background subtraction (MOG2 or a from-scratch running
                    average) + morphological opening/closing + connected
                    components -> moving-object masks and boxes.
"""
from __future__ import annotations

import cv2
import numpy as np
from scipy import ndimage as ndi


def _gray(img):
    return img if img.ndim == 2 else cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)


def _largest_component(mask: np.ndarray) -> np.ndarray:
    n, lab, stats, _ = cv2.connectedComponentsWithStats(mask.astype(np.uint8), 8)
    if n <= 1:
        return mask.astype(bool)
    k = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    return lab == k


def _roi(img_shape, box, pad_ratio: float = 0.08):
    h, w = img_shape[:2]
    x1, y1, x2, y2 = box
    px, py = pad_ratio * (x2 - x1), pad_ratio * (y2 - y1)
    X1, Y1 = int(max(0, x1 - px)), int(max(0, y1 - py))
    X2, Y2 = int(min(w, x2 + px)), int(min(h, y2 + py))
    return X1, Y1, X2, Y2


# ----------------------------------------------------------------------------
# Otsu (from scratch)
# ----------------------------------------------------------------------------
def otsu_threshold(gray: np.ndarray) -> int:
    """Maximise between-class variance sigma_B^2(k) = [mu_T w(k) - mu(k)]^2 / [w(k)(1-w(k))]."""
    hist = np.bincount(gray.ravel(), minlength=256).astype(np.float64)
    p = hist / hist.sum()
    omega = np.cumsum(p)
    mu = np.cumsum(p * np.arange(256))
    mu_t = mu[-1]
    with np.errstate(divide="ignore", invalid="ignore"):
        sb = (mu_t * omega - mu) ** 2 / (omega * (1 - omega))
    sb = np.nan_to_num(sb)
    return int(np.argmax(sb))


# ----------------------------------------------------------------------------
# ROI segmenters: each returns a boolean mask the size of the ROI crop
# ----------------------------------------------------------------------------
def seg_box(crop, inner):
    m = np.zeros(crop.shape[:2], bool)
    x1, y1, x2, y2 = inner
    m[y1:y2, x1:x2] = True
    return m


def seg_edges(crop, inner, detector: str = "canny"):
    """Edge-based: edges -> close gaps -> fill enclosed regions -> biggest blob."""
    g = cv2.GaussianBlur(_gray(crop), (5, 5), 1.2)
    if detector == "canny":
        v = np.median(g)
        edges = cv2.Canny(g, int(max(0, 0.66 * v)), int(min(255, 1.33 * v)))
    else:  # sobel + Otsu on the gradient magnitude
        gx = cv2.Sobel(g, cv2.CV_64F, 1, 0)
        gy = cv2.Sobel(g, cv2.CV_64F, 0, 1)
        mag = cv2.normalize(np.hypot(gx, gy), None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        edges = (mag > otsu_threshold(mag)).astype(np.uint8) * 255
    x1, y1, x2, y2 = inner
    box_mask = np.zeros(edges.shape, np.uint8)
    box_mask[y1:y2, x1:x2] = 1
    edges = edges * box_mask
    k = max(3, int(0.06 * (x2 - x1)) | 1)
    closed = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k)))
    filled = ndi.binary_fill_holes(closed > 0)
    opened = cv2.morphologyEx(filled.astype(np.uint8), cv2.MORPH_OPEN,
                              cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)))
    return _largest_component(opened) if opened.any() else seg_box(crop, inner)


def _border_background_stats(crop_lab, inner, margin=3):
    x1, y1, x2, y2 = inner
    ring = np.ones(crop_lab.shape[:2], bool)
    ring[y1 + margin:y2 - margin, x1 + margin:x2 - margin] = False
    return crop_lab[ring].reshape(-1, 3).mean(0)


def seg_otsu(crop, inner):
    """Region-based thresholding: Otsu on the distance-to-background colour map.

    Plain grey-level Otsu has unknown polarity (dark coat on bright street vs.
    white shirt on dark wall); measuring colour distance to the ROI border
    (assumed background) makes 'foreground = far from background' consistent.
    """
    lab = cv2.cvtColor(crop, cv2.COLOR_BGR2LAB).astype(np.float64)
    bg = _border_background_stats(lab, inner)
    dist = np.linalg.norm(lab - bg, axis=2)
    dist = cv2.normalize(dist, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    dist = cv2.GaussianBlur(dist, (5, 5), 0)
    m = dist > otsu_threshold(dist)
    m &= seg_box(crop, inner)
    m = cv2.morphologyEx(m.astype(np.uint8), cv2.MORPH_CLOSE, np.ones((7, 7), np.uint8))
    m = ndi.binary_fill_holes(m)
    return _largest_component(m) if m.any() else seg_box(crop, inner)


def region_grow(img_lab: np.ndarray, seed_mask: np.ndarray, tol: float = 18.0,
                allowed=None, max_iter: int = 500) -> np.ndarray:
    """Seeded region growing (vectorised): repeatedly add 8-neighbours whose
    colour is within `tol` (CIELAB, ~perceptual) of the current region mean."""
    region = seed_mask.astype(bool).copy()
    kernel = np.ones((3, 3), np.uint8)
    for _ in range(max_iter):
        mean = img_lab[region].mean(0)
        ring = cv2.dilate(region.astype(np.uint8), kernel).astype(bool) & ~region
        if allowed is not None:
            ring &= allowed
        close = ring & (np.linalg.norm(img_lab - mean, axis=2) < tol)
        if not close.any():
            break
        region |= close
    return region


def seg_region_growing(crop, inner, tol: float = 22.0):
    """Seeds on the torso/legs axis; growth constrained to the box."""
    lab = cv2.cvtColor(cv2.GaussianBlur(crop, (5, 5), 0), cv2.COLOR_BGR2LAB).astype(np.float64)
    x1, y1, x2, y2 = inner
    cx, bw, bh = (x1 + x2) // 2, x2 - x1, y2 - y1
    allowed = seg_box(crop, inner)
    out = np.zeros(crop.shape[:2], bool)
    for fy in (0.15, 0.35, 0.55, 0.8):  # head, torso, hips, legs
        seed = np.zeros_like(out)
        sy = y1 + int(fy * bh)
        seed[max(0, sy - 3):sy + 3, max(0, cx - max(2, bw // 12)):cx + max(2, bw // 12)] = True
        out |= region_grow(lab, seed, tol, allowed)
    out = cv2.morphologyEx(out.astype(np.uint8), cv2.MORPH_CLOSE, np.ones((9, 9), np.uint8))
    return ndi.binary_fill_holes(out)


def seg_watershed(crop, inner):
    """Marker-controlled watershed on the colour image.
    Markers: sure-background = ROI border strip, sure-foreground = narrow
    vertical core of the box (a standing pedestrian's torso/legs axis)."""
    x1, y1, x2, y2 = inner
    markers = np.zeros(crop.shape[:2], np.int32)
    markers[:, :2] = markers[:, -2:] = 1
    markers[:2, :] = markers[-2:, :] = 1
    outside = ~seg_box(crop, inner)
    markers[outside] = 1
    bw, bh = x2 - x1, y2 - y1
    cx = (x1 + x2) // 2
    core_w = max(2, int(0.12 * bw))
    markers[y1 + int(0.1 * bh):y1 + int(0.9 * bh), cx - core_w:cx + core_w] = 2
    cv2.watershed(crop.copy(), markers)
    m = markers == 2
    m = cv2.morphologyEx(m.astype(np.uint8), cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
    return ndi.binary_fill_holes(m)


def seg_grabcut(crop, inner, iters: int = 4):
    """GrabCut (Rother et al. 2004): iterated graph cuts with colour GMMs,
    initialised from the rectangle -> region-based, the most accurate option."""
    x1, y1, x2, y2 = inner
    mask = np.zeros(crop.shape[:2], np.uint8)
    rect = (int(x1), int(y1), int(max(1, x2 - x1)), int(max(1, y2 - y1)))
    bgd, fgd = np.zeros((1, 65), np.float64), np.zeros((1, 65), np.float64)
    try:
        cv2.grabCut(crop, mask, rect, bgd, fgd, iters, cv2.GC_INIT_WITH_RECT)
    except cv2.error:
        return seg_box(crop, inner)
    m = (mask == cv2.GC_FGD) | (mask == cv2.GC_PR_FGD)
    m = cv2.morphologyEx(m.astype(np.uint8), cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    return _largest_component(m) if m.any() else seg_box(crop, inner)


ROI_METHODS = {
    "box": seg_box,
    "edge_canny": lambda c, i: seg_edges(c, i, "canny"),
    "edge_sobel": lambda c, i: seg_edges(c, i, "sobel"),
    "otsu": seg_otsu,
    "region_growing": seg_region_growing,
    "watershed": seg_watershed,
    "grabcut": seg_grabcut,
}
METHOD_FAMILY = {"box": "baseline", "edge_canny": "edge", "edge_sobel": "edge", "otsu": "region",
                 "region_growing": "region", "watershed": "region", "grabcut": "region"}


def segment_people(img, boxes, method: str = "grabcut", pad_ratio: float = 0.08) -> np.ndarray:
    """Union mask of all persons: run the ROI method inside each box."""
    fn = ROI_METHODS[method]
    full = np.zeros(img.shape[:2], bool)
    for b in boxes:
        X1, Y1, X2, Y2 = _roi(img.shape, b, pad_ratio)
        if X2 - X1 < 8 or Y2 - Y1 < 8:
            continue
        crop = img[Y1:Y2, X1:X2]
        inner = (int(max(0, b[0] - X1)), int(max(0, b[1] - Y1)),
                 int(min(X2 - X1, b[2] - X1)), int(min(Y2 - Y1, b[3] - Y1)))
        full[Y1:Y2, X1:X2] |= fn(crop, inner)
    return full


# ----------------------------------------------------------------------------
# Motion segmentation for static CCTV cameras
# ----------------------------------------------------------------------------
class RunningAverageBackground:
    """From-scratch background model: B_t = (1-a) B_{t-1} + a I_t, only
    updated where the pixel is currently background (selective update)."""

    def __init__(self, alpha: float = 0.02, thresh: float = 25.0):
        self.alpha, self.thresh, self.bg = alpha, thresh, None

    def apply(self, frame):
        g = cv2.GaussianBlur(_gray(frame), (5, 5), 0).astype(np.float32)
        if self.bg is None:
            self.bg = g.copy()
            return np.zeros(g.shape, np.uint8)
        fg = (np.abs(g - self.bg) > self.thresh)
        upd = ~fg
        self.bg[upd] = (1 - self.alpha) * self.bg[upd] + self.alpha * g[upd]
        return fg.astype(np.uint8) * 255


class MotionSegmenter:
    def __init__(self, method: str = "mog2", min_area_ratio: float = 0.0015, history: int = 300):
        if method == "mog2":
            self.model = cv2.createBackgroundSubtractorMOG2(history=history, varThreshold=25,
                                                            detectShadows=True)
        elif method == "knn":
            self.model = cv2.createBackgroundSubtractorKNN(history=history, detectShadows=True)
        else:
            self.model = RunningAverageBackground()
        self.min_area_ratio = min_area_ratio
        self.k_open = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        self.k_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))

    def apply(self, frame):
        raw = self.model.apply(frame)
        fg = (raw == 255).astype(np.uint8)  # MOG2/KNN mark shadows as 127 -> drop them
        fg = cv2.morphologyEx(fg, cv2.MORPH_OPEN, self.k_open)
        fg = cv2.morphologyEx(fg, cv2.MORPH_CLOSE, self.k_close)
        n, lab, stats, _ = cv2.connectedComponentsWithStats(fg, 8)
        min_area = self.min_area_ratio * fg.size
        boxes = []
        for i in range(1, n):
            x, y, w, h, a = stats[i]
            if a >= min_area:
                boxes.append((x, y, x + w, y + h))
        return fg.astype(bool), boxes
