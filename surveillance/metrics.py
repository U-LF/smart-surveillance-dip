"""Evaluation metrics required by the project brief.

* Image quality      : MSE, PSNR, SSIM (Wang et al., 2004 - Gaussian window)
* Segmentation       : IoU (Jaccard), Dice coefficient
* Recognition        : box IoU, greedy matching, Precision / Recall / F1, AP@0.5
* Real-time          : a tiny Timer helper

All implemented from first principles (NumPy/OpenCV primitives only) and
cross-checked against scikit-image in tests/test_core.py.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field

import cv2
import numpy as np


# ----------------------------------------------------------------------------
# Image quality
# ----------------------------------------------------------------------------
def mse(ref: np.ndarray, test: np.ndarray) -> float:
    ref = ref.astype(np.float64)
    test = test.astype(np.float64)
    return float(np.mean((ref - test) ** 2))


def psnr(ref: np.ndarray, test: np.ndarray, data_range: float = 255.0) -> float:
    """Peak signal-to-noise ratio in dB. Returns inf for identical images."""
    m = mse(ref, test)
    if m == 0:
        return float("inf")
    return float(10.0 * np.log10((data_range ** 2) / m))


def _ssim_single(x: np.ndarray, y: np.ndarray, data_range: float) -> float:
    k1, k2 = 0.01, 0.03
    c1, c2 = (k1 * data_range) ** 2, (k2 * data_range) ** 2
    x = x.astype(np.float64)
    y = y.astype(np.float64)
    blur = lambda z: cv2.GaussianBlur(z, (11, 11), 1.5, borderType=cv2.BORDER_REFLECT)
    mu_x, mu_y = blur(x), blur(y)
    sxx = blur(x * x) - mu_x ** 2
    syy = blur(y * y) - mu_y ** 2
    sxy = blur(x * y) - mu_x * mu_y
    num = (2 * mu_x * mu_y + c1) * (2 * sxy + c2)
    den = (mu_x ** 2 + mu_y ** 2 + c1) * (sxx + syy + c2)
    smap = num / den
    pad = 5  # ignore border like skimage does (crop (win-1)/2)
    return float(smap[pad:-pad, pad:-pad].mean())


def ssim(ref: np.ndarray, test: np.ndarray, data_range: float = 255.0) -> float:
    """Structural similarity. Colour images -> mean of per-channel SSIM."""
    if ref.ndim == 2:
        return _ssim_single(ref, test, data_range)
    return float(np.mean([_ssim_single(ref[..., c], test[..., c], data_range)
                          for c in range(ref.shape[2])]))


# ----------------------------------------------------------------------------
# Segmentation
# ----------------------------------------------------------------------------
def iou_mask(pred: np.ndarray, gt: np.ndarray) -> float:
    pred, gt = pred.astype(bool), gt.astype(bool)
    union = np.logical_or(pred, gt).sum()
    if union == 0:
        return 1.0
    return float(np.logical_and(pred, gt).sum() / union)


def dice_mask(pred: np.ndarray, gt: np.ndarray) -> float:
    pred, gt = pred.astype(bool), gt.astype(bool)
    denom = pred.sum() + gt.sum()
    if denom == 0:
        return 1.0
    return float(2.0 * np.logical_and(pred, gt).sum() / denom)


# ----------------------------------------------------------------------------
# Detection
# ----------------------------------------------------------------------------
def box_iou(a, b) -> float:
    """IoU of two boxes in (x1, y1, x2, y2) format."""
    ix1, iy1 = max(a[0], b[0]), max(a[1], b[1])
    ix2, iy2 = min(a[2], b[2]), min(a[3], b[3])
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    inter = iw * ih
    ua = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return float(inter / ua) if ua > 0 else 0.0


@dataclass
class MatchResult:
    tp: int = 0
    fp: int = 0
    fn: int = 0
    ignored: int = 0
    scored: list = field(default_factory=list)  # (score, is_tp) for AP


def match_detections(dets, gts, iou_thr: float = 0.5, ignore_min_height: float = 0.0) -> MatchResult:
    """Greedy (score-sorted) one-to-one matching, PASCAL-VOC style.

    dets : list of (box_xyxy, score)
    gts  : list of box_xyxy
    ignore_min_height : unmatched detections shorter than this are *ignored*
        rather than counted as FP. Penn-Fudan explicitly leaves small /
        heavily occluded people unlabelled, so tiny correct detections would
        otherwise be punished. Set to 0 to disable.
    """
    res = MatchResult()
    used = np.zeros(len(gts), dtype=bool)
    for box, score in sorted(dets, key=lambda d: -d[1]):
        best, best_j = 0.0, -1
        for j, g in enumerate(gts):
            if used[j]:
                continue
            o = box_iou(box, g)
            if o > best:
                best, best_j = o, j
        if best >= iou_thr:
            used[best_j] = True
            res.tp += 1
            res.scored.append((score, 1))
        elif ignore_min_height and (box[3] - box[1]) < ignore_min_height:
            res.ignored += 1
        else:
            res.fp += 1
            res.scored.append((score, 0))
    res.fn = int((~used).sum())
    return res


def prf(tp: int, fp: int, fn: int):
    p = tp / (tp + fp) if tp + fp else 0.0
    r = tp / (tp + fn) if tp + fn else 0.0
    f = 2 * p * r / (p + r) if p + r else 0.0
    return p, r, f


def average_precision(scored, n_gt: int) -> float:
    """All-point interpolated AP (VOC 2010+) from (score, is_tp) pairs."""
    if n_gt == 0 or not scored:
        return 0.0
    scored = sorted(scored, key=lambda s: -s[0])
    tps = np.cumsum([s[1] for s in scored])
    fps = np.cumsum([1 - s[1] for s in scored])
    rec = tps / n_gt
    prec = tps / np.maximum(tps + fps, 1e-9)
    mrec = np.concatenate([[0.0], rec, [1.0]])
    mpre = np.concatenate([[0.0], prec, [0.0]])
    for i in range(len(mpre) - 2, -1, -1):
        mpre[i] = max(mpre[i], mpre[i + 1])
    idx = np.where(mrec[1:] != mrec[:-1])[0]
    return float(np.sum((mrec[idx + 1] - mrec[idx]) * mpre[idx + 1]))


class Accumulator:
    """Accumulates detection matches over a dataset."""

    def __init__(self):
        self.tp = self.fp = self.fn = self.ignored = self.n_gt = 0
        self.scored = []

    def add(self, m: MatchResult, n_gt: int):
        self.tp += m.tp
        self.fp += m.fp
        self.fn += m.fn
        self.ignored += m.ignored
        self.n_gt += n_gt
        self.scored.extend(m.scored)

    def summary(self) -> dict:
        p, r, f = prf(self.tp, self.fp, self.fn)
        return {"TP": self.tp, "FP": self.fp, "FN": self.fn, "Precision": p, "Recall": r,
                "F1": f, "AP50": average_precision(self.scored, self.n_gt)}


# ----------------------------------------------------------------------------
# Timing
# ----------------------------------------------------------------------------
class Timer:
    """Context manager: `with Timer() as t: ...; t.ms`"""

    def __enter__(self):
        self.t0 = time.perf_counter()
        return self

    def __exit__(self, *exc):
        self.ms = (time.perf_counter() - self.t0) * 1000.0
