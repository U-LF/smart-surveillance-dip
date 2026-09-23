"""Integrated, adaptive surveillance pipeline.

    frame -> [0 resize] -> [1 diagnose] -> [2 restore: reconstruct / impulse /
    periodic / Gaussian noise / illumination] -> [3 detect] -> [4 segment]
    -> [5 protect (privacy)] -> [6 compress (ROI-aware)] -> result + timings

Key idea (addresses 'Real-time vs quality' and 'Resource constraints vs
performance'): the pipeline does NOT run every filter on every frame. Blind
diagnostics decide which restoration is needed, and a *profile* decides how
expensive each chosen operation may be:

  fast     : 0.5x resolution, cheap filters, detector every 3rd frame, box masks
  balanced : 0.75x, edge-preserving filters, notch filter, detector every frame
  quality  : full resolution, adaptive median / NLM / homomorphic, YOLO@608
Segmentation defaults to marker-controlled watershed: exp4 found it both the most
accurate (IoU 0.69 vs 0.64 for GrabCut) and ~100x faster (13 ms vs 1.4 s).

Module interdependence is explicit: restoration output feeds detection, whose
boxes drive segmentation, privacy masking and ROI compression.
"""
from __future__ import annotations

from dataclasses import dataclass, field, replace

import cv2
import numpy as np

from . import compression as C
from . import frequency as F
from . import privacy as PV
from . import restoration as R
from . import segmentation as SG
from . import spatial as SP
from .metrics import Timer
from .recognition import Detection, make_detector


@dataclass
class Profile:
    name: str
    scale: float
    impulse: str            # median3 | median5 | adaptive_median
    denoise: str            # gaussian | bilateral | nlm
    periodic: bool
    homomorphic: bool
    lowlight: str           # gamma | clahe | gamma+clahe
    sharpen: bool
    detector_size: int
    detect_every: int
    segmentation: str       # box | otsu | grabcut | ...
    q_roi: int = 90
    q_bg: int = 30


PROFILES = {
    "fast": Profile("fast", 0.5, "median3", "gaussian", False, False, "gamma", False, 320, 3, "box", 85, 25),
    "balanced": Profile("balanced", 0.75, "median3", "bilateral", True, False, "gamma+clahe", False, 416, 1,
                        "watershed", 90, 30),
    "quality": Profile("quality", 1.0, "adaptive_median", "nlm", True, True, "gamma+clahe", True, 608, 1,
                       "watershed", 92, 40),
}

# Decision thresholds (calibrated in experiments/exp0_diagnostics.py)
T_IMPULSE = 0.004      # isolated 0/255 pixels: clean <= 0.001, 5% S&P >= 0.033
T_NOISE = 7.5          # edge-aware sigma: clean 0.6-8.1 (95th pct 5.8), sigma=15 -> >= 9.4
T_DARK = 50.0          # mean luminance: clean >= 62, simulated night 12-34
T_DARK_REGIONS = 0.30  # fraction of 8x8 tiles darker than 40 (shadows / lamp pools)
T_LOST = 0.002         # pixels in all-black 16x16 blocks: clean 0, 3% loss ~ 0.028


@dataclass
class PipelineResult:
    enhanced: np.ndarray
    detections: list
    mask: np.ndarray | None
    protected: np.ndarray
    compressed_bytes: int
    diagnostics: dict
    actions: list
    timings: dict = field(default_factory=dict)

    @property
    def total_ms(self):
        return sum(self.timings.values())


class SurveillancePipeline:
    def __init__(self, profile: str | Profile = "balanced", detector: str = "yolo",
                 privacy: str = "head", scramble_key: bytes | None = None,
                 illumination: str = "auto", **overrides):
        """illumination: 'auto' (dark-region rule), 'always' (camera known to
        face lamp pools / shadows - set per camera at installation) or 'off'.
        Exp0 shows uneven lighting cannot be told apart from natural shadows
        by frame statistics alone, hence the per-camera switch."""
        p = PROFILES[profile] if isinstance(profile, str) else profile
        self.profile = replace(p, **overrides)  # always a copy: never mutate PROFILES
        det_kw = {"input_size": self.profile.detector_size} if detector.startswith("yolo") else {}
        self.detector = make_detector(detector, **det_kw) if detector != "none" else None
        self.privacy = privacy
        self.illumination = illumination
        self.scrambler = PV.KeyedScrambler(scramble_key) if scramble_key else None
        self._last_dets: list = []
        self._frame_idx = 0

    # ------------------------------------------------------------------ restore
    def restore(self, img, diag: R.FrameDiagnostics | None = None):
        """Adaptive restoration; returns (image, actions, diagnostics)."""
        p = self.profile
        diag = diag or R.diagnose(img)
        actions, out = [], img

        if diag.lost_block_ratio > T_LOST:
            out = R.reconstruct_lost_blocks(out)
            actions.append("inpaint")

        if diag.impulse_ratio > T_IMPULSE:
            if p.impulse == "adaptive_median" or diag.impulse_ratio > 0.1:
                out = SP.adaptive_median_filter(out, 7)
            else:
                out = SP.median_filter(out, 3 if p.impulse == "median3" else 5)
            actions.append(f"impulse:{p.impulse}")

        if p.periodic:
            out2, peaks = F.remove_periodic_noise(out)
            if peaks:
                out = out2
                actions.append(f"notch:{len(peaks)}")

        sigma = R.estimate_noise_sigma(out)  # re-estimate after impulse / notch removal
        luma0 = float(cv2.cvtColor(out, cv2.COLOR_BGR2GRAY).mean()) if out.ndim == 3 else float(out.mean())
        # Noise that is harmless now becomes visible after low-light enhancement
        # multiplies intensities by ~gain, so judge the noise *after* that gain
        # (exp3b: denoise->enhance beats enhance->denoise by ~2 dB).
        gain = float(np.clip(110.0 / max(luma0, 1.0), 1.0, 4.0)) if luma0 < T_DARK else 1.0
        if sigma * gain > T_NOISE:
            if p.denoise == "nlm":
                out = SP.nlm_filter(out, sigma)
            elif p.denoise == "bilateral":
                out = SP.bilateral_filter(out, sigma)
            else:
                out = SP.gaussian_filter(out, max(0.8, sigma / 15.0))
            actions.append(f"denoise:{p.denoise}(s={sigma:.1f},gain={gain:.1f})")
            if p.sharpen:
                out = SP.unsharp_mask(out, 1.0, 0.35)
                actions.append("unsharp")

        luma = float(cv2.cvtColor(out, cv2.COLOR_BGR2GRAY).mean()) if out.ndim == 3 else float(out.mean())
        if luma < T_DARK:
            out = R.enhance_low_light(out, p.lowlight)
            actions.append(f"lowlight:{p.lowlight}")
        elif self.illumination == "always" or (self.illumination == "auto"
                                               and diag.dark_regions > T_DARK_REGIONS):
            out = F.homomorphic_filter(out) if p.homomorphic else R.clahe(out, 2.0)
            actions.append("homomorphic" if p.homomorphic else "clahe")
        return out, actions, diag

    # ------------------------------------------------------------------ full
    def process(self, frame, motion_mask=None) -> PipelineResult:
        p, t = self.profile, {}
        # Diagnose + repair *pixel-level* defects at native resolution: area
        # resampling would average impulses / lost blocks into blobs that no
        # filter can identify afterwards (order matters - see report Sec. IV).
        with Timer() as tm:
            diag = R.diagnose(frame)
            pre_actions, work = [], frame
            if diag.lost_block_ratio > T_LOST:
                work = R.reconstruct_lost_blocks(work)
                pre_actions.append("inpaint")
            if diag.impulse_ratio > T_IMPULSE:
                work = SP.adaptive_median_filter(work, 7) if (p.impulse == "adaptive_median"
                                                              or diag.impulse_ratio > 0.1) \
                    else SP.median_filter(work, 3)
                pre_actions.append(f"impulse:{p.impulse}")
        t["diagnose+prefix"] = tm.ms
        with Timer() as tm:
            if p.scale != 1.0:
                work = cv2.resize(work, None, fx=p.scale, fy=p.scale, interpolation=cv2.INTER_AREA)
        t["resize"] = tm.ms
        with Timer() as tm:
            clean_diag = replace(diag, impulse_ratio=0.0, lost_block_ratio=0.0)
            enhanced, actions, _ = self.restore(work, clean_diag)
            actions = pre_actions + actions
        t["restore"] = tm.ms

        with Timer() as tm:
            run_det = self.detector is not None and (self._frame_idx % p.detect_every == 0)
            if run_det:
                self._last_dets = self.detector.detect(enhanced)
            dets = self._last_dets
        t["detect"] = tm.ms

        with Timer() as tm:
            persons = [d.box for d in dets if d.label == "person"]
            if motion_mask is not None:
                mask = motion_mask
            elif persons:
                mask = SG.segment_people(enhanced, persons, p.segmentation)
            else:
                mask = None
        t["segment"] = tm.ms

        with Timer() as tm:
            if self.privacy == "scramble" and self.scrambler:
                protected = self.scrambler.scramble(enhanced, [PV.head_region(b) for b in persons],
                                                    self._frame_idx)
            elif self.privacy == "none":
                protected = enhanced
            else:
                protected = PV.anonymize(enhanced, dets, region=self.privacy, mask=mask)
        t["protect"] = tm.ms

        with Timer() as tm:
            _, nbytes = C.roi_jpeg(protected, [d.box for d in dets], p.q_roi, p.q_bg)
        t["compress"] = tm.ms

        # boxes back to original resolution
        if p.scale != 1.0:
            s = 1.0 / p.scale
            dets = [Detection(tuple(v * s for v in d.box), d.score, d.label) for d in dets]
        self._frame_idx += 1
        return PipelineResult(enhanced, dets, mask, protected, nbytes, diag.as_dict(), actions, t)


def overlay(frame, result: PipelineResult, fps: float | None = None):
    """Visualisation used by demo.py (boxes + mask tint + HUD)."""
    from .recognition import draw_detections
    vis = result.protected.copy()
    if result.mask is not None and result.mask.shape == vis.shape[:2]:
        tint = vis.copy()
        tint[result.mask] = (0.5 * tint[result.mask] + [0, 0, 127]).astype(np.uint8)
        vis = tint
    s = vis.shape[1] / frame.shape[1]
    dets = [Detection(tuple(v * s for v in d.box), d.score, d.label) for d in result.detections]
    vis = draw_detections(vis, dets)
    hud = f"{result.total_ms:.0f} ms" + (f" | {fps:.1f} FPS" if fps else "") + \
          f" | {', '.join(result.actions) or 'clean'}"
    cv2.putText(vis, hud, (6, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 2, cv2.LINE_AA)
    cv2.putText(vis, hud, (6, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 0), 1, cv2.LINE_AA)
    return vis
