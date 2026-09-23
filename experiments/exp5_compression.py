"""Experiment 5 - Compression vs image fidelity (Technical Task 5).

(a) Rate-distortion curves (bpp vs PSNR / SSIM): libjpeg, our from-scratch
    DCT codec, and wavelet codecs (Haar, CDF 9/7 = bior4.4).
(b) Task-aware fidelity: pedestrian detection F1 as a function of JPEG quality
    (how far can we compress before the *analytics* break?).
(c) ROI coding vs uniform JPEG at equal file size: quality inside person boxes.
"""
from common import parse_args, pick_indices, save_fig, save_table, plt  # noqa: I001
import numpy as np
import pandas as pd

from surveillance import compression as C
from surveillance.datasets import PennFudan
from surveillance.metrics import Accumulator, Timer, match_detections, psnr, ssim
from surveillance.recognition import YOLOTinyDetector

IGNORE_H = 90


def main():
    a = parse_args(__doc__, default_n=24, quick_n=4)
    ds = PennFudan()
    idx = pick_indices(len(ds), a.n)
    imgs = [ds[i].image for i in idx]

    # (a) rate-distortion
    rows = []
    for im in imgs:
        for q in (5, 10, 20, 30, 50, 70, 90):
            for codec, fn in (("JPEG (libjpeg)", lambda x: C.jpeg_roundtrip(x, q)),
                              ("DCT codec (ours)", lambda x: C.DCTCodec(q).roundtrip(x))):
                with Timer() as t:
                    rec, n = fn(im)
                rows.append({"codec": codec, "param": q, "bpp": C.bpp(n, im.shape), "PSNR": psnr(im, rec),
                             "SSIM": ssim(im, rec), "ratio": im.nbytes / n, "ms": t.ms})
        for wv in ("haar", "bior4.4"):
            for delta in (3, 6, 12, 24, 48, 96):
                with Timer() as t:
                    rec, n = C.WaveletCodec(wv, 3, delta).roundtrip(im)
                rows.append({"codec": f"Wavelet {wv}", "param": delta, "bpp": C.bpp(n, im.shape),
                             "PSNR": psnr(im, rec), "SSIM": ssim(im, rec), "ratio": im.nbytes / n, "ms": t.ms})
    df = pd.DataFrame(rows).groupby(["codec", "param"], sort=False).mean().reset_index()
    save_table(df, "exp5a_rate_distortion")

    fig, axes = plt.subplots(1, 2, figsize=(7.5, 2.8))
    for codec, d in df.groupby("codec", sort=False):
        d = d.sort_values("bpp")
        axes[0].plot(d.bpp, d.PSNR, "o-", ms=3, label=codec)
        axes[1].plot(d.bpp, d.SSIM, "o-", ms=3, label=codec)
    for ax, yl in zip(axes, ("PSNR (dB)", "SSIM")):
        ax.set_xlabel("bits per pixel"); ax.set_ylabel(yl); ax.set_xlim(0, 4)
    axes[1].legend(fontsize=7)
    fig.tight_layout()
    save_fig(fig, "exp5_rate_distortion")

    # BD-style summary: quality at ~0.5 and ~1.0 bpp (linear interpolation)
    summ = []
    for codec, d in df.groupby("codec", sort=False):
        d = d.sort_values("bpp")
        for target in (0.5, 1.0):
            summ.append({"codec": codec, "bpp": target, "PSNR": float(np.interp(target, d.bpp, d.PSNR)),
                         "SSIM": float(np.interp(target, d.bpp, d.SSIM))})
    save_table(pd.DataFrame(summ), "exp5a_quality_at_fixed_bpp")

    # (b) detection vs JPEG quality
    yolo = YOLOTinyDetector(416)
    idx_b = pick_indices(len(ds), max(a.n, 60) if not a.quick else 6)
    rows = []
    for q in (100, 70, 40, 20, 10, 5):
        acc, bpps = Accumulator(), []
        for i in idx_b:
            s = ds[i]
            rec, n = (s.image, s.image.nbytes) if q == 100 else C.jpeg_roundtrip(s.image, q)
            bpps.append(C.bpp(n, s.image.shape) if q != 100 else 24.0)
            dets = [(d.box, d.score) for d in yolo.detect(rec) if d.label == "person"]
            acc.add(match_detections(dets, s.boxes, 0.5, IGNORE_H), len(s.boxes))
        sm = acc.summary()
        rows.append({"JPEG quality": "raw" if q == 100 else q, "bpp": float(np.mean(bpps)),
                     "Precision": sm["Precision"], "Recall": sm["Recall"], "F1": sm["F1"], "AP50": sm["AP50"]})
    save_table(pd.DataFrame(rows), "exp5b_detection_vs_jpeg")

    # (c) ROI coding vs uniform JPEG at (no more than) the same size
    rows = []
    for i in idx:
        s = ds[i]
        im = s.image
        roi_rec, roi_bytes = C.roi_jpeg(im, s.boxes, q_roi=90, q_bg=15)
        best_q, uni_rec, uni_bytes = 1, None, 0
        for q in range(5, 96):
            r, n = C.jpeg_roundtrip(im, q)
            if n > roi_bytes:
                break
            best_q, uni_rec, uni_bytes = q, r, n
        m = C.roi_mask_from_boxes(im.shape, s.boxes, 0)
        for name, rec, n in (("ROI (q90 people / q15 bg)", roi_rec, roi_bytes),
                             ("uniform JPEG (same size)", uni_rec, uni_bytes)):
            rows.append({"scheme": name, "bpp": C.bpp(n, im.shape), "PSNR_people": psnr(im[m], rec[m]),
                         "PSNR_background": psnr(im[~m], rec[~m]), "PSNR_frame": psnr(im, rec),
                         "uniform_q": best_q})
    save_table(pd.DataFrame(rows).groupby("scheme", sort=False).mean().reset_index(), "exp5c_roi_coding")


if __name__ == "__main__":
    main()
