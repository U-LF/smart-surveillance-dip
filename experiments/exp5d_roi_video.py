"""Experiment 5d - ROI coding on a wide-angle CCTV view (people are small).

Complements exp5c: on Penn-Fudan the pedestrians fill ~25% of the frame, so ROI
coding has little room to help. Here YOLO boxes on vtest.avi define the ROI and
we compare against uniform JPEG at the same (or smaller) file size.
"""
from common import parse_args, save_table  # noqa: I001
import numpy as np
import pandas as pd

from surveillance import compression as C
from surveillance.config import VIDEO_DIR
from surveillance.datasets import iter_video
from surveillance.metrics import psnr
from surveillance.recognition import YOLOTinyDetector


def main():
    a = parse_args(__doc__, default_n=30, quick_n=5)
    det = YOLOTinyDetector(416)
    rows = []
    frames = [f for i, f in iter_video(VIDEO_DIR / "vtest.avi") if i % 20 == 0][: a.n]
    for f in frames:
        boxes = [d.box for d in det.detect(f)]
        if not boxes:
            continue
        m = C.roi_mask_from_boxes(f.shape, boxes, 0)
        for q_bg in (15, 30):
            roi, nb = C.roi_jpeg(f, boxes, 90, q_bg)
            q_u, uni, nu = 5, None, 0
            for q in range(5, 96):
                r, n = C.jpeg_roundtrip(f, q)
                if n > nb:
                    break
                q_u, uni, nu = q, r, n
            for name, rec, n in ((f"ROI q90/q{q_bg}", roi, nb), (f"uniform (<= size of q90/q{q_bg})", uni, nu)):
                rows.append({"scheme": name, "roi_area_%": 100 * m.mean(), "bpp": C.bpp(n, f.shape),
                             "PSNR_objects": psnr(f[m], rec[m]), "PSNR_background": psnr(f[~m], rec[~m]),
                             "uniform_q": q_u})
    save_table(pd.DataFrame(rows).groupby("scheme", sort=False).mean().reset_index(), "exp5d_roi_coding_cctv")


if __name__ == "__main__":
    main()
