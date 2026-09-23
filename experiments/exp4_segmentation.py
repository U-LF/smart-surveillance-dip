"""Experiment 4 - Segmentation (Technical Task 4), evaluated with IoU and Dice.

(a) Edge-based vs region-based person segmentation on Penn-Fudan pixel masks,
    with oracle (ground-truth) boxes and with YOLO boxes (end-to-end system).
(b) Interdependence: segmentation on noisy frames with / without restoration.
(c) Motion segmentation (background subtraction + morphology) on video:
    MOG2 vs from-scratch running average; box-level P/R on CAVIAR if present.
"""
from common import image_grid, parse_args, pick_indices, rng, save_table  # noqa: I001
import cv2
import numpy as np
import pandas as pd

from surveillance import degradation as D
from surveillance import segmentation as SG
from surveillance.config import FIG_DIR, VIDEO_DIR
from surveillance.datasets import CaviarSequence, PennFudan, available_caviar, iter_video
from surveillance.metrics import Accumulator, Timer, dice_mask, iou_mask, match_detections
from surveillance.pipeline import SurveillancePipeline
from surveillance.recognition import YOLOTinyDetector


def seg_eval(samples, boxes_fn, methods, label):
    rows = []
    for s, img in samples:
        boxes = boxes_fn(s, img)
        for m in methods:
            with Timer() as t:
                pred = SG.segment_people(img, boxes, m)
            rows.append({"boxes": label, "method": m, "family": SG.METHOD_FAMILY[m],
                         "IoU": iou_mask(pred, s.mask), "Dice": dice_mask(pred, s.mask), "ms": t.ms})
    return pd.DataFrame(rows)


def main():
    a = parse_args(__doc__, default_n=100, quick_n=8)
    ds = PennFudan()
    idx = pick_indices(len(ds), a.n)
    samples = [(ds[i], ds[i].image) for i in idx]
    yolo = YOLOTinyDetector(416)
    det_cache = {}

    def yolo_boxes(s, img):
        key = (s.name, img.shape, int(img[::7, ::7].sum()))
        if key not in det_cache:
            det_cache[key] = [d.box for d in yolo.detect(img) if d.label == "person"]
        return det_cache[key]

    methods = list(SG.ROI_METHODS)
    df = pd.concat([seg_eval(samples, lambda s, im: s.boxes, methods, "ground-truth boxes"),
                    seg_eval(samples, yolo_boxes, methods, "YOLO boxes")])
    summary = df.groupby(["boxes", "method", "family"], sort=False).agg(
        IoU=("IoU", "mean"), IoU_std=("IoU", "std"), Dice=("Dice", "mean"), ms=("ms", "mean")).reset_index()
    save_table(summary, "exp4a_segmentation")

    s, img = samples[len(samples) // 2]
    ims, titles = [img, (s.mask * 255).astype(np.uint8)], ["image", "ground truth"]
    for m in ("box", "edge_canny", "otsu", "region_growing", "watershed", "grabcut"):
        pred = SG.segment_people(img, s.boxes, m)
        ims.append((pred * 255).astype(np.uint8))
        titles.append(f"{m}\nIoU={iou_mask(pred, s.mask):.2f}")
    image_grid(ims, titles, 4, "exp4_segmentation_examples")

    # (b) noisy frames, with and without the adaptive restoration stage
    g = rng(4)
    pipe = SurveillancePipeline("balanced", detector="none")
    pipe_q = SurveillancePipeline("quality", detector="none")
    sub = samples[:: max(1, len(samples) // 30)]
    noisy = [(s, D.add_gaussian_noise(im, 30, g)) for s, im in sub]
    restored = [(s, pipe.restore(im)[0]) for s, im in noisy]
    restored_q = [(s, pipe_q.restore(im)[0]) for s, im in noisy]
    m2 = ["edge_canny", "otsu", "watershed", "grabcut"]
    inter = pd.concat([seg_eval(sub, yolo_boxes, m2, "clean"),
                       seg_eval(noisy, yolo_boxes, m2, "noisy sigma=30"),
                       seg_eval(restored, yolo_boxes, m2, "restored (balanced: bilateral)"),
                       seg_eval(restored_q, yolo_boxes, m2, "restored (quality: NLM)")])
    save_table(inter.groupby(["boxes", "method"], sort=False)[["IoU", "Dice"]].mean().reset_index(),
               "exp4b_segmentation_vs_noise")

    # (c) motion segmentation
    rows = []
    vid = VIDEO_DIR / "vtest.avi"
    if vid.exists():
        for method in ("mog2", "knn", "running_avg"):
            ms_ = SG.MotionSegmenter(method)
            times, frame_show = [], None
            for i, fr in iter_video(vid, max_frames=200 if not a.quick else 60):
                with Timer() as t:
                    mask, boxes = ms_.apply(fr)
                times.append(t.ms)
                if i == (150 if not a.quick else 50):
                    frame_show = (fr, mask, boxes)
            rows.append({"source": "vtest.avi", "method": method, "ms/frame": float(np.mean(times[5:])),
                         "Precision": np.nan, "Recall": np.nan, "F1": np.nan})
            if frame_show is not None:
                fr, mask, boxes = frame_show
                vis = fr.copy()
                for b in boxes:
                    cv2.rectangle(vis, b[:2], b[2:], (0, 255, 0), 2)
                cv2.imwrite(str(FIG_DIR / f"exp4_motion_{method}.png"),
                            np.hstack([vis, cv2.cvtColor((mask * 255).astype(np.uint8), cv2.COLOR_GRAY2BGR)]))
    for name in available_caviar()[:3]:
        seq = CaviarSequence(name)
        for method in ("mog2", "running_avg"):
            ms_, acc, times = SG.MotionSegmenter(method, min_area_ratio=0.002), Accumulator(), []
            for n, fr, gt in seq:
                with Timer() as t:
                    _, boxes = ms_.apply(fr)
                times.append(t.ms)
                if n < 50:  # background model warm-up
                    continue
                m = match_detections([(b, 1.0) for b in boxes], gt, 0.3)
                acc.add(m, len(gt))
            sm = acc.summary()
            rows.append({"source": f"CAVIAR/{name}", "method": method, "ms/frame": float(np.mean(times)),
                         "Precision": sm["Precision"], "Recall": sm["Recall"], "F1": sm["F1"]})
    if rows:
        save_table(pd.DataFrame(rows), "exp4c_motion_segmentation")


if __name__ == "__main__":
    main()
