"""Experiment 6 - Object recognition (Technical Task 6) and module interdependence.

(a) Detector benchmark on Penn-Fudan: HOG+SVM vs YOLOv4-tiny at 3 input sizes
    (Precision, Recall, F1, AP@0.5, ms/image).
(b) Robustness: F1 on each degradation scenario, raw vs after the adaptive
    restoration of the 'balanced' and 'quality' profiles. This is the key
    evidence that the enhancement modules matter for the end task.
(c) Privacy vs accessibility: detection on anonymised frames.
(d) Vehicles: robustness on the KITTI street clip using detections on the clean
    video as pseudo ground truth (no public vehicle GT without registration).
(e) CAVIAR (real CCTV) detection if the sequences are downloaded.

Penn-Fudan does not label small / heavily occluded people, so unmatched
detections shorter than IGNORE_H px are ignored (neither TP nor FP).
"""
from common import parse_args, pick_indices, rng, save_fig, save_table, plt  # noqa: I001
import numpy as np
import pandas as pd

from surveillance import degradation as D
from surveillance import privacy as PV
from surveillance.config import VIDEO_DIR, VEHICLE_CLASSES, SURVEILLANCE_CLASSES
from surveillance.datasets import CaviarSequence, PennFudan, available_caviar, iter_video
from surveillance.metrics import Accumulator, Timer, match_detections
from surveillance.pipeline import SurveillancePipeline
from surveillance.recognition import HOGDetector, YOLOTinyDetector

IGNORE_H = 90
VEHICLE_NAMES = {SURVEILLANCE_CLASSES[c] for c in VEHICLE_CLASSES}


def evaluate(detect_fn, samples, ignore_h=IGNORE_H, iou=0.5):
    acc, times = Accumulator(), []
    for gt_boxes, img in samples:
        with Timer() as t:
            dets = detect_fn(img)
        times.append(t.ms)
        acc.add(match_detections([(d.box, d.score) for d in dets if d.label == "person"], gt_boxes, iou,
                                 ignore_h), len(gt_boxes))
    out = acc.summary()
    out["ms"] = float(np.mean(times))
    return out


def main():
    a = parse_args(__doc__, default_n=170, quick_n=8)
    ds = PennFudan()
    idx = pick_indices(len(ds), a.n)
    clean = [(ds[i].boxes, ds[i].image) for i in idx]
    g = rng(6)

    # (a) detector benchmark
    rows = []
    for det in (HOGDetector(), YOLOTinyDetector(320), YOLOTinyDetector(416), YOLOTinyDetector(608)):
        r = evaluate(det.detect, clean)
        rows.append({"detector": det.name, **r})
    try:
        from surveillance.recognition import UltralyticsDetector
        det = UltralyticsDetector()
        rows.append({"detector": det.name, **evaluate(det.detect, clean)})
    except Exception:  # noqa: BLE001 - optional dependency
        pass
    save_table(pd.DataFrame(rows), "exp6a_detector_benchmark")

    # (b) robustness with / without adaptive restoration
    n_rb = min(len(idx), 30 if not a.quick else 5)
    sub = [clean[k] for k in pick_indices(len(clean), n_rb)]
    yolo = YOLOTinyDetector(416)
    pipes = {"+ restore (balanced)": SurveillancePipeline("balanced", detector="yolo", privacy="none"),
             "+ restore (quality)": SurveillancePipeline("quality", detector="yolo", privacy="none",
                                                         illumination="always")}
    for p in pipes.values():
        p.detector = yolo  # same detector everywhere -> only restoration differs
    rows = []
    for sc in ["clean"] + list(D.SCENARIOS):
        deg = sub if sc == "clean" else [(b, D.SCENARIOS[sc](im, g)) for b, im in sub]
        r0 = evaluate(yolo.detect, deg)
        row = {"scenario": sc, "raw F1": r0["F1"], "raw AP50": r0["AP50"]}
        for name, p in pipes.items():
            if name.endswith("(quality)"):
                p.illumination = "always" if sc == "uneven_light" else "auto"
            r = evaluate(lambda im: p.process(im).detections, deg)
            row[f"{name} F1"], row[f"{name} AP50"], row[f"{name} ms"] = r["F1"], r["AP50"], r["ms"]
        rows.append(row)
    df = pd.DataFrame(rows)
    save_table(df, "exp6b_robustness")
    fig, ax = plt.subplots(figsize=(7.5, 2.8))
    x = np.arange(len(df))
    for k, (col, lab) in enumerate((("raw F1", "degraded"), ("+ restore (balanced) F1", "restored (balanced)"),
                                    ("+ restore (quality) F1", "restored (quality)"))):
        ax.bar(x + (k - 1) * 0.27, df[col], 0.27, label=lab)
    ax.set_xticks(x); ax.set_xticklabels(df.scenario, rotation=30, ha="right", fontsize=7)
    ax.set_ylabel("pedestrian F1"); ax.set_ylim(0, 1); ax.legend(fontsize=7, ncol=3)
    save_fig(fig, "exp6_robustness")

    # (c) privacy vs accessibility
    rows = []
    for label, fn in (("original", lambda im, d: im),
                      ("heads pixelated", lambda im, d: PV.anonymize(im, d, "head", "pixelate")),
                      ("heads blurred", lambda im, d: PV.anonymize(im, d, "head", "blur")),
                      ("whole body pixelated", lambda im, d: PV.anonymize(im, d, "body", "pixelate"))):
        prot = [(b, fn(im, yolo.detect(im))) for b, im in sub]
        rows.append({"published frame": label, **evaluate(yolo.detect, prot)})
    save_table(pd.DataFrame(rows), "exp6c_privacy_vs_detection")

    # (d) vehicles, pseudo ground truth from the clean video
    vid = VIDEO_DIR / "kitti-inference-vid.mp4"
    if vid.exists():
        frames = [f for i, f in iter_video(vid) if i % 10 == 0][: (30 if not a.quick else 5)]
        ref = YOLOTinyDetector(608, conf=0.5)
        pseudo = [[d.box for d in ref.detect(f) if d.label in VEHICLE_NAMES] for f in frames]
        pipe = SurveillancePipeline("balanced", detector="yolo", privacy="none")
        pipe.detector = yolo
        rows = []
        for sc in ("clean", "gaussian_s30", "saltpepper_5", "periodic", "low_light", "night_compound"):
            deg = frames if sc == "clean" else [D.SCENARIOS[sc](f, g) for f in frames]
            row = {"scenario": sc}
            for name, fn in (("raw", yolo.detect), ("restored", lambda im: pipe.process(im).detections)):
                acc = Accumulator()
                for gt, im in zip(pseudo, deg):
                    dets = [(d.box, d.score) for d in fn(im) if d.label in VEHICLE_NAMES]
                    acc.add(match_detections(dets, gt, 0.5), len(gt))
                row[f"{name} F1"] = acc.summary()["F1"]
            rows.append(row)
        save_table(pd.DataFrame(rows), "exp6d_vehicles_pseudo_gt")

    # (e) CAVIAR real CCTV
    rows = []
    for name in available_caviar():
        seq = CaviarSequence(name)
        samples = [(gt, fr) for k, (n, fr, gt) in enumerate(seq) if k % 10 == 0 and gt]
        for det in (HOGDetector(), YOLOTinyDetector(416), YOLOTinyDetector(608)):
            rows.append({"sequence": name, "detector": det.name, **evaluate(det.detect, samples, 0, 0.4)})
    if rows:
        save_table(pd.DataFrame(rows), "exp6e_caviar_detection")


if __name__ == "__main__":
    main()
