"""Technical Task 6 - Object recognition (pedestrians and vehicles).

Three interchangeable detectors with one interface `detect(img) -> [Detection]`:

  HOGDetector        Dalal & Triggs (2005) HOG + linear SVM (OpenCV's pretrained
                     people model). Classical DIP features, no GPU, no downloads.
  YOLOTinyDetector   YOLOv4-tiny (COCO, 80 classes) run through OpenCV's DNN
                     module - no PyTorch needed, ~24 MB weights, CPU real-time.
                     Detects persons AND vehicles (car, bus, truck, motorbike, bicycle).
  UltralyticsDetector (optional) YOLOv8/YOLO11 if `pip install ultralytics` is done.

YOLOv4-tiny is the default because it satisfies the 'resource constraints vs
performance' requirement: it runs on a laptop CPU / Raspberry-Pi-class edge
device with no GPU and no PyTorch.
"""
from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from .config import MODEL_DIR, SURVEILLANCE_CLASSES


_CV5_MSG = ("OpenCV {v} is installed, but OpenCV 5.x removed {what}. Install OpenCV 4.x:\n"
            "    python -m pip uninstall -y opencv-python opencv-python-headless opencv-contrib-python\n"
            "    python -m pip install \"opencv-python>=4.8,<5\"")


def _require(attr_owner, attr: str, what: str):
    if not hasattr(attr_owner, attr):
        raise ImportError(_CV5_MSG.format(v=cv2.__version__, what=what))


@dataclass
class Detection:
    box: tuple  # (x1, y1, x2, y2) in pixels
    score: float
    label: str = "person"

    @property
    def height(self):
        return self.box[3] - self.box[1]


def nms(dets, iou_thr: float = 0.45):
    """Greedy non-maximum suppression (from scratch)."""
    if not dets:
        return []
    boxes = np.array([d.box for d in dets], np.float64)
    scores = np.array([d.score for d in dets])
    order = scores.argsort()[::-1]
    keep = []
    area = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
    while order.size:
        i = order[0]
        keep.append(i)
        xx1 = np.maximum(boxes[i, 0], boxes[order[1:], 0])
        yy1 = np.maximum(boxes[i, 1], boxes[order[1:], 1])
        xx2 = np.minimum(boxes[i, 2], boxes[order[1:], 2])
        yy2 = np.minimum(boxes[i, 3], boxes[order[1:], 3])
        inter = np.clip(xx2 - xx1, 0, None) * np.clip(yy2 - yy1, 0, None)
        iou = inter / (area[i] + area[order[1:]] - inter + 1e-9)
        order = order[1:][iou < iou_thr]
    return [dets[i] for i in keep]


class HOGDetector:
    name = "HOG+SVM"

    def __init__(self, max_side: int = 480, hit_threshold: float = 0.0, scale: float = 1.05,
                 score_thr: float = 0.3):
        _require(cv2, "HOGDescriptor", "the HOG people detector")
        self.hog = cv2.HOGDescriptor()
        self.hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())
        self.max_side, self.hit_threshold, self.scale, self.score_thr = max_side, hit_threshold, scale, score_thr

    def detect(self, img):
        f = min(1.0, self.max_side / max(img.shape[:2]))
        small = cv2.resize(img, None, fx=f, fy=f) if f < 1 else img
        rects, weights = self.hog.detectMultiScale(small, hitThreshold=self.hit_threshold,
                                                   winStride=(8, 8), padding=(8, 8), scale=self.scale)
        dets = []
        for (x, y, w, h), s in zip(rects, np.ravel(weights) if len(rects) else []):
            if s < self.score_thr:
                continue
            # the default model's box includes ~10% margin around the person -> tighten
            px, py = 0.1 * w, 0.05 * h
            dets.append(Detection(((x + px) / f, (y + py) / f, (x + w - px) / f, (y + h - py) / f),
                                  float(s), "person"))
        return nms(dets, 0.4)


class YOLOTinyDetector:
    name = "YOLOv4-tiny"

    def __init__(self, input_size: int = 416, conf: float = 0.35, nms_thr: float = 0.45,
                 classes=None, model_dir=MODEL_DIR):
        cfg, weights = model_dir / "yolov4-tiny.cfg", model_dir / "yolov4-tiny.weights"
        if not weights.exists():
            raise FileNotFoundError(f"{weights} missing - run: python scripts/download_data.py --models")
        _require(cv2.dnn, "readNetFromDarknet", "the Darknet/YOLO model loader")
        net = cv2.dnn.readNetFromDarknet(str(cfg), str(weights))
        net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
        net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)
        self.model = cv2.dnn_DetectionModel(net)
        self.model.setInputParams(size=(input_size, input_size), scale=1 / 255.0, swapRB=True)
        self.conf, self.nms_thr = conf, nms_thr
        self.classes = classes if classes is not None else SURVEILLANCE_CLASSES
        self.name = f"YOLOv4-tiny@{input_size}"

    def detect(self, img):
        if img.ndim == 2:
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        cls, scores, boxes = self.model.detect(img, self.conf, self.nms_thr)
        dets = []
        for c, s, (x, y, w, h) in zip(np.ravel(cls), np.ravel(scores), boxes):
            c = int(c)
            if c in self.classes:
                dets.append(Detection((float(x), float(y), float(x + w), float(y + h)), float(s),
                                      self.classes[c]))
        return dets


class UltralyticsDetector:
    """Optional stronger detector (needs PyTorch): pip install ultralytics"""

    def __init__(self, weights: str = "yolov8n.pt", conf: float = 0.35, imgsz: int = 640):
        from ultralytics import YOLO  # noqa: lazy import
        self.model, self.conf, self.imgsz = YOLO(weights), conf, imgsz
        self.name = f"Ultralytics-{weights.split('.')[0]}"

    def detect(self, img):
        r = self.model.predict(img, conf=self.conf, imgsz=self.imgsz, verbose=False)[0]
        dets = []
        for b, s, c in zip(r.boxes.xyxy.cpu().numpy(), r.boxes.conf.cpu().numpy(),
                           r.boxes.cls.cpu().numpy().astype(int)):
            if c in SURVEILLANCE_CLASSES:
                dets.append(Detection(tuple(map(float, b)), float(s), SURVEILLANCE_CLASSES[c]))
        return dets


def make_detector(name: str = "yolo", **kw):
    name = name.lower()
    if name in ("hog", "hog+svm"):
        return HOGDetector(**kw)
    if name in ("yolo", "yolo-tiny", "yolov4-tiny"):
        return YOLOTinyDetector(**kw)
    if name.startswith("ultra"):
        return UltralyticsDetector(**kw)
    raise ValueError(name)


def draw_detections(img, dets, color_map=None):
    out = img.copy()
    color_map = color_map or {"person": (0, 200, 0)}
    for d in dets:
        x1, y1, x2, y2 = map(int, d.box)
        col = color_map.get(d.label, (0, 140, 255))
        cv2.rectangle(out, (x1, y1), (x2, y2), col, 2)
        cv2.putText(out, f"{d.label} {d.score:.2f}", (x1, max(12, y1 - 4)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, col, 1, cv2.LINE_AA)
    return out
